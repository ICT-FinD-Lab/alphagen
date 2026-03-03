import math
from itertools import count
from typing import List, Optional, Tuple, Iterable, Dict, Any, Set
from abc import ABCMeta, abstractmethod

import numpy as np
import torch

from .alpha_pool import AlphaPoolBase
from ..data.calculator import AlphaCalculator, TensorAlphaCalculator
from ..data.expression import Expression, OutOfDataRangeError
from ..data.pool_update import PoolUpdate, AddRemoveAlphas
from ..utils.correlation import batch_pearsonr


class LinearAlphaPool(AlphaPoolBase, metaclass=ABCMeta):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__(capacity, calculator, device)
        self.exprs: List[Optional[Expression]] = [None for _ in range(capacity + 1)]
        self.single_ics: np.ndarray = np.zeros(capacity + 1)
        self._weights: np.ndarray = np.zeros(capacity + 1)
        self._mutual_ics: np.ndarray = np.identity(capacity + 1)
        self._extra_info = [None for _ in range(capacity + 1)]
        self._ic_lower_bound = -1. if ic_lower_bound is None else ic_lower_bound
        self.best_obj = -1.
        self.update_history: List[PoolUpdate] = []
        self._failure_cache: Set[str] = set()

    @property
    def weights(self) -> np.ndarray:
        "Get the weights of the linear model as a numpy array of shape (size,)."
        return self._weights[:self.size]

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        "Set the weights of the linear model with a numpy array of shape (size,)."
        assert value.shape == (self.size,), f"Invalid weights shape: {value.shape}"
        self._weights[:self.size] = value

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "exprs": self.exprs[:self.size],
            "ics_ret": list(self.single_ics[:self.size]),
            "weights": list(self.weights),
            "best_ic_ret": self.best_ic_ret
        }

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "exprs": [str(expr) for expr in self.exprs[:self.size]],
            "weights": list(self.weights)
        }

    def try_new_expr(self, expr: Expression) -> bool:
        """Try to add a new expression to the pool"""
        try:
            ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=0.99)
        except OutOfDataRangeError:
            print(f"[Warning] Expression requires data outside available range, skipping")
            return False
        except Exception as e:
            print(f"[Warning] Failed to evaluate expression: {type(e).__name__}: {e}")
            return False

        if ic_ret is None or ic_mut is None or np.isnan(ic_ret) or np.isnan(ic_mut).any():
            return False
        if str(expr) in self._failure_cache:
            return self.best_obj

        self.eval_cnt += 1
        old_pool: List[Expression] = self.exprs[:self.size]     # type: ignore
        self._add_factor(expr, ic_ret, ic_mut)
        if self.size > 1:
            new_weights = self.optimize()
            worst_idx = None
            if self.size > self.capacity:   # Need to remove one
                worst_idx = int(np.argmin(np.abs(new_weights)))
                # The one added this time is the worst, revert the changes
                if worst_idx == self.capacity:
                    self._pop(worst_idx)
                    self._failure_cache.add(str(expr))
                    return self.best_obj
            removed_idx = [worst_idx] if worst_idx is not None else []
            self.weights = new_weights
            self.update_history.append(AddRemoveAlphas(
                added_exprs=[expr],
                removed_idx=removed_idx,
                old_pool=old_pool,
                old_pool_ic=self.best_ic_ret,
                new_pool_ic=ic_ret
            ))
            if worst_idx is not None:
                self._pop(worst_idx)
        else:
            self.update_history.append(AddRemoveAlphas(
                added_exprs=[expr],
                removed_idx=[],
                old_pool=[],
                old_pool_ic=0.,
                new_pool_ic=ic_ret
            ))

        self._failure_cache = set()
        new_ic_ret, new_obj = self.calculate_ic_and_objective()
        self._maybe_update_best(new_ic_ret, new_obj)
        return new_obj

    def force_load_exprs(self, exprs: List[Expression], weights: Optional[List[float]] = None) -> None:
        self._failure_cache = set()
        old_ic = self.evaluate_ensemble()
        old_pool: List[Expression] = self.exprs[:self.size] # type: ignore
        added = []
        for expr in exprs:
            if self.size >= self.capacity:
                break
            try:
                ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=None)
            except (OutOfDataRangeError, TypeError):
                continue
            assert ic_ret is not None and ic_mut is not None
            self._add_factor(expr, ic_ret, ic_mut)
            added.append(expr)
            assert self.size <= self.capacity
        if weights is not None:
            if len(weights) != self.size:
                raise ValueError(f"Invalid weights length: got {len(weights)}, expected {self.size}")
            self.weights = np.array(weights)
        else:
            self.weights = self.optimize()
        new_ic, new_obj = self.calculate_ic_and_objective()
        self._maybe_update_best(new_ic, new_obj)
        self.update_history.append(AddRemoveAlphas(
            added_exprs=added,
            removed_idx=[],
            old_pool=old_pool,
            old_pool_ic=old_ic,
            new_pool_ic=new_ic
        ))

    def calculate_ic_and_objective(self) -> Tuple[float, float]:
        ic = self.evaluate_ensemble()
        obj = self._calc_main_objective()
        if obj is None:
            obj = ic
        return ic, obj

    def _calc_main_objective(self) -> Optional[float]:
        "Get the main optimization objective, return None for the default (ensemble IC)."

    def _maybe_update_best(self, ic: float, obj: float) -> bool:
        if obj <= self.best_obj:
            return False
        self.best_obj = obj
        self.best_ic_ret = ic
        return True

    @abstractmethod
    def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:
        "Optimize the weights of the linear model and return the new weights as a numpy array."

    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float]:
        return calculator.calc_pool_all_ret(self.exprs[:self.size], self.weights)      # type: ignore

    def evaluate_ensemble(self) -> float:
        if self.size == 0:
            return 0.
        return self.calculator.calc_pool_IC_ret(self.exprs[:self.size], self.weights)  # type: ignore

    @property
    def _under_thres_alpha(self) -> bool:
        if self._ic_lower_bound is None or self.size > 1:
            return False
        return self.size == 0 or abs(self.single_ics[0]) < self._ic_lower_bound

    def _calc_ics(
        self,
        expr: Expression,
        ic_mut_threshold: Optional[float] = None
    ) -> Tuple[float, Optional[List[float]]]:
        single_ic = self.calculator.calc_single_IC_ret(expr)
        if not self._under_thres_alpha and single_ic < self._ic_lower_bound:
            return single_ic, None

        mutual_ics = []
        for i in range(self.size):
            mutual_ic = self.calculator.calc_mutual_IC(expr, self.exprs[i])     # type: ignore
            if ic_mut_threshold is not None and mutual_ic > ic_mut_threshold:
                return single_ic, None
            mutual_ics.append(mutual_ic)

        return single_ic, mutual_ics
    
    def _get_extra_info(self, expr: Expression) -> Any:
        "Override this method to save extra data for a newly added expression."

    def _add_factor(
        self,
        expr: Expression,
        ic_ret: float,
        ic_mut: List[float]
    ):
        if self._under_thres_alpha and self.size == 1:
            self._pop()
        n = self.size
        self.exprs[n] = expr
        self.single_ics[n] = ic_ret
        for i in range(n):
            self._mutual_ics[i][n] = self._mutual_ics[n][i] = ic_mut[i]
        self._extra_info[n] = self._get_extra_info(expr)
        new_weight = max(ic_ret, 0.01) if n == 0 else self.weights.mean()
        self._weights[n] = new_weight   # Assign an initial weight
        self.size += 1

    def _pop(self, index_hint: Optional[int] = None) -> None:
        if self.size <= self.capacity:
            return
        index = int(np.argmin(np.abs(self.weights))) if index_hint is None else index_hint
        self._swap_idx(index, self.capacity)
        self.size = self.capacity

    def most_significant_indices(self, k: int) -> List[int]:
        if self.size == 0:
            return []
        ranks = (-np.abs(self.weights)).argsort().argsort()
        return [i for i in range(self.size) if ranks[i] < k]

    def leave_only(self, indices: Iterable[int]) -> None:
        "Leaves only the alphas at the given indices intact, and removes all others."
        self._failure_cache = set()
        indices = sorted(indices)
        for i, j in enumerate(indices):
            self._swap_idx(i, j)
        self.size = len(indices)

    def bulk_edit(self, removed_indices: Iterable[int], added_exprs: List[Expression]) -> None:
        self._failure_cache = set()
        old_ic = self.evaluate_ensemble()
        old_pool: List[Expression] = self.exprs[:self.size] # type: ignore
        removed_indices = set(removed_indices)
        remain = [i for i in range(self.size) if i not in removed_indices]
        old_exprs = {id(self.exprs[i]): i for i in range(self.size)}
        old_update_history_count = len(self.update_history)
        self.leave_only(remain)
        for e in added_exprs:
            self.try_new_expr(e)
        self.update_history = self.update_history[:old_update_history_count]
        new_exprs = {id(e): e for e in self.exprs[:self.size]}
        added_exprs = [e for e in self.exprs[:self.size] if id(e) not in old_exprs] # type: ignore
        removed_indices = list(sorted(i for eid, i in old_exprs.items() if eid not in new_exprs))
        new_ic, new_obj = self.calculate_ic_and_objective()
        self._maybe_update_best(new_ic, new_obj)
        self.update_history.append(AddRemoveAlphas(
            added_exprs=added_exprs,
            removed_idx=removed_indices,
            old_pool=old_pool,
            old_pool_ic=old_ic,
            new_pool_ic=new_ic
        ))

    def _swap_idx(self, i: int, j: int) -> None:
        if i == j:
            return
        
        def swap_in_list(lst, i: int, j: int) -> None:
            lst[i], lst[j] = lst[j], lst[i]

        swap_in_list(self.exprs, i, j)
        swap_in_list(self.single_ics, i, j)
        self._mutual_ics[:, [i, j]] = self._mutual_ics[:, [j, i]]
        self._mutual_ics[[i, j], :] = self._mutual_ics[[j, i], :]
        swap_in_list(self._weights, i, j)
        swap_in_list(self._extra_info, i, j)


class MseAlphaPool(LinearAlphaPool):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__(capacity, calculator, ic_lower_bound, device)
        self._l1_alpha = l1_alpha

    def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:
        alpha = self._l1_alpha
        if math.isclose(alpha, 0.):     # No L1 regularization, use the faster least-squares method
            return self._optimize_lstsq()
            
        ics_ret = torch.tensor(self.single_ics[:self.size], device=self.device)
        ics_mut = torch.tensor(self._mutual_ics[:self.size, :self.size], device=self.device)
        weights = torch.tensor(self.weights, device=self.device, requires_grad=True)
        optim = torch.optim.Adam([weights], lr=lr)
    
        loss_ic_min = float("inf")
        best_weights = weights
        tolerance_count = 0
        for step in count():
            ret_ic_sum = (weights * ics_ret).sum()
            mut_ic_sum = (torch.outer(weights, weights) * ics_mut).sum()
            loss_ic = mut_ic_sum - 2 * ret_ic_sum + 1
            loss_ic_curr = loss_ic.item()
    
            loss_l1 = torch.norm(weights, p=1)  # type: ignore
            loss = loss_ic + alpha * loss_l1
    
            optim.zero_grad()
            loss.backward()
            optim.step()
    
            if loss_ic_min - loss_ic_curr > 1e-6:
                tolerance_count = 0
            else:
                tolerance_count += 1
    
            if loss_ic_curr < loss_ic_min:
                best_weights = weights
                loss_ic_min = loss_ic_curr
    
            if tolerance_count >= tolerance or step >= max_steps:
                break
    
        return best_weights.cpu().detach().numpy()
    
    def _optimize_lstsq(self) -> np.ndarray:
        try:
            return np.linalg.lstsq(self._mutual_ics[:self.size, :self.size],self.single_ics[:self.size])[0]
        except (np.linalg.LinAlgError, ValueError):
            return self.weights


# Note: Currently the weights are only updated when the new IC is higher.
# It might be better to update the weights according to the actual objective,
# in this case the ICIR or the LCB of the IC.

class MeanStdAlphaPool(LinearAlphaPool):
    def __init__(
        self,
        capacity: int,
        calculator: TensorAlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        lcb_beta: Optional[float] = None,
        device: torch.device = torch.device("cpu")
    ):
        """
        l1_alpha: the L1 regularization coefficient.
        lcb_beta: for optimizing the lower-confidence-bound: LCB = mean - beta * std, \
                  when this is None, optimize ICIR (mean / std) instead.
        """
        super().__init__(capacity, calculator, ic_lower_bound, device)
        self.calculator: TensorAlphaCalculator
        self._l1_alpha = l1_alpha
        self._lcb_beta = lcb_beta

    def _get_extra_info(self, expr: Expression) -> Any:
        return self.calculator.evaluate_alpha(expr)
    
    def _calc_main_objective(self) -> float:
        alpha_values = torch.stack(self._extra_info[:self.size])    # type: ignore | shape: n * days * stocks
        weights = torch.tensor(self.weights, device=self.device)
        return self._calc_obj_impl(alpha_values, weights).item()
    
    def _calc_obj_impl(self, alpha_values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        target_value = self.calculator.target
        weighted = (weights[:, None, None] * alpha_values).sum(dim=0)
        ics = batch_pearsonr(weighted, target_value)
        mean, std = ics.mean(), ics.std()
        if self._lcb_beta is not None:
            return mean - self._lcb_beta * std
        else:
            return mean / std

    def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:
        alpha_values = torch.stack(self._extra_info[:self.size])    # type: ignore | shape: n * days * stocks
        weights = torch.tensor(self.weights, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([weights], lr=lr)
    
        min_loss = float("inf")
        best_weights = weights
        tol_count = 0
        for step in count():
            obj = self._calc_obj_impl(alpha_values, weights)
            loss_l1 = torch.norm(weights, p=1)
            loss = self._l1_alpha * loss_l1 - obj   # Maximize the objective
            curr_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if min_loss - curr_loss > 1e-6:
                tol_count = 0
            else:
                tol_count += 1
    
            if curr_loss < min_loss:
                best_weights = weights
                min_loss = curr_loss
    
            if tol_count >= tolerance or step >= max_steps:
                break
    
        return best_weights.cpu().detach().numpy()
