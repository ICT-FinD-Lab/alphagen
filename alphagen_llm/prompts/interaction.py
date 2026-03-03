from typing import List, Optional, Callable, Sequence, Tuple, Dict, Any
from abc import ABCMeta, abstractmethod
from logging import Logger
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin

from openai import BadRequestError
import numpy as np

from alphagen.data.calculator import AlphaCalculator
from alphagen.data.exception import InvalidExpressionException
from alphagen.data.expression import Expression
from alphagen.data.parser import ExpressionParser
from alphagen.data.pool_update import PoolUpdate, AddRemoveAlphas
from alphagen.models.linear_alpha_pool import LinearAlphaPool

from .common import alpha_phrase, safe_parse_list
from ..client.base import ChatClient
from ..llm_alpha_stats_utils import record_llm_alpha


class InterativeSession(metaclass=ABCMeta):
    def __init__(
        self,
        parser: ExpressionParser,
        client: ChatClient,
        pool_factory: Callable[[List[Expression]], LinearAlphaPool],
        calculator_train: AlphaCalculator,
        calculators_test: Optional[Sequence[AlphaCalculator]] = None
    ):
        self._parser = parser
        self._client = client
        self._calc_train = calculator_train
        self._calcs_test = calculators_test or []
        self._logger = client.logger
        self._pool_factory = pool_factory

    @property
    def logger(self) -> Logger: return self._logger

    @property
    def client(self) -> ChatClient: return self._client

    def run(self, exprs: Optional[List[Expression]] = None, n_updates: int = 20) -> LinearAlphaPool:
        exprs = exprs or []
        pool = self._pool_factory(exprs)
        self._initialize(pool, exprs)
        self.update_pool(pool, n_updates)
        return pool

    def update_pool(self, pool: LinearAlphaPool, n_updates: int = 20) -> None:
        try:
            if pool.size == 0:
                self._initialize(pool, [])
            for i in range(n_updates):
                self._client.log_message(("script", f"Starting iteration {i + 1}"))
                if not self._update(i, pool):
                    break
        except BadRequestError as e:
            self._client.log_message(("script", f"Interaction ended because: {e.message}"))
            raise
        except InvalidExpressionException as e:
            self._client.log_message(("script", f"Interaction ended due to invalid expression: {e}"))
            raise

    def _initialize(self, pool: LinearAlphaPool, exprs: List[Expression]) -> None:
        "Provide the chat client with initial instructions and expressions."

    @abstractmethod
    def _update(self, iter: int, pool: LinearAlphaPool) -> bool:
        "The loop body, return True if the loop should continue."


@dataclass
class DefaultReport(DataClassJsonMixin):
    pool_state: List[Tuple[str, float]]
    train_ic: float
    train_ric: float
    test_ics: List[float]
    test_rics: List[float]


class DefaultInteraction(InterativeSession):
    def __init__(
        self,
        parser: ExpressionParser,
        client: ChatClient,
        pool_factory: Callable[[List[Expression]], LinearAlphaPool],
        calculator_train: AlphaCalculator,
        calculators_test: Optional[Sequence[AlphaCalculator]] = None,
        replace_k: int = 3,
        force_remove: bool = False,
        forgetful: bool = False,
        no_actual_weights: bool = False,
        also_report_history: bool = False,
        on_pool_update: Optional[Callable[[DefaultReport, int], None]] = None,
        llm_alpha_stats: Optional[Dict[str, Any]] = None
    ):
        """
        no_actual_weights: Do not output the actual weights of the alphas in the prompt, just sort them.
        llm_alpha_stats: Optional dictionary to track valid/invalid alphas generated
        """
        super().__init__(parser, client, pool_factory, calculator_train, calculators_test)
        self._replace_k = replace_k
        self._forgetful = forgetful
        self._force_remove = force_remove
        self._no_actual_weights = no_actual_weights
        self._also_report_history = also_report_history
        self._on_pool_update = on_pool_update or (lambda r, i: None)
        self._reports: List[DefaultReport] = []
        self._llm_alpha_stats = llm_alpha_stats

    @property
    def reports(self) -> List[DefaultReport]: return self._reports

    def reset_reports(self) -> None: self._reports.clear()

    def _initialize(self, pool: LinearAlphaPool, exprs: List[Expression]) -> None:
        if len(exprs) != 0:
            return
        p = (f"Please generate {alpha_phrase(pool.capacity)} that you think would be "
             "indicative of future stock price trend. Each alpha should be "
             "on its own line without numbering. Please do not output anything else.")
        self._parse_and_add(p, pool)
        report = self._evaluate_pool(pool)
        self._reports.append(report)
        self._on_pool_update(report, 0)
        self._client.reset()    # Reset here to unify later steps

    def _update(self, iter: int, pool: LinearAlphaPool) -> bool:
        PREFIX0 = ("Here are a set of formulaic alphas generated by an automated system. {}"
                   "These alphas are combined with a linear model into the final predictive signal. "
                   "The alphas and the combined signal are tested on real-world dataset, ")
        PREFIX1A = ("and the alphas are sorted based on their weights in the linear model, the most significant ones "
                    "(larger absolute weights) come at the top, and the insignificant ones go to the bottom.\n")
        PREFIX1B = ("and the IC/Rank IC metrics of them, together with the alphas' weights in the "
                    "linear model is reported as follows:\n")
        PREFIX2 = ("The updated alpha set is tested again on the dataset:\n")
        PREFIX_UPDATE = ("The update history of the alpha set, together with how the edits influenced "
                         "the IC performance of the set, is listed below:\n")
        REPLACE = ("\nAccording to the result, please generate {}, not similar to the insignificant ones. "
                   "The most insignificant alphas will be replaced with the new ones to potentially boost the performance. "
                   "Again, one on each line without numbering and metrics, and do not output anything else.")
        SIG_THRES = 1e-4

        if self._forgetful:
            self._client.reset()
        history = ""
        if self._also_report_history:            
            desc = "".join(_describe_update(h) for h in pool.update_history)
            history = f"{PREFIX_UPDATE}{desc}"
        prefix0 = PREFIX0.format(history)
        prefix1 = PREFIX1A if self._no_actual_weights else PREFIX1B
        prefix = (prefix0 + prefix1) if self._forgetful or iter == 0 else PREFIX2
        report_str, report = self._generate_report(pool)
        abs_weights = np.abs(pool.weights)
        insig_count = np.count_nonzero(abs_weights <= SIG_THRES)
        weight_rank = abs_weights.argsort().argsort()
        replaced_count = max(insig_count, self._replace_k)
        removed_idx = []
        if self._force_remove:
            removed_idx = [i for i, r in enumerate(weight_rank) if r < replaced_count]
        replace_prompt = REPLACE.format(alpha_phrase(replaced_count, "more"))
        exprs = self._chat_and_parse(prefix + report_str + replace_prompt)
        if len(exprs) == 0:
            if self._forgetful:    # If using "forgetful" strategy, just reset the client and try again
                return self._update(iter, pool)
            retry = ("Your answer seems to be formatted incorrectly, or that all the alphas you generated are invalid. "
                     "Please follow the instructions carefully, output an alpha per line *without numbering and metrics and anything else*! "
                     f"Try again and generate {alpha_phrase(replaced_count)} again, following the instructions.")
            exprs = self._chat_and_parse(retry)
            if len(exprs) == 0:
                return False
           
        pool.bulk_edit(removed_idx, exprs)
        self._reports.append(report)
        self._on_pool_update(report, iter + 1)
        return True
    
    def _chat_and_parse(self, prompt: str) -> List[Expression]:
        lines = self._client.chat_complete(prompt)
        exprs, invalid = safe_parse_list(lines.split('\n'), self._parser)
        if len(invalid) != 0:
            self._client.log_message(("script", f"Invalid expressions: {invalid}"))
        # Track valid/invalid alphas if stats dict is provided
        if self._llm_alpha_stats is not None:
            for expr in exprs:
                record_llm_alpha(self._llm_alpha_stats, 'original', valid=True, fixed=False, expr=expr)
            for expr in invalid:
                record_llm_alpha(self._llm_alpha_stats, 'original', valid=False, fixed=False, expr=expr)
        return exprs

    def _parse_and_add(self, prompt: str, pool: LinearAlphaPool) -> bool:
        exprs = self._chat_and_parse(prompt)
        pool.force_load_exprs(exprs)
        return len(exprs) != 0

    def _evaluate_pool(self, pool: LinearAlphaPool) -> DefaultReport:
        train = pool.test_ensemble(self._calc_train)
        tests = [pool.test_ensemble(c) for c in self._calcs_test]
        return DefaultReport(
            pool_state=[(str(e), w) for e, w in zip(pool.exprs[:pool.size], pool.weights)],
            train_ic=train[0], train_ric=train[1],
            test_ics=[t[0] for t in tests], test_rics=[t[1] for t in tests]
        )

    def _generate_report(self, pool: LinearAlphaPool) -> Tuple[str, DefaultReport]:
        SIGNIFICANT = "  This is a good alpha!"
        INSIGNIFICANT = "  This alpha doesn't contribute much."
        SIG_THRES = 1e-4

        def format_ic(ic: float, ric: float) -> str:
            return f"IC = {ic:.4f}, Rank IC = {ric:.4f}"

        performances = []
        exprs = list(zip(pool.weights, pool.single_ics[:pool.size], pool.exprs[:pool.size]))
        exprs.sort(key=lambda t: abs(t[0]), reverse=True)
        insig_n = sum(1 for w, _, _ in exprs if abs(w) <= SIG_THRES)
        for w, i, e in exprs:
            sig_phrase = ""
            if insig_n != 0 and not self._no_actual_weights:
                sig = abs(w) > SIG_THRES
                sig_phrase = SIGNIFICANT if sig else INSIGNIFICANT
            ic_weight = ""
            if not self._no_actual_weights:
                ic_weight = f": IC = {i:.4f}, weight = {w:.4f}{sig_phrase}"
            performances.append(f"{str(e).replace('$', '')}{ic_weight}")
        report = self._evaluate_pool(pool)

        train_res = format_ic(report.train_ic, report.train_ric)
        self._client.log_message(("script", f"Ensemble on train: {train_res}"))
        for i, (ic, ric) in enumerate(zip(report.test_ics, report.test_rics)):
            self._client.log_message(("script", f"Ensemble on test #{i + 1}: {format_ic(ic, ric)}"))
        performances.append(f"Ensemble: {train_res}")
        return "\n".join(performances), report


def _describe_update(u: PoolUpdate) -> str:
    if isinstance(u, AddRemoveAlphas) and len(u.added_exprs) == 0 and len(u.removed_idx) == 0:
        return ""
    return u.describe() + '\n'
