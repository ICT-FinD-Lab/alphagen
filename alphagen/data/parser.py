import re
from typing import Type, List, Dict, Set, Union, Optional, cast, overload, Literal
from .expression import *
from ..utils.misc import find_last_if


_PATTERN = re.compile(r'([+-]?[\d.]+|\W|\w+)')
_NUMERIC = re.compile(r'[+-]?[\d.]+')
_StackItem = Union[List[Type[Operator]], Expression]
_OpMap = Dict[str, List[Type[Operator]]]
_DTLike = Union[float, Constant, DeltaTime]


class ExpressionParsingError(Exception):
    pass


class ExpressionParser:
    def __init__(
        self,
        operators: List[Type[Operator]],
        ignore_case: bool = False,
        time_deltas_need_suffix: bool = False,
        non_positive_time_deltas_allowed: bool = True,
        feature_need_dollar_sign: bool = False,
        additional_operator_mapping: Optional[_OpMap] = None
    ):
        self._ignore_case = ignore_case
        self._allow_np_dt = non_positive_time_deltas_allowed
        self._suffix_needed = time_deltas_need_suffix
        self._dollar_needed = feature_need_dollar_sign
        self._features = {f.name.lower(): f for f in FeatureType}
        self._operators: _OpMap = {op.__name__: [op] for op in operators}
        if additional_operator_mapping is not None:
            self._merge_op_mapping(additional_operator_mapping)
        if ignore_case:
            self._operators = {k.lower(): v for k, v in self._operators.items()}
        self._stack: List[_StackItem] = []
        self._tokens: List[str] = []

    def parse(self, expr: str) -> Expression:
        self._stack = []
        self._tokens = [t for t in _PATTERN.findall(expr) if not t.isspace()]
        self._tokens.reverse()
        while len(self._tokens) > 0:
            self._stack.append(self._get_next_item())
            self._process_punctuation()
        if len(self._stack) != 1:
            raise ExpressionParsingError("Multiple items remain in the stack")
        if len(self._stack) == 0:
            raise ExpressionParsingError("Nothing was parsed")
        if isinstance(self._stack[0], Expression):
            return self._stack[0]
        raise ExpressionParsingError(f"{self._stack[0]} is not a valid expression")

    def _merge_op_mapping(self, map: _OpMap) -> None:
        for name, ops in map.items():
            if (old_ops := self._operators.get(name)) is not None:
                self._operators[name] = list(dict.fromkeys(old_ops + ops))
            else:
                self._operators[name] = ops

    def _get_next_item(self) -> _StackItem:
        top = self._pop_token()
        
        # Handle $-prefixed features when feature_need_dollar_sign=False
        # This allows parsing both "open" and "$open" when dollar signs are optional
        if not self._dollar_needed and top.startswith('$'):
            feature_name = top[1:]  # Strip the $
            if (feature := self._features.get(feature_name)) is not None:
                return Feature(feature)
            else:
                raise ExpressionParsingError(f"Can't find the feature {feature_name}")
        
        if top == '$':      # Feature next
            top = self._pop_token()
            if (feature := self._features.get(top)) is None:
                raise ExpressionParsingError(f"Can't find the feature {top}")
            return Feature(feature)
        elif self._tokens_eq(top, "Constant"):
            if self._pop_token() != '(':
                raise ExpressionParsingError("\"Constant\" should be followed by a left parenthesis")
            value = self._to_float(self._pop_token())
            if self._pop_token() != ')':
                raise ExpressionParsingError("\"Constant\" should be closed by a right parenthesis")
            return Constant(value)
        elif _NUMERIC.fullmatch(top) is not None:
            value = self._to_float(top)
            if self._peek_token() == 'd':
                self._pop_token()
                return self._as_delta_time(value)
            else:
                return Constant(value)
        else:
            if not self._dollar_needed and (feature := self._features.get(top)) is not None:
                return Feature(feature)
            elif (ops := self._operators.get(top)) is not None:
                return ops
            else:
                raise ExpressionParsingError(f"Cannot find the operator/feature name {top}")

    def _process_punctuation(self) -> None:
        if len(self._tokens) == 0:
            return
        top = self._pop_token()
        stack_top_is_ops = len(self._stack) != 0 and not isinstance(self._stack[-1], Expression)
        if (top == '(') != stack_top_is_ops:
            raise ExpressionParsingError("A left parenthesis should follow an operator name")
        if top == '(' or top == ',':
            return
        elif top == ')':
            self._build_one_subexpr()       # Pop an operator with its operands
            self._process_punctuation()     # There might be consecutive right parens
        else:
            raise ExpressionParsingError(f"Unexpected token {top}")

    def _build_one_subexpr(self) -> None:
        if (op_idx := find_last_if(self._stack, lambda item: isinstance(item, list))) == -1:
            raise ExpressionParsingError("Unmatched right parenthesis")
        ops = cast(List[Type[Operator]], self._stack[op_idx])
        operands = self._stack[op_idx + 1:]
        self._stack = self._stack[:op_idx]
        if any(not isinstance(item, Expression) for item in operands):
            raise ExpressionParsingError("An operator name cannot be used as an operand")
        operands = cast(List[Expression], operands)
        dt_operands = operands
        if (not self._suffix_needed and
                isinstance(operands[-1], Constant) and
                (dt := self._as_delta_time(operands[-1], noexcept=True)) is not None):
            dt_operands = operands.copy()
            dt_operands[-1] = dt
        msgs: Set[str] = set()
        for op in ops:
            used_operands = operands
            if issubclass(op, (RollingOperator, PairRollingOperator)):
                used_operands = dt_operands
            if (msg := op.validate_parameters(*used_operands)).is_none:
                self._stack.append(op(*used_operands))      # type: ignore
                return
            else:
                msgs.add(msg.value_or(""))
        raise ExpressionParsingError("; ".join(msgs))

    def _tokens_eq(self, lhs: str, rhs: str) -> bool:
        if self._ignore_case:
            return lhs.lower() == rhs.lower()
        else:
            return lhs == rhs

    @classmethod
    def _to_float(cls, token: str) -> float:
        try:
            return float(token)
        except:
            raise ExpressionParsingError(f"{token} can't be converted to float")

    def _pop_token(self) -> str:
        if len(self._tokens) == 0:
            raise ExpressionParsingError("No more tokens left")
        top = self._tokens.pop()
        return top.lower() if self._ignore_case else top

    def _peek_token(self) -> Optional[str]:
        return self._tokens[-1] if len(self._tokens) != 0 else None
    
    @overload
    def _as_delta_time(self, value: _DTLike, noexcept: Literal[False] = False) -> DeltaTime: ...
    @overload
    def _as_delta_time(self, value: _DTLike, noexcept: Literal[True]) -> Optional[DeltaTime]: ...
    @overload
    def _as_delta_time(self, value: _DTLike, noexcept: bool) -> Optional[DeltaTime]: ...

    def _as_delta_time(self, value: _DTLike, noexcept: bool = False):
        def maybe_raise(message: str) -> None:
            if not noexcept:
                raise ExpressionParsingError(message)
        
        if isinstance(value, DeltaTime):
            return value
        if isinstance(value, Constant):
            value = value.value
        if not float(value).is_integer():
            maybe_raise(f"A DeltaTime should be integral, but {value} is not")
            return
        if int(value) <= 0 and not self._allow_np_dt:
            maybe_raise(f"A DeltaTime should refer to a positive time difference, but got {int(value)}d")
            return
        return DeltaTime(int(value))


def parse_expression(expr: str) -> Expression:
    "Parse an expression using the default expression parser."
    return ExpressionParser(Operators).parse(expr)
