from enum import IntEnum
from typing import Optional, List, Tuple
from num2words import num2words
from alphagen.data.expression import Expression
from alphagen.data.parser import ExpressionParser


class MetricDescriptionMode(IntEnum):
    NOT_INCLUDED = 0    # Description of this metric is not included in the prompt.
    INCLUDED = 1        # Description of this metric is included in the prompt.
    SORTED_BY = 2       # Description is included, and the alphas will be sorted according to this metric.


def alpha_word(n: int) -> str: return "alpha" if n == 1 else "alphas"


def alpha_phrase(n: int, adjective: Optional[str] = None) -> str:
    n_word = str(n) if n > 10 else num2words(n)
    adjective = f" {adjective}" if adjective is not None else ""
    return f"{n_word}{adjective} {alpha_word(n)}"


def safe_parse(parser: ExpressionParser, expr_str: str) -> Tuple[Optional[Expression], Optional[str]]:
    """Parse expression and return (expr, error_msg).
    
    Returns:
        (Expression, None) on success
        (None, error_message) on failure
    """
    try:
        # prevent error
        clean_expr = expr_str.replace('$', '')
        return parser.parse(clean_expr), None  # returns (expression, None)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"PARSING FAILED: {expr_str}")
        print(f"  Error: {error_msg}")
        return None, error_msg


def safe_parse_list(lines: List[str], parser: ExpressionParser) -> Tuple[List[Expression], List[Tuple[str, str]]]:
    """Parse list of expressions and return (valid_exprs, invalid_with_errors).
    
    Returns:
        valid_exprs: List of successfully parsed Expression objects
        invalid_with_errors: List of (expr_str, error_message) tuples for failed parses
    """
    parsed, invalid = [], []
    for line in lines:
        if line == "":
            continue
        expr, error = safe_parse(parser, line)
        if expr is not None:
            parsed.append(expr)
        else:
            invalid.append((line, error))
    return parsed, invalid


def get_fixer_prompt(alpha_str: str, error_msg: str) -> str:
    """Generate prompt for fixer LLM with error context.
    
    Args:
        alpha_str: The invalid alpha expression string
        error_msg: The error message from parsing
    
    Returns:
        The complete prompt for the fixer LLM
    """
    error_context = f"Error encountered: {error_msg}\n\n"
    
    prompt = f"Fix this alpha expression that has syntax errors:\n" \
         f"Alpha: {alpha_str}\n" \
         f"Error: {error_context}\n" \
         f"Rules:\n" \
         f"1. Balance all parentheses - every opening paren must have a closing paren\n" \
         f"2. Ensure all operators have correct number of arguments:\n" \
         f"   - Unary operators (Abs, Log): 1 argument\n" \
         f"   - Binary operators (Add, Sub, Mul, Div, Greater, Less): 2 arguments\n" \
         f"   - Time-series unary (Ref, Mean, Sum, Std, Var, Max, Min, Med, Mad, Delta, WMA, EMA): 1 argument + time span (e.g., 5d)\n" \
         f"   - Time-series binary (Cov, Corr): 2 arguments + time span\n" \
         f"3. Output ONLY the corrected formula, no explanations and no metrics\n" \
         f"\n" \
         f"ACCEPTABLE examples:\n" \
         f"  Abs(Sub(Ref(close,5d),Ref(open,5d)))\n" \
         f"  Cov(Ref(volume,5d),close,30d)\n" \
         f"\n" \
         f"UNACCEPTABLE examples (and how to fix):\n" \
         f"  Abs(Sub(Ref(close,5d),Ref(open,5d) -> missing closing paren\n" \
         f"  Mean(close) -> missing time span -> Mean(close,10d)\n" \
         f"  Max(Mean(high,10d), Mean(low,10d)) -> Max(x,t) requires a time window for the second argument\n" \
         f"  Mean(close, 10d): IC = 0.0511 -> extra output\n" \
         f"  Mean(close - open, 10d) -> unexpected token -; use Sub(close, open) instead\n" \
         f"\n" \
         f"POTENTIAL ERRORS AND FIXES:\n" \
         f"- Multiple items remain in the stack: Add any missing parenthesis ).\n" \
         f"- Unexpected token [token]: Remove invalid symbols or fix typos in operator names.\n" \
         f"- Can't find the feature [name]: Use a valid feature name defined in FeatureType (e.g., $open, $close).\n" \
         f"- [Op] expects X operand(s), but received Y: Check function requirements and provide the correct number of arguments for the function (e.g., Add needs 2).\n" \
         f"- issubclass() arg X must be a class: Check function requirements (should it be an expression or a time delta)\n"
    return prompt
