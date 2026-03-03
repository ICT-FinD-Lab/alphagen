from typing import List, Dict, Any, Optional
import numpy as np
import re
from alphagen.data.expression import Expression
from alphagen.data.parser import ExpressionParser
from alphagen.models.linear_alpha_pool import LinearAlphaPool
from alphagen_llm.prompts.common import safe_parse, get_fixer_prompt
from alphagen_llm.llm_alpha_stats_utils import record_llm_alpha
def count_expression_tokens(expr_str: str) -> int:
    """
    Count the number of tokens in an expression string.
    Tokens are operators, operands, and literal values.
    
    Example: "Add(Ref(close,5d),Mul(open,10d))" has 7 tokens:
      Add, Ref, close, 5d, Mul, open, 10d
    """
    # Remove whitespace
    expr_str = expr_str.strip()
    
    # Split on common delimiters: parentheses and commas
    # This gives us the operators and operands
    tokens = re.split(r'[(),]', expr_str)
    
    # Filter out empty strings and count non-empty tokens
    tokens = [t.strip() for t in tokens if t.strip()]
    
    return len(tokens)

def _get_client(chat_session):
    """Helper to extract ChatClient from either a ChatClient or InteractiveSession"""
    if hasattr(chat_session, 'client'):
        # It's an InteractiveSession
        return chat_session.client
    else:
        # It's already a ChatClient
        return chat_session

def compute_alpha_metrics(alpha_expr: Expression, pool: LinearAlphaPool) -> Dict[str, Any]:
    """
    Compute detailed metrics including Orthogonality, Complexity, and Feature Exposure.
    """
    metrics = {
        "valid": True,
        "expr_str": str(alpha_expr),
        "ic": None,
        "rank_ic": None,
        "icir": None,
        "rank_icir": None,
        "max_mutual_ic": 0.0,
        "clashing_alpha": None,  # The expression of the most similar alpha
        "complexity": 0,         # Length of string representation
        "parsimony_score": 0.0,  # IC / Complexity
    }

    # 1. Complexity (number of tokens, not string length)
    expr_str = str(alpha_expr)
    metrics["complexity"] = count_expression_tokens(expr_str)

    # 2. Performance (Strength)
    try:
        metrics["ic"] = pool.calculator.calc_single_IC_ret(alpha_expr)
        # Try to get rank IC if available
        try:
            metrics["rank_ic"] = pool.calculator.calc_single_Rank_IC_ret(alpha_expr)
        except:
            metrics["rank_ic"] = None
        
        # Calculate ICIR (Information Coefficient Information Ratio)
        # ICIR = IC / std(IC) - but here we approximate as IC / complexity for simplicity
        if metrics["ic"] is not None:
            metrics["icir"] = abs(metrics["ic"]) / max(metrics["complexity"], 1)
            metrics["rank_icir"] = abs(metrics["rank_ic"]) / max(metrics["complexity"], 1) if metrics["rank_ic"] else None
            metrics["parsimony_score"] = (metrics["ic"] * 100) / max(metrics["complexity"], 1)
    except Exception:
        metrics["ic"] = None
        metrics["valid"] = False
        return metrics

    # 3. Orthogonality (Dissimilarity)
    try:
        max_mut = 0.0
        clashing_expr = None
        
        # Iterate over existing alphas in the pool
        for i in range(pool.size):
            existing_expr = pool.exprs[i]
            # Skip self-comparison if it happens
            if str(existing_expr) == expr_str:
                continue
                
            try:
                mut = pool.calculator.calc_mutual_IC(alpha_expr, existing_expr)
                if abs(mut) > abs(max_mut):
                    max_mut = mut
                    clashing_expr = str(existing_expr)
            except Exception:
                continue
        
        metrics["max_mutual_ic"] = max_mut
        metrics["clashing_alpha"] = clashing_expr
        
    except Exception:
        metrics["max_mutual_ic"] = None

    # Validity Check: IC must be a number (not NaN)
    metrics["valid"] = metrics["ic"] is not None and not np.isnan(metrics["ic"])
    return metrics

def generate_critique_text(metrics: Dict[str, Any]) -> str:
    """
    Generates a structured critique based on the computed metrics.
    """
    if not metrics["valid"]:
        return "CRITIQUE: Invalid alpha. It evaluates to NaN or raised an error."

    critiques = []
    
    # 1. Orthogonality Critique
    if metrics["max_mutual_ic"] is not None and abs(metrics["max_mutual_ic"]) > 0.70:
        critiques.append(
            f"[High Correlation] This alpha is {abs(metrics['max_mutual_ic'])*100:.1f}% correlated "
            f"with an existing alpha: `{metrics['clashing_alpha']}`. "
            "It adds very little new information. Try a different mathematical concept."
        )
    elif abs(metrics["max_mutual_ic"] or 0) < 0.3:
        critiques.append("[Good Orthogonality] This alpha is unique and distinct from the pool.")

    # 2. Strength & Parsimony Critique (IC & ICIR)
    ic = metrics["ic"]
    rank_ic = metrics.get("rank_ic")
    icir = metrics.get("icir")
    rank_icir = metrics.get("rank_icir")
    
    if abs(ic) < 0.02:
        critiques.append(f"[Weak IC] IC is {ic:.4f}, which is too low. Needs improvement.")
    else:
        critiques.append(f"[Good IC] IC={ic:.4f}")
    
    if rank_ic is not None:
        if abs(rank_ic) < 0.02:
            critiques.append(f"[Weak Rank IC] Rank IC is {rank_ic:.4f}, too low.")
        else:
            critiques.append(f"[Good Rank IC] Rank IC={rank_ic:.4f}")
    
    # 3. Information Ratio (ICIR)
    if icir is not None:
        if icir < 0.01:
            critiques.append(f"[Low ICIR] ICIR is {icir:.6f}, suggesting poor risk-adjusted return.")
        else:
            critiques.append(f"[ICIR] ICIR={icir:.6f}")
    
    if rank_icir is not None:
        if rank_icir < 0.01:
            critiques.append(f"[Low Rank ICIR] Rank ICIR is {rank_icir:.6f}, weak rank-adjusted return.")
        else:
            critiques.append(f"[Rank ICIR] Rank ICIR={rank_icir:.6f}")
    
    # 4. Parsimony Critique
    if abs(metrics["parsimony_score"]) < 0.05: 
        critiques.append(
            f"[Overly Complex] Formula length is {metrics['complexity']} tokens. "
            "Try to simplify to achieve the same IC with fewer terms."
        )
    else:
        critiques.append(f"[Good Parsimony] Formula complexity={metrics['complexity']} tokens.")

    return "\n".join(critiques)

def call_critique_llm(chat_session, alpha_str: str, metrics: Dict[str, Any]) -> str:
    """Call critique LLM and reset chat session"""
    # Generate the data-driven critique first
    system_critique = generate_critique_text(metrics)
    
    # Format metrics string with all available metrics - add null checks
    ic = metrics['ic'] if metrics['ic'] is not None else 0.0
    metrics_str = f"IC={ic:.4f}"
    if metrics.get('rank_ic') is not None:
        metrics_str += f", Rank IC={metrics['rank_ic']:.4f}"
    if metrics.get('icir') is not None:
        metrics_str += f", ICIR={metrics['icir']:.4f}"
    if metrics.get('rank_icir') is not None:
        metrics_str += f", Rank ICIR={metrics['rank_icir']:.4f}"
    if metrics.get('max_mutual_ic') is not None and abs(metrics['max_mutual_ic']) > 0.9:
        metrics_str += f", Max Corr={metrics['max_mutual_ic']:.2f}, Length={metrics['complexity']}"
    
    prompt = (
        f"You are a strict Alpha Reviewer. Analyze this alpha:\n"
        f"Alpha: {alpha_str}\n"
        f"Metrics: {metrics_str}\n"
        f"Automated Analysis: {system_critique}\n\n"
        "Task: Based on the Automated Analysis, provide a 1-sentence specific instruction on how to improve this alpha so that it is indicative of future stock price and improves metrics. "
    )
    
    try:
        if chat_session is None:
            return f"(MOCK) {system_critique}"
        client = _get_client(chat_session)
        result = client.chat_complete(prompt)
        client.reset()
        return result
    except Exception as e:
        print(f"[CRITIQUE ERROR] Exception occurred: {type(e).__name__}: {e}")
        return f"System Critique: {system_critique}"

def call_fixer_llm(chat_session, alpha_str: str, error_msg: str) -> str:
    """Call fixer LLM with error message context and reset chat session"""
    prompt = get_fixer_prompt(alpha_str, error_msg)
    try:
        client = _get_client(chat_session)
        result = client.chat_complete(prompt).strip()
        client.reset()
        return result
    except Exception as e:
        print(f"[FIXER ERROR] Exception occurred: {type(e).__name__}: {e}")
        print(f"[FIXER ERROR] Returning original alpha unchanged: {alpha_str}")
        return alpha_str

def call_improvement_llm(chat_session, alpha_str: str, critique_text: str) -> str:
    """Call improvement LLM and reset chat session"""
    prompt = (
        f"Refine this alpha based on the critique.\n"
        f"Original: {alpha_str}\n"
        f"Critique: {critique_text}\n"
        "Instruction: Generate a NEW alpha that solves the critique and is indicative of future stock price trend."
        "Output ONLY the new alpha formula."
    )
    try:
        client = _get_client(chat_session)
        result = client.chat_complete(prompt).strip()
        client.reset()
        return result
    except Exception as e:
        print(f"[IMPROVEMENT ERROR] Exception occurred: {type(e).__name__}: {e}")
        print(f"[IMPROVEMENT ERROR] Returning original alpha unchanged: {alpha_str}")
        return alpha_str

def run_feedback_loop_on_alphas(
        initial_alphas: List[Expression],
        pool,
        chat_session,
        feedback_iter: int = 2,
        parser=None,
        test_calculators=None,
        llm_alpha_stats=None
    ) -> List[Expression]:
    """
    Run feedback loop on alphas.
    
    feedback_mode: "full" for critique/fix/improve, "fix-only" for syntax fixing only
    """
    
    candidates = initial_alphas
    print(f"[FEEDBACK LOOP] Starting with {len(candidates)} candidates in 'full' mode")
    
    # if feedback_mode == "fix-only":
    #     print(f"[FEEDBACK LOOP] Running fix-only mode")
    #     # Fix-only mode: just parse and fix syntax, no critique/improve
    #     final_exprs = []
    #     fixed_count = 0
    #     for alpha_str in candidates:
    #         alpha_expr = safe_parse(parser, alpha_str)
    #         if alpha_expr is None:
    #             print(f"[FEEDBACK LOOP] Alpha failed to parse: {alpha_str}")
    #             fixed_str = call_fixer_llm(chat_session, alpha_str)
    #             print(f"[FEEDBACK LOOP] Fixer LLM returned: {fixed_str}")
    #             alpha_expr = safe_parse(parser, fixed_str)
    #             if alpha_expr is not None:
    #                 fixed_count += 1
    #         if alpha_expr is not None:
    #             final_exprs.append(alpha_expr)
        
    #     print(f"[FEEDBACK LOOP] Fix-only: {fixed_count} alphas were fixed via LLM")
    #     # Deduplication
    #     seen = set()
    #     dedup_exprs = []
    #     for expr in final_exprs:
    #         s = str(expr)
    #         if s not in seen:
    #             seen.add(s)
    #             dedup_exprs.append(expr)
    #     print(f"[FEEDBACK LOOP] After deduplication: {len(dedup_exprs)}/{len(final_exprs)} expressions remain")
    #     return dedup_exprs
    
    # Full feedback mode: critique/fix/improve
    print(f"[FEEDBACK LOOP] Running full feedback mode with {feedback_iter} iterations")
    for round_i in range(feedback_iter):
        print(f"[FEEDBACK LOOP] Round {round_i + 1}/{feedback_iter}: Processing {len(candidates)} candidates")
        next_candidates = []
        fixed_in_round = 0
        improved_in_round = 0
        
        for alpha_expr in candidates:
            # 2. Compute Metrics
            metrics = compute_alpha_metrics(alpha_expr, pool)
            print(f"[FEEDBACK LOOP] Called metrics: {metrics}")

            # 3. Critique & Improve
            critique = call_critique_llm(chat_session, str(alpha_expr), metrics)
            print(f"[FEEDBACK LOOP] Called Critique LLM: {critique}")
            improved_str = call_improvement_llm(chat_session, str(alpha_expr), critique)
            print(f"[FEEDBACK LOOP] Called Improvement LLM: {improved_str}")
            
            # Verify the improvement parses
            improved_expr, parse_error = safe_parse(parser, improved_str)
            if improved_expr:
                next_candidates.append(improved_expr)
                improved_in_round += 1
                # Record as valid improvement
                if llm_alpha_stats is not None:
                    record_llm_alpha(llm_alpha_stats, 'improvement', valid=True, fixed=False, expr=improved_expr)
            else:  
                print(f"[FEEDBACK LOOP] Improved alpha failed to parse: {improved_str}")
                # Record as invalid improvement before trying to fix
                if llm_alpha_stats is not None:
                    llm_alpha_stats['improvement']['invalid'] += 1
                
                fixed_str = call_fixer_llm(chat_session, improved_str, parse_error)
                fixed_exprs, _ = safe_parse(parser, fixed_str)
                if fixed_exprs is None:
                    # If improvement failed to parse and fix, keep original if it was at least valid
                    print(f"[FEEDBACK LOOP] Improved alpha failed to fix: {improved_str} -> {fixed_str}")
                    next_candidates.append(alpha_expr)
                else:
                    next_candidates.append(fixed_exprs)
                    fixed_in_round += 1
                    # Record as fixed improvement
                    if llm_alpha_stats is not None:
                        record_llm_alpha(llm_alpha_stats, 'improvement', valid=True, fixed=True, expr=fixed_exprs)
                        llm_alpha_stats['improvement']['fixed'] += 1
        
        print(f"[FEEDBACK LOOP] Round {round_i + 1} complete: Fixed={fixed_in_round}, Improved={improved_in_round}")        
        # Convert back to strings for next iteration (except last round)
        candidates = next_candidates # Keep as Expressions

    # Final Deduplication
    print(f"[FEEDBACK LOOP] Starting final deduplication with {len(candidates)} candidates")
    seen = set()
    finals = []
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            finals.append(c)
    
    print(f"[FEEDBACK LOOP] Final deduplication complete: {len(finals)}/{len(candidates)} expressions remain")
    print(f"[FEEDBACK LOOP] Feedback loop finished, returning {len(finals)} final alphas")
    return finals
