from typing import List
import numpy as np

from .interaction import DefaultInteraction
from .common import alpha_phrase, safe_parse_list, safe_parse
from ..feedback import run_feedback_loop_on_alphas, call_fixer_llm
from ..llm_alpha_stats_utils import record_llm_alpha
from alphagen.data.expression import Expression


class FeedbackInteraction(DefaultInteraction):
	"""
	Extends DefaultInteraction with a feedback loop during _update.
	On each iteration, generates candidate alphas, runs them through critique/fix/improve feedback,
	then bulk-edits the pool with the finalized alphas.
	"""
	
	def __init__(self, *args, feedback_iter: int = 3, feedback_mode: str = "full", llm_alpha_stats = None, **kwargs):
		"""Initialize with feedback_iter, feedback_mode and llm_alpha_stats parameters; all other args/kwargs passed to DefaultInteraction.
		
		feedback_mode: "full" for critique/fix/improve loop, "fix-only" for syntax fixing only
		llm_alpha_stats: Optional dictionary to track valid/invalid alphas (passed to parent DefaultInteraction)
		"""
		# Pass llm_alpha_stats to parent DefaultInteraction
		kwargs['llm_alpha_stats'] = llm_alpha_stats
		super().__init__(*args, **kwargs)
		self.feedback_iter = feedback_iter
		self.feedback_mode = feedback_mode

	def _chat_and_parse(self, prompt: str) -> List[Expression]:
		"""Override to fix invalid alphas during parsing."""
		lines = self._client.chat_complete(prompt)
		exprs, invalid = safe_parse_list(lines.split('\n'), self._parser)
		
		# Track valid alphas
		if self._llm_alpha_stats is not None:
			for expr in exprs:
				record_llm_alpha(self._llm_alpha_stats, 'original', valid=True, fixed=False, expr=expr)
			# Track that these were invalid (before fixing)
			self._llm_alpha_stats['original']['invalid'] += len(invalid)
		
		# Try to fix invalid alphas
		fixed_exprs = []
		if len(invalid) != 0:
			self._client.log_message(("script", f"Invalid expressions: {len(invalid)}"))
			print(f"[FEEDBACK] Found {len(invalid)} invalid alphas in _chat_and_parse, attempting to fix...")
			for invalid_str, error_msg in invalid:
				print(f"[FEEDBACK] Fixing invalid alpha: {invalid_str} (error: {error_msg})")
				fixed_str = call_fixer_llm(self, invalid_str, error_msg)
				fixed_expr, _ = safe_parse(self._parser, fixed_str)
				if fixed_expr is not None:
					fixed_exprs.append(fixed_expr)
					# Record as fixed, and increment fixed counter
					if self._llm_alpha_stats is not None:
						record_llm_alpha(self._llm_alpha_stats, 'original', valid=True, fixed=True, expr=fixed_expr)
						self._llm_alpha_stats['original']['fixed'] += 1
					print(f"[FEEDBACK] Fixed: {invalid_str} -> {fixed_str}")
		
		self._client.reset()  
		# Return valid + fixed expressions
		return exprs + fixed_exprs


	def _initialize(self, pool, exprs: List[Expression]) -> None:
		"""Override initialization to fix invalid alphas during pool creation."""
		if len(exprs) != 0:
			return
		
		print("[FEEDBACK-INIT] Initializing pool with feedback loop enabled...")
		# Generate initial alphas from LLM
		p = (f"Please generate {alpha_phrase(pool.capacity)} that you think would be "
		     "indicative of future stock price trend. Each alpha should be "
		     "on its own line without numbering. Please do not output anything else.")
		lines = self._client.chat_complete(p)
		self._client.reset()  
		print(f"[FEEDBACK-INIT] Generated initial alphas from LLM")
		
		# Parse and identify invalid alphas
		valid_exprs, invalid_with_errors = safe_parse_list(lines.split('\n'), self._parser)
		print(f"[FEEDBACK-INIT] Parsed {len(valid_exprs)} valid and {len(invalid_with_errors)} invalid alphas")
		
		# Track valid alphas and count invalid ones
		if self._llm_alpha_stats is not None:
			for expr in valid_exprs:
				record_llm_alpha(self._llm_alpha_stats, 'original', valid=True, fixed=False, expr=expr)
			# Track that these were invalid before fixing
			self._llm_alpha_stats['original']['invalid'] += len(invalid_with_errors)
		
		# Fix invalid alphas using the fixer LLM
		if len(invalid_with_errors) > 0:
			print(f"[FEEDBACK-INIT] Attempting to fix {len(invalid_with_errors)} invalid alphas...")
			self._client.log_message(("script", f"Found {len(invalid_with_errors)} invalid alphas during initialization, attempting to fix..."))
			fixed_count = 0
			for invalid_str, error_msg in invalid_with_errors:
				print(f"[FEEDBACK-INIT] Fixing invalid alpha: {invalid_str} (error: {error_msg})")
				fixed_str = call_fixer_llm(self, invalid_str, error_msg)
				fixed_expr, _ = safe_parse(self._parser, fixed_str)
				if fixed_expr is not None:
					valid_exprs.append(fixed_expr)
					fixed_count += 1
					# Record as fixed, and increment fixed counter
					if self._llm_alpha_stats is not None:
						record_llm_alpha(self._llm_alpha_stats, 'original', valid=True, fixed=True, expr=fixed_expr)
						self._llm_alpha_stats['original']['fixed'] += 1
					print(f"[FEEDBACK-INIT] Successfully fixed: {invalid_str} -> {fixed_str}")
					self._client.log_message(("script", f"Fixed invalid alpha: {invalid_str} -> {fixed_str}"))
			print(f"[FEEDBACK-INIT] Fixed {fixed_count}/{len(invalid_with_errors)} invalid alphas")
		
		# Load all valid (and fixed) expressions into pool
		print(f"[FEEDBACK] Loading {len(valid_exprs)} valid expressions into pool")
		pool.force_load_exprs(valid_exprs)
		report = self._evaluate_pool(pool)
		self._reports.append(report)
		self._on_pool_update(report, 0)
		self._client.reset()
		print(f"[FEEDBACK] Pool initialization complete")

	def _update(self, iter: int, pool) -> bool:
		"""Override _update to apply feedback loop to generated alphas."""
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
		           "Again, one on each line without numbering, and do not output anything else.")
		SIG_THRES = 1e-4

		if self._forgetful:
			self._client.reset()
		history = ""
		if self._also_report_history:
			from .interaction import _describe_update
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
		replace_prompt = REPLACE.format(self._alpha_phrase(replaced_count, "more"))
		
		# Generate initial candidates, try to fix once
		exprs = self._chat_and_parse(prefix + report_str + replace_prompt)
	
		# Run feedback loop on the generated alphas
		print(f"[FEEDBACK] Starting feedback loop (mode={self.feedback_mode}, iter={self.feedback_iter})")
		print(f"[FEEDBACK] Generated {len(exprs)} candidate alphas for feedback processing: {exprs}")
		if self.feedback_mode == "full":
			final_exprs = run_feedback_loop_on_alphas(
				# Only contains valid + fixed expressions
				initial_alphas=exprs,
				pool=pool,
				test_calculators=self._calcs_test,
				chat_session=self._client,
				feedback_iter=self.feedback_iter,
				parser=self._parser,
				llm_alpha_stats=self._llm_alpha_stats,
			)
			print(f"[FEEDBACK] Feedback loop complete. {len(final_exprs)} final alphas after processing")
		else: 
			final_exprs = exprs
		pool.bulk_edit(removed_idx, final_exprs)
		self._reports.append(report)
		self._on_pool_update(report, iter + 1)
		print(f"[FEEDBACK] Pool update complete for iteration {iter}")
		return True
	
	def _alpha_phrase(self, k: int, suffix: str = "") -> str:
		"""Helper to generate alpha phrase (e.g., '3 alphas', '1 alpha more')."""
		return alpha_phrase(k, suffix)
