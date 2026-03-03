from typing import Optional, List
from logging import Logger
from datetime import datetime
import json
from itertools import accumulate

import fire
import torch
from openai import OpenAI

from alphagen.data.expression import Expression
from alphagen.data.parser import ExpressionParser
from alphagen.data.expression import *
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import StockData, initialize_qlib
from alphagen_generic.features import target
from alphagen_llm.client import OpenAIClient, ChatConfig
from alphagen_llm.prompts.interaction import DefaultInteraction, DefaultReport
from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC
from alphagen.utils import get_logger
from alphagen.utils.misc import pprint_arguments


def build_chat(system_prompt: str, logger: Optional[Logger] = None):
    return OpenAIClient(
        OpenAI(),
        ChatConfig(
            system_prompt=system_prompt,
            logger=logger
        )
    )


def build_parser(use_additional_mapping: bool = False) -> ExpressionParser:
    mapping = {
        "Max": [Greater],
        "Min": [Less],
        "Delta": [Sub]
    }
    return ExpressionParser(
        Operators,
        ignore_case=True,
        additional_operator_mapping=mapping if use_additional_mapping else None,
        non_positive_time_deltas_allowed=False
    )


def build_test_data(instruments: str, device: torch.device, n_half_years: int) -> List[Tuple[str, StockData]]:
    halves = (("01-01", "06-30"), ("07-01", "12-31"))

    def get_dataset(i: int) -> Tuple[str, StockData]:
        year = 2022 + i // 2
        start, end = halves[i % 2]
        return (
            f"{year}h{i % 2 + 1}",
            StockData(
                instrument=instruments,
                start_time=f"{year}-{start}",
                end_time=f"{year}-{end}",
                device=device
            )
        )

    return [get_dataset(i) for i in range(n_half_years)]


def run_experiment(
    pool_size: int = 20,
    n_replace: int = 3,
    n_updates: int = 20,
    without_weights: bool = False,
    contextful: bool = False,
    prefix: Optional[str] = None,
    force_remove: bool = False,
    also_report_history: bool = False
):
    """
    :param pool_size: Maximum alpha pool size
    :param n_replace: Replace n alphas on each iteration
    :param n_updates: Run n iterations
    :param without_weights: Do not report the weights of the alphas to the LLM
    :param contextful: Keep context in the conversation
    :param prefix: Output location prefix
    :param force_remove: Force remove worst old alphas
    :param also_report_history: Also report alpha pool update history to the LLM
    """

    args = pprint_arguments()

    initialize_qlib(f"~/.qlib/qlib_data/cn_data")
    instruments = "csi300"
    device = torch.device("cuda:0")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    prefix = str(prefix) + "-" if prefix is not None else ""
    out_path = f"./out/llm-tests/interaction/{prefix}{timestamp}"
    logger = get_logger(name="llm", file_path=f"{out_path}/llm.log")

    with open(f"{out_path}/config.json", "w") as f:
        json.dump(args, f)

    data_train = StockData(
        instrument=instruments,
        start_time="2012-01-01",
        end_time="2021-12-31",
        device=device
    )
    data_test = build_test_data(instruments, device, n_half_years=3)
    calculator_train = QLibStockDataCalculator(data_train, target)
    calculator_test = [QLibStockDataCalculator(d, target) for _, d in data_test]

    def make_pool(exprs: List[Expression]) -> MseAlphaPool:
        pool = MseAlphaPool(
            capacity=max(pool_size, len(exprs)),
            calculator=calculator_train,
            device=device
        )
        pool.force_load_exprs(exprs)
        return pool

    def show_iteration(_, iter: int):
        print(f"Iteration {iter} finished...")

    inter = DefaultInteraction(
        parser=build_parser(),
        client=build_chat(EXPLAIN_WITH_TEXT_DESC, logger=logger),
        pool_factory=make_pool,
        calculator_train=calculator_train,
        calculators_test=calculator_test,
        replace_k=n_replace,
        force_remove=force_remove,
        forgetful=not contextful,
        no_actual_weights=without_weights,
        also_report_history=also_report_history,
        on_pool_update=show_iteration
    )
    inter.run(n_updates=n_updates)

    with open(f"{out_path}/report.json", "w") as f:
        json.dump([r.to_dict() for r in inter.reports], f)

    cum_days = list(accumulate(d.n_days for _, d in data_test))
    mean_ic_results = {}
    mean_ics, mean_rics = [], []

    def get_rolling_means(ics: List[float]) -> List[float]:
        cum_ics = accumulate(ic * tup[1].n_days for ic, tup in zip(ics, data_test))
        return [s / n for s, n in zip(cum_ics, cum_days)]

    for report in inter.reports:
        mean_ics.append(get_rolling_means(report.test_ics))
        mean_rics.append(get_rolling_means(report.test_rics))

    for i, (name, _) in enumerate(data_test):
        mean_ic_results[name] = {
            "ics": [step[i] for step in mean_ics],
            "rics": [step[i] for step in mean_rics]
        }
    
    with open(f"{out_path}/rolling_mean_ic.json", "w") as f:
        json.dump(mean_ic_results, f)


if __name__ == "__main__":
    fire.Fire(run_experiment)
