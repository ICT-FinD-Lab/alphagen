from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import json
from logging import Logger
from datetime import datetime
from openai import OpenAI
import fire

from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen_llm.client import ChatClient, OpenAIClient, ChatConfig
from alphagen_llm.prompts.common import safe_parse
from alphagen.utils import get_logger
from alphagen_llm.prompts.system_prompt import *


_GENERATE_ALPHAS_DEFAULT_PROMPT = "Generate me ten alphas that you think would be indicative of future stock price trend. Each alpha should be on its own line without numbering. Please do not output anything else."


@dataclass
class ValidityTestResult:
    n_parsers: int
    "Number of parsers used in the test."
    lines: Dict[str, List[Optional[Expression]]]
    "Map of lines to the expressions parsed by each parser."
    n_duplicates: int
    "Number of duplicated lines found."

    @property
    def n_total_lines(self) -> int:
        "Total number of lines output by the client. Includes duplicated expressions."
        return len(self.lines) + self.n_duplicates

    @property
    def duplicate_rate(self) -> float:
        "Fraction of total lines that are duplicates."
        return self.n_duplicates / self.n_total_lines

    @property
    def validity_stats(self) -> List[Tuple[int, float]]:
        "Number and fraction of valid lines for each parser. Excluding the duplicates."
        counts: List[int] = [sum(1 for parsed in self.lines.values() if parsed[i] is not None)
                             for i in range(self.n_parsers)]
        n_lines = len(self.lines)
        return [(c, c / n_lines) for c in counts]
    
    def __str__(self) -> str:
        generic = f"Lines: {self.n_total_lines} ({len(self.lines)} + {self.n_duplicates} duplicates) | Valid: "
        return generic + ", ".join(f"{n} ({r * 100:.1f}%)" for n, r in self.validity_stats)
    

def test_validity(
    client: ChatClient,
    parsers: List[ExpressionParser],
    n_repeats: int,
    generate_alphas_prompt: str = _GENERATE_ALPHAS_DEFAULT_PROMPT,
    with_tqdm: bool = False
) -> ValidityTestResult:
    lines: Dict[str, List[Optional[Expression]]] = {}
    duplicates = 0
    range_ = range
    if with_tqdm:
        from tqdm import trange
        range_ = trange
    for _ in range_(n_repeats):
        client.reset()
        response = client.chat_complete(generate_alphas_prompt)
        output_lines = [stripped for line in response.splitlines() if (stripped := line.strip()) != ""]
        for line in output_lines:
            if line in lines:
                duplicates += 1
                continue
            lines[line] = [safe_parse(parser, line) for parser in parsers]
    return ValidityTestResult(len(parsers), lines, duplicates)


def build_chat_client(system_prompt: str, logger: Optional[Logger] = None):
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


def run_experiment(n_repeats: int = 10):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_prefix = f"./out/llm-tests/test_validity/{timestamp}"
    logger: Logger = get_logger(name="llm", file_path=f"{file_prefix}.log")
    parsers = [build_parser(use_additional_mapping=use) for use in (False, True)]
    chat = build_chat_client(EXPLAIN_WITH_TEXT_DESC, logger)
    results = test_validity(chat, parsers, n_repeats=n_repeats, with_tqdm=True)
    print(results)
    with open(f"{file_prefix}.json", "w") as f:
        parsed, invalid = [], []
        for line, output in results.lines.items():
            if all(e is None for e in output):
                invalid.append(line)
            else:
                parsed.append(line)
        json.dump(dict(parsed=parsed, invalid=invalid), f, indent=4)


if __name__ == "__main__":
    fire.Fire(run_experiment)
