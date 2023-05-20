from pathlib import Path
from typing import Union

import pandas as pd
from pytrec_eval import RelevanceEvaluator, parse_qrel, parse_run


def load_qrels_from_file(file: Union[Path, str]) -> dict[str, dict[str, int]]:
    with Path(file).open("r") as f:
        return parse_qrel(f)


def load_run_from_file(file: Union[Path, str]) -> dict[str, dict[str, float]]:
    with Path(file).open("r") as f:
        return parse_run(f)


def trec_evaluation(qrels, run, metrics: list[str], relevance_level: int = 1) -> dict[str, float]:
    evaluator = RelevanceEvaluator(qrels, metrics, relevance_level=relevance_level)
    eval = evaluator.evaluate(run)
    df = pd.DataFrame([*eval.values()])
    return df.mean().to_dict()
