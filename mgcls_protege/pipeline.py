"""Carregar as características PCA e repetir a etapa do Protege."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import contract as C
from .config import FROZEN_RESULTS, ProtegeSettings
from .data import evaluation_labels, load_catalogue
from .metrics import top_k_confusion
from .protege import ReplayResult, replay


def evaluation_inputs(
    pca_scores: pd.DataFrame,
    catalogue: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Selecionar o subconjunto de avaliação na tabela PCA completa."""
    labels = evaluation_labels(catalogue)
    scores = pca_scores.copy()
    scores.index = scores.index.astype(str)
    return scores.loc[labels.index], labels


def replay_saved_result(
    results_dir: str | Path = FROZEN_RESULTS,
    settings: ProtegeSettings = ProtegeSettings(),
) -> tuple[ReplayResult, pd.DataFrame, pd.Series]:
    """Repetir o Protege a partir das características PCA fornecidas."""
    root = Path(results_dir)
    catalogue = load_catalogue(root / "catalogue.csv")
    pca_scores = pd.read_parquet(root / "pca_features.parquet")
    features, labels = evaluation_inputs(pca_scores, catalogue)
    return replay(features, labels, settings), features, labels


def compare_with_saved(
    result: ReplayResult,
    results_dir: str | Path = FROZEN_RESULTS,
) -> dict:
    """Comparar um replay com o ranking fornecido em ``results_dir``."""
    saved = pd.read_csv(Path(results_dir) / "evaluation_ranking.csv").set_index("objid")
    ranking = result.ranking
    same_sources = set(ranking.index) == set(saved.index)

    if same_sources:
        score_gap = float(
            (ranking["trained_score"].reindex(saved.index) - saved["trained_score"])
            .abs()
            .max()
        )
        rank_shift = int(
            (ranking["rank"].reindex(saved.index) - saved["rank"]).abs().max()
        )
    else:
        score_gap = float("inf")
        rank_shift = -1

    saved_trace = pd.read_csv(Path(results_dir) / "active_learning_trace.csv")

    def trace_key(table: pd.DataFrame) -> list[tuple]:
        return sorted(zip(table["n_labels"], table["objid"], table["human_label"]))

    confusion = top_k_confusion(ranking.reset_index())
    observed = (
        confusion.true_positive,
        confusion.false_positive,
        confusion.false_negative,
        confusion.true_negative,
    )
    expected = (C.TRUE_POSITIVE, C.FALSE_POSITIVE, C.FALSE_NEGATIVE, C.TRUE_NEGATIVE)

    report = {
        "same_ranking_order": ranking.index.tolist() == saved.index.tolist(),
        "max_abs_score_difference": score_gap,
        "max_rank_shift": rank_shift,
        "same_top_100": set(ranking.index[:100]) == set(saved.index[:100]),
        "same_active_learning_trace": trace_key(result.trace) == trace_key(saved_trace),
        "confusion": confusion.as_dict(),
    }
    report["exact"] = report["same_ranking_order"] and score_gap < 1e-8
    report["equivalent"] = bool(
        report["same_top_100"]
        and report["same_active_learning_trace"]
        and score_gap < 1e-3
        and observed == expected
    )
    if not report["equivalent"]:
        raise ValueError(f"o replay difere do resultado fornecido: {report}")
    return report
