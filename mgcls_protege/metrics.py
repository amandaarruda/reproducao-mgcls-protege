"""Métricas usadas para avaliar o ranking final."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import EvaluationSettings


@dataclass(frozen=True)
class Confusion:
    top_k: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def positives(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def total(self) -> int:
        return self.positives + self.false_positive + self.true_negative

    @property
    def recall(self) -> float:
        return self.true_positive / self.positives

    @property
    def precision(self) -> float:
        return self.true_positive / self.top_k

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total

    @property
    def f1(self) -> float:
        numerator = 2 * self.true_positive
        return numerator / (numerator + self.false_positive + self.false_negative)

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "recall": self.recall,
            "precision": self.precision,
            "accuracy": self.accuracy,
            "f1": self.f1,
        }


def top_k_confusion(
    ranking: pd.DataFrame,
    top_k: int = EvaluationSettings.top_k,
    min_label: int = EvaluationSettings.high_interest_min_label,
) -> Confusion:
    """Retornar a matriz de confusão das primeiras ``top_k`` fontes."""
    if "ground_truth_label" not in ranking.columns:
        raise ValueError("o ranking precisa da coluna 'ground_truth_label'")

    labels = ranking.sort_values("rank")["ground_truth_label"]
    positive = labels.ge(min_label).to_numpy()
    true_positive = int(positive[:top_k].sum())
    false_negative = int(positive.sum()) - true_positive
    return Confusion(
        top_k=top_k,
        true_positive=true_positive,
        false_positive=top_k - true_positive,
        false_negative=false_negative,
        true_negative=len(positive) - top_k - false_negative,
    )


def recall_at(
    ranking: pd.DataFrame,
    ks: tuple[int, ...] = (10, 20, 50, 100, 200, 300, 500, 1031),
) -> pd.DataFrame:
    """Retornar recall e enriquecimento para diferentes limites de inspeção."""
    positive = ranking.sort_values("rank")["ground_truth_label"].ge(4).to_numpy()
    cumulative = np.cumsum(positive)
    total = int(positive.sum())
    n_sources = len(positive)

    rows = []
    for k in ks:
        found = int(cumulative[k - 1])
        random_expected = k * total / n_sources
        rows.append(
            {
                "K": k,
                "found": found,
                "recall": found / total,
                "random_expected": random_expected,
                "enrichment": found / random_expected,
            }
        )
    return pd.DataFrame(rows)


def roc_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Calcular ROC AUC a partir das posições, tratando empates como metade."""
    from scipy.stats import rankdata

    ranks = rankdata(scores)
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    u_statistic = ranks[positive].sum() - n_positive * (n_positive + 1) / 2
    return float(u_statistic / (n_positive * n_negative))
