"""Carregar e validar o resultado compacto fornecido com o repositório."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import contract as C
from .config import FROZEN_RESULTS
from .metrics import top_k_confusion


def load_results(results_dir: str | Path = FROZEN_RESULTS) -> dict[str, Any]:
    root = Path(results_dir).expanduser().resolve()
    return {
        "root": root,
        "ranking": pd.read_csv(root / "evaluation_ranking.csv"),
        "trace": pd.read_csv(root / "active_learning_trace.csv"),
        "loss": pd.read_csv(root / "byol_loss.csv"),
        "catalogue": pd.read_csv(root / "catalogue.csv"),
        "metrics": json.loads(
            (root / "protege_evaluation_metrics.json").read_text(encoding="utf-8")
        ),
        "derived": json.loads(
            (root / "derived_metrics.json").read_text(encoding="utf-8")
        ),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def validate_results(results_dir: str | Path = FROZEN_RESULTS) -> dict[str, Any]:
    """Verificar dimensões, rótulos, ordenação e métricas do resultado."""
    data = load_results(results_dir)
    root = data["root"]
    ranking = data["ranking"]
    trace = data["trace"]
    loss = data["loss"]
    catalogue = data["catalogue"]
    metrics = data["metrics"]

    _require(len(catalogue) == C.CATALOGUE_SOURCES, "tamanho inesperado do catálogo")
    _require(not catalogue["objid"].duplicated().any(), "identificadores duplicados no catálogo")
    _require(len(ranking) == C.EVALUATION_SOURCES, "tamanho inesperado do ranking")
    _require(not ranking["objid"].duplicated().any(), "identificadores duplicados no ranking")
    _require(ranking["objid"].isin(catalogue["objid"]).all(), "fonte desconhecida no ranking")
    _require(ranking["rank"].tolist() == list(range(len(ranking))), "ranking fora de ordem")
    _require(ranking["trained_score"].is_monotonic_decreasing, "escores não estão decrescentes")
    _require(len(loss) == C.BYOL_EPOCHS, "número inesperado de épocas BYOL")
    _require(np.isfinite(loss[["epoch", "training_loss"]].to_numpy()).all(), "valor inválido na perda")

    labels = catalogue.set_index("objid")["evaluation_subset_author_ML_score"]
    evaluated = labels[labels >= 0]
    _require(len(evaluated) == C.EVALUATION_SOURCES, "tamanho inesperado do subconjunto de avaliação")
    _require(evaluated.value_counts().sort_index().to_dict() == C.EVALUATION_LABEL_COUNTS,
             "as quantidades de rótulos de avaliação diferem")
    expected_labels = labels.reindex(ranking["objid"]).to_numpy()
    _require((expected_labels == ranking["ground_truth_label"].to_numpy()).all(),
             "os rótulos do ranking diferem dos rótulos do catálogo")

    high_interest = ranking["ground_truth_label"].ge(4)
    _require((ranking["human_high_interest"] == high_interest).all(),
             "a indicação de alto interesse é inconsistente")
    _require((ranking["cumulative_high_interest"].to_numpy() == high_interest.cumsum().to_numpy()).all(),
             "a contagem acumulada é inconsistente")

    labelled = ranking[ranking["human_label"] >= 0]
    _require(len(labelled) == C.LABELS_USED, "número inesperado de rótulos consultados")
    _require((labelled["human_label"] == labelled["ground_truth_label"]).all(),
             "os rótulos consultados diferem dos rótulos de avaliação")

    rounds = sorted(trace["n_labels"].unique().tolist())
    _require(rounds == list(range(10, 101, 10)), "rodadas inesperadas de aprendizado ativo")
    snapshots = {n: set(group["objid"]) for n, group in trace.groupby("n_labels")}
    _require(all(len(snapshots[n]) == n for n in rounds), "tamanho dos instantâneos difere")
    _require(all(snapshots[a] < snapshots[b] for a, b in zip(rounds, rounds[1:])),
             "os instantâneos não estão aninhados")
    _require(snapshots[100] == set(labelled["objid"]), "o trace final difere do ranking")

    confusion = top_k_confusion(ranking).as_dict()
    expected_counts = {
        "true_positive": C.TRUE_POSITIVE,
        "false_positive": C.FALSE_POSITIVE,
        "false_negative": C.FALSE_NEGATIVE,
        "true_negative": C.TRUE_NEGATIVE,
    }
    for name, value in expected_counts.items():
        _require(confusion[name] == value, f"valor inesperado para {name}")

    expected_metrics = {
        "recall": C.RECALL_AT_100,
        "precision": C.PRECISION_AT_100,
        "accuracy": C.ACCURACY,
        "f1": C.F1,
    }
    for name, value in expected_metrics.items():
        _require(_close(confusion[name], value), f"valor inesperado para {name}")

    _require(metrics["features_shape"] == [C.CATALOGUE_SOURCES, C.FEATURE_DIMENSION],
             "formato inesperado da tabela de características")
    _require(metrics["pca_shape"] == [C.CATALOGUE_SOURCES, C.PCA_COMPONENTS],
             "formato inesperado da tabela PCA")
    _require(metrics["recall_at_100"] == C.TRUE_POSITIVE, "a contagem de recall salva difere")

    derived_names = {
        "recall": "recall_fraction",
        "precision": "precision_top100",
        "accuracy": "accuracy_top100",
        "f1": "f1_top100",
    }
    for metric_name, stored_name in derived_names.items():
        _require(_close(data["derived"][stored_name], confusion[metric_name], 1e-12),
                 f"a métrica salva {metric_name} difere")

    pca_features = pd.read_parquet(root / "pca_features.parquet")
    _require(pca_features.shape == (C.CATALOGUE_SOURCES, C.PCA_COMPONENTS),
             "formato inesperado da tabela de características PCA")
    pca_components = np.load(root / "pca_components.npy")
    _require(pca_components.shape == (C.PCA_COMPONENTS, C.FEATURE_DIMENSION),
             "formato inesperado da matriz de componentes PCA")

    return {
        "status": "ok",
        "results_dir": str(root),
        "confusion": confusion,
        "loss_first": float(loss["training_loss"].iloc[0]),
        "loss_last": float(loss["training_loss"].iloc[-1]),
    }
