import numpy as np
import pandas as pd

from mgcls_protege.metrics import recall_at, roc_auc, top_k_confusion


def ranking(labels):
    return pd.DataFrame({"rank": range(len(labels)), "ground_truth_label": labels})


def test_contagens_da_matriz_e_metricas():
    counts = top_k_confusion(ranking([5, 1, 4, 2, 1, 1, 4, 1]), top_k=4)
    assert (
        counts.true_positive,
        counts.false_positive,
        counts.false_negative,
        counts.true_negative,
    ) == (2, 2, 1, 3)
    assert counts.recall == 2 / 3
    assert counts.precision == 1 / 2
    assert counts.accuracy == 5 / 8


def test_curva_de_recall_e_monotona():
    table = recall_at(ranking([5, 1, 4, 2, 1, 4]), ks=(1, 3, 6))
    assert table["recall"].is_monotonic_increasing
    assert table["recall"].iloc[-1] == 1.0


def test_roc_auc_para_escores_ordenados():
    positive = np.array([True, True, False, False])
    assert roc_auc(np.array([4.0, 3.0, 2.0, 1.0]), positive) == 1.0
    assert roc_auc(np.array([1.0, 2.0, 3.0, 4.0]), positive) == 0.0
