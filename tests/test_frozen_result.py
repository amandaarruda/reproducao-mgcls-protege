"""Testes de regressão para o resultado fornecido."""

from mgcls_protege import contract as C
from mgcls_protege.pipeline import compare_with_saved, replay_saved_result
from mgcls_protege.results import validate_results


def test_resultado_fornecido_e_consistente():
    report = validate_results()
    counts = report["confusion"]
    assert (
        counts["true_positive"],
        counts["false_positive"],
        counts["false_negative"],
        counts["true_negative"],
    ) == (43, 57, 43, 888)
    assert counts["recall"] == C.RECALL_AT_100 == 0.5


def test_replay_corresponde_ao_ranking_fornecido():
    result, _, _ = replay_saved_result()
    report = compare_with_saved(result)
    assert report["equivalent"]
    assert report["same_top_100"]
    assert report["same_active_learning_trace"]
    assert report["confusion"]["true_positive"] == 43
