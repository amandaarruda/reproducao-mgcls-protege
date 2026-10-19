"""Testes unitários dos componentes do Protege em dados sintéticos pequenos."""
import numpy as np
import pandas as pd

from mgcls_protege.config import ProtegeSettings
from mgcls_protege.protege import expected_improvement, initial_query, replay


def synthetic(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, 3)), index=[f"s{i}" for i in range(n)])
    labels = pd.Series(np.clip(np.round(2.5 + 1.5 * x[0] + rng.normal(0, .3, n)), 1, 5).astype(int), index=x.index)
    return x, labels


def test_melhoria_esperada_aumenta_com_incerteza_e_media():
    mean, std = np.array([2.0, 2.0, 4.0]), np.array([0.1, 1.0, 0.1])
    ei = expected_improvement(mean, std, best=3.0, tradeoff=0.5)
    assert ei[1] > ei[0]      # mais incerteza, mesma média -> maior ganho possível
    assert ei[2] > ei[0]      # média maior, mesma incerteza -> maior ganho possível
    assert (ei >= 0).all()


def test_primeira_consulta_cobre_o_primeiro_componente():
    x, _ = synthetic()
    picked = initial_query(x, 10)
    assert len(set(picked)) == 10
    assert x.loc[picked, 0].min() == x[0].min() and x.loc[picked, 0].max() == x[0].max()


def test_replay_consulta_rotula_e_ordena_todas_as_fontes():
    x, labels = synthetic()
    result = replay(x, labels, ProtegeSettings(n_labels=30, batch_size=10))
    assert [r.n_labels for r in result.rounds] == [10, 20, 30]
    assert len(result.ranking) == len(x) and result.ranking["rank"].tolist() == list(range(len(x)))
    assert (result.ranking["human_label"] >= 0).sum() == 30      # somente fontes consultadas têm rótulo
    assert result.ranking["trained_score"].is_monotonic_decreasing
    assert result.ranking.loc[result.ranking["human_label"] >= 0, "ground_truth_label"].equals(
        result.ranking.loc[result.ranking["human_label"] >= 0, "human_label"])


def test_replay_nao_repete_fonte_ja_pontuada():
    x, labels = synthetic()
    asked = replay(x, labels, ProtegeSettings(n_labels=40, batch_size=10)).rounds[-1].labelled
    assert len(asked) == len(set(asked)) == 40
