"""Replay do aprendizado ativo do Protege.

O modelo ajusta um processo gaussiano aos escores coletados. A melhoria
esperada seleciona cada novo lote, e a média final do processo define o ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from .config import EvaluationSettings, ProtegeSettings


def fit_gp(features: np.ndarray, labels: np.ndarray) -> GaussianProcessRegressor:
    """Ajustar o processo gaussiano às fontes já pontuadas."""
    gp = GaussianProcessRegressor(kernel=Matern() + WhiteKernel())
    return gp.fit(features, labels)


def expected_improvement(mean: np.ndarray, std: np.ndarray, best: float, tradeoff: float) -> np.ndarray:
    """Melhoria esperada sobre ``best + tradeoff`` para uma predição gaussiana.

    ``mean`` e ``std`` são a predição do GP para cada fonte, e ``best`` é o maior
    escore atribuído até o momento. Com ``u = mean - best - tradeoff``::

        EI = u * Phi(u / std) + std * phi(u / std)

    em que Phi e phi são a função de distribuição e a densidade normal padrão.
    """
    u = mean - best - tradeoff
    z = u / std
    return u * ndtr(z) + std * norm.pdf(z)


@dataclass
class RoundState:
    """Estado do modelo depois do ajuste em ``n_labels`` escores."""

    n_labels: int
    labelled: list[str]               # fontes pontuadas, na ordem das consultas
    mean: pd.Series                   # escore previsto de cada fonte
    std: pd.Series                    # incerteza do modelo para cada fonte
    acquisition: pd.Series            # melhoria esperada de cada fonte
    gp: GaussianProcessRegressor = field(repr=False)


@dataclass
class ReplayResult:
    ranking: pd.DataFrame             # ranking final, uma linha por fonte avaliada
    trace: pd.DataFrame               # fontes consultadas acumuladas por rodada
    rounds: list[RoundState]          # um estado por ajuste (10, 20, ..., 100 rótulos)


def initial_query(features: pd.DataFrame, size: int) -> list[str]:
    """Primeiro lote: fontes distribuídas ao longo do primeiro componente principal.

    A ordenação pelo eixo de maior variância e a seleção de posições equidistantes
    cobrem o espaço de características, em vez de escolher fontes ao acaso.
    """
    ordered = features.sort_values(features.columns[0])
    positions = np.linspace(0, len(ordered) - 1, size, dtype=int)
    return list(ordered.index[positions])


def _fit_round(features: pd.DataFrame, labels: pd.Series, labelled: list[str], tradeoff: float) -> RoundState:
    answers = labels.loc[labelled]
    gp = fit_gp(features.loc[labelled].to_numpy(), answers.to_numpy())
    mean, std = gp.predict(features.to_numpy(), return_std=True)
    acquisition = expected_improvement(mean, std, float(answers.max()), tradeoff)
    as_series = lambda values: pd.Series(values, index=features.index)  # noqa: E731
    return RoundState(len(labelled), list(labelled), as_series(mean), as_series(std), as_series(acquisition), gp)


def replay(
    features: pd.DataFrame,
    labels: pd.Series,
    settings: ProtegeSettings = ProtegeSettings(),
) -> ReplayResult:
    """Repetir uma sessão do Protege sobre as fontes de avaliação.

    Parameters
    ----------
    features:
        Coordenadas dos componentes principais das fontes de avaliação (fontes x 30).
    labels:
        Escore atribuído (1 a 5) a cada fonte. O valor é consultado somente para
        as fontes solicitadas pelo algoritmo.
    """
    labelled = initial_query(features, settings.batch_size)
    state = _fit_round(features, labels, labelled, settings.ei_tradeoff)
    rounds = [state]

    while len(labelled) < settings.n_labels:
        batch = min(settings.batch_size, settings.n_labels - len(labelled))
        # Consultar as fontes ainda não pontuadas com maior melhoria esperada.
        candidates = state.acquisition.drop(labelled).sort_values(ascending=False).index[:batch]
        labelled = labelled + list(candidates)
        state = _fit_round(features, labels, labelled, settings.ei_tradeoff)
        rounds.append(state)

    return ReplayResult(_final_ranking(state, labels), _trace(rounds, features.index, labels), rounds)


def _final_ranking(state: RoundState, labels: pd.Series) -> pd.DataFrame:
    """Ordenar todas as fontes pelo escore previsto final, do maior para o menor."""
    ranking = pd.DataFrame({"trained_score": state.mean, "acquisition": state.acquisition})
    ranking["human_label"] = -1  # -1 indica ausência de avaliação humana
    ranking.loc[state.labelled, "human_label"] = labels.loc[state.labelled]
    ranking = ranking.sort_values("trained_score", ascending=False)
    ranking.insert(0, "rank", np.arange(len(ranking)))  # zero-based position
    ranking["ground_truth_label"] = labels.loc[ranking.index].astype(int)
    # Derivado dos rótulos completos, e não apenas das 100 consultas.
    ranking["human_high_interest"] = ranking["ground_truth_label"] >= EvaluationSettings.high_interest_min_label
    ranking["cumulative_high_interest"] = ranking["human_high_interest"].cumsum()
    ranking.index.name = "objid"
    return ranking[
        ["rank", "human_label", "trained_score", "acquisition", "ground_truth_label",
         "human_high_interest", "cumulative_high_interest"]
    ]


def _trace(rounds: list[RoundState], index: pd.Index, labels: pd.Series) -> pd.DataFrame:
    """Instantâneos acumulados das fontes consultadas após cada ajuste."""
    rows = []
    for state in rounds:
        asked = set(state.labelled)
        rows += [
            {"n_labels": state.n_labels, "iteration": state.n_labels, "objid": obj, "human_label": int(labels[obj])}
            for obj in index if obj in asked
        ]
    return pd.DataFrame(rows)


def rounds_table(rounds: list[RoundState], batch_size: int) -> pd.DataFrame:
    """Uma linha por (rodada, fonte): escore, incerteza e melhoria esperados.

    ``is_scored`` marca as fontes já pontuadas, e ``is_next_query`` marca as
    ``batch_size`` fontes que seriam consultadas em seguida.
    """
    frames = []
    for state in rounds:
        table = pd.DataFrame({"mean": state.mean, "std": state.std, "acquisition": state.acquisition})
        table.insert(0, "n_labels", state.n_labels)
        table["is_scored"] = table.index.isin(state.labelled)
        nxt = state.acquisition.drop(state.labelled).sort_values(ascending=False).index[:batch_size]
        table["is_next_query"] = table.index.isin(nxt) & (state is not rounds[-1])
        frames.append(table)
    out = pd.concat(frames)
    out.index.name = "objid"
    return out.reset_index()
