"""Configurações usadas no replay em CPU e na avaliação."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_RESULTS = PROJECT_ROOT / "results" / "final"


@dataclass(frozen=True)
class ProtegeSettings:
    """Etapa 3: laço de aprendizado ativo."""

    n_labels: int = 100              # total de rótulos solicitados ao usuário
    batch_size: int = 10             # rótulos por rodada, também o lote inicial
    ei_tradeoff: float = 3.0         # epsilon da melhoria esperada, entre os escores 1 e 5


@dataclass(frozen=True)
class EvaluationSettings:
    """Etapa 4: avaliação do ranking."""

    top_k: int = 100                 # as 100 primeiras fontes são as recomendadas
    high_interest_min_label: int = 4 # escores 4 e 5 definem alto interesse
