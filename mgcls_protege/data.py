"""Dados: catálogo de fontes de rádio, uma imagem por fonte e escores atribuídos.

Catálogo (``catalogue.csv``): uma linha por fonte, 6.161 no total.

=================================  ==========================================================
coluna                             significado
=================================  ==========================================================
``objid``                          identificador da fonte, por exemplo ``img80_isl1861``
``protege_rank``                   posição no ranking dos autores; também é o nome da imagem
``ra``, ``dec``                    coordenadas celestes em graus
``original_image``                 campo MGCLS de onde veio o recorte
``evaluation_subset_author_ML_score``  escore de interesse de 1 a 5 para as 1.031 fontes de avaliação;
                                   ``-1`` para as 5.130 fontes sem escore
=================================  ==========================================================

As imagens são arquivos PNG RGBA de 185 x 185 pixels, nomeados ``<protege_rank>.png``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LABEL_COLUMN = "evaluation_subset_author_ML_score"
NOT_SCORED = -1


def load_catalogue(path: str | Path) -> pd.DataFrame:
    """Ler o catálogo, indexado por ``objid`` como texto."""
    catalogue = pd.read_csv(path)
    missing = {"objid", "protege_rank", LABEL_COLUMN} - set(catalogue.columns)
    if missing:
        raise ValueError(f"faltam colunas no catálogo: {sorted(missing)}")
    if catalogue["objid"].duplicated().any():
        raise ValueError("o catálogo contém identificadores objid duplicados")
    catalogue["objid"] = catalogue["objid"].astype(str)
    return catalogue.set_index("objid")


def evaluation_labels(catalogue: pd.DataFrame) -> pd.Series:
    """Escores de 1 a 5 do subconjunto de avaliação."""
    labels = catalogue[LABEL_COLUMN]
    return labels[labels != NOT_SCORED].astype(int)


def image_path(image_dir: str | Path, protege_rank: int) -> Path:
    """Caminho da imagem: ``<image_dir>/<protege_rank>.png``."""
    return Path(image_dir) / f"{int(protege_rank)}.png"


def check_images(catalogue: pd.DataFrame, image_dir: str | Path) -> None:
    """Informar erro se faltar alguma imagem do catálogo."""
    missing = [
        str(image_path(image_dir, rank))
        for rank in catalogue["protege_rank"]
        if not image_path(image_dir, rank).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"faltam {len(missing)} imagens; exemplos: {missing[:3]}")
