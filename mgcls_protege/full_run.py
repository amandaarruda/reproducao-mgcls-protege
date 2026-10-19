"""Reprodução completa do MGCLS em uma máquina com CUDA.

Este caminho opcional treina o BYOL nos recortes publicados, ajusta o PCA a
todas as características e executa a avaliação do Protege. Os comandos
``check``, ``metrics`` e ``replay`` não importam PyTorch nem Astronomaly.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .data import evaluation_labels, load_catalogue
from .metrics import top_k_confusion
from .protege import replay


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def train_byol(
    catalogue_path: Path,
    image_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    workers: int,
    seed: int,
) -> pd.DataFrame:
    """Treinar a implementação de BYOL e retornar as características."""
    try:
        import torch
        from astronomaly.feature_extraction.byol_features import BYOL_Features
        from .full_data import build_thumbnail_dataset
    except ImportError as error:
        raise RuntimeError(
            "As dependências da etapa com GPU estão ausentes. Instale primeiro "
            "requirements/full-cu121.txt."
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError("A reprodução completa exige uma GPU compatível com CUDA.")

    _seed_everything(seed)
    dataset = build_thumbnail_dataset(catalogue_path, image_dir, output_dir)
    print(f"Treinando BYOL em {len(dataset.index)} fontes")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = BYOL_Features(
        output_dir=str(output_dir),
        image_size=128,
        projection_layer_size=100,
        normalize=False,
        base_learning_rate=learning_rate,
        n_epochs=epochs,
        batch_size=batch_size,
        num_workers=workers,
        save_model=True,
        load_model=False,
        model_output_file_root="byol",
        train_model=True,
    )
    model.train_byol(dataset)
    features = model.run_on_dataset(dataset)
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features, index=dataset.index)
    features.index = features.index.astype(str)
    features.index.name = "objid"
    features.to_parquet(output_dir / "byol_features.parquet")

    metadata = {
        "sources": int(len(features)),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "workers": workers,
        "seed": seed,
        "feature_shape": list(features.shape),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    (output_dir / "byol_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return features


def run_full(
    catalogue_path: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    workers: int = 8,
    seed: int = 42,
) -> dict:
    """Executar BYOL, PCA e Protege e salvar os resultados posteriores."""
    catalogue_path = Path(catalogue_path).expanduser().resolve()
    image_dir = Path(image_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features = train_byol(
        catalogue_path, image_dir, output_dir, epochs, batch_size,
        learning_rate, workers, seed,
    )

    pca = PCA(n_components=0.95)
    reduced = pd.DataFrame(pca.fit_transform(features), index=features.index)
    reduced.index.name = "objid"
    reduced.to_parquet(output_dir / "pca_features.parquet")
    np.save(output_dir / "pca_components.npy", pca.components_)

    catalogue = load_catalogue(catalogue_path)
    labels = evaluation_labels(catalogue)
    result = replay(reduced.loc[labels.index], labels)
    ranking = result.ranking.reset_index()
    ranking.to_csv(output_dir / "evaluation_ranking.csv", index=False)
    result.trace.to_csv(output_dir / "active_learning_trace.csv", index=False)

    confusion = top_k_confusion(ranking)
    metrics = {
        "feature_shape": list(features.shape),
        "pca_shape": list(reduced.shape),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "evaluation_sources": len(labels),
        "high_interest_sources": int(labels.ge(4).sum()),
        "recall_at_100": confusion.recall,
        "precision_at_100": confusion.precision,
        "accuracy_at_100": confusion.accuracy,
        "f1_at_100": confusion.f1,
    }
    (output_dir / "full_run_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics
