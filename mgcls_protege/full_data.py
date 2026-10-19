"""Adaptador entre o catálogo MGCLS e o leitor de imagens do Astronomaly."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import image_path, load_catalogue


def build_thumbnail_dataset(catalogue_path: str | Path, image_dir: str | Path, output_dir: str | Path):
    """Carregar os recortes publicados na ordem de canais esperada pelo BYOL."""
    from astronomaly.data_management.image_reader import ImageThumbnailsDataset

    catalogue = load_catalogue(catalogue_path).copy()
    catalogue["filename"] = catalogue["protege_rank"].map(
        lambda rank: str(image_path(image_dir, rank))
    )

    class ChannelFirstThumbnails(ImageThumbnailsDataset):
        """Converter o formato da imagem sem alterar os valores dos pixels."""

        def get_sample(self, index):
            sample = np.asarray(super().get_sample(index))
            if sample.ndim == 2:
                sample = np.repeat(sample[None, :, :], 3, axis=0)
            elif sample.ndim == 3 and sample.shape[-1] in (3, 4):
                sample = np.moveaxis(sample[..., :3], -1, 0)
            elif sample.ndim == 3 and sample.shape[0] in (3, 4):
                sample = sample[:3]
            else:
                raise ValueError(f"formato inesperado do recorte: {sample.shape}")
            return np.ascontiguousarray(sample)

    return ChannelFirstThumbnails(
        catalogue=catalogue,
        output_dir=str(Path(output_dir).resolve()),
        transform_function=None,
        display_transform_function=None,
        fits_format=False,
        check_corrupt_data=False,
    )
