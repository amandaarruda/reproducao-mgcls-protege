"""Gera figuras quantitativas a partir dos CSVs em ``results/additional``."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "additional"


def plot(results_dir: Path) -> list[Path]:
    metrics = pd.read_csv(results_dir / "representation_metrics.csv")
    variance = pd.read_csv(results_dir / "pca_variance.csv")
    figure_dir = results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    created = []

    pca = metrics[metrics.configuration.isin(
        ["PCA-5 + GP", "PCA-10 + GP", "PCA-20 + GP", "PCA-30 + GP", "PCA-30 whitening + GP"]
    )].copy()
    labels = ["5", "10", "20", "30", "30 + whitening"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(labels, pca.hits_at_100, color=["#4C78A8", "#72B7B2", "#F2CF5B", "#1B4F72", "#E45756"])
    ax.axhline(43, color="#333333", linestyle="--", linewidth=1, label="Baseline: 43/86")
    ax.set_xlabel("Componentes PCA usadas pelo processo gaussiano")
    ax.set_ylabel("Fontes de alto interesse no top-100")
    ax.set_ylim(0, 55)
    for bar, hits in zip(bars, pca.hits_at_100):
        ax.annotate(f"{int(hits)}/86", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    created.extend(_save(fig, figure_dir / "pca_ablation"))

    names = ["PCA-30 + GP", "PCA-5 + GP", "PCA-10 + GP", "PCA-20 + GP",
             "PCA-30 whitening + GP", "Isolation Forest PCA-30"]
    colors = ["#1B4F72", "#4C78A8", "#72B7B2", "#F2CF5B", "#E45756", "#7A5195"]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for name, color in zip(names, colors):
        ranking = pd.read_csv(results_dir / _ranking_file(name))
        high_interest = ranking.high_interest.to_numpy(dtype=int)[:300]
        ax.plot(np.arange(1, len(high_interest) + 1), np.cumsum(high_interest),
                label=name, color=color, linewidth=2 if name == "PCA-30 + GP" else 1.5)
    ax.plot(np.arange(1, 301), np.arange(1, 301) * 86 / 1031, "--", color="#777777",
            linewidth=1.3, label="Aleatório (esperado)")
    ax.axvline(100, color="#444444", linestyle=":", linewidth=1)
    ax.set(xlim=(1, 300), xlabel="Fontes inspecionadas (K)", ylabel="Fontes de alto interesse recuperadas")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    created.extend(_save(fig, figure_dir / "recovery_curves"))

    variance = variance[variance.components.isin([5, 10, 20, 30])]
    recall = pca[pca.configuration.isin([f"PCA-{n} + GP" for n in (5, 10, 20, 30)])]
    recall = recall.set_index("components")
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    colors = {5: "#4C78A8", 10: "#72B7B2", 20: "#F2CF5B", 30: "#1B4F72"}
    for _, row in variance.iterrows():
        n = int(row.components)
        y = 100 * float(recall.loc[n, "recall_at_100"])
        x = 100 * float(row.explained_variance)
        ax.scatter(x, y, color=colors[n], s=65, zorder=3)
        ax.annotate(str(n), (x, y), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Variância retida estimada (%)")
    ax.set_ylabel("Recall@100 (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    created.extend(_save(fig, figure_dir / "variance_recall"))
    files = sorted(path for path in results_dir.rglob("*")
                   if path.is_file() and path.name != "SHA256SUMS")
    checksums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(results_dir).as_posix()}\n"
        for path in files
    )
    (results_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    return created


def _ranking_file(name: str) -> str:
    return {
        "PCA-5 + GP": "ranking_pca_5.csv",
        "PCA-10 + GP": "ranking_pca_10.csv",
        "PCA-20 + GP": "ranking_pca_20.csv",
        "PCA-30 + GP": "ranking_pca_30.csv",
        "PCA-30 whitening + GP": "ranking_pca_30_whitening.csv",
        "Isolation Forest PCA-30": "ranking_isolation_forest_pca30.csv",
    }[name]


def _save(figure, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in paths:
        figure.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(figure)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    paths = plot(args.results_dir.expanduser().resolve())
    print(f"Figuras salvas em {args.results_dir / 'figures'} ({len(paths)} arquivos).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
