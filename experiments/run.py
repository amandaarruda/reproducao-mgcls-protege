"""Executa as análises adicionais da reprodução a partir dos resultados salvos.

O script não treina o BYOL nem ajusta novamente o PCA. Ele usa as coordenadas
PCA e o catálogo incluídos em ``results/final``. Para cada análise, grava uma
tabela de métricas e os rankings necessários para conferir ou refazer as figuras.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
import sklearn

from mgcls_protege.config import FROZEN_RESULTS, ProtegeSettings
from mgcls_protege.protege import replay

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "additional"
HIGH_INTEREST_LABEL = 4
N_POSITIVES = 86
N_EVALUATION = 1031
N_CATALOGUE = 6161
N_QUERIES = 100
PCA_30_VARIANCE = 0.9523180723190308


def load_inputs(results_dir: Path = FROZEN_RESULTS) -> dict:
    """Carrega os resultados públicos e confere as dimensões esperadas."""
    catalogue = pd.read_csv(results_dir / "catalogue.csv", dtype={"objid": str})
    catalogue = catalogue.set_index("objid")
    features = pd.read_parquet(results_dir / "pca_features.parquet")
    features.index = features.index.astype(str)
    ranking = pd.read_csv(results_dir / "evaluation_ranking.csv", dtype={"objid": str})
    queried = ranking.loc[ranking.human_label >= 0].set_index("objid").human_label

    labels = catalogue["evaluation_subset_author_ML_score"]
    evaluation_ids = labels[labels >= 0].index.astype(str)
    labels = labels.loc[evaluation_ids].astype(int)
    if features.shape != (N_CATALOGUE, 30):
        raise ValueError(f"Esperadas {N_CATALOGUE} linhas e 30 componentes PCA; recebidas {features.shape}.")
    if len(labels) != N_EVALUATION or int(labels.ge(HIGH_INTEREST_LABEL).sum()) != N_POSITIVES:
        raise ValueError("O subconjunto de avaliação não corresponde aos resultados fornecidos.")
    if len(queried) != N_QUERIES or not queried.index.is_unique:
        raise ValueError("O resultado fornecido não contém exatamente 100 fontes consultadas.")
    if not np.array_equal(queried.to_numpy(), labels.loc[queried.index].to_numpy()):
        raise ValueError("Os rótulos consultados divergem do catálogo fornecido.")
    if not features.index.is_unique or not features.index.isin(catalogue.index).all():
        raise ValueError("Índices das características inválidos ou sem correspondência no catálogo.")
    return {"catalogue": catalogue, "features": features, "evaluation_ids": evaluation_ids,
            "labels": labels, "queried": queried, "results_dir": results_dir}


def make_ranking(ids: pd.Index, scores: np.ndarray, labels: pd.Series,
                 queried: pd.Index | None = None) -> pd.DataFrame:
    """Monta um ranking padronizado; posições começam em zero."""
    values = np.asarray(scores, dtype=float)
    if len(values) != len(ids) or not np.isfinite(values).all():
        raise ValueError("Os escores precisam ser finitos e corresponder a todas as fontes.")
    order = np.argsort(-values, kind="stable")
    ordered_ids = ids[order].astype(str)
    ordered_labels = labels.loc[ordered_ids].astype(int).to_numpy()
    result = pd.DataFrame({"objid": ordered_ids, "rank": np.arange(len(order)),
                           "score": values[order], "ground_truth_label": ordered_labels})
    result["high_interest"] = result.ground_truth_label >= HIGH_INTEREST_LABEL
    result["cumulative_high_interest"] = result.high_interest.cumsum()
    result["queried"] = result.objid.isin(queried) if queried is not None else False
    return result


def measure(name: str, ranking: pd.DataFrame, **details) -> dict:
    """Calcula métricas para um ranking, incluindo o corte principal K=100."""
    positive = ranking.high_interest.to_numpy(dtype=bool)
    score = ranking.score.to_numpy(dtype=float)
    hits = int(positive[:100].sum())
    total_positive = int(positive.sum())
    row = {"configuration": name, "evaluation_sources": len(ranking),
           "high_interest_sources": total_positive, "hits_at_100": hits,
           "recall_at_100": hits / total_positive, "precision_at_100": hits / 100,
           "roc_auc": float(roc_auc_score(positive, score)),
           "average_precision": float(average_precision_score(positive, score))}
    for k in (20, 50, 200, 300):
        found = int(positive[:k].sum())
        row[f"hits_at_{k}"] = found
        row[f"recall_at_{k}"] = found / total_positive
        row[f"precision_at_{k}"] = found / k
    return {**row, **details}


def fixed_label_gp(features: pd.DataFrame, ids: pd.Index, labels: pd.Series,
                   queried: pd.Series) -> pd.DataFrame:
    """Ajusta o processo gaussiano às mesmas 100 notas em todas as variantes."""
    model = GaussianProcessRegressor(kernel=Matern() + WhiteKernel())
    model.fit(features.loc[queried.index].to_numpy(), queried.to_numpy())
    scores = model.predict(features.loc[ids].to_numpy())
    return make_ranking(ids, scores, labels, queried.index)


def representation_experiments(data: dict, output: Path) -> pd.DataFrame:
    features, ids = data["features"], data["evaluation_ids"]
    labels, queried = data["labels"], data["queried"]
    all_variance = features.to_numpy(dtype=np.float64).var(axis=0, ddof=1)
    variance_fraction = all_variance.cumsum() / all_variance.sum() * PCA_30_VARIANCE
    rows, rankings = [], {}

    for n_components in (5, 10, 20, 30):
        name = f"PCA-{n_components} + GP"
        rank = fixed_label_gp(features.iloc[:, :n_components], ids, labels, queried)
        rankings[name] = rank
        rows.append(measure(name, rank, components=n_components,
                            explained_variance=float(variance_fraction[n_components - 1]),
                            labels_used=N_QUERIES, method="Gaussian process with fixed labels"))
        rank.to_csv(output / f"ranking_pca_{n_components}.csv", index=False)

    scales = features.iloc[:, :30].std(axis=0, ddof=0).replace(0, 1)
    whitened = features.iloc[:, :30].divide(scales, axis=1)
    name = "PCA-30 whitening + GP"
    rank = fixed_label_gp(whitened, ids, labels, queried)
    rankings[name] = rank
    rows.append(measure(name, rank, components=30, explained_variance=float(variance_fraction[-1]),
                        labels_used=N_QUERIES, method="Gaussian process with fixed labels; whitening"))
    rank.to_csv(output / "ranking_pca_30_whitening.csv", index=False)

    forest = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
    forest.fit(features.iloc[:, :30].to_numpy())
    anomaly_scores = -forest.score_samples(features.loc[ids].iloc[:, :30].to_numpy())
    rank = make_ranking(ids, anomaly_scores, labels)
    rankings["Isolation Forest PCA-30"] = rank
    rows.append(measure("Isolation Forest PCA-30", rank, components=30,
                        explained_variance=float(variance_fraction[-1]), labels_used=0,
                        method="Unsupervised isolation forest"))
    rank.to_csv(output / "ranking_isolation_forest_pca30.csv", index=False)

    metrics = pd.DataFrame(rows)
    baseline_hits = int(metrics.loc[metrics.configuration == "PCA-30 + GP", "hits_at_100"].iloc[0])
    metrics["delta_hits_vs_pca30"] = metrics.hits_at_100 - baseline_hits
    metrics.to_csv(output / "representation_metrics.csv", index=False)
    pd.DataFrame([{"components": n, "explained_variance": float(variance_fraction[n - 1])}
                  for n in (5, 10, 20, 30)]).to_csv(output / "pca_variance.csv", index=False)

    baseline = rankings["PCA-30 + GP"].set_index("objid")["rank"]
    alternative = rankings["PCA-5 + GP"].set_index("objid")["rank"]
    positive_ids = labels[labels >= HIGH_INTEREST_LABEL].index.astype(str)
    cases = pd.DataFrame({"label": labels.loc[positive_ids],
                          "baseline_rank": baseline.loc[positive_ids] + 1,
                          "pca5_rank": alternative.loc[positive_ids] + 1})
    cases.index.name = "objid"
    cases["rank_change"] = cases.pca5_rank - cases.baseline_rank
    cases["top100_change"] = np.select(
        [(cases.baseline_rank > 100) & (cases.pca5_rank <= 100),
         (cases.baseline_rank <= 100) & (cases.pca5_rank > 100)],
        ["entered", "left"], default="unchanged")
    cases.sort_values("rank_change").to_csv(output / "qualitative_cases_pca5.csv")
    return metrics


def active_replay_experiments(data: dict, output: Path) -> pd.DataFrame:
    rows = []
    for components in (20, 30):
        features = data["features"].loc[data["evaluation_ids"]].iloc[:, :components]
        for tradeoff in (0.0, 3.0):
            setting = ProtegeSettings(n_labels=N_QUERIES, batch_size=10, ei_tradeoff=tradeoff)
            result = replay(features, data["labels"], setting)
            name = f"PCA-{components} epsilon-{tradeoff:g}"
            ranking = result.ranking.reset_index().rename(
                columns={"trained_score": "score", "human_high_interest": "high_interest"})
            ranking.to_csv(output / f"ranking_replay_pca{components}_epsilon{tradeoff:g}.csv", index=False)
            row = measure(name, ranking, components=components, ei_tradeoff=tradeoff,
                          labels_used=N_QUERIES, method="Active PROTEGE replay")
            row["matches_frozen_top100"] = (
                set(ranking.sort_values("rank").head(100).objid)
                == set(pd.read_csv(data["results_dir"] / "evaluation_ranking.csv").head(100).objid))
            rows.append(row)
            result.trace.to_csv(output / f"trace_replay_pca{components}_epsilon{tradeoff:g}.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(output / "active_replay_metrics.csv", index=False)
    return table


def svm_experiments(data: dict, output: Path) -> pd.DataFrame:
    rows = []
    query_ids, query_labels = data["queried"].index, data["queried"]
    target = (query_labels >= HIGH_INTEREST_LABEL).astype(int)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    parameter_grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1.0]}
    for components in (20, 30):
        features = data["features"].loc[data["evaluation_ids"]].iloc[:, :components]
        model = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), parameter_grid,
                             scoring="average_precision", cv=folds, n_jobs=1, refit=True)
        model.fit(features.loc[query_ids].to_numpy(), target.to_numpy())
        scores = model.best_estimator_.decision_function(features.to_numpy())
        ranking = make_ranking(data["evaluation_ids"], scores, data["labels"], query_ids)
        name = f"Balanced RBF SVM PCA-{components}"
        ranking.to_csv(output / f"ranking_svm_pca{components}.csv", index=False)
        rows.append(measure(name, ranking, components=components, labels_used=N_QUERIES,
                            method="Balanced RBF SVM; 5-fold AP grid search",
                            best_C=float(model.best_params_["C"]), best_gamma=model.best_params_["gamma"],
                            cross_validation_average_precision=float(model.best_score_)))
        cv = pd.DataFrame(model.cv_results_).sort_values("rank_test_score")
        cv[["param_C", "param_gamma", "mean_test_score", "std_test_score", "rank_test_score"]].to_csv(
            output / f"svm_grid_pca{components}.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "svm_metrics.csv", index=False)
    return metrics


def write_run_info(output: Path) -> None:
    summary = {"input": "results/final (frozen catalogue, PCA features and evaluation labels)",
               "byol_retrained": False, "pca_refitted": False,
               "evaluation_sources": N_EVALUATION, "high_interest_rule": "score >= 4",
               "high_interest_sources": N_POSITIVES, "fixed_label_budget": N_QUERIES,
               "software": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
                            "numpy": np.__version__, "pandas": pd.__version__},
               "parameters": {
                   "fixed_label_gp": {"kernel": "Matern() + WhiteKernel()", "tradeoff": 3.0,
                                       "query_labels": 100},
                   "pca_dimensions": [5, 10, 20, 30],
                   "whitening": "divide each PCA-30 component by its population standard deviation over all sources; do not center",
                   "isolation_forest": {"estimators": 200, "random_state": 42, "fit_sources": N_CATALOGUE},
                   "active_replay": {"dimensions": [20, 30], "epsilon": [0, 3], "batch_size": 10,
                                     "labels_per_run": 100},
                   "svm": {"kernel": "RBF", "class_weight": "balanced", "C": [0.1, 1, 10, 100],
                           "gamma": ["scale", 0.01, 0.1, 1.0], "cv_folds": 5,
                           "cv_scoring": "average_precision", "cv_random_state": 42}},
               "interpretation": "Exploratory comparisons on the supplied run; they do not replace the PROTEGE reproduction.",
               "outputs": sorted(path.relative_to(output).as_posix()
                                  for path in output.rglob("*")
                                  if path.is_file() and path.name not in {"run_info.json", "SHA256SUMS"})}
    (output / "run_info.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    files = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    checksums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output).as_posix()}\n"
        for path in files
    )
    (output / "SHA256SUMS").write_text(checksums, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=FROZEN_RESULTS,
                        help="pasta dos resultados congelados; padrão: results/final")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="destino dos arquivos; o padrão não é sobrescrito")
    parser.add_argument("--analysis", choices=("representations", "replay", "svm", "all"), default="all")
    parser.add_argument("--refresh", action="store_true",
                        help="atualizar os arquivos gerados na pasta de saída indicada")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.refresh:
        raise SystemExit(f"O diretório de saída já contém arquivos; preservando-os: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data = load_inputs(args.results_dir.expanduser().resolve())
    if args.analysis in ("representations", "all"):
        representation_experiments(data, output)
    if args.analysis in ("replay", "all"):
        active_replay_experiments(data, output)
    if args.analysis in ("svm", "all"):
        svm_experiments(data, output)
    write_run_info(output)
    print(f"Análises concluídas. Arquivos em: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
