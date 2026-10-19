"""Interface de linha de comando para verificação e reprodução completa."""

from __future__ import annotations

import argparse
import json
import sys


def _check(_: argparse.Namespace) -> int:
    from .results import validate_results

    report = validate_results()
    counts = report["confusion"]
    print("Os arquivos de resultado são consistentes.")
    print(
        "100 primeiras posições: "
        f"TP={counts['true_positive']}, FP={counts['false_positive']}, "
        f"FN={counts['false_negative']}, TN={counts['true_negative']}"
    )
    positives = counts["true_positive"] + counts["false_negative"]
    print(
        f"Recall@100: {counts['true_positive']}/{positives} = "
        f"{counts['recall']:.2%}"
    )
    return 0


def _metrics(_: argparse.Namespace) -> int:
    import pandas as pd

    from .config import FROZEN_RESULTS
    from .metrics import recall_at, top_k_confusion

    ranking = pd.read_csv(FROZEN_RESULTS / "evaluation_ranking.csv")
    counts = top_k_confusion(ranking)
    print("Resultado final")
    print(f"  recall@100 : {counts.true_positive}/{counts.positives} = {counts.recall:.2%}")
    print(f"  precisão   : {counts.precision:.2%}")
    print(f"  acurácia   : {counts.accuracy:.2%}")
    print(f"  F1         : {counts.f1:.2%}\n")

    table = recall_at(ranking)
    table["recall"] = table["recall"].map(lambda value: f"{value:.1%}")
    table["random_expected"] = table["random_expected"].round(1)
    table["enrichment"] = table["enrichment"].round(1)
    table = table.rename(columns={
        "K": "K",
        "found": "encontradas",
        "recall": "recall",
        "random_expected": "aleatório_esperado",
        "enrichment": "enriquecimento",
    })
    print(table.to_string(index=False))
    return 0


def _replay(_: argparse.Namespace) -> int:
    from .pipeline import compare_with_saved, replay_saved_result

    result, _, _ = replay_saved_result()
    report = compare_with_saved(result)
    description = "idêntico" if report["exact"] else "equivalente"
    print(f"O replay do Protege é {description} ao resultado fornecido.")
    print(f"  mesmo conjunto das 100 primeiras : {report['same_top_100']}")
    print(f"  mesma sequência de consultas     : {report['same_active_learning_trace']}")
    print(f"  maior diferença entre escores    : {report['max_abs_score_difference']:.2e}")
    confusion = report["confusion"]
    positives = confusion["true_positive"] + confusion["false_negative"]
    print(
        f"  recall@100         : {confusion['true_positive']}/{positives} = "
        f"{confusion['recall']:.2%}"
    )
    return 0


def _full(args: argparse.Namespace) -> int:
    from .full_run import run_full

    metrics = run_full(
        args.catalogue,
        args.image_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        workers=args.workers,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mgcls_protege",
        description="Verificar, repetir ou reproduzir o resultado do MGCLS Protege.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="validar os arquivos de resultado").set_defaults(run=_check)
    commands.add_parser("metrics", help="recalcular as métricas apresentadas").set_defaults(run=_metrics)
    commands.add_parser("replay", help="repetir o Protege a partir das características PCA").set_defaults(run=_replay)
    full = commands.add_parser("full", help="treinar o BYOL e executar o pipeline completo na GPU")
    full.add_argument("--catalogue", required=True)
    full.add_argument("--image-dir", required=True)
    full.add_argument("--output-dir", required=True)
    full.add_argument("--epochs", type=int, default=100)
    full.add_argument("--batch-size", type=int, default=32)
    full.add_argument("--learning-rate", type=float, default=5e-4)
    full.add_argument("--workers", type=int, default=8)
    full.add_argument("--seed", type=int, default=42)
    full.set_defaults(run=_full)
    args = parser.parse_args(argv)

    try:
        return args.run(args)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        print(f"ERRO: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
