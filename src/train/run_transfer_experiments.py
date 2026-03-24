"""
Loop de experimentos Transfer Learning.

Percorre uma lista de configs TL e dispara o treino automaticamente
para cada variacao usando o TransferLearningExperimentRunner.

Uso:
    python -m src.train.run_transfer_experiments
    python -m src.train.run_transfer_experiments --configs tl_baseline tl_lr_high
    python -m src.train.run_transfer_experiments --limit 50   # teste rapido
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.transfer_experiment_runner import TransferLearningExperimentRunner
from src.models.cnn_config import list_available_configs


TL_EXPERIMENT_CONFIGS = [
    "tl_baseline",
    "tl_lr_high",
    "tl_lr_low",
    "tl_bs_large",
    "tl_unfreeze_deep",
    "tl_unfreeze_shallow",
    "tl_dropout_high",
    "tl_dropout_low",
]


def run_all_tl_experiments(
    config_names: list[str] | None = None,
    limit_samples: int | None = None,
    verbose: int = 1,
) -> list[dict]:
    """
    Percorre lista de configs TL e dispara treino para cada uma.

    Args:
        config_names: Lista de nomes de config YAML (sem .yaml).
                      Se None, usa TL_EXPERIMENT_CONFIGS.
        limit_samples: Limita amostras para teste rapido.
        verbose: Verbosidade do treinamento Keras.

    Returns:
        Lista de dicts com resultado de cada experimento.
    """
    configs = config_names or TL_EXPERIMENT_CONFIGS
    available = list_available_configs()

    print("=" * 60)
    print("LOOP DE EXPERIMENTOS — TRANSFER LEARNING")
    print(f"Configs a executar: {configs}")
    if limit_samples:
        print(f"Limite de amostras: {limit_samples}")
    print("=" * 60)

    missing = [c for c in configs if c not in available]
    if missing:
        print(f"\nERRO: configs nao encontradas: {missing}")
        print(f"Disponiveis (tl_*): {[c for c in available if c.startswith('tl_')]}")
        sys.exit(1)

    results = []

    for i, config_name in enumerate(configs, 1):
        print(f"\n{'#' * 60}")
        print(f"# EXPERIMENTO TL {i}/{len(configs)}: {config_name}")
        print(f"{'#' * 60}")

        try:
            runner = TransferLearningExperimentRunner(config_name)
            result = runner.run_full_pipeline(
                limit_samples=limit_samples,
                verbose=verbose,
            )
            results.append(result)
        except Exception as e:
            print(f"\nERRO em {config_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"config_name": config_name, "error": str(e)})

    _print_summary(results)
    return results


def _print_summary(results: list[dict]) -> None:
    """Printa tabela comparativa dos resultados."""
    print(f"\n{'=' * 70}")
    print("RESUMO COMPARATIVO — TRANSFER LEARNING")
    print(f"{'=' * 70}")
    print(
        f"{'Config':<25} {'Acc':>8} {'F1':>8} {'AUC':>8} "
        f"{'Bal Acc':>8} {'Epochs':>7}"
    )
    print("-" * 70)

    for r in results:
        name = r.get("config_name", "?")
        if "error" in r:
            print(f"{name:<25} {'ERRO':>8}")
        else:
            acc = r.get("val_accuracy", 0)
            f1 = r.get("val_f1", 0)
            auc = r.get("val_auc_roc", 0) or 0
            bal = r.get("val_balanced_accuracy", 0)
            ep = r.get("total_epochs", 0)
            print(
                f"{name:<25} {acc:>8.4f} {f1:>8.4f} {auc:>8.4f} "
                f"{bal:>8.4f} {ep:>7}"
            )

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loop de experimentos Transfer Learning"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Nomes das configs (sem .yaml). Default: todas tl_*",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limitar amostras (para teste rapido)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosidade do treinamento (0=silent, 1=progress, 2=epoch)",
    )
    args = parser.parse_args()

    run_all_tl_experiments(
        config_names=args.configs,
        limit_samples=args.limit,
        verbose=args.verbose,
    )
