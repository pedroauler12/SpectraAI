"""
Loop de experimentos CNN.

Percorre uma lista de configs (E1, E2, E3, E4) e dispara o treino
automaticamente para cada variacao usando o ExperimentRunner.

Uso:
    python -m src.train.run_experiments
    python -m src.train.run_experiments --configs E1_baseline E3_high_dropout
    python -m src.train.run_experiments --limit 50   # teste rapido
"""

import argparse
import sys
from pathlib import Path

# Garantir que o root do projeto esta no sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.experiment_runner import ExperimentRunner
from src.models.cnn_config import list_available_configs


EXPERIMENT_CONFIGS = [
    "E1_baseline",
    "E2_more_filters",
    "E3_high_dropout",
    "E4_wide_dense",
]


def run_all_experiments(
    config_names: list[str] | None = None,
    limit_samples: int | None = None,
    verbose: int = 1,
) -> list[dict]:
    """
    Percorre lista de configs e dispara treino para cada uma.

    Args:
        config_names: Lista de nomes de config YAML (sem .yaml).
                      Se None, usa EXPERIMENT_CONFIGS (E1-E4).
        limit_samples: Limita amostras para teste rapido.
        verbose: Verbosidade do treinamento Keras.

    Returns:
        Lista de dicts com resultado de cada experimento.
    """
    configs = config_names or EXPERIMENT_CONFIGS
    available = list_available_configs()

    print("=" * 60)
    print("LOOP DE EXPERIMENTOS CNN")
    print(f"Configs a executar: {configs}")
    print(f"Configs disponiveis: {available}")
    if limit_samples:
        print(f"Limite de amostras: {limit_samples}")
    print("=" * 60)

    # Validar que todas as configs existem antes de comecar
    missing = [c for c in configs if c not in available]
    if missing:
        print(f"\nERRO: configs nao encontradas: {missing}")
        print(f"Disponiveis: {available}")
        sys.exit(1)

    results = []

    for i, config_name in enumerate(configs, 1):
        print(f"\n{'#' * 60}")
        print(f"# EXPERIMENTO {i}/{len(configs)}: {config_name}")
        print(f"{'#' * 60}")

        try:
            runner = ExperimentRunner(config_name)
            result = runner.run_full_pipeline(
                limit_samples=limit_samples,
                verbose=verbose,
            )
            results.append(result)
        except Exception as e:
            print(f"\nERRO em {config_name}: {e}")
            results.append({"config_name": config_name, "error": str(e)})

    # Resumo comparativo
    _print_summary(results)

    return results


def _print_summary(results: list[dict]) -> None:
    """Printa tabela comparativa dos resultados."""
    print(f"\n{'=' * 60}")
    print("RESUMO COMPARATIVO")
    print(f"{'=' * 60}")
    print(f"{'Config':<20} {'Train Acc':>10} {'Val Acc':>10} {'Val Loss':>10}")
    print("-" * 52)

    for r in results:
        name = r.get("config_name", "?")
        if "error" in r:
            print(f"{name:<20} {'ERRO':>10} {'-':>10} {'-':>10}")
        else:
            train_acc = r.get("final_train_acc", 0)
            val_acc = r.get("final_val_acc", 0)
            val_loss = r.get("final_val_loss", 0)
            print(f"{name:<20} {train_acc:>10.4f} {val_acc:>10.4f} {val_loss:>10.4f}")

    print("=" * 52)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loop de experimentos CNN")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Nomes das configs (sem .yaml). Default: E1-E4",
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

    run_all_experiments(
        config_names=args.configs,
        limit_samples=args.limit,
        verbose=args.verbose,
    )
