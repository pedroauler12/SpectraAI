"""Entrypoint oficial da entrega A11."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline end-to-end do A11")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
        help="Caminho para o arquivo config.yaml do A11.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Limita o numero de amostras para smoke tests.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reutiliza o modelo salvo em outputs/models/best_model.keras.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Pula exportacao de predicoes e visualizacoes finais.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretorio alternativo para salvar outputs do A11.",
    )
    return parser.parse_args(argv)


def _load_runtime_dependencies() -> dict[str, Any]:
    try:
        from artefatos.a11_pipeline_e2e.src.evaluation import (
            build_summary,
            save_summary_files,
            save_visualizations,
        )
        from artefatos.a11_pipeline_e2e.src.inference import export_test_predictions
        from artefatos.a11_pipeline_e2e.src.preprocessing import (
            ensure_output_dirs,
            load_pipeline_config,
            prepare_split_data,
            validate_input_files,
        )
        from artefatos.a11_pipeline_e2e.src.training import run_training_pipeline
    except ModuleNotFoundError as exc:
        raise SystemExit(_format_missing_dependency_error(exc)) from exc

    return {
        "build_summary": build_summary,
        "ensure_output_dirs": ensure_output_dirs,
        "export_test_predictions": export_test_predictions,
        "load_pipeline_config": load_pipeline_config,
        "prepare_split_data": prepare_split_data,
        "run_training_pipeline": run_training_pipeline,
        "save_summary_files": save_summary_files,
        "save_visualizations": save_visualizations,
        "validate_input_files": validate_input_files,
    }


def run_pipeline(
    args: argparse.Namespace,
    deps: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deps = deps or _load_runtime_dependencies()
    config = deps["load_pipeline_config"](args.config)
    deps["validate_input_files"](config)
    output_paths = deps["ensure_output_dirs"](config, args.output_dir)

    split_data = deps["prepare_split_data"](config, limit_samples=args.limit_samples)
    execution = deps["run_training_pipeline"](
        config=config,
        split_data=split_data,
        output_paths=output_paths,
        skip_train=args.skip_train,
    )

    predictions_path = None
    if not args.skip_inference:
        predictions_df = deps["export_test_predictions"](
            model=execution["runner"].model,
            test_dataset=execution["tf_data"]["test_ds"],
            image_ids=split_data["image_ids_test"],
            threshold=float(config["evaluation"]["threshold_default"]),
            output_path=output_paths["predictions"] / "test_predictions.csv",
        )
        predictions_path = output_paths["predictions"] / "test_predictions.csv"
        deps["save_visualizations"](
            predictions_df=predictions_df,
            output_dir=output_paths["visualizations"],
        )

    summary = deps["build_summary"](
        config=config,
        split_meta=split_data["split_meta"],
        result=execution["result"],
        model_path=execution["model_path"],
        history_path=execution["history_path"],
        predictions_path=predictions_path,
    )
    deps["save_summary_files"](
        summary=summary,
        json_path=output_paths["metrics"] / "summary.json",
        csv_path=output_paths["metrics"] / "summary.csv",
    )
    return summary


def _format_missing_dependency_error(exc: ModuleNotFoundError) -> str:
    missing_module = exc.name or "dependencia_desconhecida"
    requirements_path = Path("artefatos/a11_pipeline_e2e/requirements.txt")
    return (
        "Dependencia Python ausente para executar o A11.\n"
        f"Modulo nao encontrado: {missing_module}\n\n"
        "Instale as dependencias do artefato com:\n"
        f"python3 -m pip install -r {requirements_path}\n\n"
        "Se estiver usando ambiente virtual, ative-o antes de executar o pipeline."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_pipeline(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
