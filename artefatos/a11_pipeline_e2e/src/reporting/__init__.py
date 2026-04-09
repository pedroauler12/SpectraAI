from __future__ import annotations

import os
from pathlib import Path


def execute_notebook_report(
    *,
    notebook_path: str | Path,
    repo_root: str | Path,
    output_dir: str | Path | None = None,
    timeout_seconds: int = 3600,
    kernel_name: str = "python3",
) -> dict[str, Path]:
    """
    Executa o notebook oficial do A11 de forma headless e salva uma copia
    executada para reproducao dos resultados analiticos.
    """
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "dependencia_desconhecida"
        raise ModuleNotFoundError(
            "Dependencia ausente para executar o notebook do A11. "
            "Instale nbconvert, nbformat e ipykernel pelo requirements do artefato. "
            f"Modulo nao encontrado: {missing_module}"
        ) from exc

    notebook_path = Path(notebook_path).resolve()
    repo_root = Path(repo_root).resolve()

    if output_dir is None:
        artifact_root = notebook_path.parents[1]
        output_dir = artifact_root / "outputs" / "notebooks"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    executed_notebook_path = output_dir / f"{notebook_path.stem}.executed.ipynb"
    notebook_visualizations_dir = (
        notebook_path.parents[1] / "outputs" / "notebook_visualizations"
    ).resolve()
    mpl_config_dir = output_dir / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)

    with notebook_path.open("r", encoding="utf-8") as file:
        notebook = nbformat.read(file, as_version=4)

    previous_mplconfigdir = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    try:
        executor = ExecutePreprocessor(
            timeout=timeout_seconds,
            kernel_name=kernel_name,
        )
        executor.preprocess(notebook, {"metadata": {"path": str(repo_root)}})
    finally:
        if previous_mplconfigdir is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_mplconfigdir

    with executed_notebook_path.open("w", encoding="utf-8") as file:
        nbformat.write(notebook, file)

    return {
        "executed_notebook_path": executed_notebook_path,
        "notebook_visualizations_dir": notebook_visualizations_dir,
    }
