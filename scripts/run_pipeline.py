"""
run_pipeline.py — Entrypoint unificado do pipeline Spectra (A11).

Encadeia automaticamente as etapas:
  preprocess → train → evaluate → infer

Uso básico:
    python run_pipeline.py                          # train + infer com config baseline
    python run_pipeline.py --config E1_baseline     # config específica
    python run_pipeline.py --stages train infer     # etapas explícitas
    python run_pipeline.py --stages preprocess train infer  # pipeline completo
    python run_pipeline.py --limit 200 --verbose 0  # execução rápida para teste
    python run_pipeline.py --list-configs           # lista configs disponíveis

Etapas disponíveis:
  preprocess  Baixa e empilha tiles ASTER via NASA EarthAccess (requer .netrc)
  train       Carrega dados, constrói CNN, treina e avalia (métricas no val set)
  infer       Roda batch_predict no val set e salva CSV de predições
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Garantir que o root do projeto está no sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─── Constantes ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG = "baseline"
DEFAULT_STAGES = ["train", "infer"]


# ─── Etapas do pipeline ───────────────────────────────────────────────────────

def stage_preprocess(args: argparse.Namespace) -> None:
    """
    Etapa 1: Download e empilhamento de tiles ASTER.

    Requer arquivo .netrc com credenciais NASA EarthAccess.
    Configurações lidas de src/tiles/config.py.
    """
    _header("ETAPA 1/3 — PRÉ-PROCESSAMENTO (tiles ASTER)")

    try:
        from src.tiles.pipeline import run as tiles_run
        from src.tiles.config import Config as TilesConfig
    except ImportError as e:
        _abort(f"Dependência não encontrada para pré-processamento: {e}")

    netrc_path = args.netrc or str(Path.home() / ".netrc")
    if not Path(netrc_path).exists():
        _abort(
            f"Arquivo .netrc não encontrado em '{netrc_path}'.\n"
            "Crie-o com suas credenciais NASA EarthAccess antes de usar --stages preprocess."
        )

    cfg = TilesConfig()
    print(f"  netrc:    {netrc_path}")
    print(f"  out_root: {cfg.out_root}")

    tiles_run(cfg, netrc_path=netrc_path, limit_rows=args.limit)
    print("\nPré-processamento concluído.")


def stage_train(args: argparse.Namespace):
    """
    Etapas 2-4: Carrega dados, treina CNN e avalia.

    Usa ExperimentRunner com a config YAML selecionada.
    Retorna o runner após o pipeline completo (para reutilizar em infer).
    """
    _header("ETAPA 2/3 — TREINAMENTO + AVALIAÇÃO")

    from src.models.experiment_runner import ExperimentRunner

    runner = ExperimentRunner(args.config)
    result = runner.run_full_pipeline(
        limit_samples=args.limit,
        verbose=args.verbose,
    )

    print(f"\n  Experimento salvo em: {result['experiment_dir']}")
    return runner, result


def stage_infer(args: argparse.Namespace, runner=None, result: dict = None) -> dict:
    """
    Etapa 3/3: Inferência em lote e salvamento de predições CSV.

    Se runner for fornecido (logo após treinamento), usa o modelo e val set
    já em memória. Caso contrário, carrega o modelo do disco a partir de
    --model-path.
    """
    _header("ETAPA 3/3 — INFERÊNCIA EM LOTE")

    from src.inference.batch_predict import batch_predict

    # Resolver modelo e dados de entrada
    if runner is not None and runner.model is not None:
        model = runner.model
        X_infer = runner.X_val
        y_true = runner.y_val
        source = "val set (do treinamento atual)"
    elif args.model_path:
        import numpy as np
        import tensorflow as tf

        model_path = Path(args.model_path)
        if not model_path.exists():
            _abort(f"Modelo não encontrado: {model_path}")
        print(f"  Carregando modelo: {model_path}")
        model = tf.keras.models.load_model(model_path)

        if not args.infer_data:
            _abort(
                "Sem runner ativo, forneça --infer-data com caminho para CSV de pixels."
            )
        import pandas as pd
        from src.models.cnn_data_prep import prepare_cnn_inputs

        df = pd.read_csv(args.infer_data)
        result_prep = prepare_cnn_inputs(df, labels=None)
        X_infer = result_prep["X"]
        y_true = None
        source = args.infer_data
    else:
        _abort(
            "Nenhum modelo disponível para inferência.\n"
            "Execute a etapa 'train' antes de 'infer', ou forneça --model-path."
        )

    print(f"  Fonte dos dados: {source}")
    print(f"  Amostras:        {X_infer.shape[0]}")

    infer_results = batch_predict(
        model,
        X_infer,
        batch_size=args.infer_batch_size,
        return_proba=True,
    )

    # Resolver diretório de predições a partir do config (sem hardcode)
    if runner is not None:
        pred_dir_str = runner.config.get("output", {}).get("predictions_dir", "outputs/predictions")
    else:
        from src.models.cnn_config import load_config as _load_config
        _cfg = _load_config(args.config)
        pred_dir_str = _cfg.get("output", {}).get("predictions_dir", "outputs/predictions")
    predictions_dir = Path(pred_dir_str) if Path(pred_dir_str).is_absolute() else ROOT / pred_dir_str
    predictions_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = predictions_dir / f"predictions_{args.config}_{timestamp}.csv"
    df_preds = infer_results["dataframe"]

    if y_true is not None:
        import numpy as np
        df_preds["true_label"] = y_true

    df_preds.to_csv(csv_path, index=False)
    print(f"\n  Predições salvas: {csv_path}")
    print(f"  Tempo de inferência: {infer_results['inference_time']:.2f}s")

    return infer_results


# ─── Orquestrador principal ───────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.list_configs:
        from src.models.cnn_config import list_available_configs
        configs = list_available_configs()
        print("Configurações disponíveis:")
        for c in configs:
            print(f"  {c}")
        return

    stages = args.stages
    _validate_stages(stages)

    # Carregar config para obter seed antes de qualquer operação aleatória
    from src.models.cnn_config import load_config as _load_config
    from src.reprodutibilidade import set_global_seed
    _cfg = _load_config(args.config)
    set_global_seed(_cfg.get("seed", 42))

    print("\n" + "=" * 60)
    print("SPECTRA — PIPELINE E2E (A11)")
    print("=" * 60)
    print(f"  Config:    {args.config}")
    print(f"  Seed:      {_cfg.get('seed', 42)}")
    print(f"  Etapas:    {' → '.join(stages)}")
    if args.limit:
        print(f"  Limite:    {args.limit} amostras")
    print("=" * 60)

    wall_start = time.time()
    runner = None
    train_result = None

    if "preprocess" in stages:
        stage_preprocess(args)

    if "train" in stages:
        runner, train_result = stage_train(args)

    if "infer" in stages:
        stage_infer(args, runner=runner, result=train_result)

    elapsed = time.time() - wall_start
    print("\n" + "=" * 60)
    print(f"Pipeline concluído em {elapsed:.1f}s")
    print("=" * 60)


# ─── Argumentos CLI ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline E2E Spectra — executa preprocess → train → infer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        metavar="NAME",
        help=f"Nome da config YAML em src/models/configs/ (sem .yaml). Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=DEFAULT_STAGES,
        choices=["preprocess", "train", "infer"],
        metavar="STAGE",
        help=(
            "Etapas a executar em ordem. Escolha: preprocess, train, infer. "
            f"Default: {' '.join(DEFAULT_STAGES)}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limitar N amostras de treinamento (útil para testes rápidos).",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosidade do Keras: 0=silencioso, 1=barra de progresso, 2=por época. Default: 1",
    )

    # Opções específicas de preprocess
    preproc = parser.add_argument_group("preprocess")
    preproc.add_argument(
        "--netrc",
        type=str,
        default=None,
        metavar="PATH",
        help="Caminho para .netrc com credenciais NASA EarthAccess. Default: ~/.netrc",
    )

    # Opções específicas de infer (modo standalone)
    infer = parser.add_argument_group("infer (modo standalone)")
    infer.add_argument(
        "--model-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Caminho para modelo .keras salvo (quando infer roda sem train).",
    )
    infer.add_argument(
        "--infer-data",
        type=str,
        default=None,
        metavar="CSV",
        help="CSV de pixels para inferência standalone (quando --model-path é usado).",
    )
    infer.add_argument(
        "--infer-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Tamanho do mini-batch na inferência. Default: processa tudo de uma vez.",
    )

    # Utilitários
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Lista todas as configs YAML disponíveis e sai.",
    )

    return parser.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _abort(message: str) -> None:
    print(f"\nERRO: {message}", file=sys.stderr)
    sys.exit(1)


def _validate_stages(stages: list) -> None:
    valid = {"preprocess", "train", "infer"}
    invalid = set(stages) - valid
    if invalid:
        _abort(f"Etapas inválidas: {invalid}. Escolha entre: {valid}")

    # Verificar ordenação lógica: preprocess antes de train, train antes de infer
    order = {"preprocess": 0, "train": 1, "infer": 2}
    if stages != sorted(stages, key=lambda s: order[s]):
        _abort(
            f"Etapas fora de ordem: {stages}. "
            "A ordem correta é: preprocess → train → infer"
        )

    # Avisar se infer está sem train e sem --model-path (validado depois, no stage)
    if "infer" in stages and "train" not in stages:
        print(
            "AVISO: 'infer' sem 'train' — certifique-se de fornecer --model-path.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
