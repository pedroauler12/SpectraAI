#!/usr/bin/env python3
"""
Script de consolidação de métricas do A11 (Pipeline Final).

Extrai métricas do summary.json do A11 e exporta em formato padronizado (CSV/JSON)
para fácil referência no artigo e validação de reprodutibilidade.

Uso:
    python -m src.utils.consolidate_a11_metrics

Saída:
    - outputs/a11_metrics_final.csv
    - outputs/a11_metrics_final.json
"""

import json
import sys
from pathlib import Path

import pandas as pd


def main():
    # Localizar caminhos
    project_root = Path(__file__).parent.parent
    a11_summary = project_root / "artefatos" / "a11_pipeline_e2e" / "outputs" / "metrics" / "summary.json"
    output_dir = project_root / "outputs"

    # Validar entrada
    if not a11_summary.exists():
        print(f" Erro: arquivo não encontrado: {a11_summary}")
        print("\n Dica: Execute o A11 primeiro com:")
        print("   python -m artefatos.a11_pipeline_e2e")
        sys.exit(1)

    # Carregar dados
    print(f" Lendo: {a11_summary}")
    with open(a11_summary, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Criar DataFrame com colunas bem organizadas
    metrics_row = {
        "Sprint": "A11",
        "Artefato": "Pipeline End-to-End",
        "Modelo": summary.get("model_name", "—"),
        "Seed": summary.get("seed", "—"),
        "Timestamp": summary.get("timestamp", "—"),
        "N_Total": summary.get("n_total", "—"),
        "N_Treino": summary.get("n_train", "—"),
        "N_Validação": summary.get("n_val", "—"),
        "N_Teste": summary.get("n_test", "—"),
        "Acurácia": f"{summary.get('test_accuracy', 0):.4f}",
        "Precisão": f"{summary.get('test_precision', 0):.4f}",
        "Recall": f"{summary.get('test_recall', 0):.4f}",
        "F1-Score": f"{summary.get('test_f1', 0):.4f}",
        "Balanced_Accuracy": f"{summary.get('test_balanced_accuracy', 0):.4f}",
        "ROC-AUC": f"{summary.get('test_roc_auc', 0):.4f}",
        "PR-AUC": f"{summary.get('test_pr_auc', 0):.4f}",
        "Threshold": summary.get("threshold", "—"),
        "Head_Epochs": summary.get("head_epochs", "—"),
        "FT_Epochs": summary.get("ft_epochs", "—"),
        "Total_Epochs": summary.get("total_epochs", "—"),
        "Tempo_Treino_s": f"{summary.get('training_time_seconds', 0):.2f}",
        "Modelo_Path": summary.get("model_path", "—"),
        "Predicoes_Path": summary.get("predictions_path", "—"),
    }

    df = pd.DataFrame([metrics_row])

    # Salvar em CSV
    output_csv = output_dir / "a11_metrics_final.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f" Exportado (CSV): {output_csv}")

    # Salvar em JSON também
    output_json = output_dir / "a11_metrics_final.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics_row, f, indent=2, ensure_ascii=False)
    print(f" Exportado (JSON): {output_json}")

    # Exibir resumo
    print("\n" + "═" * 70)
    print("RESUMO DAS MÉTRICAS DO A11")
    print("═" * 70)
    summary_metrics = {
        "Modelo": metrics_row["Modelo"],
        "Acurácia": metrics_row["Acurácia"],
        "Precisão": metrics_row["Precisão"],
        "Recall": metrics_row["Recall"],
        "F1-Score": metrics_row["F1-Score"],
        "ROC-AUC": metrics_row["ROC-AUC"],
        "PR-AUC": metrics_row["PR-AUC"],
        "Amostras (Teste)": metrics_row["N_Teste"],
        "Épocas": metrics_row["Total_Epochs"],
        "Tempo Treino": f"{metrics_row['Tempo_Treino_s']}s",
        "Timestamp": metrics_row["Timestamp"],
    }
    for key, val in summary_metrics.items():
        print(f"  {key:<20} {val:>15}")
    print("═" * 70)
    print("\n✔ Consolidação concluída com sucesso!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
