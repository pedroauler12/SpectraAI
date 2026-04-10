from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap


def resolve_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "src").exists():
        return cwd
    for parent in [cwd, *cwd.parents]:
        if (parent / "src").exists():
            return parent
    raise FileNotFoundError("Nao foi possivel localizar a raiz do projeto.")


PROJECT_ROOT = resolve_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


METADATA_SHEET = "Banco de Dados Positivo-Negativ"

VISUAL_SECTIONS = [
    (
        "gradcam_examples.png",
        "Onde o modelo prestou mais atencao",
        "Mostra quais partes da imagem mais pesaram na leitura do modelo em alguns exemplos.",
    ),
    (
        "spectral_analysis.png",
        "Comportamento espectral",
        "Resume como os sinais das bandas mudam entre grupos e ajuda a ligar o resultado ao contexto geologico.",
    ),
    (
        "channel_adapter_importance.png",
        "Bandas que mais ajudaram",
        "Indica quais bandas carregaram mais informacao util para separar areas mais e menos promissoras.",
    ),
    (
        "prospectivity_maps.png",
        "Distribuicao das areas promissoras",
        "Mostra como os sinais mais fortes aparecem no territorio e ajuda a localizar concentracoes de interesse.",
    ),
]

TECHNICAL_FIGURES = [
    (
        "roc_pr_curves.png",
        "Curvas de desempenho",
        "Ajudam a entender se o modelo separa bem as areas mais promissoras das menos promissoras.",
    ),
    (
        "threshold_sweep.png",
        "Efeito da regra de decisao",
        "Mostra como a leitura muda quando a regra fica mais aberta ou mais conservadora.",
    ),
    (
        "prob_distributions.png",
        "Distribuicao das chances estimadas",
        "Mostra como as amostras se espalham entre sinais mais fracos e mais fortes.",
    ),
    (
        "prob_boxplot.png",
        "Faixa de valores por grupo",
        "Resume a variacao das chances estimadas em cada grupo.",
    ),
    (
        "confusion_matrix.png",
        "Resumo dos acertos e erros",
        "Mostra os principais tipos de acerto e erro na avaliacao final.",
    ),
]


def _format_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2%}"


def normalize_tier_label(tier: str | None) -> str:
    normalized = str(tier or "").strip().lower()
    if normalized in {"muito alto", "muito_alto"}:
        return "Muito Alto"
    if normalized == "alto":
        return "Alto"
    if normalized in {"medio", "médio", "moderado"}:
        return "Moderado"
    return "Baixo"


def classify_explainability_readiness(summary: dict[str, Any]) -> str:
    roc_auc = float(summary.get("test_roc_auc", 0.0))
    f1_score = float(summary.get("test_f1", 0.0))
    if roc_auc >= 0.90 and f1_score >= 0.80:
        return "Bom para apoiar a priorizacao"
    if roc_auc >= 0.80 and f1_score >= 0.70:
        return "Util com revisao da equipe"
    return "Uso exploratorio com cautela"


def build_executive_takeaways(summary: dict[str, Any]) -> list[str]:
    threshold = float(summary.get("threshold", 0.5))
    f1_score = float(summary.get("test_f1", 0.0))
    recall = float(summary.get("test_recall", 0.0))
    precision = float(summary.get("test_precision", 0.0))
    roc_auc = float(summary.get("test_roc_auc", 0.0))
    pr_auc = float(summary.get("test_pr_auc", 0.0))
    n_test = int(summary.get("n_test", 0))

    takeaways = [
        f"Na avaliacao final, o modelo mostrou boa capacidade de separar areas mais promissoras, com F1 de {_format_pct(f1_score)}, ROC-AUC de {_format_pct(roc_auc)} e PR-AUC de {_format_pct(pr_auc)}.",
        f"Ele consegue recuperar boa parte dos alvos positivos e ao mesmo tempo evita inflar demais a lista de prioridades.",
        f"A regra atual de decisao e {threshold:.2f}; por isso o app deve ser usado para priorizar visitas, nao para substituir a validacao geologica.",
    ]

    if n_test < 100:
        takeaways.append(
            f"A avaliacao final foi feita com {n_test} amostras de teste; o resultado e promissor, mas ainda ganha forca com mais dados de campo."
        )

    return takeaways


def build_operational_cautions(summary: dict[str, Any]) -> list[str]:
    cautions = [
        "Chance alta indica prioridade de investigacao, nao garantia de mineralizacao economicamente viavel.",
        "As imagens de apoio ajudam a entender o resultado, mas nao substituem o olhar geologico nem a checagem da qualidade da cena.",
    ]

    n_test = int(summary.get("n_test", 0))
    if n_test < 100:
        cautions.append("A base ainda e pequena; novos dados de campo vao deixar a leitura mais confiavel.")

    return cautions


def summarize_cross_validation(cv_df: pd.DataFrame) -> dict[str, float]:
    if cv_df.empty:
        return {}

    metric_columns = ["accuracy", "f1", "roc_auc", "pr_auc", "balanced_accuracy"]
    available = [column for column in metric_columns if column in cv_df.columns]
    if not available:
        return {}

    summary: dict[str, float] = {}
    for column in available:
        summary[f"{column}_mean"] = float(cv_df[column].mean())
        summary[f"{column}_std"] = float(cv_df[column].std(ddof=0))
    return summary


def prepare_explainability_target_table(
    ranking_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    *,
    top_n: int = 15,
    min_score: float = 0.5,
    only_positive: bool = True,
) -> pd.DataFrame:
    if ranking_df.empty:
        return ranking_df.copy()

    df = ranking_df.copy()
    df["image_id"] = df["image_id"].astype(str)
    if "rank" not in df.columns:
        df = df.sort_values("y_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

    if "tier" not in df.columns:
        df["tier"] = df["y_score"].apply(
            lambda score: "Muito Alto" if score >= 0.80 else "Alto" if score >= 0.65 else "Moderado" if score >= 0.45 else "Baixo"
        )
    else:
        df["tier"] = df["tier"].map(normalize_tier_label)

    if only_positive and "y_pred" in df.columns:
        df = df[df["y_pred"] == 1].copy()

    df = df[df["y_score"] >= float(min_score)].copy()

    metadata = metadata_df.copy()
    metadata["numero_amostra"] = metadata["numero_amostra"].astype(str)
    merged = df.merge(
        metadata[
            [
                "numero_amostra",
                "latitude_wgs84_decimal",
                "longitude_wgs84_decimal",
                "classe_balanceamento",
                "litologia_padronizada",
            ]
        ],
        left_on="image_id",
        right_on="numero_amostra",
        how="left",
    )

    merged["classe_real"] = merged.get("y_true", pd.Series(index=merged.index, dtype=float)).map(
        {0: "Negativo", 1: "Positivo"}
    )
    merged["classe_prevista"] = merged.get("y_pred", pd.Series(index=merged.index, dtype=float)).map(
        {0: "Negativo", 1: "Positivo"}
    )
    merged = merged.sort_values(["rank", "y_score"], ascending=[True, False]).head(int(top_n)).reset_index(drop=True)
    return merged


@st.cache_data(show_spinner=True)
def load_a11_explainability_assets(project_root: Path) -> dict[str, Any]:
    outputs_dir = project_root / "artefatos" / "a11_pipeline_e2e" / "outputs"
    summary_path = outputs_dir / "metrics" / "summary.json"
    ranking_path = outputs_dir / "notebook_visualizations" / "prospectivity_ranking.csv"
    cv_results_path = outputs_dir / "notebook_visualizations" / "cross_validation_results.csv"
    hp_search_path = outputs_dir / "notebook_visualizations" / "hyperparameter_search.csv"
    metadata_path = project_root / "data" / "banco.xlsx"

    missing_paths = [path for path in [summary_path, ranking_path, metadata_path] if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Artefatos obrigatorios de explicabilidade nao encontrados: {missing_text}")

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    ranking_df = pd.read_csv(ranking_path)
    metadata_df = pd.read_excel(metadata_path, sheet_name=METADATA_SHEET)
    cv_results_df = pd.read_csv(cv_results_path) if cv_results_path.exists() else pd.DataFrame()
    hp_search_df = pd.read_csv(hp_search_path) if hp_search_path.exists() else pd.DataFrame()

    image_paths = {}
    for filename, _, _ in [*VISUAL_SECTIONS, *TECHNICAL_FIGURES]:
        path = outputs_dir / "notebook_visualizations" / filename
        if path.exists():
            image_paths[filename] = path

    return {
        "summary": summary,
        "ranking_df": ranking_df,
        "metadata_df": metadata_df,
        "cv_results_df": cv_results_df,
        "hp_search_df": hp_search_df,
        "image_paths": image_paths,
    }


def render_visual_gallery(image_paths: dict[str, Path], sections: list[tuple[str, str, str]]) -> None:
    for filename, title, caption in sections:
        image_path = image_paths.get(filename)
        if image_path is None:
            continue
        st.subheader(title)
        st.caption(caption)
        st.image(str(image_path), use_container_width=True)


@st.cache_data(show_spinner=True)
def load_sample_gradcam(
    project_root: Path,
    sample_id: str,
    *,
    model_key: str = "a11_pipeline_e2e",
) -> dict[str, Any]:
    from inference import generate_dataset_sample_gradcam

    return generate_dataset_sample_gradcam(
        sample_id=sample_id,
        project_root=project_root,
        model_key=model_key,
    )


def summarize_gradcam_heatmap(heatmap: Any) -> dict[str, str]:
    heat = pd.DataFrame(heatmap).to_numpy(dtype=float)
    if heat.ndim != 2 or heat.size == 0:
        return {
            "title": "Mapa de atencao indisponivel",
            "summary": "Nao foi possivel resumir a atencao espacial desta amostra.",
        }

    total_energy = float(heat.sum())
    if total_energy <= 0.0:
        return {
            "title": "Atencao difusa",
            "summary": "O mapa de atencao ficou muito fraco. Isso sugere pouca concentracao clara do modelo nesta amostra.",
        }

    height, width = heat.shape
    y0, y1 = int(height * 0.25), int(height * 0.75)
    x0, x1 = int(width * 0.25), int(width * 0.75)
    center_energy = float(heat[y0:y1, x0:x1].sum()) / total_energy
    strong_area_ratio = float((heat >= 0.6).mean())

    if center_energy >= 0.45 and strong_area_ratio <= 0.30:
        title = "Atencao focada"
        summary = (
            "O modelo concentrou a maior parte da atencao em uma regiao mais central e relativamente compacta. "
            "Isso costuma indicar uma leitura mais objetiva do padrao presente no chip."
        )
    elif center_energy >= 0.30:
        title = "Atencao moderadamente concentrada"
        summary = (
            "O modelo olhou para uma area principal, mas ainda espalhou parte da atencao por outras regioes. "
            "Vale interpretar junto com litologia e amostras vizinhas."
        )
    else:
        title = "Atencao dispersa"
        summary = (
            "A atencao ficou mais espalhada ou puxada para bordas. "
            "Isso pode indicar um caso menos claro e que merece revisao adicional antes de priorizacao."
        )

    return {
        "title": title,
        "summary": summary,
    }


@st.cache_data(show_spinner=False)
def build_region_cluster_assets(
    ranking_df: pd.DataFrame,
    *,
    min_score: float,
    radius_km: float,
    min_samples: int,
) -> pd.DataFrame:
    from inference import build_promising_region_clusters

    return build_promising_region_clusters(
        ranking_df,
        min_score=min_score,
        radius_km=radius_km,
        min_samples=min_samples,
    )


def make_promising_regions_map(
    ranking_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    *,
    min_heatmap_score: float = 0.0,
) -> folium.Map:
    from inference import build_sample_heatmap_points

    valid_geo = ranking_df.dropna(subset=["latitude_wgs84_decimal", "longitude_wgs84_decimal"]).copy()
    if valid_geo.empty:
        return folium.Map(location=[-14.5, -47.5], zoom_start=4, control_scale=True)

    center_lat = float(valid_geo["latitude_wgs84_decimal"].mean())
    center_lon = float(valid_geo["longitude_wgs84_decimal"].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=7, control_scale=True, tiles=None)
    folium.TileLayer("CartoDB positron", name="Base").add_to(fmap)

    heatmap_points = build_sample_heatmap_points(
        ranking_df,
        min_score=min_heatmap_score,
    )
    if heatmap_points:
        HeatMap(
            heatmap_points,
            name="Mapa de calor",
            min_opacity=0.35,
            radius=25,
            blur=18,
            max_zoom=12,
        ).add_to(fmap)

    for _, row in valid_geo.iterrows():
        color = "#b91c1c" if float(row["y_score"]) >= 0.65 else "#2563eb"
        folium.CircleMarker(
            location=[float(row["latitude_wgs84_decimal"]), float(row["longitude_wgs84_decimal"])],
            radius=4 + 7 * float(row["y_score"]),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.55,
            weight=1,
            popup=(
                f"Amostra {row['image_id']}<br>"
                f"Rank #{int(row['rank'])}<br>"
                f"Score: {float(row['y_score']):.2%}<br>"
                f"Tier: {row['tier']}"
            ),
        ).add_to(fmap)

    for _, cluster_row in cluster_df.iterrows():
        folium.Circle(
            location=[float(cluster_row["latitude_centro"]), float(cluster_row["longitude_centro"])],
            radius=max(120.0, 180.0 * float(cluster_row["n_amostras"])),
            color="#92400e",
            weight=2,
            fill=True,
            fill_opacity=0.18,
            fill_color="#f59e0b",
            popup=(
                f"{cluster_row['region_label']}<br>"
                f"Amostras: {int(cluster_row['n_amostras'])}<br>"
                f"Score medio: {float(cluster_row['score_medio']):.2%}<br>"
                f"Score maximo: {float(cluster_row['score_maximo']):.2%}<br>"
                f"Top amostras: {cluster_row['top_amostras']}"
            ),
        ).add_to(fmap)

        folium.Marker(
            location=[float(cluster_row["latitude_centro"]), float(cluster_row["longitude_centro"])],
            tooltip=cluster_row["region_label"],
            icon=folium.DivIcon(
                html=(
                    "<div style='font-size: 12px; font-weight: 700; color: #78350f; "
                    "background: rgba(255,255,255,0.9); padding: 4px 6px; border-radius: 8px;'>"
                    f"{cluster_row['region_label']}</div>"
                )
            ),
        ).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def classify_sample_confidence(score: float, threshold: float) -> str:
    margin = float(score) - float(threshold)
    if margin >= 0.20:
        return "Alta"
    if margin >= 0.08:
        return "Media"
    if margin >= 0.0:
        return "Baixa"
    if margin >= -0.08:
        return "Baixa"
    return "Fora da faixa prioritaria"


def build_nontechnical_sample_explanation(
    row: pd.Series,
    *,
    threshold: float,
    total_samples: int,
) -> dict[str, str]:
    sample_id = str(row.get("image_id", "-"))
    score = float(row.get("y_score", 0.0))
    rank = int(row.get("rank", 0) or 0)
    tier = normalize_tier_label(row.get("tier"))
    predicted_label = str(row.get("classe_prevista", "Sem classificacao"))
    lithology = row.get("litologia_padronizada")
    confidence = classify_sample_confidence(score, threshold)
    percentile = 1.0 - ((max(rank, 1) - 1) / max(total_samples, 1))
    actual_label = row.get("classe_real")

    if predicted_label == "Positivo" and tier == "Muito Alto":
        headline = f"A amostra {sample_id} entrou entre os alvos mais fortes do modelo."
        business_meaning = "Ela deve ser vista como candidata prioritaria para validacao geologica e planejamento de campo."
    elif predicted_label == "Positivo" and tier in {"Alto", "Moderado"}:
        headline = f"A amostra {sample_id} apresenta sinal promissor, mas ainda pede revisao."
        business_meaning = "Ela merece entrar em shortlist, principalmente se estiver perto de outros alvos fortes ou em contexto litologico favoravel."
    else:
        headline = f"A amostra {sample_id} nao aparece como prioridade imediata."
        business_meaning = "No estado atual, ela pode ficar em monitoramento ou servir como comparacao com alvos mais fortes."

    if confidence == "Alta":
        confidence_text = "O score esta bem acima do threshold operacional, entao o modelo mostra seguranca relativamente boa para priorizacao."
    elif confidence == "Media":
        confidence_text = "O score esta acima do threshold, mas sem muita folga. Vale combinar essa leitura com contexto geologico e analise visual."
    elif confidence == "Baixa":
        confidence_text = "O score esta muito perto do threshold. E um caso limitrofe, entao pequenas variacoes podem mudar a decisao."
    else:
        confidence_text = "O score ficou abaixo da faixa prioritaria. Isso reduz a urgencia operacional desta amostra."

    if pd.notna(lithology):
        geology_text = f"O ponto esta associado a litologia `{lithology}`, o que ajuda a equipe a contextualizar o alvo geologicamente."
    else:
        geology_text = "Nao ha litologia associada a esta amostra nos metadados atuais."

    if pd.notna(actual_label):
        actual_label_text = (
            f"No conjunto historico rotulado, esta amostra aparece como `{actual_label}`. Isso serve como referencia para calibrar a confianca da equipe."
        )
    else:
        actual_label_text = "Nao ha rotulo historico visivel para esta amostra nesta tela."

    if predicted_label == "Positivo":
        next_step = "Proximo passo sugerido: incluir na shortlist, comparar com amostras vizinhas e revisar evidencias geologicas antes de campo."
    elif score >= threshold * 0.8:
        next_step = "Proximo passo sugerido: manter em observacao e revisar junto com amostras do mesmo contexto geologico."
    else:
        next_step = "Proximo passo sugerido: nao priorizar agora e concentrar recursos nos alvos de score superior."

    return {
        "headline": headline,
        "business_meaning": business_meaning,
        "confidence_title": confidence,
        "confidence_text": confidence_text,
        "geology_text": geology_text,
        "actual_label_text": actual_label_text,
        "next_step": next_step,
        "plain_score": (
            f"Score de {_format_pct(score)}. Isso coloca a amostra aproximadamente entre os {_format_pct(percentile)} mais altos do recorte analisado."
        ),
    }
