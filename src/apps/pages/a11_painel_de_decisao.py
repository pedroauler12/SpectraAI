from __future__ import annotations

import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


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


MODEL_OPTIONS = {
    "a11_pipeline_e2e": "Pipeline Final (A11)",
    "a08_transfer_learning": "Transfer Learning (A08)",
}

THRESHOLD_OPTIONS = {
    "artifact_default": "Regra recomendada",
    "threshold_0.5": "Regra mais conservadora",
}

TIER_ORDER = ["Muito Alto", "Alto", "Medio", "Baixo"]


def normalize_threshold_mode(model_key: str, threshold_choice: str) -> str:
    if threshold_choice == "threshold_0.5":
        return "threshold_0.5"
    return "threshold_default" if model_key == "a11_pipeline_e2e" else "threshold_f1"


def classify_probability_tier(probability: float) -> str:
    if probability >= 0.80:
        return "Muito Alto"
    if probability >= 0.65:
        return "Alto"
    if probability >= 0.45:
        return "Medio"
    return "Baixo"


def classify_operational_priority(probability: float) -> str:
    if probability >= 0.80:
        return "Campo imediato"
    if probability >= 0.65:
        return "Validacao prioritaria"
    if probability >= 0.45:
        return "Monitoramento"
    return "Baixa prioridade"


def build_target_recommendation(row: pd.Series) -> str:
    tier = str(row.get("tier", "Baixo"))
    label = str(row.get("pred_label_threshold", "Negativo"))
    lithology = row.get("litologia_padronizada")
    lithology_text = f" Contexto litologico: {lithology}." if pd.notna(lithology) else ""

    if tier == "Muito Alto" and label == "Positivo":
        return "Priorizar para avaliacao de campo no curto prazo." + lithology_text
    if tier == "Alto" and label == "Positivo":
        return "Entrar na shortlist de validacao geologica." + lithology_text
    if tier == "Medio":
        return "Manter em observacao e comparar com alvos vizinhos." + lithology_text
    return "Baixa prioridade operacional no estado atual." + lithology_text


def enrich_ranking_frame(ranking_df: pd.DataFrame) -> pd.DataFrame:
    df = ranking_df.copy()
    df["tier"] = df["prob_pos"].map(classify_probability_tier)
    df["prioridade_operacional"] = df["prob_pos"].map(classify_operational_priority)
    df["recomendacao"] = df.apply(build_target_recommendation, axis=1)
    return df


def filter_operational_frame(
    ranking_df: pd.DataFrame,
    *,
    sample_query: str = "",
    top_n: int = 100,
    min_probability: float = 0.0,
    tiers: list[str] | None = None,
    predicted_labels: list[str] | None = None,
    lithologies: list[str] | None = None,
) -> pd.DataFrame:
    filtered_df = ranking_df.copy()
    filtered_df["numero_amostra"] = filtered_df["numero_amostra"].astype(str)

    if sample_query.strip():
        filtered_df = filtered_df[
            filtered_df["numero_amostra"].str.contains(sample_query.strip(), case=False, na=False)
        ]

    filtered_df = filtered_df[filtered_df["prob_pos"] >= float(min_probability)].copy()

    if tiers:
        filtered_df = filtered_df[filtered_df["tier"].isin(tiers)]
    if predicted_labels:
        filtered_df = filtered_df[filtered_df["pred_label_threshold"].isin(predicted_labels)]
    if lithologies:
        filtered_df = filtered_df[filtered_df["litologia_padronizada"].isin(lithologies)]

    return filtered_df.sort_values("rank", ascending=True).head(int(top_n)).reset_index(drop=True)


def get_shortlist_state() -> set[str]:
    if "ranking_shortlist" not in st.session_state:
        st.session_state["ranking_shortlist"] = set()
    return st.session_state["ranking_shortlist"]


def shortlist_dataframe(ranking_df: pd.DataFrame, shortlist_ids: set[str]) -> pd.DataFrame:
    if not shortlist_ids:
        return ranking_df.iloc[0:0].copy()
    return ranking_df[ranking_df["numero_amostra"].astype(str).isin(shortlist_ids)].sort_values("rank").copy()


def build_target_snapshot(row: pd.Series) -> dict[str, object]:
    return {
        "Amostra": str(row["numero_amostra"]),
        "Rank": int(row["rank"]),
        "Probabilidade positiva": f"{float(row['prob_pos']):.2%}",
        "Tier": str(row["tier"]),
        "Prioridade operacional": str(row["prioridade_operacional"]),
        "Classe prevista": str(row["pred_label_threshold"]),
        "Classe base": row.get("classe_balanceamento"),
        "Litologia": row.get("litologia_padronizada"),
        "Latitude": row.get("latitude_wgs84_decimal"),
        "Longitude": row.get("longitude_wgs84_decimal"),
        "Threshold": f"{float(row['decision_threshold']):.4f}",
        "Regra de threshold": str(row["decision_threshold_name"]),
        "Recomendacao": str(row["recomendacao"]),
    }


def make_operational_map(
    filtered_df: pd.DataFrame,
    highlighted_sample: str | None = None,
    shortlisted_samples: set[str] | None = None,
) -> folium.Map:
    shortlisted_samples = shortlisted_samples or set()
    if filtered_df.empty:
        return folium.Map(location=[-14.5, -47.5], zoom_start=4, control_scale=True)

    valid_geo = filtered_df.dropna(
        subset=["latitude_wgs84_decimal", "longitude_wgs84_decimal"]
    ).copy()
    if valid_geo.empty:
        return folium.Map(location=[-14.5, -47.5], zoom_start=4, control_scale=True)

    center_lat = float(valid_geo["latitude_wgs84_decimal"].mean())
    center_lon = float(valid_geo["longitude_wgs84_decimal"].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=5, control_scale=True, tiles=None)
    folium.TileLayer("CartoDB positron", name="Base").add_to(fmap)

    for _, row in valid_geo.iterrows():
        sample_id = str(row["numero_amostra"])
        is_highlighted = highlighted_sample == sample_id
        is_shortlisted = sample_id in shortlisted_samples
        color = "#b91c1c" if row["pred_label_threshold"] == "Positivo" else "#2563eb"
        radius = 4 + (8 * float(row["prob_pos"]))
        popup = (
            f"Amostra {sample_id}<br>"
            f"Rank #{int(row['rank'])}<br>"
            f"Prob. positiva: {float(row['prob_pos']):.2%}<br>"
            f"Tier: {row['tier']}<br>"
            f"Prioridade: {row['prioridade_operacional']}<br>"
            f"Predicao: {row['pred_label_threshold']}"
        )
        folium.CircleMarker(
            location=[row["latitude_wgs84_decimal"], row["longitude_wgs84_decimal"]],
            radius=radius,
            color="#111827" if is_highlighted else color,
            weight=4 if is_highlighted else 2,
            fill=True,
            fill_color=color,
            fill_opacity=0.95 if is_highlighted or is_shortlisted else 0.75,
            popup=popup,
        ).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


@st.cache_data(show_spinner=True)
def load_probability_ranking(
    project_root: Path,
    model_key: str,
    threshold_mode: str,
    force_refresh: bool,
) -> pd.DataFrame:
    from inference import build_dataset_probability_ranking

    return build_dataset_probability_ranking(
        project_root=project_root,
        model_key=model_key,
        threshold_mode=threshold_mode,
        force_refresh=force_refresh,
    )


def render_page() -> None:
    st.set_page_config(
        page_title="SpectraAI - Painel Operacional",
        layout="wide",
    )

    st.title("Painel de Decisao")
    st.caption("Uma visao rapida para decidir quais alvos merecem atencao primeiro.")

    shortlist_ids = get_shortlist_state()

    with st.sidebar:
        st.header("Refinar analise")
        model_key = st.selectbox(
            "Modelo",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda key: MODEL_OPTIONS[key],
            index=0,
        )
        threshold_choice = st.radio(
            "Regra de decisao",
            options=list(THRESHOLD_OPTIONS.keys()),
            format_func=lambda key: THRESHOLD_OPTIONS[key],
            index=0,
        )
        top_n = st.slider("Quantidade de amostras", min_value=10, max_value=295, value=100, step=5)
        min_probability = st.slider("Chance minima", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        sample_query = st.text_input("Buscar codigo da amostra")
        force_refresh = st.checkbox("Atualizar lista", value=False)

    threshold_mode = normalize_threshold_mode(model_key, threshold_choice)

    try:
        ranking_df = load_probability_ranking(
            PROJECT_ROOT,
            model_key=model_key,
            threshold_mode=threshold_mode,
            force_refresh=force_refresh,
        )
    except Exception as exc:
        st.error(str(exc))
        return

    ranking_df = enrich_ranking_frame(ranking_df)

    tier_options = [tier for tier in TIER_ORDER if tier in set(ranking_df["tier"])]
    label_options = sorted(ranking_df["pred_label_threshold"].dropna().astype(str).unique().tolist())
    lithology_options = sorted(ranking_df["litologia_padronizada"].dropna().astype(str).unique().tolist())

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        selected_tiers = st.multiselect("Nivel de prioridade", options=tier_options, default=tier_options)
    with filter_col2:
        selected_labels = st.multiselect("Resultado sugerido", options=label_options, default=label_options)
    with filter_col3:
        selected_lithologies = st.multiselect("Litologia", options=lithology_options, default=[])

    filtered_df = filter_operational_frame(
        ranking_df,
        sample_query=sample_query,
        top_n=top_n,
        min_probability=min_probability,
        tiers=selected_tiers,
        predicted_labels=selected_labels,
        lithologies=selected_lithologies or None,
    )

    total_samples = int(len(ranking_df))
    visible_samples = int(len(filtered_df))
    prob_mean = float(filtered_df["prob_pos"].mean()) if not filtered_df.empty else 0.0
    shortlisted_df = shortlist_dataframe(ranking_df, shortlist_ids)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Total de amostras", total_samples)
    metric_col2.metric("Amostras mostradas", visible_samples)
    metric_col3.metric("Chance media nesta tela", f"{prob_mean:.2%}")
    metric_col4.metric("Alvos separados", len(shortlist_ids))

    focus_options = filtered_df["numero_amostra"].astype(str).tolist()
    focused_sample = st.selectbox(
        "Amostra em destaque",
        options=focus_options,
        index=0 if focus_options else None,
        placeholder="Nenhuma amostra encontrada com os filtros atuais",
    ) if focus_options else None

    focused_row = None
    if focused_sample is not None:
        focused_row = filtered_df.loc[filtered_df["numero_amostra"].astype(str) == focused_sample].iloc[0]

    action_col1, action_col2, action_col3 = st.columns([1.0, 1.0, 2.0])
    with action_col1:
        if focused_sample is not None and st.button("Separar para campanha", disabled=focused_sample in shortlist_ids):
            shortlist_ids.add(str(focused_sample))
            st.rerun()
    with action_col2:
        if focused_sample is not None and st.button("Remover da campanha", disabled=focused_sample not in shortlist_ids):
            shortlist_ids.discard(str(focused_sample))
            st.rerun()
    with action_col3:
        if focused_row is not None:
            st.info(build_target_recommendation(focused_row))

    map_col, detail_col = st.columns([1.05, 0.95], gap="large")

    with map_col:
        st.subheader("Mapa de prioridades")
        fmap = make_operational_map(
            filtered_df,
            highlighted_sample=focused_sample,
            shortlisted_samples=shortlist_ids,
        )
        st_folium(
            fmap,
            key=f"operational_map_{model_key}_{threshold_mode}",
            height=620,
            returned_objects=[],
            use_container_width=True,
        )

    with detail_col:
        st.subheader("Resumo do alvo")
        if focused_row is None:
            st.write("Escolha uma amostra em destaque para ver o resumo.")
        else:
            st.json(build_target_snapshot(focused_row))

    table_col, shortlist_col = st.columns([1.2, 0.8], gap="large")

    with table_col:
        st.subheader("Tabela de apoio")
        display_df = filtered_df[
            [
                "rank",
                "numero_amostra",
                "prob_pos",
                "tier",
                "prioridade_operacional",
                "pred_label_threshold",
                "classe_balanceamento",
                "litologia_padronizada",
                "latitude_wgs84_decimal",
                "longitude_wgs84_decimal",
            ]
        ].copy()
        display_df.insert(0, "shortlist", display_df["numero_amostra"].astype(str).isin(shortlist_ids))
        st.dataframe(
            display_df,
            width="stretch",
            height=620,
            hide_index=True,
            column_config={
                "shortlist": st.column_config.CheckboxColumn("separado", disabled=True),
                "prob_pos": st.column_config.ProgressColumn(
                    "chance_estimada",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                ),
            },
        )

    with shortlist_col:
        st.subheader("Alvos separados para campanha")
        if shortlisted_df.empty:
            st.write("Nenhum alvo foi separado ainda.")
        else:
            shortlist_view = shortlisted_df[
                ["rank", "numero_amostra", "prob_pos", "tier", "prioridade_operacional", "recomendacao"]
            ].copy()
            st.dataframe(
                shortlist_view,
                width="stretch",
                height=430,
                hide_index=True,
                column_config={
                    "prob_pos": st.column_config.ProgressColumn(
                        "prob_pos",
                        format="%.2f",
                        min_value=0.0,
                        max_value=1.0,
                    ),
                },
            )
            st.download_button(
                "Baixar lista da campanha",
                data=shortlisted_df.to_csv(index=False).encode("utf-8"),
                file_name=f"shortlist_campanha_{model_key}.csv",
                mime="text/csv",
            )
            if st.button("Limpar lista"):
                shortlist_ids.clear()
                st.rerun()


if __name__ == "__main__":
    render_page()
