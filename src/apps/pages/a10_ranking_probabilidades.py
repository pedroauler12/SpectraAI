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
    "artifact_default": "Threshold do artefato",
    "threshold_0.5": "0.5 (mais conservador)",
}


def normalize_threshold_mode(model_key: str, threshold_choice: str) -> str:
    if threshold_choice == "threshold_0.5":
        return "threshold_0.5"
    return "threshold_default" if model_key == "a11_pipeline_e2e" else "threshold_f1"


def filter_ranking_frame(
    ranking_df: pd.DataFrame,
    *,
    sample_query: str = "",
    top_n: int = 100,
    min_probability: float = 0.0,
) -> pd.DataFrame:
    filtered_df = ranking_df.copy()
    filtered_df["numero_amostra"] = filtered_df["numero_amostra"].astype(str)

    if sample_query.strip():
        filtered_df = filtered_df[
            filtered_df["numero_amostra"].str.contains(sample_query.strip(), case=False, na=False)
        ]

    filtered_df = filtered_df[filtered_df["prob_pos"] >= float(min_probability)].copy()
    filtered_df = filtered_df.sort_values("rank", ascending=True).head(int(top_n)).reset_index(drop=True)
    return filtered_df


def make_probability_ranking_map(filtered_df: pd.DataFrame) -> folium.Map:
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
        color = "#b91c1c" if row["pred_label_threshold"] == "Positivo" else "#2563eb"
        radius = 4 + (8 * float(row["prob_pos"]))
        popup = (
            f"Amostra {row['numero_amostra']}<br>"
            f"Rank #{int(row['rank'])}<br>"
            f"Prob. positiva: {float(row['prob_pos']):.2%}<br>"
            f"Predicao: {row['pred_label_threshold']}"
        )
        folium.CircleMarker(
            location=[row["latitude_wgs84_decimal"], row["longitude_wgs84_decimal"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
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
        page_title="SpectraAI - Ranking de Probabilidades",
        layout="wide",
    )

    st.title("Ranking de Probabilidades")
    st.caption("Ranking completo das amostras da base com ordenacao por probabilidade prevista.")

    with st.sidebar:
        st.header("Filtros")
        model_key = st.selectbox(
            "Modelo",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda key: MODEL_OPTIONS[key],
            index=0,
        )
        threshold_choice = st.radio(
            "Threshold",
            options=list(THRESHOLD_OPTIONS.keys()),
            format_func=lambda key: THRESHOLD_OPTIONS[key],
            index=0,
        )
        top_n = st.slider("Top N amostras", min_value=10, max_value=295, value=100, step=5)
        min_probability = st.slider("Probabilidade minima", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        sample_query = st.text_input("Buscar amostra")
        force_refresh = st.checkbox("Recalcular ranking", value=False)

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

    filtered_df = filter_ranking_frame(
        ranking_df,
        sample_query=sample_query,
        top_n=top_n,
        min_probability=min_probability,
    )

    total_samples = int(len(ranking_df))
    visible_samples = int(len(filtered_df))
    prob_mean = float(filtered_df["prob_pos"].mean()) if not filtered_df.empty else 0.0

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Amostras no ranking", total_samples)
    metric_col2.metric("Amostras visiveis", visible_samples)
    metric_col3.metric("Probabilidade media visivel", f"{prob_mean:.2%}")

    st.caption(
        f"Modelo ativo: {MODEL_OPTIONS[model_key]} | "
        f"threshold: {filtered_df['decision_threshold_name'].iloc[0] if not filtered_df.empty else threshold_mode}"
    )

    map_col, table_col = st.columns([1.0, 1.2], gap="large")

    with map_col:
        st.subheader("Mapa do ranking")
        fmap = make_probability_ranking_map(filtered_df)
        st_folium(
            fmap,
            key=f"ranking_map_{model_key}_{threshold_mode}",
            width=None,
            height=620,
            returned_objects=[],
        )

    with table_col:
        st.subheader("Tabela ranqueada")
        display_columns = [
            "rank",
            "numero_amostra",
            "prob_pos",
            "pred_label_threshold",
            "classe_balanceamento",
            "litologia_padronizada",
            "latitude_wgs84_decimal",
            "longitude_wgs84_decimal",
        ]
        st.dataframe(
            filtered_df[display_columns],
            width="stretch",
            height=620,
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
            "Baixar CSV filtrado",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name=f"ranking_probabilidades_{model_key}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    render_page()
