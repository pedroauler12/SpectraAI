from __future__ import annotations

import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium


THRESHOLD_OPTIONS = {
    "threshold_f1": "threshold_f1 (recomendado)",
    "threshold_0.5": "0.5 (mais conservador)",
}


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

from inference import build_dataset_probability_ranking  # noqa: E402


@st.cache_data(show_spinner=False)
def load_probability_ranking_frame(
    project_root: str,
    threshold_mode: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    return build_dataset_probability_ranking(
        project_root=Path(project_root),
        threshold_mode=threshold_mode,
        force_refresh=force_refresh,
    )


def filter_ranking_frame(
    ranking_df: pd.DataFrame,
    *,
    sample_query: str = "",
    top_n: int | None = None,
    min_probability: float = 0.0,
) -> pd.DataFrame:
    filtered = ranking_df.copy()
    if sample_query.strip():
        filtered = filtered[
            filtered["numero_amostra"].astype(str).str.contains(sample_query.strip(), case=False, na=False)
        ]

    filtered = filtered[filtered["prob_pos"] >= float(min_probability)].copy()
    filtered = filtered.sort_values(["rank", "numero_amostra"], ascending=[True, True]).reset_index(drop=True)

    if top_n is not None:
        filtered = filtered.head(int(top_n)).copy()

    return filtered.reset_index(drop=True)


def make_probability_ranking_map(points_df: pd.DataFrame) -> folium.Map:
    if points_df.empty:
        return folium.Map(location=[-18.5, -46.5], zoom_start=5, control_scale=True, tiles="CartoDB positron")

    center_lat = float(points_df["latitude_wgs84_decimal"].mean())
    center_lon = float(points_df["longitude_wgs84_decimal"].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=6, control_scale=True, tiles=None)
    folium.TileLayer("CartoDB positron", name="Base").add_to(fmap)

    colormap = LinearColormap(
        colors=["#1a9850", "#fee08b", "#d73027"],
        vmin=0.0,
        vmax=1.0,
        caption="Probabilidade de terra rara",
    )

    for _, row in points_df.iterrows():
        prob = float(row["prob_pos"])
        popup_html = (
            f"<b>Rank:</b> {int(row['rank'])}<br>"
            f"<b>Amostra:</b> {row['numero_amostra']}<br>"
            f"<b>Probabilidade:</b> {prob:.2%}<br>"
            f"<b>Litologia:</b> {row.get('litologia_padronizada', '-')}"
        )
        folium.CircleMarker(
            location=[float(row["latitude_wgs84_decimal"]), float(row["longitude_wgs84_decimal"])],
            radius=5 + (prob * 5),
            color=colormap(prob),
            fill=True,
            fill_color=colormap(prob),
            fill_opacity=0.75,
            weight=1,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"#{int(row['rank'])} - {row['numero_amostra']} ({prob:.1%})",
        ).add_to(fmap)

    colormap.add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def main() -> None:
    st.set_page_config(
        page_title="SpectraAI - Ranking de Probabilidades",
        layout="wide",
    )

    st.title("Ranking de probabilidades por ponto")
    st.caption(
        "Ordenacao decrescente das amostras do pixels_dataset.csv pela probabilidade prevista "
        "de potencial para terra rara usando o modelo A08."
    )

    with st.sidebar:
        st.header("Filtros")
        threshold_mode = st.radio(
            "Threshold de decisao",
            options=list(THRESHOLD_OPTIONS.keys()),
            format_func=lambda key: THRESHOLD_OPTIONS[key],
            index=0,
        )
        sample_query = st.text_input("Buscar por numero_amostra", value="")
        top_n = st.number_input("Top N", min_value=10, max_value=295, value=100, step=5)
        min_probability = st.slider("Probabilidade minima", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        force_refresh = st.button("Recalcular ranking completo")

    with st.spinner("Carregando ranking completo da base..."):
        ranking_df = load_probability_ranking_frame(
            str(PROJECT_ROOT),
            threshold_mode=threshold_mode,
            force_refresh=force_refresh,
        )

    filtered_df = filter_ranking_frame(
        ranking_df,
        sample_query=sample_query,
        top_n=int(top_n),
        min_probability=float(min_probability),
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Total de pontos", f"{len(ranking_df)}")
    metric_col2.metric("Maior probabilidade", f"{ranking_df['prob_pos'].max():.2%}")
    metric_col3.metric("Media de prob_pos", f"{ranking_df['prob_pos'].mean():.2%}")
    metric_col4.metric(
        "Acima do threshold",
        f"{int((ranking_df['prob_pos'] >= ranking_df['decision_threshold'].iloc[0]).sum())}",
    )

    st.markdown("**Mapa de probabilidade**")
    probability_map = make_probability_ranking_map(filtered_df)
    st_folium(
        probability_map,
        key="probability_ranking_map",
        width=None,
        height=520,
        returned_objects=[],
    )

    st.markdown("**Lista ordenada**")
    st.caption(f"Exibindo {len(filtered_df)} ponto(s) apos os filtros aplicados.")
    table_columns = [
        "rank",
        "numero_amostra",
        "prob_pos",
        "prob_neg",
        "pred_label_threshold",
        "latitude_wgs84_decimal",
        "longitude_wgs84_decimal",
        "classe_balanceamento",
        "litologia_padronizada",
    ]
    st.dataframe(
        filtered_df[table_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "prob_pos": st.column_config.NumberColumn("prob_pos", format="%.4f"),
            "prob_neg": st.column_config.NumberColumn("prob_neg", format="%.4f"),
            "latitude_wgs84_decimal": st.column_config.NumberColumn("latitude", format="%.6f"),
            "longitude_wgs84_decimal": st.column_config.NumberColumn("longitude", format="%.6f"),
        },
    )


if __name__ == "__main__":
    main()
