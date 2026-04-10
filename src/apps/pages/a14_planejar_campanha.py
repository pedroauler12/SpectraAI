from __future__ import annotations

import sys
from pathlib import Path

_CURRENT_FILE = Path(__file__).resolve()
_SRC_DIR = _CURRENT_FILE.parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from apps.explainability_shared import (
    PROJECT_ROOT,
    _format_pct,
    build_region_cluster_assets,
    load_a11_explainability_assets,
    prepare_explainability_target_table,
)
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


def _get_campaign_state_key() -> str:
    return "campaign_plan_df"


def _build_campaign_map(plan_df: pd.DataFrame) -> folium.Map:
    valid_df = plan_df.dropna(subset=["latitude_wgs84_decimal", "longitude_wgs84_decimal"]).copy()
    if valid_df.empty:
        return folium.Map(location=[-14.5, -47.5], zoom_start=4, control_scale=True)

    center_lat = float(valid_df["latitude_wgs84_decimal"].mean())
    center_lon = float(valid_df["longitude_wgs84_decimal"].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=8, control_scale=True, tiles=None)
    folium.TileLayer("CartoDB positron", name="Base").add_to(fmap)

    priority_colors = {
        "Alta": "#b91c1c",
        "Media": "#d97706",
        "Observacao": "#2563eb",
    }

    for _, row in valid_df.iterrows():
        sample_id = str(row["image_id"])
        color = priority_colors.get(str(row.get("prioridade", "Observacao")), "#2563eb")
        popup = (
            f"Amostra {sample_id}<br>"
            f"Score: {float(row['y_score']):.2%}<br>"
            f"Prioridade: {row.get('prioridade', '-') }<br>"
            f"Regiao: {row.get('region_label', '-') }<br>"
            f"Status: {row.get('status', '-') }"
        )
        folium.CircleMarker(
            location=[float(row["latitude_wgs84_decimal"]), float(row["longitude_wgs84_decimal"])],
            radius=6 + 7 * float(row["y_score"]),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=popup,
        ).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def _initialize_campaign_plan(suggested_df: pd.DataFrame) -> None:
    state_key = _get_campaign_state_key()
    if state_key not in st.session_state:
        st.session_state[state_key] = suggested_df.copy()


def render_page() -> None:
    st.set_page_config(
        page_title="SpectraAI - Campanha Sugerida",
        layout="wide",
    )

    st.title("Campanha Sugerida")
    st.caption("Monte uma sugestao inicial de visita e ajuste a lista com a sua equipe.")

    with st.sidebar:
        st.header("Montar campanha")
        max_targets = st.slider("Numero maximo de alvos", min_value=3, max_value=15, value=8, step=1)
        min_score = st.slider("Chance minima para entrar na sugestao", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
        region_min_score = st.slider("Chance minima para formar uma regiao", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
        cluster_radius_km = st.slider("Distancia para agrupar pontos (km)", min_value=0.05, max_value=2.0, value=0.25, step=0.05)
        cluster_min_samples = st.slider("Minimo de amostras por regiao", min_value=2, max_value=5, value=2, step=1)

    try:
        assets = load_a11_explainability_assets(PROJECT_ROOT)
    except Exception as exc:
        st.error(str(exc))
        return

    ranking_df = assets["ranking_df"]
    metadata_df = assets["metadata_df"]
    full_target_df = prepare_explainability_target_table(
        ranking_df,
        metadata_df,
        top_n=max(len(ranking_df), 1),
        min_score=0.0,
        only_positive=False,
    )
    region_cluster_df = build_region_cluster_assets(
        full_target_df,
        min_score=region_min_score,
        radius_km=cluster_radius_km,
        min_samples=cluster_min_samples,
    )

    from inference import build_campaign_suggestion

    suggested_df = build_campaign_suggestion(
        full_target_df,
        region_cluster_df,
        max_targets=max_targets,
        min_score=min_score,
    )

    if suggested_df.empty:
        st.warning("Nenhuma sugestao de campanha foi formada com os filtros atuais.")
        return

    _initialize_campaign_plan(suggested_df)
    state_key = _get_campaign_state_key()

    action_col1, action_col2 = st.columns([1.0, 2.0])
    with action_col1:
        if st.button("Refazer sugestao"):
            st.session_state[state_key] = suggested_df.copy()
            st.rerun()
    with action_col2:
        st.caption("A sugestao inicial usa chance estimada, posicao no ranking e distribuicao regional. Depois disso, a equipe pode editar livremente.")

    addable_samples = full_target_df["image_id"].astype(str).tolist()
    add_sample_id = st.selectbox("Adicionar amostra manualmente", options=addable_samples, index=None, placeholder="Escolha uma amostra")
    if add_sample_id and st.button("Adicionar na campanha"):
        current_df = st.session_state[state_key].copy()
        if str(add_sample_id) not in current_df["image_id"].astype(str).tolist():
            add_row = full_target_df.loc[full_target_df["image_id"].astype(str) == str(add_sample_id)].iloc[0].to_dict()
            add_row.update(
                {
                    "region_label": "Inclusao manual",
                    "prioridade": "Media",
                    "status": "Incluido manualmente",
                    "janela_campo": "A definir",
                    "responsavel": "",
                    "notas": "",
                    "motivo_sugestao": "Inclusao manual da equipe.",
                }
            )
            st.session_state[state_key] = pd.concat([current_df, pd.DataFrame([add_row])], ignore_index=True)
            st.rerun()

    editable_df = st.session_state[state_key].copy()
    editable_df["incluir"] = editable_df.get("incluir", True)
    edited_df = st.data_editor(
        editable_df,
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "incluir": st.column_config.CheckboxColumn("Incluir", default=True),
            "prioridade": st.column_config.SelectboxColumn("Prioridade", options=["Alta", "Media", "Observacao"]),
            "status": st.column_config.SelectboxColumn(
                "Status",
                options=["Sugerido", "Validar", "Aprovado", "Em campo", "Descartado", "Incluido manualmente"],
            ),
            "janela_campo": st.column_config.SelectboxColumn(
                "Momento da visita",
                options=["Imediata", "Proxima campanha", "Se houver capacidade", "A definir"],
            ),
            "y_score": st.column_config.ProgressColumn("chance_estimada", format="%.2f", min_value=0.0, max_value=1.0),
        },
        disabled=[
            "rank",
            "image_id",
            "y_score",
            "tier",
            "classe_prevista",
            "classe_real",
            "litologia_padronizada",
            "latitude_wgs84_decimal",
            "longitude_wgs84_decimal",
            "region_label",
            "motivo_sugestao",
        ],
    )
    st.session_state[state_key] = edited_df.copy()

    included_df = edited_df[edited_df["incluir"] == True].copy()  # noqa: E712
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Alvos incluidos", int(len(included_df)))
    metric_col2.metric("Chance media", _format_pct(included_df["y_score"].mean()) if not included_df.empty else "-")
    metric_col3.metric("Alvos com prioridade alta", int((included_df["prioridade"] == "Alta").sum()) if not included_df.empty else 0)
    metric_col4.metric("Regioes cobertas", int(included_df["region_label"].nunique()) if not included_df.empty else 0)

    left_col, right_col = st.columns([1.1, 0.9], gap="large")
    with left_col:
        st.subheader("Mapa da campanha")
        campaign_map = _build_campaign_map(included_df)
        st_folium(
            campaign_map,
            key=f"a14_campaign_map_{len(included_df)}",
            height=620,
            returned_objects=[],
            use_container_width=True,
        )
    with right_col:
        st.subheader("Resumo para decidir")
        if included_df.empty:
            st.info("Nenhum alvo esta marcado para entrar na campanha.")
        else:
            top_region = included_df["region_label"].mode().iloc[0]
            st.markdown(f"- Regiao com mais alvos na carteira: `{top_region}`")
            st.markdown(f"- Melhor score incluido: `{_format_pct(included_df['y_score'].max())}`")
            st.markdown(f"- Litologia mais frequente: `{included_df['litologia_padronizada'].mode().iloc[0] if included_df['litologia_padronizada'].notna().any() else '-'}`")
            st.download_button(
                "Baixar campanha em CSV",
                data=included_df.to_csv(index=False).encode("utf-8"),
                file_name="campanha_sugerida_editavel.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    render_page()
