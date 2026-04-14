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
    build_nontechnical_sample_explanation,
    build_region_cluster_assets,
    load_a11_explainability_assets,
    load_sample_gradcam,
    make_promising_regions_map,
    prepare_explainability_target_table,
)
import streamlit as st
from streamlit_folium import st_folium


def render_page() -> None:
    st.set_page_config(
        page_title="SpectraAI - Regioes e Comparacao",
        layout="wide",
    )

    st.title("Regioes Promissoras e Comparacao")
    st.caption("Veja onde os sinais mais fortes se concentram e compare amostras lado a lado.")

    with st.sidebar:
        st.header("Refinar mapa")
        min_score = st.slider("Chance minima para o mapa", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        region_min_score = st.slider("Chance minima para formar uma regiao", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
        cluster_radius_km = st.slider("Distancia para agrupar pontos (km)", min_value=0.05, max_value=2.0, value=0.25, step=0.05)
        cluster_min_samples = st.slider("Minimo de amostras por regiao", min_value=2, max_value=5, value=2, step=1)

    try:
        assets = load_a11_explainability_assets(PROJECT_ROOT)
    except Exception as exc:
        st.error(str(exc))
        return

    summary = assets["summary"]
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

    tabs = st.tabs(["Regioes promissoras", "Comparativo de amostras"])

    with tabs[0]:
        st.subheader("Mapa de calor e regioes promissoras")
        st.caption("As cores mostram onde os sinais ficam mais fortes. Os circulos destacam concentracoes de interesse.")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Regioes encontradas", int(len(region_cluster_df)))
        metric_col2.metric(
            "Melhor chance da regiao",
            _format_pct(region_cluster_df["score_maximo"].max()) if not region_cluster_df.empty else "-",
        )
        metric_col3.metric(
            "Maior concentracao",
            int(region_cluster_df["n_amostras"].max()) if not region_cluster_df.empty else 0,
        )

        region_map = make_promising_regions_map(
            full_target_df,
            region_cluster_df,
            min_heatmap_score=min_score,
        )
        st_folium(
            region_map,
            key=f"a13_regions_map_{min_score}_{region_min_score}_{cluster_radius_km}_{cluster_min_samples}",
            height=620,
            returned_objects=[],
            use_container_width=True,
        )

        if region_cluster_df.empty:
            st.info("Nenhuma regiao promissora apareceu com os filtros atuais.")
        else:
            from inference import describe_region_opportunity

            focus_region_label = st.selectbox(
                "Regiao em destaque",
                options=region_cluster_df["region_label"].tolist(),
                index=0,
            )
            focus_region = region_cluster_df.loc[region_cluster_df["region_label"] == focus_region_label].iloc[0]
            st.success(describe_region_opportunity(focus_region))

            left_col, right_col = st.columns([1.1, 0.9], gap="large")
            with left_col:
                st.dataframe(
                    region_cluster_df[
                        [
                            "region_label",
                            "n_amostras",
                            "score_medio",
                            "score_maximo",
                            "litologia_dominante",
                            "top_amostras",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "score_medio": st.column_config.ProgressColumn(
                            "score_medio",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "score_maximo": st.column_config.ProgressColumn(
                            "score_maximo",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                    },
                )
            with right_col:
                region_samples = full_target_df[
                    full_target_df["image_id"].astype(str).isin(list(focus_region["amostras"]))
                ].copy()
                if not region_samples.empty:
                    st.subheader("Amostras de referencia")
                    st.dataframe(
                        region_samples[
                            [
                                "rank",
                                "image_id",
                                "y_score",
                                "tier",
                                "litologia_padronizada",
                            ]
                        ],
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "y_score": st.column_config.ProgressColumn(
                                "y_score",
                                format="%.2f",
                                min_value=0.0,
                                max_value=1.0,
                            ),
                        },
                    )

    with tabs[1]:
        st.subheader("Comparar amostras")
        st.caption("Use esta area para discutir dois ou tres alvos lado a lado com a equipe.")

        compare_default = full_target_df["image_id"].astype(str).head(2).tolist()
        compare_sample_ids = st.multiselect(
            "Escolha ate 3 amostras",
            options=full_target_df["image_id"].astype(str).tolist(),
            default=compare_default,
            max_selections=3,
        )

        if len(compare_sample_ids) < 2:
            st.info("Selecione pelo menos duas amostras para ativar a comparacao.")
        else:
            from inference import build_sample_comparison_table

            comparison_df = build_sample_comparison_table(full_target_df, compare_sample_ids)
            st.dataframe(
                comparison_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "y_score": st.column_config.ProgressColumn(
                        "y_score",
                        format="%.2f",
                        min_value=0.0,
                        max_value=1.0,
                    ),
                },
            )

            compare_columns = st.columns(len(compare_sample_ids))
            for column, sample_id in zip(compare_columns, compare_sample_ids):
                sample_row = full_target_df.loc[full_target_df["image_id"].astype(str) == str(sample_id)].iloc[0]
                sample_story = build_nontechnical_sample_explanation(
                    sample_row,
                    threshold=float(summary.get("threshold", 0.5)),
                    total_samples=len(full_target_df),
                )
                with column:
                    st.markdown(f"**Amostra {sample_id}**")
                    st.metric("Chance estimada", f"{float(sample_row['y_score']):.2%}")
                    st.metric("Nivel de prioridade", str(sample_row.get("tier", "-")))
                    st.caption(sample_story["business_meaning"])
                    try:
                        compare_gradcam = load_sample_gradcam(PROJECT_ROOT, str(sample_id), model_key="a11_pipeline_e2e")
                    except Exception as exc:
                        st.warning(f"Imagem explicativa indisponivel: {exc}")
                    else:
                        st.image(
                            compare_gradcam["overlay"],
                            caption="Partes da imagem que mais pesaram",
                            use_container_width=True,
                        )


if __name__ == "__main__":
    render_page()
