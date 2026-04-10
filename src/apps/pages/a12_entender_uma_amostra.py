from __future__ import annotations

import sys
from pathlib import Path

_CURRENT_FILE = Path(__file__).resolve()
_SRC_DIR = _CURRENT_FILE.parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from apps.explainability_shared import (
    PROJECT_ROOT,
    TECHNICAL_FIGURES,
    VISUAL_SECTIONS,
    _format_pct,
    build_executive_takeaways,
    build_nontechnical_sample_explanation,
    build_operational_cautions,
    classify_explainability_readiness,
    load_a11_explainability_assets,
    load_sample_gradcam,
    normalize_tier_label,
    prepare_explainability_target_table,
    render_visual_gallery,
    summarize_cross_validation,
    summarize_gradcam_heatmap,
)
import pandas as pd
import streamlit as st


def _stringify_value_column(df: pd.DataFrame, column_name: str = "Valor") -> pd.DataFrame:
    out = df.copy()
    if column_name in out.columns:
        out[column_name] = out[column_name].map(lambda value: "-" if pd.isna(value) else str(value))
    return out


def render_page() -> None:
    st.set_page_config(
        page_title="SpectraAI - Entendendo o Resultado",
        layout="wide",
    )

    st.title("Entendendo o Resultado")
    st.caption("Veja por que uma amostra chamou atencao e como usar isso para decidir melhor.")

    with st.sidebar:
        st.header("Refinar leitura")
        top_n = st.slider("Quantidade de alvos na tela", min_value=5, max_value=30, value=12, step=1)
        min_score = st.slider("Chance minima", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        only_positive = st.checkbox("Mostrar apenas os que parecem promissores", value=True)

    try:
        assets = load_a11_explainability_assets(PROJECT_ROOT)
    except Exception as exc:
        st.error(str(exc))
        return

    summary = assets["summary"]
    ranking_df = assets["ranking_df"]
    metadata_df = assets["metadata_df"]
    cv_results_df = assets["cv_results_df"]
    hp_search_df = assets["hp_search_df"]
    image_paths = assets["image_paths"]

    readiness = classify_explainability_readiness(summary)
    takeaways = build_executive_takeaways(summary)
    cautions = build_operational_cautions(summary)
    target_df = prepare_explainability_target_table(
        ranking_df,
        metadata_df,
        top_n=top_n,
        min_score=min_score,
        only_positive=only_positive,
    )

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric("Nivel de confianca geral", readiness)
    metric_col2.metric("F1 em teste", _format_pct(summary.get("test_f1")))
    metric_col3.metric("ROC-AUC", _format_pct(summary.get("test_roc_auc")))
    metric_col4.metric("PR-AUC", _format_pct(summary.get("test_pr_auc")))
    metric_col5.metric("Regra de decisao", f"{float(summary.get('threshold', 0.5)):.2f}")

    tabs = st.tabs(
        [
            "Leitura da amostra",
            "Resumo executivo",
            "Imagens de apoio",
            "Alvos em destaque",
            "Leitura tecnica",
        ]
    )

    with tabs[0]:
        st.subheader("Leitura personalizada por amostra")
        st.caption("Traduz o resultado do modelo em uma recomendacao simples para priorizacao.")

        if target_df.empty:
            st.warning("Nenhuma amostra atende aos filtros atuais para gerar um parecer individual.")
        else:
            sample_options = target_df["image_id"].astype(str).tolist()
            selected_sample_id = st.selectbox("Escolha a amostra", options=sample_options, index=0)
            sample_row = target_df.loc[target_df["image_id"].astype(str) == selected_sample_id].iloc[0]
            sample_story = build_nontechnical_sample_explanation(
                sample_row,
                threshold=float(summary.get("threshold", 0.5)),
                total_samples=len(target_df),
            )

            if sample_row.get("classe_prevista") == "Positivo" and sample_story["confidence_title"] in {"Alta", "Media"}:
                st.success(sample_story["headline"])
            elif sample_row.get("classe_prevista") == "Positivo":
                st.warning(sample_story["headline"])
            else:
                st.info(sample_story["headline"])

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("Amostra", str(sample_row["image_id"]))
            metric_col2.metric("Chance estimada", f"{float(sample_row['y_score']):.2%}")
            metric_col3.metric("Nivel de prioridade", normalize_tier_label(sample_row.get("tier")))
            metric_col4.metric("Confianca desta leitura", sample_story["confidence_title"])

            st.progress(float(sample_row["y_score"]))
            st.caption(sample_story["plain_score"])

            left_col, right_col = st.columns([1.1, 0.9], gap="large")
            with left_col:
                st.subheader("Resumo simples")
                st.markdown(sample_story["business_meaning"])
                st.markdown(sample_story["confidence_text"])
                st.markdown(sample_story["geology_text"])
                st.markdown(sample_story["actual_label_text"])

            with right_col:
                detail_frame = pd.DataFrame(
                    [
                        {"Campo": "Classe prevista", "Valor": sample_row.get("classe_prevista", "-")},
                        {"Campo": "Classe real", "Valor": sample_row.get("classe_real", "-")},
                        {"Campo": "Rank no recorte", "Valor": int(sample_row.get("rank", 0))},
                        {"Campo": "Litologia", "Valor": sample_row.get("litologia_padronizada", "-")},
                        {"Campo": "Latitude", "Valor": sample_row.get("latitude_wgs84_decimal", "-")},
                        {"Campo": "Longitude", "Valor": sample_row.get("longitude_wgs84_decimal", "-")},
                    ]
                )
                st.dataframe(_stringify_value_column(detail_frame), width="stretch", hide_index=True)

            st.subheader("Proximo passo")
            st.info(sample_story["next_step"])

            st.subheader("Partes da imagem que mais pesaram")
            try:
                gradcam_data = load_sample_gradcam(PROJECT_ROOT, selected_sample_id, model_key="a11_pipeline_e2e")
                gradcam_summary = summarize_gradcam_heatmap(gradcam_data["heatmap"])
            except Exception as exc:
                st.warning(f"Nao foi possivel gerar o Grad-CAM desta amostra: {exc}")
            else:
                if sample_row.get("classe_prevista") == "Positivo":
                    st.success(gradcam_summary["title"])
                else:
                    st.info(gradcam_summary["title"])
                st.markdown(gradcam_summary["summary"])

                image_col1, image_col2, image_col3 = st.columns(3)
                image_col1.image(
                    gradcam_data["false_color_preview"],
                    caption="Imagem de apoio da area",
                    use_container_width=True,
                )
                image_col2.image(
                    gradcam_data["heatmap"],
                    caption="Mapa de atencao",
                    use_container_width=True,
                    clamp=True,
                )
                image_col3.image(
                    gradcam_data["overlay"],
                    caption="Imagem com destaque das areas mais relevantes",
                    use_container_width=True,
                )
                st.caption(
                    f"Chance estimada: {_format_pct(gradcam_data['prob_pos'])} | regra de decisao: {gradcam_data['decision_threshold']:.2f}"
                )

    with tabs[1]:
        left_col, right_col = st.columns([1.1, 0.9], gap="large")
        with left_col:
            st.subheader("O que isso significa na pratica")
            for takeaway in takeaways:
                st.markdown(f"- {takeaway}")

            st.subheader("Cuidados importantes")
            for caution in cautions:
                st.markdown(f"- {caution}")

        with right_col:
            st.subheader("Resumo geral")
            summary_frame = pd.DataFrame(
                [
                    {"Indicador": "Modelo", "Valor": summary.get("model_name", "A11")},
                    {"Indicador": "Amostras validas", "Valor": int(summary.get("n_valid", 0))},
                    {"Indicador": "Amostras de treino", "Valor": int(summary.get("n_train", 0))},
                    {"Indicador": "Amostras de validacao", "Valor": int(summary.get("n_val", 0))},
                    {"Indicador": "Amostras de teste", "Valor": int(summary.get("n_test", 0))},
                    {"Indicador": "Tempo de treino", "Valor": f"{float(summary.get('training_time_seconds', 0.0)):.1f}s"},
                ]
            )
            st.dataframe(_stringify_value_column(summary_frame), width="stretch", hide_index=True)

            cv_summary = summarize_cross_validation(cv_results_df)
            if cv_summary:
                cv_frame = pd.DataFrame(
                    [
                        {"Indicador": "CV accuracy media", "Valor": _format_pct(cv_summary.get("accuracy_mean"))},
                        {"Indicador": "CV F1 media", "Valor": _format_pct(cv_summary.get("f1_mean"))},
                        {"Indicador": "CV ROC-AUC media", "Valor": _format_pct(cv_summary.get("roc_auc_mean"))},
                        {"Indicador": "CV PR-AUC media", "Valor": _format_pct(cv_summary.get("pr_auc_mean"))},
                    ]
                )
                st.subheader("Estabilidade em diferentes testes")
                st.dataframe(_stringify_value_column(cv_frame), width="stretch", hide_index=True)

    with tabs[2]:
        render_visual_gallery(image_paths, VISUAL_SECTIONS)

    with tabs[3]:
        st.subheader("Alvos em destaque")
        display_columns = [
            "rank",
            "image_id",
            "y_score",
            "tier",
            "classe_prevista",
            "classe_real",
            "litologia_padronizada",
            "latitude_wgs84_decimal",
            "longitude_wgs84_decimal",
        ]
        available_columns = [column for column in display_columns if column in target_df.columns]
        st.dataframe(
            target_df[available_columns],
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
        st.download_button(
            "Baixar tabela de alvos",
            data=target_df.to_csv(index=False).encode("utf-8"),
            file_name="a11_explicabilidade_alvos.csv",
            mime="text/csv",
        )

    with tabs[4]:
        tech_left, tech_right = st.columns([1.0, 1.0], gap="large")
        with tech_left:
            render_visual_gallery(image_paths, TECHNICAL_FIGURES[:3])
        with tech_right:
            render_visual_gallery(image_paths, TECHNICAL_FIGURES[3:])

        if not hp_search_df.empty:
            st.subheader("Comparacao entre configuracoes testadas")
            st.dataframe(
                hp_search_df.sort_values("val_f1", ascending=False).head(5),
                width="stretch",
                hide_index=True,
            )


if __name__ == "__main__":
    render_page()
