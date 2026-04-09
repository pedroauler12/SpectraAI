from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


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
        "Onde o modelo concentra atencao",
        "Grad-CAM mostra as regioes do chip que mais influenciaram a decisao em exemplos do teste.",
    ),
    (
        "spectral_analysis.png",
        "Leitura espectral",
        "Resume como as respostas espectrais diferem entre grupos e ajuda a conectar o modelo ao comportamento mineralogico.",
    ),
    (
        "channel_adapter_importance.png",
        "Importancia dos canais",
        "Indica quais bandas adaptadas para a CNN carregam mais sinal util para a classificacao.",
    ),
    (
        "prospectivity_maps.png",
        "Distribuicao espacial da prospectividade",
        "Mostra como os scores se distribuem no territorio, facilitando a leitura de clusters e hotspots.",
    ),
]

TECHNICAL_FIGURES = [
    (
        "roc_pr_curves.png",
        "Curvas ROC e PR",
        "Evidenciam separacao entre classes e desempenho em cenario desbalanceado.",
    ),
    (
        "threshold_sweep.png",
        "Sensibilidade ao threshold",
        "Ajuda a explicar como a troca do ponto de corte altera recall, precisao e F1.",
    ),
    (
        "prob_distributions.png",
        "Distribuicao de probabilidades",
        "Mostra a distancia entre positivos e negativos no espaco de score.",
    ),
    (
        "prob_boxplot.png",
        "Boxplot de scores por classe",
        "Resume dispersao e sobreposicao das probabilidades previstas.",
    ),
    (
        "confusion_matrix.png",
        "Matriz de confusao",
        "Consolida o tipo de erro que o pipeline final comete no conjunto de teste.",
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
        return "Forte para priorizacao operacional"
    if roc_auc >= 0.80 and f1_score >= 0.70:
        return "Bom para priorizacao com revisao geologica"
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
        f"No conjunto de teste, o pipeline final atingiu F1 de {_format_pct(f1_score)}, ROC-AUC de {_format_pct(roc_auc)} e PR-AUC de {_format_pct(pr_auc)}.",
        f"O recall de {_format_pct(recall)} mostra boa capacidade de recuperar alvos positivos, enquanto a precisao de {_format_pct(precision)} ajuda a manter a shortlist mais limpa.",
        f"O threshold operacional atual e {threshold:.2f}; por isso o app deve ser lido como sistema de priorizacao de alvos, nao como confirmacao geologica.",
    ]

    if n_test < 100:
        takeaways.append(
            f"A avaliacao final foi feita com {n_test} amostras de teste; o sinal e forte, mas ainda merece validacao de campo e acumulacao de novos dados."
        )

    return takeaways


def build_operational_cautions(summary: dict[str, Any]) -> list[str]:
    cautions = [
        "Probabilidade alta indica prioridade de investigacao, nao garantia de mineralizacao economicamente viavel.",
        "Os mapas de atencao ajudam a explicar o comportamento do modelo, mas nao substituem interpretacao geologica e checagem de qualidade da cena.",
    ]

    n_test = int(summary.get("n_test", 0))
    if n_test < 100:
        cautions.append("O conjunto de teste ainda e pequeno; novos dados de campo vao melhorar a confianca do sistema.")

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


def render_page() -> None:
    st.set_page_config(
        page_title="SpectraAI - Explicabilidade",
        layout="wide",
    )

    st.title("Explicabilidade do Modelo")
    st.caption("Leitura guiada do Pipeline Final (A11) para apoiar decisao geologica e priorizacao de campo.")

    with st.sidebar:
        st.header("Exploracao")
        top_n = st.slider("Top alvos explicados", min_value=5, max_value=30, value=12, step=1)
        min_score = st.slider("Score minimo", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        only_positive = st.checkbox("Somente preditos como positivos", value=True)

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
    metric_col1.metric("Readiness", readiness)
    metric_col2.metric("F1 em teste", _format_pct(summary.get("test_f1")))
    metric_col3.metric("ROC-AUC", _format_pct(summary.get("test_roc_auc")))
    metric_col4.metric("PR-AUC", _format_pct(summary.get("test_pr_auc")))
    metric_col5.metric("Threshold", f"{float(summary.get('threshold', 0.5)):.2f}")

    tabs = st.tabs(
        [
            "Parecer da amostra",
            "Resumo executivo",
            "Evidencia visual",
            "Amostras prioritarias",
            "Leitura tecnica",
        ]
    )

    with tabs[0]:
        st.subheader("Leitura personalizada por amostra")
        st.caption("Traduz o score do modelo em uma recomendacao simples para priorizacao de campo.")

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
            metric_col2.metric("Score", f"{float(sample_row['y_score']):.2%}")
            metric_col3.metric("Tier", normalize_tier_label(sample_row.get("tier")))
            metric_col4.metric("Confianca da leitura", sample_story["confidence_title"])

            st.progress(float(sample_row["y_score"]))
            st.caption(sample_story["plain_score"])

            left_col, right_col = st.columns([1.1, 0.9], gap="large")
            with left_col:
                st.subheader("Em linguagem simples")
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
                st.dataframe(detail_frame, width="stretch", hide_index=True)

            st.subheader("Proximo passo sugerido")
            st.info(sample_story["next_step"])
            st.caption(
                "Esta leitura individual combina score, posicao no ranking e contexto geologico. Ela foi desenhada para apoiar a decisao, nao para substituir validacao de campo."
            )

            st.subheader("Onde o modelo olhou nesta amostra")
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
                    caption="Imagem base em falsa cor",
                    use_container_width=True,
                )
                image_col2.image(
                    gradcam_data["heatmap"],
                    caption="Mapa de atencao Grad-CAM",
                    use_container_width=True,
                    clamp=True,
                )
                image_col3.image(
                    gradcam_data["overlay"],
                    caption="Sobreposicao do Grad-CAM",
                    use_container_width=True,
                )
                st.caption(
                    f"Score previsto: {_format_pct(gradcam_data['prob_pos'])} | threshold operacional: {gradcam_data['decision_threshold']:.2f}"
                )

    with tabs[1]:
        left_col, right_col = st.columns([1.1, 0.9], gap="large")
        with left_col:
            st.subheader("O que isso significa para a Frontera")
            for takeaway in takeaways:
                st.markdown(f"- {takeaway}")

            st.subheader("Cuidados de uso")
            for caution in cautions:
                st.markdown(f"- {caution}")

        with right_col:
            st.subheader("Resumo do experimento")
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
            st.dataframe(summary_frame, width="stretch", hide_index=True)

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
                st.subheader("Robustez em validacao cruzada")
                st.dataframe(cv_frame, width="stretch", hide_index=True)

    with tabs[2]:
        render_visual_gallery(image_paths, VISUAL_SECTIONS)

    with tabs[3]:
        st.subheader("Top alvos para leitura explicavel")
        st.caption(
            "A tabela conecta o ranking de prospectividade do A11 aos metadados geologicos para facilitar a triagem."
        )
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
            "Baixar tabela de alvos explicados",
            data=target_df.to_csv(index=False).encode("utf-8"),
            file_name="a11_explicabilidade_alvos.csv",
            mime="text/csv",
        )

        if not target_df.empty:
            sample_id = st.selectbox("Amostra em destaque", options=target_df["image_id"].astype(str).tolist(), index=0)
            sample_row = target_df.loc[target_df["image_id"].astype(str) == sample_id].iloc[0]

            sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
            sample_col1.metric("Amostra", str(sample_row["image_id"]))
            sample_col2.metric("Rank", int(sample_row["rank"]))
            sample_col3.metric("Score", f"{float(sample_row['y_score']):.2%}")
            sample_col4.metric("Tier", str(sample_row.get("tier", "-")))

            detail_frame = pd.DataFrame(
                [
                    {"Campo": "Classe prevista", "Valor": sample_row.get("classe_prevista", "-")},
                    {"Campo": "Classe real", "Valor": sample_row.get("classe_real", "-")},
                    {"Campo": "Litologia", "Valor": sample_row.get("litologia_padronizada", "-")},
                    {"Campo": "Classe de base", "Valor": sample_row.get("classe_balanceamento", "-")},
                    {"Campo": "Latitude", "Valor": sample_row.get("latitude_wgs84_decimal", "-")},
                    {"Campo": "Longitude", "Valor": sample_row.get("longitude_wgs84_decimal", "-")},
                ]
            )
            st.dataframe(detail_frame, width="stretch", hide_index=True)
            st.info(
                "Leitura recomendada: combine o score desta amostra com a evidencia visual de Grad-CAM e com o contexto litologico antes de entrar em campo."
            )

    with tabs[4]:
        tech_left, tech_right = st.columns([1.0, 1.0], gap="large")
        with tech_left:
            render_visual_gallery(image_paths, TECHNICAL_FIGURES[:3])
        with tech_right:
            render_visual_gallery(image_paths, TECHNICAL_FIGURES[3:])

        if not hp_search_df.empty:
            st.subheader("Comparacao de hiperparametros")
            st.dataframe(
                hp_search_df.sort_values("val_f1", ascending=False).head(5),
                width="stretch",
                hide_index=True,
            )


if __name__ == "__main__":
    render_page()
