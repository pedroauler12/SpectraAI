from __future__ import annotations

import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


THRESHOLD_OPTIONS = {
    "artifact_default": "Threshold do artefato (recomendado)",
    "threshold_0.5": "0.5 (mais conservador)",
}

QUALITY_MESSAGES = {
    "baixo_percentual_de_pixels_finitos": "Muitos pixels invalidos ou faltantes.",
    "muitos_pixels_nulos_ou_sem_resposta": "Grande parte do chip tem resposta nula ou muito baixa.",
    "bandas_com_baixa_variacao_espectral": "Varias bandas estao com pouca variacao espectral.",
    "possivel_nuvem_ou_saturacao": "Ha sinais de nuvem, brilho excessivo ou saturacao.",
    "possivel_sombra_ou_no_data": "Ha sinais de sombra forte ou ausencia de dados.",
    "chip_sem_pixels_validos": "O chip nao contem pixels validos para inferencia.",
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


def build_model_catalog(project_root: Path) -> dict[str, dict[str, object]]:
    a11_model_path = project_root / "artefatos" / "a11_pipeline_e2e" / "outputs" / "models" / "best_model.keras"
    a08_model_path = project_root / "outputs" / "a08_transfer_learning" / "best_model.keras"

    return {
        "a11_pipeline_e2e": {
            "label": "Pipeline Final (A11)",
            "available": a11_model_path.exists(),
            "description": (
                "Modelo final treinado no pipeline end-to-end do A11."
                if a11_model_path.exists()
                else f"Artefato ausente: {a11_model_path}"
            ),
        },
        "a08_transfer_learning": {
            "label": "Transfer Learning (A08)",
            "available": a08_model_path.exists(),
            "description": (
                "Modelo geoespacial pronto para inferencia direta em chips ASTER 9 bandas."
                if a08_model_path.exists()
                else f"Artefato ausente: {a08_model_path}"
            ),
        },
        "a03_mlp_pca": {
            "label": "MLP PCA (A03)",
            "available": False,
            "description": (
                "Modelo salvo, mas indisponivel no app porque o scaler/PCA do treino "
                "nao foram exportados como artefatos."
            ),
        },
    }


MODEL_CATALOG = build_model_catalog(PROJECT_ROOT)


st.set_page_config(
    page_title="SpectraAI - Demo Geoespacial ASTER",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_points_frame(project_root: Path) -> pd.DataFrame:
    excel_path = project_root / "data" / "banco.xlsx"
    df = pd.read_excel(excel_path, sheet_name="Banco de Dados Positivo-Negativ")
    df = df.dropna(subset=["latitude_wgs84_decimal", "longitude_wgs84_decimal"]).copy()
    df["numero_amostra"] = df["numero_amostra"].astype(str)
    return df


@st.cache_resource(show_spinner=True)
def get_inference_bundle(project_root: Path, model_key: str, threshold_mode: str):
    from inference import (
        load_a11_transfer_inference_bundle,
        load_transfer_inference_bundle,
    )

    if model_key == "a11_pipeline_e2e":
        if threshold_mode == "threshold_0.5":
            return load_a11_transfer_inference_bundle(
                project_root=project_root,
                decision_threshold=0.5,
                decision_threshold_name="threshold_0.5",
            )
        return load_a11_transfer_inference_bundle(
            project_root=project_root,
        )

    if model_key != "a08_transfer_learning":
        raise RuntimeError(
            "Este modelo ainda nao esta pronto para a demo geoespacial. "
            "Use o Pipeline Final (A11) ou o Transfer Learning (A08)."
        )

    if threshold_mode == "threshold_0.5":
        return load_transfer_inference_bundle(
            project_root=project_root,
            decision_threshold=0.5,
            decision_threshold_name="threshold_0.5",
        )

    return load_transfer_inference_bundle(
        project_root=project_root,
        decision_threshold_name="threshold_f1",
    )


def clear_local_cache(cache_root: Path) -> None:
    from inference import clear_cache_dir

    clear_cache_dir(cache_root)


def run_point_inference(bundle, **kwargs):
    from inference import predict_point_from_earthdata

    return predict_point_from_earthdata(bundle, **kwargs)


def reset_last_result() -> None:
    st.session_state["last_result"] = None
    st.session_state["last_error"] = None


def quality_summary_message(severity: str) -> tuple[str, str]:
    if severity == "critical":
        return "error", "Qualidade critica do chip: use o resultado com muita cautela."
    if severity == "warning":
        return "warning", "Qualidade intermediaria: a cena pode estar degradando a confianca."
    return "success", "Qualidade visual aceitavel para uma leitura exploratoria."


def make_base_map(points_df: pd.DataFrame, selected_point: tuple[float, float] | None) -> folium.Map:
    center_lat = float(points_df["latitude_wgs84_decimal"].mean())
    center_lon = float(points_df["longitude_wgs84_decimal"].mean())

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=5, control_scale=True, tiles=None)
    folium.TileLayer("CartoDB positron", name="Base").add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    folium.LatLngPopup().add_to(fmap)

    for _, row in points_df.head(300).iterrows():
        folium.CircleMarker(
            location=[row["latitude_wgs84_decimal"], row["longitude_wgs84_decimal"]],
            radius=3,
            color="#2b8cbe",
            fill=True,
            fill_opacity=0.55,
            popup=f"Amostra {row['numero_amostra']}",
        ).add_to(fmap)

    if selected_point is not None:
        folium.CircleMarker(
            location=[selected_point[0], selected_point[1]],
            radius=8,
            color="#d7301f",
            fill=True,
            fill_color="#d7301f",
            fill_opacity=0.9,
            tooltip="Ponto selecionado",
        ).add_to(fmap)

    return fmap


st.title("SpectraAI - Demo Geoespacial ASTER")
st.caption(
    "Mapa clicavel, modelos A11/A08, previews RGB/false-color e "
    "checagem simples de qualidade do chip."
)

points_df = load_points_frame(PROJECT_ROOT)
cache_root = PROJECT_ROOT / "outputs" / "a09_streamlit_cache"

with st.sidebar:
    st.header("Configuracao")

    model_key = st.selectbox(
        "Modelo",
        options=list(MODEL_CATALOG.keys()),
        format_func=lambda key: MODEL_CATALOG[key]["label"],
    )
    threshold_mode = st.radio(
        "Threshold de decisao",
        options=list(THRESHOLD_OPTIONS.keys()),
        format_func=lambda key: THRESHOLD_OPTIONS[key],
        index=0,
    )

    selected_model_info = MODEL_CATALOG[model_key]
    if selected_model_info["available"]:
        st.success(selected_model_info["description"])
    else:
        st.warning(selected_model_info["description"])

    netrc_path = st.text_input("Caminho do .netrc EarthData", value=str(PROJECT_ROOT / ".netrc"))
    start_date = st.text_input("Data inicial", value="2000-01-01")
    end_date = st.text_input("Data final", value="2007-12-31")
    chip_side_m = st.number_input("Lado do chip (m)", min_value=500, max_value=5000, value=1200, step=100)
    margin_m = st.number_input("Margem interna (m)", min_value=0, max_value=1000, value=150, step=50)
    max_granules = st.number_input("Maximo de granules", min_value=1, max_value=25, value=10, step=1)
    force_refresh = st.checkbox("Forcar novo download", value=False)
    block_critical_quality = st.checkbox("Bloquear leitura quando qualidade for critica", value=True)

    if st.button("Limpar cache local"):
        clear_local_cache(cache_root)
        st.cache_data.clear()
        st.cache_resource.clear()
        reset_last_result()
        st.success("Cache removido.")

    st.markdown(
        """
        **Notas**
        - O A11 usa o artefato final salvo em `artefatos/a11_pipeline_e2e/outputs/models/`.
        - O A08 continua disponivel como baseline geoespacial anterior.
        - O app mostra RGB natural e false-color mineralogico.
        - A página "Ranking de Probabilidades" mostra o ranking completo das amostras.
        - MLP/SVM/RF ainda exigem artefatos extras de preprocessamento.
        """
    )


if "selected_point" not in st.session_state:
    st.session_state["selected_point"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None


map_col, result_col = st.columns([1.25, 1.0], gap="large")

with map_col:
    st.subheader("Mapa interativo")
    fmap = make_base_map(points_df, st.session_state["selected_point"])
    map_state = st_folium(
        fmap,
        key="spectraai_folium_map",
        width="stretch",
        height=620,
        returned_objects=["last_clicked"],
    )

    last_clicked = map_state.get("last_clicked")
    if last_clicked:
        new_point = (
            float(last_clicked["lat"]),
            float(last_clicked["lng"]),
        )
        if st.session_state["selected_point"] != new_point:
            st.session_state["selected_point"] = new_point
            reset_last_result()

    if st.session_state["selected_point"] is not None:
        lat_sel, lon_sel = st.session_state["selected_point"]
        st.info(f"Ponto selecionado: lat={lat_sel:.6f}, lon={lon_sel:.6f}")
    else:
        st.warning("Selecione um ponto no mapa para habilitar a inferencia.")


with result_col:
    st.subheader("Resultado da inferencia")

    if st.session_state["selected_point"] is None:
        st.write("Aguardando selecao de ponto.")
    else:
        lat_sel, lon_sel = st.session_state["selected_point"]
        can_run = selected_model_info["available"]

        if st.button("Rodar inferencia EarthData", type="primary", disabled=not can_run):
            reset_last_result()
            try:
                with st.spinner("Buscando granule ASTER, recortando chip e rodando o modelo..."):
                    bundle = get_inference_bundle(PROJECT_ROOT, model_key, threshold_mode)
                    result = run_point_inference(
                        bundle,
                        lat=lat_sel,
                        lon=lon_sel,
                        netrc_path=netrc_path,
                        cache_root=cache_root,
                        start_date=start_date,
                        end_date=end_date,
                        chip_side_m=float(chip_side_m),
                        margin_m=float(margin_m),
                        max_granules=int(max_granules),
                        force_refresh=force_refresh,
                    )

                result_payload = result.to_dict()
                result_payload["model_name"] = bundle.model_name
                result_payload["decision_threshold"] = float(bundle.decision_threshold)
                result_payload["decision_threshold_name"] = bundle.decision_threshold_name
                result_payload["label_at_threshold_05"] = bundle.class_names[int(result.prob_pos >= 0.5)]
                result_payload["label_at_bundle_threshold"] = bundle.class_names[
                    int(result.prob_pos >= bundle.decision_threshold)
                ]
                st.session_state["last_result"] = result_payload
            except Exception as exc:
                st.session_state["last_error"] = str(exc)

        if st.session_state.get("last_error"):
            st.error(st.session_state["last_error"])

        result = st.session_state.get("last_result")
        if result:
            quality_report = result.get("quality_report") or {}
            quality_severity = quality_report.get("severity", "ok")
            quality_box_type, quality_box_message = quality_summary_message(quality_severity)

            if quality_box_type == "error":
                st.error(quality_box_message)
            elif quality_box_type == "warning":
                st.warning(quality_box_message)
            else:
                st.success(quality_box_message)

            prob_pos = float(result["prob_pos"])
            threshold_value = float(result["decision_threshold"])
            pred_threshold_label = result["label_at_bundle_threshold"]
            pred_05_label = result["label_at_threshold_05"]

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Probabilidade positiva", f"{prob_pos:.2%}")
                st.metric("Modelo", result["model_name"])
            with metric_col2:
                st.metric("Classe no threshold do app", pred_threshold_label)
                st.metric("Classe no threshold 0.5", pred_05_label)

            st.progress(prob_pos)
            st.caption(
                f"Threshold ativo: {result['decision_threshold_name']} = {threshold_value:.4f}"
            )

            if quality_severity == "critical" and block_critical_quality:
                st.warning(
                    "O app bloqueou a leitura operacional deste resultado porque a qualidade do chip "
                    "foi classificada como critica."
                )

            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                if result.get("preview_rgb_path"):
                    st.image(result["preview_rgb_path"], caption="RGB ASTER (B03N, B02, B01)")
                else:
                    st.info("Preview RGB indisponivel.")
            with preview_col2:
                if result.get("preview_false_color_path"):
                    st.image(
                        result["preview_false_color_path"],
                        caption="False-color ASTER (B06, B05, B02)",
                    )
                elif result.get("preview_png_path"):
                    st.image(result["preview_png_path"], caption="False-color ASTER")
                else:
                    st.info("Preview false-color indisponivel.")

            st.markdown("**Qualidade do chip**")
            quality_metrics = {
                "severity": quality_severity,
                "finite_ratio": quality_report.get("finite_ratio"),
                "zero_ratio": quality_report.get("zero_ratio"),
                "bright_ratio": quality_report.get("bright_ratio"),
                "dark_ratio": quality_report.get("dark_ratio"),
                "low_dynamic_bands": quality_report.get("low_dynamic_bands"),
            }
            st.json(quality_metrics)

            quality_warnings = quality_report.get("warnings") or []
            if quality_warnings:
                st.markdown("**Alertas de qualidade**")
                for warning_key in quality_warnings:
                    st.write(f"- {QUALITY_MESSAGES.get(warning_key, warning_key)}")

            st.markdown("**Metadados do granule**")
            st.json(
                {
                    "granule_id": result.get("granule_id"),
                    "acquisition_time": result.get("acquisition_time"),
                    "cloud_cover": result.get("cloud_cover"),
                    "bbox_wgs84": result.get("bbox_wgs84"),
                    "chip_shape": result.get("chip_shape"),
                    "chip_path": result.get("chip_path"),
                }
            )

            st.markdown(
                """
                **Leitura de confianca**
                - A probabilidade e mais informativa do que o rotulo isolado.
                - O threshold_f1 tende a recuperar mais positivos do que o corte em 0.5.
                - Cena ruim, nuvem, sombra ou bandas pobres podem puxar a probabilidade para baixo.
                - O app e exploratorio e nao substitui a analise geologica.
                """
            )
