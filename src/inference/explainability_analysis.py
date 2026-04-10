"""
Analises reutilizaveis para explicabilidade operacional do ranking de prospectividade.
"""

from __future__ import annotations

from collections import Counter
from math import asin, cos, radians, sin, sqrt
from typing import Any

import pandas as pd


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distancia entre dois pontos WGS84 em quilometros.
    """
    earth_radius_km = 6371.0
    dlat = radians(float(lat2) - float(lat1))
    dlon = radians(float(lon2) - float(lon1))
    a = (
        sin(dlat / 2.0) ** 2
        + cos(radians(float(lat1))) * cos(radians(float(lat2))) * sin(dlon / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * asin(sqrt(a))


def _connected_components(adjacency: list[list[int]]) -> list[list[int]]:
    visited = [False] * len(adjacency)
    components: list[list[int]] = []

    for start_idx in range(len(adjacency)):
        if visited[start_idx]:
            continue
        stack = [start_idx]
        visited[start_idx] = True
        component: list[int] = []

        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        components.append(sorted(component))

    return components


def build_promising_region_clusters(
    ranking_df: pd.DataFrame,
    *,
    score_column: str = "y_score",
    sample_column: str = "image_id",
    lat_column: str = "latitude_wgs84_decimal",
    lon_column: str = "longitude_wgs84_decimal",
    min_score: float = 0.65,
    radius_km: float = 0.25,
    min_samples: int = 2,
) -> pd.DataFrame:
    """
    Agrupa amostras proximas geograficamente e com score alto em regioes promissoras.
    """
    if ranking_df.empty:
        return pd.DataFrame()

    required_columns = {score_column, sample_column, lat_column, lon_column}
    missing_columns = required_columns - set(ranking_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"DataFrame sem colunas obrigatorias para clusterizacao: {missing}")

    candidates = ranking_df.copy()
    candidates = candidates.dropna(subset=[lat_column, lon_column]).copy()
    candidates = candidates[candidates[score_column] >= float(min_score)].reset_index(drop=True)

    if candidates.empty:
        return pd.DataFrame()

    adjacency: list[list[int]] = [[] for _ in range(len(candidates))]
    points = candidates[[lat_column, lon_column]].to_numpy(dtype=float)
    for idx in range(len(points)):
        for jdx in range(idx + 1, len(points)):
            distance_km = haversine_distance_km(points[idx][0], points[idx][1], points[jdx][0], points[jdx][1])
            if distance_km <= float(radius_km):
                adjacency[idx].append(jdx)
                adjacency[jdx].append(idx)

    components = _connected_components(adjacency)
    clusters: list[dict[str, Any]] = []

    for cluster_idx, component in enumerate(components, start=1):
        cluster_rows = candidates.iloc[component].copy()
        if len(cluster_rows) < int(min_samples):
            continue

        score_values = cluster_rows[score_column].astype(float)
        lithologies = cluster_rows.get("litologia_padronizada", pd.Series(index=cluster_rows.index, dtype=object))
        lithologies = lithologies.dropna().astype(str).tolist()
        top_samples = cluster_rows.sort_values(score_column, ascending=False)[sample_column].astype(str).head(3).tolist()
        top_samples_text = ", ".join(top_samples)
        dominant_lithology = Counter(lithologies).most_common(1)[0][0] if lithologies else "-"

        clusters.append(
            {
                "cluster_id": cluster_idx,
                "region_label": f"Regiao Promissora {cluster_idx}",
                "n_amostras": int(len(cluster_rows)),
                "score_medio": float(score_values.mean()),
                "score_maximo": float(score_values.max()),
                "score_minimo": float(score_values.min()),
                "rank_medio": float(cluster_rows["rank"].mean()) if "rank" in cluster_rows.columns else None,
                "latitude_centro": float(cluster_rows[lat_column].mean()),
                "longitude_centro": float(cluster_rows[lon_column].mean()),
                "litologia_dominante": dominant_lithology,
                "top_amostras": top_samples_text,
                "amostras": top_samples,
            }
        )

    if not clusters:
        return pd.DataFrame()

    cluster_df = pd.DataFrame(clusters)
    cluster_df = cluster_df.sort_values(
        ["score_maximo", "score_medio", "n_amostras"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return cluster_df


def build_sample_heatmap_points(
    ranking_df: pd.DataFrame,
    *,
    score_column: str = "y_score",
    lat_column: str = "latitude_wgs84_decimal",
    lon_column: str = "longitude_wgs84_decimal",
    min_score: float = 0.0,
) -> list[list[float]]:
    """
    Converte o ranking em pontos ponderados para o HeatMap do folium.
    """
    if ranking_df.empty:
        return []

    valid = ranking_df.dropna(subset=[lat_column, lon_column]).copy()
    valid = valid[valid[score_column] >= float(min_score)]
    if valid.empty:
        return []

    return [
        [
            float(row[lat_column]),
            float(row[lon_column]),
            float(row[score_column]),
        ]
        for _, row in valid.iterrows()
    ]


def build_sample_comparison_table(
    ranking_df: pd.DataFrame,
    sample_ids: list[str],
    *,
    sample_column: str = "image_id",
) -> pd.DataFrame:
    """
    Monta uma tabela comparativa entre amostras selecionadas.
    """
    if ranking_df.empty or not sample_ids:
        return pd.DataFrame()

    df = ranking_df.copy()
    df[sample_column] = df[sample_column].astype(str)
    filtered = df[df[sample_column].isin([str(sample_id) for sample_id in sample_ids])].copy()
    if filtered.empty:
        return pd.DataFrame()

    preferred_columns = [
        "rank",
        sample_column,
        "y_score",
        "tier",
        "classe_prevista",
        "classe_real",
        "litologia_padronizada",
        "latitude_wgs84_decimal",
        "longitude_wgs84_decimal",
    ]
    ordered_columns = [column for column in preferred_columns if column in filtered.columns]
    ordered_columns += [column for column in filtered.columns if column not in ordered_columns]
    return filtered[ordered_columns].sort_values(["rank", sample_column]).reset_index(drop=True)


def describe_region_opportunity(cluster_row: pd.Series) -> str:
    """
    Traduz um cluster promissor em linguagem executiva.
    """
    n_samples = int(cluster_row.get("n_amostras", 0))
    score_maximo = float(cluster_row.get("score_maximo", 0.0))
    score_medio = float(cluster_row.get("score_medio", 0.0))
    lithology = cluster_row.get("litologia_dominante", "-")
    top_samples = str(cluster_row.get("top_amostras", "-"))

    if score_maximo >= 0.80 and n_samples >= 3:
        lead = "Esta e a regiao mais forte para priorizacao de campo."
    elif score_maximo >= 0.70:
        lead = "Esta regiao mostra um agrupamento consistente de sinais promissores."
    else:
        lead = "Esta regiao merece monitoramento, mas ainda com prioridade intermediaria."

    return (
        f"{lead} O agrupamento reune {n_samples} amostras proximas, com score medio de {score_medio:.2%} "
        f"e pico de {score_maximo:.2%}. Litologia dominante: {lithology}. "
        f"Amostras de referencia: {top_samples}."
    )


def build_campaign_suggestion(
    ranking_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    *,
    max_targets: int = 8,
    min_score: float = 0.60,
) -> pd.DataFrame:
    """
    Gera uma carteira sugerida de campanha a partir do ranking e das regioes promissoras.
    """
    if ranking_df.empty:
        return pd.DataFrame()

    candidates = ranking_df.copy()
    candidates["image_id"] = candidates["image_id"].astype(str)
    candidates = candidates[candidates["y_score"] >= float(min_score)].copy()
    if candidates.empty:
        return pd.DataFrame()

    cluster_lookup: dict[str, str] = {}
    for _, cluster_row in cluster_df.iterrows():
        for sample_id in cluster_row.get("amostras", []):
            cluster_lookup[str(sample_id)] = str(cluster_row["region_label"])

    selected_rows: list[pd.Series] = []
    selected_ids: set[str] = set()

    if not cluster_df.empty:
        for _, cluster_row in cluster_df.sort_values(["score_maximo", "n_amostras"], ascending=[False, False]).iterrows():
            cluster_samples = candidates[candidates["image_id"].isin([str(sample) for sample in cluster_row.get("amostras", [])])]
            if cluster_samples.empty:
                continue
            top_cluster_sample = cluster_samples.sort_values(["y_score", "rank"], ascending=[False, True]).iloc[0]
            sample_id = str(top_cluster_sample["image_id"])
            if sample_id not in selected_ids:
                selected_rows.append(top_cluster_sample)
                selected_ids.add(sample_id)
            if len(selected_rows) >= int(max_targets):
                break

    if len(selected_rows) < int(max_targets):
        for _, row in candidates.sort_values(["y_score", "rank"], ascending=[False, True]).iterrows():
            sample_id = str(row["image_id"])
            if sample_id in selected_ids:
                continue
            selected_rows.append(row)
            selected_ids.add(sample_id)
            if len(selected_rows) >= int(max_targets):
                break

    if not selected_rows:
        return pd.DataFrame()

    suggestion_df = pd.DataFrame(selected_rows).copy().reset_index(drop=True)
    suggestion_df["region_label"] = suggestion_df["image_id"].map(lambda sample_id: cluster_lookup.get(str(sample_id), "Alvo isolado"))
    suggestion_df["prioridade"] = suggestion_df["y_score"].map(
        lambda score: "Alta" if float(score) >= 0.80 else "Media" if float(score) >= 0.65 else "Observacao"
    )
    suggestion_df["status"] = "Sugerido"
    suggestion_df["janela_campo"] = suggestion_df["prioridade"].map(
        {
            "Alta": "Imediata",
            "Media": "Proxima campanha",
            "Observacao": "Se houver capacidade",
        }
    )
    suggestion_df["responsavel"] = ""
    suggestion_df["notas"] = ""
    suggestion_df["motivo_sugestao"] = suggestion_df.apply(
        lambda row: (
            f"Score de {float(row['y_score']):.2%}, tier {row.get('tier', '-')}, "
            f"regiao {row['region_label']}."
        ),
        axis=1,
    )
    return suggestion_df.sort_values(["prioridade", "y_score", "rank"], ascending=[True, False, True]).reset_index(drop=True)
