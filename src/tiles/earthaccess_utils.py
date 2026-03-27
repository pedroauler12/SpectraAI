from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import Config


def login_earthdata(netrc_path: str) -> None:
    """
    Faz login via .netrc. No Colab, aponte para um arquivo no Drive.
    Ex: /content/drive/MyDrive/Aster/.netrc
    """
    import earthaccess

    os.environ.setdefault("NETRC", netrc_path)
    earthaccess.login(strategy="netrc")


def read_points(cfg: Config) -> pd.DataFrame:
    df = pd.read_excel(cfg.excel_path, sheet_name=cfg.sheet)
    df = df.dropna(subset=[cfg.lat_col, cfg.lon_col])
    return df


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_search_kwargs(
    cfg: Config, bbox_wgs84: Tuple[float, float, float, float]
) -> Dict[str, Any]:
    return dict(
        concept_id=cfg.concept_id,
        bounding_box=bbox_wgs84,
        temporal=(cfg.start_date, cfg.end_date),
        count=cfg.max_granules,
    )


def granule_start_time(g: Any) -> Optional[datetime]:
    """
    Extrai BeginningDateTime do UMM (quando disponível).
    """
    umm = getattr(g, "umm", None)
    if not umm:
        return None
    te = umm.get("TemporalExtent", {}) or {}
    rr = te.get("RangeDateTime", {}) or {}
    start = rr.get("BeginningDateTime")
    if not start:
        return None
    # exemplo: "2006-05-17T00:00:00.000Z"
    try:
        return datetime.fromisoformat(start.replace("Z", "+00:00"))
    except Exception:
        return None


def filter_granules_by_date(
    granules: List[Any], start_date: str, end_date: str
) -> List[Any]:
    """
    Hard filter: garante que você NÃO pegue cenas fora do intervalo,
    mesmo que a busca retorne algo estranho.
    """
    start_dt = datetime.fromisoformat(start_date + "T00:00:00+00:00")
    end_dt = datetime.fromisoformat(end_date + "T23:59:59+00:00")
    kept: List[Any] = []
    for g in granules:
        t = granule_start_time(g)
        if t and (start_dt <= t <= end_dt):
            kept.append(g)
    return kept


def cloud_cover_value(g: Any) -> float:
    """
    Heurística: tenta achar CloudCover no UMM.
    Se não achar, devolve 999.
    """
    try:
        umm = getattr(g, "umm", None) or {}
        for aa in (umm.get("AdditionalAttributes", []) or []):
            name = str(aa.get("Name", "")).lower()
            if name in ("cloudcover", "cloud_cover", "cloud cover"):
                vals = aa.get("Values", [])
                if vals:
                    return float(vals[0])
        return 999.0
    except Exception:
        return 999.0


def choose_best_granule(granules: List[Any]) -> Optional[Any]:
    """
    Escolhe o melhor granule (menor nuvem).
    """
    if not granules:
        return None
    return sorted(granules, key=cloud_cover_value)[0]


def download_granule(granule: Any, outdir: Path, dry_run: bool) -> None:
    import earthaccess

    if dry_run:
        print(f"  [DRY_RUN] download -> {outdir}")
        return
    earthaccess.download([granule], outdir)
