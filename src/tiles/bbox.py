from __future__ import annotations

import random
from typing import Optional, Tuple

from pyproj import CRS, Geod


# --- Geodesia/CRS ---
CRS_WGS84 = CRS.from_epsg(4326)
WGS84_GEOD = Geod(ellps="WGS84")


def bbox_with_point_inside(
    lat: float,
    lon: float,
    chip_side_m: float,
    margin_m: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """
    Retorna bbox (W,S,E,N) em WGS84 de um quadrado de lado chip_side_m,
    garantindo que o ponto (lat, lon) esteja DENTRO do chip (com margem),
    mas NÃO necessariamente no centro.
    """
    if seed is not None:
        random.seed(seed)

    half = chip_side_m / 2.0
    if margin_m < 0 or margin_m >= half:
        raise ValueError("margin_m deve ser >= 0 e < chip_side_m/2")

    max_off = half - margin_m
    off_east_m = random.uniform(-max_off, +max_off)
    off_north_m = random.uniform(-max_off, +max_off)

    # move para achar centro do chip
    az_e = 90 if off_east_m >= 0 else 270
    lon_c, lat_c, _ = WGS84_GEOD.fwd(lon, lat, az_e, abs(off_east_m))

    az_n = 0 if off_north_m >= 0 else 180
    lon_c, lat_c, _ = WGS84_GEOD.fwd(lon_c, lat_c, az_n, abs(off_north_m))

    # bbox centrado no centro do chip
    lon_n, lat_n, _ = WGS84_GEOD.fwd(lon_c, lat_c, 0, half)
    lon_s, lat_s, _ = WGS84_GEOD.fwd(lon_c, lat_c, 180, half)
    lon_e, lat_e, _ = WGS84_GEOD.fwd(lon_c, lat_c, 90, half)
    lon_w, lat_w, _ = WGS84_GEOD.fwd(lon_c, lat_c, 270, half)

    return (lon_w, lat_s, lon_e, lat_n)
