from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rasterio.enums import Resampling


@dataclass
class Config:
    # Excel
    excel_path: str = "banco.xlsx"
    sheet: str = "Banco de Dados Positivo-Negativ"
    lat_col: str = "latitude_wgs84_decimal"
    lon_col: str = "longitude_wgs84_decimal"
    id_col: str = "numero_amostra"
    label_col: Optional[str] = None

    # Earthdata (USE SOMENTE concept_id)
    concept_id: str = "C3306884993-LPCLOUD"  # AST_07XT v004 (COG)

    # Datas
    start_date: str = "2000-01-01"
    end_date: str = "2007-12-31"

    # Busca
    max_granules: int = 10

    # Output
    out_root: str = "/content/drive/MyDrive/ASTER_IMG"

    # Chip
    chip_side_m: float = 2000.0
    margin_m: float = 200.0
    jitter_seed_mode: str = "row_index"
    fixed_seed: int = 123

    # Multi-bandas
    stack_resampling: Resampling = Resampling.nearest

    # Debug
    dry_run: bool = False
    keep_downloads: bool = True
