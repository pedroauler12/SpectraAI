from __future__ import annotations

from typing import Optional

import earthaccess

from .bbox import bbox_with_point_inside
from .config import Config
from .earthaccess_utils import (
    build_search_kwargs,
    choose_best_granule,
    cloud_cover_value,
    download_granule,
    ensure_dir,
    granule_start_time,
    login_earthdata,
    read_points,
)
from .multiband import crop_and_stack_multiband, list_band_tifs, summarize_tif


def process_one_point(
    cfg: Config, sample_id: str, lat: float, lon: float, row_seed: int
) -> None:
    out_root = ensure_dir(cfg.out_root)
    point_dir = ensure_dir(out_root / str(sample_id))

    # bbox: ponto dentro do chip (não precisa ser centro)
    if cfg.jitter_seed_mode == "fixed":
        seed = cfg.fixed_seed
    else:
        seed = row_seed

    bbox = bbox_with_point_inside(
        lat=lat,
        lon=lon,
        chip_side_m=cfg.chip_side_m,
        margin_m=cfg.margin_m,
        seed=seed,
    )

    print(f"\n[{sample_id}] lat={lat} lon={lon}")
    print(f"  chip_side={cfg.chip_side_m:.0f}m margin={cfg.margin_m:.0f}m seed={seed}")
    print(f"  bbox(W,S,E,N)={bbox}")

    # Busca
    search_kwargs = build_search_kwargs(cfg, bbox)
    if cfg.dry_run:
        print(f"  [DRY_RUN] search_data({search_kwargs})")
        return

    granules = earthaccess.search_data(**search_kwargs)
    print(f"  search retornou: {len(granules)} granule(s)")

    # Hard filter por data (pra não cair 2025)
    print(f"  usando temporal do CMR: {cfg.start_date}–{cfg.end_date} (sem hard-filter local)")

    # Escolhe melhor (menor nuvem)
    best = choose_best_granule(granules)
    bt = granule_start_time(best)
    cc = cloud_cover_value(best)
    print(f"  best granule time={bt} cloud≈{cc}")

    # Download
    print("  baixando...")
    download_granule(best, point_dir, cfg.dry_run)

    # Listar tifs (bandas)
    tif_paths = list_band_tifs(point_dir)
    print(f"  tifs encontrados p/ empilhar: {len(tif_paths)}")
    for p in tif_paths[:10]:
        print(f"   - {p.name}")

    if len(tif_paths) < 2:
        print("  ⚠️ veio <2 tif. Isso significa que seu download não trouxe múltiplas bandas como arquivos separados.")
        print("     (Pode ser o produto errado/derivado. Cole aqui os nomes dos arquivos que baixaram que eu te digo o ajuste.)")
        # ainda assim, se veio 1 tif, vamos gerar chip com 1 banda para não perder
        if len(tif_paths) == 1:
            out_chip = point_dir / f"chip_{int(cfg.chip_side_m)}m_multiband.tif"
            chip = crop_and_stack_multiband(
                tif_paths=tif_paths,
                out_tif=out_chip,
                bbox_wgs84=bbox,
                resampling=cfg.stack_resampling,
            )
            if chip:
                info = summarize_tif(chip)
                print(
                    f"  ✅ chip salvo (1 banda): {out_chip.name} -> bands={info['bands']} "
                    f"shape={info['height']}x{info['width']}"
                )
        return

    # Empilhar multibanda
    out_chip = point_dir / f"chip_{int(cfg.chip_side_m)}m_multiband.tif"
    chip = crop_and_stack_multiband(
        tif_paths=tif_paths,
        out_tif=out_chip,
        bbox_wgs84=bbox,
        resampling=cfg.stack_resampling,
    )

    if not chip:
        print("  ❌ falha ao gerar chip multibanda.")
        return

    info = summarize_tif(chip)
    print(f"  ✅ chip multibanda salvo: {out_chip.name}")
    print(f"     bands={info['bands']} shape={info['height']}x{info['width']} crs={info['crs']}")

    # Opcional: limpar downloads grandes
    if not cfg.keep_downloads:
        for p in tif_paths:
            try:
                p.unlink()
            except Exception:
                pass
        print("  🧹 downloads originais removidos (keep_downloads=False)")


def run(cfg: Config, netrc_path: str, limit_rows: Optional[int] = None) -> None:
    print("[auth] login earthdata via netrc...")
    login_earthdata(netrc_path)

    out_root = ensure_dir(cfg.out_root)
    print(f"[out] root: {out_root}")
    print(f"[cfg] dates={cfg.start_date}..{cfg.end_date}")
    print(
        f"[cfg] chip_side_m={cfg.chip_side_m} margin={cfg.margin_m} max_granules={cfg.max_granules}"
    )
    print(f"[cfg] dry_run={cfg.dry_run} keep_downloads={cfg.keep_downloads}")

    df = read_points(cfg)
    if limit_rows is not None:
        df = df.head(limit_rows)

    print(f"[data] pontos: {len(df)}")

    for i, row in df.iterrows():
        sample_id = row.get(cfg.id_col, f"row_{i}")
        try:
            lat = float(row[cfg.lat_col])
            lon = float(row[cfg.lon_col])
        except Exception as exc:
            print(f"\n[{sample_id}] erro lat/lon: {exc}. Pulando.")
            continue

        process_one_point(cfg, str(sample_id), lat, lon, row_seed=i)

    print("\nFim.")
