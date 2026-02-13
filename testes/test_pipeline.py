from pathlib import Path
import pandas as pd

from src.tiles.config import Config
import src.tiles.pipeline as pipeline


def test_run_calls_process_one_point(monkeypatch, tmp_path: Path):
    calls = []

    def fake_login(path):
        return None

    def fake_read_points(cfg):
        return pd.DataFrame(
            {
                cfg.id_col: ["A", "B"],
                cfg.lat_col: [0.0, 1.0],
                cfg.lon_col: [0.0, 1.0],
            }
        )

    def fake_process(cfg, sample_id, lat, lon, row_seed):
        calls.append((sample_id, lat, lon, row_seed))

    monkeypatch.setattr(pipeline, "login_earthdata", fake_login)
    monkeypatch.setattr(pipeline, "read_points", fake_read_points)
    monkeypatch.setattr(pipeline, "process_one_point", fake_process)

    cfg = Config(out_root=str(tmp_path), dry_run=True)
    pipeline.run(cfg, netrc_path=str(tmp_path / ".netrc"), limit_rows=None)

    assert len(calls) == 2
    assert calls[0][0] == "A"
    assert calls[1][0] == "B"
