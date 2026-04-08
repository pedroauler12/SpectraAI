import importlib
from datetime import datetime
from pathlib import Path

import pytest


def test_ensure_dir(tmp_path: Path):
    mod = importlib.import_module("src.tiles.earthaccess_utils")
    out = mod.ensure_dir(tmp_path / "a" / "b")
    assert out.exists()
    assert out.is_dir()


def test_build_search_kwargs():
    mod = importlib.import_module("src.tiles.earthaccess_utils")
    cfg = mod.Config(
        concept_id="TEST",
        start_date="2000-01-01",
        end_date="2000-12-31",
        max_granules=5,
    )
    bbox = (-10.0, -5.0, 10.0, 5.0)
    kwargs = mod.build_search_kwargs(cfg, bbox)
    assert kwargs["concept_id"] == "TEST"
    assert kwargs["bounding_box"] == bbox
    assert kwargs["temporal"] == ("2000-01-01", "2000-12-31")
    assert kwargs["count"] == 5


def test_granule_time_and_filter():
    mod = importlib.import_module("src.tiles.earthaccess_utils")

    class G:
        def __init__(self, dt):
            self.umm = {
                "TemporalExtent": {
                    "RangeDateTime": {"BeginningDateTime": dt}
                }
            }

    g1 = G("2001-01-01T00:00:00.000Z")
    g2 = G("2001-06-01T00:00:00.000Z")
    g3 = G("2002-01-01T00:00:00.000Z")

    assert mod.granule_start_time(g1) == datetime.fromisoformat("2001-01-01T00:00:00+00:00")

    kept = mod.filter_granules_by_date([g1, g2, g3], "2001-01-01", "2001-12-31")
    assert kept == [g1, g2]


def test_cloud_cover_and_choose_best():
    mod = importlib.import_module("src.tiles.earthaccess_utils")

    class G:
        def __init__(self, cloud):
            self.umm = {
                "AdditionalAttributes": [
                    {"Name": "CloudCover", "Values": [str(cloud)]}
                ]
            }

    g1 = G(50)
    g2 = G(10)

    assert mod.cloud_cover_value(g1) == 50.0
    assert mod.choose_best_granule([g1, g2]) == g2


def test_login_and_download_mock(monkeypatch, tmp_path: Path):
    earthaccess = pytest.importorskip("earthaccess")
    mod = importlib.import_module("src.tiles.earthaccess_utils")

    called = {"login": False, "download": False}

    def fake_login(strategy):
        assert strategy == "netrc"
        called["login"] = True

    def fake_download(granules, outdir):
        assert outdir == tmp_path
        called["download"] = True

    monkeypatch.setattr(earthaccess, "login", fake_login)
    monkeypatch.setattr(earthaccess, "download", fake_download)

    mod.login_earthdata("/tmp/.netrc")
    mod.download_granule("g", tmp_path, dry_run=False)

    assert called["login"] is True
    assert called["download"] is True
