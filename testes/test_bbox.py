from src.tiles.bbox import bbox_with_point_inside


def test_bbox_contains_point():
    lat, lon = -23.5, -46.6
    bbox = bbox_with_point_inside(lat, lon, chip_side_m=1000, margin_m=0, seed=42)
    lon_w, lat_s, lon_e, lat_n = bbox
    assert lon_w <= lon <= lon_e
    assert lat_s <= lat <= lat_n


def test_bbox_invalid_margin():
    try:
        bbox_with_point_inside(0, 0, chip_side_m=100, margin_m=50)
    except ValueError as e:
        assert "margin_m" in str(e)
    else:
        raise AssertionError("Expected ValueError for invalid margin")


def test_bbox_deterministic_seed():
    b1 = bbox_with_point_inside(1, 2, chip_side_m=1000, margin_m=10, seed=123)
    b2 = bbox_with_point_inside(1, 2, chip_side_m=1000, margin_m=10, seed=123)
    assert b1 == b2
