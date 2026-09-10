"""Unit tests for dashboard data helpers (Module 6)."""
import pandas as pd
import pytest

from src.dashboard.data import (
    CLASS_COLORS,
    build_field_map,
    class_color,
    class_counts,
    histogram_bins,
    load_detections,
    summary_metrics,
)


def _sample_df():
    return pd.DataFrame({
        "krop_id": [1, 2, 3, 4, 5],
        "weight_class": ["light", "light", "medium", "heavy", "heavy"],
        "weight_g": [300, 320, 420, 600, 700],
        "lat": [51.25, 51.2501, 51.2502, 51.2503, 51.2504],
        "lon": [5.95, 5.9501, 5.9502, 5.9503, 5.9504],
    })


def test_load_detections_validates_columns(tmp_path):
    good = tmp_path / "good.csv"
    _sample_df().to_csv(good, index=False)
    df = load_detections(str(good))
    assert len(df) == 5

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_detections(str(bad))


def test_class_counts_covers_all_classes():
    counts = class_counts(_sample_df())
    assert counts == {"light": 2, "medium": 1, "heavy": 2}
    # empty frame still returns all three keys at zero
    empty = pd.DataFrame({"krop_id": [], "weight_class": []})
    assert class_counts(empty) == {"light": 0, "medium": 0, "heavy": 0}


def test_class_color_mapping():
    assert class_color("light") == CLASS_COLORS["light"]
    assert class_color("heavy") == CLASS_COLORS["heavy"]
    assert class_color("unknown") == "#808080"


def test_summary_metrics():
    m = summary_metrics(_sample_df())
    assert m["total"] == 5
    assert m["counts"]["heavy"] == 2
    assert m["mean_weight_g"] == pytest.approx(468.0, abs=0.1)
    assert m["total_yield_kg"] == pytest.approx(2.34, abs=0.01)


def test_histogram_bins():
    hb = histogram_bins(_sample_df(), bins=5)
    assert len(hb["counts"]) == 5
    assert len(hb["edges"]) == 6
    assert sum(hb["counts"]) == 5


def test_build_field_map_returns_folium_map():
    import folium
    fmap = build_field_map(_sample_df())
    assert isinstance(fmap, folium.Map)
    # a map with no coordinates still returns a valid (centred) map
    fmap2 = build_field_map(pd.DataFrame({"krop_id": [], "weight_class": []}))
    assert isinstance(fmap2, folium.Map)
