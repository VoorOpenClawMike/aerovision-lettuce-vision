"""Unit tests for the end-to-end pipeline orchestrator."""
import csv
import math

from src.data.synth_generator import SynthConfig, generate_dataset
from src.pipeline import FIELD_LAT, FIELD_LON, pixel_to_latlon, run_pipeline


def test_pixel_to_latlon_near_reference():
    lat, lon = pixel_to_latlon(0, 0, tile_row=0, tile_col=0,
                               gsd_cm_per_px=0.25, image_h_px=640)
    # origin tile, bottom-left pixel maps very close to the field reference
    assert abs(lat - FIELD_LAT) < 0.001
    assert abs(lon - FIELD_LON) < 0.001
    # moving east (larger col) increases longitude
    _, lon_east = pixel_to_latlon(0, 0, 0, 3, 0.25, 640)
    assert lon_east > lon


def test_run_pipeline_fallback_produces_csv(tmp_path):
    cfg = SynthConfig(num_images=3, heads_per_image=12, seed=5, out_dir=str(tmp_path))
    summary = generate_dataset(cfg)
    out_csv = str(tmp_path / "field.csv")

    result = run_pipeline(
        summary["coco_path"], summary["csv_path"], summary["images_dir"],
        out_csv, gsd_cm_per_px=0.25, model_path=None)

    # with no model, it must fall back to ground-truth polygons
    assert result["detection_source"] == "ground_truth_fallback"
    assert result["num_detections"] == summary["num_instances"]
    assert sum(result["class_counts"].values()) == summary["num_instances"]

    rows = list(csv.DictReader(open(out_csv)))
    assert len(rows) == summary["num_instances"]
    expected_cols = {"krop_id", "image", "cx", "cy", "area_px", "area_cm2",
                     "weight_g", "weight_class", "lat", "lon"}
    assert expected_cols.issubset(rows[0].keys())
    for r in rows:
        assert r["weight_class"] in ("light", "medium", "heavy")
        assert float(r["weight_g"]) > 0
        assert 50.0 < float(r["lat"]) < 52.0    # plausibly near Midden-Limburg
        assert 5.0 < float(r["lon"]) < 7.0


def test_missing_model_path_falls_back(tmp_path):
    cfg = SynthConfig(num_images=2, heads_per_image=6, seed=1, out_dir=str(tmp_path))
    summary = generate_dataset(cfg)
    out_csv = str(tmp_path / "field.csv")
    result = run_pipeline(summary["coco_path"], summary["csv_path"],
                          summary["images_dir"], out_csv,
                          model_path="/does/not/exist.pt")
    assert result["detection_source"] == "ground_truth_fallback"
    assert result["num_detections"] == summary["num_instances"]
