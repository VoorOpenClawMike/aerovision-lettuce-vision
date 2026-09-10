"""Unit tests for the synthetic dataset generator (Module 1)."""
import json
import math

import numpy as np

from src.data.synth_generator import (
    CLASS_HEAVY_MIN,
    CLASS_LIGHT_MAX,
    SynthConfig,
    area_px_to_cm2,
    generate_dataset,
    make_blob_polygon,
    polygon_area,
    weight_class,
    weight_from_area_cm2,
)


def test_polygon_area_square_and_blob():
    # exact square
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert polygon_area(square) == 100.0
    # degenerate
    assert polygon_area([(0, 0), (1, 1)]) == 0.0
    # blob area approximates the circle it perturbs (within ~35%)
    rng = np.random.default_rng(0)
    poly = make_blob_polygon(100, 100, 50, rng)
    circle_area = math.pi * 50 ** 2
    assert 0.65 * circle_area < polygon_area(poly) < 1.45 * circle_area


def test_weight_class_thresholds():
    assert weight_class(CLASS_LIGHT_MAX - 1) == "light"
    assert weight_class(CLASS_LIGHT_MAX) == "medium"       # boundary is not "light"
    assert weight_class(420) == "medium"
    assert weight_class(CLASS_HEAVY_MIN) == "medium"       # boundary is not "heavy"
    assert weight_class(CLASS_HEAVY_MIN + 1) == "heavy"


def test_weight_model_monotonic_and_positive():
    rng = np.random.default_rng(1)
    # with noise off, weight strictly increases with area
    w_small = weight_from_area_cm2(100, rng, noise_sigma=0.0)
    w_large = weight_from_area_cm2(400, rng, noise_sigma=0.0)
    assert 0 < w_small < w_large
    # area conversion uses gsd squared
    assert area_px_to_cm2(400, gsd_cm_per_px=0.5) == 100.0


def test_generate_dataset_end_to_end(tmp_path):
    cfg = SynthConfig(num_images=2, heads_per_image=8, seed=7, out_dir=str(tmp_path))
    summary = generate_dataset(cfg)

    assert summary["num_images"] == 2
    assert summary["num_instances"] == 16
    assert sum(summary["class_counts"].values()) == 16

    coco = json.load(open(summary["coco_path"]))
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 16
    ann = coco["annotations"][0]
    # each segmentation is a flat list of an even number of coords, >= 3 points
    seg = ann["segmentation"][0]
    assert len(seg) % 2 == 0 and len(seg) >= 6
    assert ann["category_id"] == 1 and ann["iscrowd"] == 0

    # images actually written
    import os
    assert len(os.listdir(summary["images_dir"])) == 2


def test_generation_is_deterministic(tmp_path):
    cfg_a = SynthConfig(num_images=1, heads_per_image=5, seed=99, out_dir=str(tmp_path / "a"))
    cfg_b = SynthConfig(num_images=1, heads_per_image=5, seed=99, out_dir=str(tmp_path / "b"))
    a = json.load(open(generate_dataset(cfg_a)["coco_path"]))
    b = json.load(open(generate_dataset(cfg_b)["coco_path"]))
    assert a["annotations"] == b["annotations"]
