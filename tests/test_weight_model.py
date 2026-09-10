"""Unit tests for the weight regression module (Module 4)."""
import math

import numpy as np
import pytest

from src.regression.weight_model import (
    FEATURE_NAMES,
    WeightRegressor,
    feature_vector,
    features_from_mask,
    features_from_polygon,
    train_and_evaluate,
)


def _regular_polygon(n, r, cx=0.0, cy=0.0):
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in ang]


def test_features_of_regular_polygon():
    poly = _regular_polygon(64, 50, 100, 100)  # near-circle
    feats = features_from_polygon(poly, gsd_cm_per_px=0.25)
    # a 64-gon approximates a circle: compactness & hull_ratio ~ 1
    assert feats["compactness"] > 0.95
    assert feats["hull_ratio"] > 0.98
    # area_cm2 ~ pi r^2 * gsd^2
    expected = math.pi * 50 ** 2 * 0.25 ** 2
    assert feats["area_cm2"] == pytest.approx(expected, rel=0.02)
    # equivalent diameter ~ 2r*gsd = 25 cm
    assert feats["eq_diameter_cm"] == pytest.approx(2 * 50 * 0.25, rel=0.02)


def test_feature_vector_ordering_and_flat_input():
    # flat [x1,y1,...] and list-of-pairs give identical features
    poly_pairs = _regular_polygon(16, 30)
    flat = [c for xy in poly_pairs for c in xy]
    f1 = features_from_polygon(poly_pairs, 0.3)
    f2 = features_from_polygon(flat, 0.3)
    assert feature_vector(f1) == pytest.approx(feature_vector(f2))
    assert len(feature_vector(f1)) == len(FEATURE_NAMES) == 5
    with pytest.raises(ValueError):
        features_from_polygon([(0, 0), (1, 1)], 0.3)  # < 3 points


def test_features_from_mask_matches_polygon_area():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:60, 20:60] = 1  # 40x40 square, area 1600 px
    feats = features_from_mask(mask, gsd_cm_per_px=0.5)
    # area 1600 px * 0.25 = 400 cm2 (contour may clip by ~1px border)
    assert feats["area_cm2"] == pytest.approx(400, rel=0.1)


def test_regressor_fit_predict_and_classes():
    # synthetic linear target: weight grows with the first feature
    rng = np.random.default_rng(0)
    X = rng.uniform(50, 900, size=(200, 5))
    y = 0.9 * X[:, 0] + 50 + rng.normal(0, 5, size=200)
    reg = WeightRegressor(alpha=1.0).fit(X, y)
    pred = reg.predict(X)
    assert np.corrcoef(pred, y)[0, 1] > 0.95
    classes = reg.predict_classes(X[:5])
    assert all(c in ("light", "medium", "heavy") for c in classes)
    # predictions are clipped positive
    assert (reg.predict(np.zeros((3, 5))) >= 1.0).all()


def test_unfitted_regressor_raises():
    with pytest.raises(RuntimeError):
        WeightRegressor().predict(np.zeros((1, 5)))


def test_train_and_evaluate_on_synthetic(tmp_path):
    # build a tiny synthetic dataset on the fly
    from src.data.synth_generator import SynthConfig, generate_dataset
    cfg = SynthConfig(num_images=4, heads_per_image=20, seed=3, out_dir=str(tmp_path))
    summary = generate_dataset(cfg)
    reg, report = train_and_evaluate(
        summary["coco_path"], summary["csv_path"], gsd_cm_per_px=0.25, seed=0)
    # features are derived from area and weight ~ area^power -> strong fit
    assert report.r2 > 0.6
    assert report.mae < 120
    assert 0.0 <= report.class_accuracy <= 1.0
    assert report.n_train + report.n_test == summary["num_instances"]
