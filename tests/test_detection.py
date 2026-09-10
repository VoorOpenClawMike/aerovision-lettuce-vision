"""Unit tests for detection: COCO->YOLO conversion + pure inference helpers."""
import json
import os

import numpy as np
import pytest

from src.detection.coco_to_yolo import (
    _split_indices,
    convert_coco_to_yolo_seg,
    normalize_polygon,
)
from src.detection.infer import _resize_mask, mask_area_centroid
from src.detection.train import build_train_config


# --------------------------- converter --------------------------- #
def test_normalize_polygon_clamps_and_scales():
    norm = normalize_polygon([0, 0, 100, 200, 150, 100], width=100, height=200)
    assert norm == [0.0, 0.0, 1.0, 1.0, 1.0, 0.5]
    # out-of-range coords are clamped to [0,1]
    assert normalize_polygon([-10, 400], 100, 200) == [0.0, 1.0]
    with pytest.raises(ValueError):
        normalize_polygon([1, 2], 0, 10)


def test_split_indices_keeps_both_splits():
    train, val = _split_indices(10, 0.2)
    assert len(val) == 2 and len(train) == 8
    assert set(train) | set(val) == set(range(10))
    # single image -> all train, no val
    assert _split_indices(1, 0.2) == ([0], [])


def _mini_coco(tmp_path):
    coco = {
        "images": [
            {"id": 0, "file_name": "a.png", "width": 100, "height": 100},
            {"id": 1, "file_name": "b.png", "width": 100, "height": 100},
        ],
        "categories": [{"id": 1, "name": "krop", "supercategory": "lettuce"}],
        "annotations": [
            {"id": 1, "image_id": 0, "category_id": 1,
             "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
             "area": 400, "bbox": [10, 10, 20, 20], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1,
             "segmentation": [[50, 50, 70, 50, 60, 70]],
             "area": 200, "bbox": [50, 50, 20, 20], "iscrowd": 0},
        ],
    }
    p = tmp_path / "coco.json"
    p.write_text(json.dumps(coco))
    return str(p)


def test_convert_creates_yolo_layout(tmp_path):
    coco_path = _mini_coco(tmp_path)
    out = str(tmp_path / "yolo")
    summary = convert_coco_to_yolo_seg(coco_path, images_dir=str(tmp_path), out_dir=out)

    assert summary["num_train"] + summary["num_val"] == 2
    assert summary["num_instances"] == 2
    assert summary["names"] == {0: "krop"}
    assert os.path.exists(os.path.join(out, "data.yaml"))
    # every label line: class id 0 then an even number of normalised coords in [0,1]
    label_files = []
    for split in ("train", "val"):
        d = os.path.join(out, "labels", split)
        label_files += [os.path.join(d, f) for f in os.listdir(d)]
    seen_lines = 0
    for lf in label_files:
        for line in open(lf).read().splitlines():
            if not line:
                continue
            seen_lines += 1
            parts = line.split()
            assert parts[0] == "0"
            coords = list(map(float, parts[1:]))
            assert len(coords) % 2 == 0 and len(coords) >= 6
            assert all(0.0 <= c <= 1.0 for c in coords)
    assert seen_lines == 2


# --------------------------- inference helpers --------------------------- #
def test_mask_area_centroid():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 4:8] = True          # 4x4 block, area 16, centroid at (5.5, 3.5)
    area, cx, cy = mask_area_centroid(mask)
    assert area == 16
    assert cx == pytest.approx(5.5) and cy == pytest.approx(3.5)
    # empty mask
    a, ex, ey = mask_area_centroid(np.zeros((5, 5)))
    assert a == 0 and np.isnan(ex) and np.isnan(ey)


def test_resize_mask_preserves_shape_and_content():
    small = np.array([[1, 0], [0, 1]], dtype=bool)
    big = _resize_mask(small, 4, 4)
    assert big.shape == (4, 4)
    assert big[0, 0] and big[-1, -1] and not big[0, -1]


# --------------------------- train config --------------------------- #
def test_build_train_config_defaults_and_validation():
    cfg = build_train_config("data.yaml", epochs=50, imgsz=640)
    assert cfg["data"] == "data.yaml"
    assert cfg["epochs"] == 50 and cfg["imgsz"] == 640
    assert cfg["exist_ok"] is True
    with pytest.raises(ValueError):
        build_train_config("d.yaml", epochs=0)
    with pytest.raises(ValueError):
        build_train_config("d.yaml", imgsz=100)   # not a multiple of 32
