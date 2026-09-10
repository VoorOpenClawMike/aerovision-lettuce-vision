"""End-to-end pipeline orchestrator: detections -> features -> weight -> CSV.

Produces ``results/field_detections.csv`` — the input the Streamlit dashboard
consumes — with columns::

    krop_id, image, cx, cy, area_px, area_cm2, weight_g, weight_class, lat, lon

Detection source:
  * If a trained YOLO model is available *and* produces detections, those masks
    are used (the real inference path).
  * Otherwise the pipeline falls back to the **ground-truth COCO polygons** as
    stand-in "detections" so the downstream demo (regression + dashboard +
    sampling) always has data. This fallback is clearly logged; with a properly
    trained detector on real imagery it is not used.

Geo-referencing is **synthetic**: the images have no real world coordinates, so
we place them on a grid near Midden-Limburg (≈51.25 N, 5.95 E) and map pixel
offsets to degrees via the GSD. Real orthomosaics carry world coordinates
(GeoTIFF) — replace :func:`pixel_to_latlon` accordingly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.regression.weight_model import (
    feature_vector,
    features_from_mask,
    features_from_polygon,
    train_and_evaluate,
)
from src.data.synth_generator import weight_class

# Synthetic field reference point (Midden-Limburg) and tile spacing.
FIELD_LAT = 51.25
FIELD_LON = 5.95
TILE_SPACING_M = 30.0     # synthetic spacing between image tiles on the map


def pixel_to_latlon(
    cx: float, cy: float, tile_row: int, tile_col: int,
    gsd_cm_per_px: float, image_h_px: int,
) -> Tuple[float, float]:
    """Map a pixel position (+ tile position) to a synthetic lat/lon."""
    # metres east/north of the field origin
    east_m = tile_col * TILE_SPACING_M + cx * gsd_cm_per_px / 100.0
    # image y grows downward; flip so north is up
    north_m = tile_row * TILE_SPACING_M + (image_h_px - cy) * gsd_cm_per_px / 100.0
    dlat = north_m / 111_320.0
    dlon = east_m / (111_320.0 * math.cos(math.radians(FIELD_LAT)))
    return FIELD_LAT + dlat, FIELD_LON + dlon


def _detections_from_coco(coco_path: str) -> Dict[str, List[dict]]:
    """Ground-truth 'detections': one polygon record per annotation, by image."""
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    id_to_name = {im["id"]: im["file_name"] for im in coco["images"]}
    id_to_hw = {im["id"]: (im["height"], im["width"]) for im in coco["images"]}
    by_image: Dict[str, List[dict]] = {}
    for ann in coco["annotations"]:
        name = id_to_name[ann["image_id"]]
        seg = ann["segmentation"][0]
        xs = seg[0::2]
        ys = seg[1::2]
        by_image.setdefault(name, []).append({
            "polygon": seg,
            "cx": float(np.mean(xs)),
            "cy": float(np.mean(ys)),
            "image_hw": id_to_hw[ann["image_id"]],
        })
    return by_image


def _try_yolo_detections(
    model_path: str, images_dir: str, conf: float, imgsz: int
) -> Optional[Dict[str, List[dict]]]:
    """Return YOLO mask detections by image, or None if unavailable/empty."""
    if not model_path or not os.path.exists(model_path):
        return None
    try:
        from src.detection.infer import predict_masks
        image_paths = sorted(
            os.path.join(images_dir, f) for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        by_image: Dict[str, List[dict]] = {}
        total = 0
        for image_path, masks in predict_masks(model_path, image_paths, conf, imgsz):
            recs = []
            for m in masks:
                ys, xs = np.nonzero(m)
                if xs.size == 0:
                    continue
                recs.append({"mask": m, "cx": float(xs.mean()),
                             "cy": float(ys.mean()), "image_hw": m.shape})
                total += 1
            by_image[os.path.basename(image_path)] = recs
        return by_image if total > 0 else None
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[pipeline] YOLO inference unavailable ({exc}); using fallback.")
        return None


def run_pipeline(
    coco_path: str,
    weights_csv: str,
    images_dir: str,
    out_csv: str,
    gsd_cm_per_px: float = 0.25,
    model_path: Optional[str] = None,
    conf: float = 0.25,
    imgsz: int = 640,
) -> dict:
    """Run the full pipeline and write the field-detection CSV."""
    # 1) train regression model on the synthetic ground truth
    reg, _ = train_and_evaluate(coco_path, weights_csv, gsd_cm_per_px)

    # 2) obtain detections (YOLO if it fires, else ground-truth polygons)
    detections = _try_yolo_detections(model_path, images_dir, conf, imgsz)
    source = "yolo"
    if detections is None:
        detections = _detections_from_coco(coco_path)
        source = "ground_truth_fallback"
        print("[pipeline] Using ground-truth COCO polygons as detection "
              "stand-in (no trained-model detections).")

    # 3) features -> weight -> class, with synthetic geo-referencing
    image_names = sorted(detections.keys())
    n_cols = max(1, int(math.ceil(math.sqrt(len(image_names)))))
    rows: List[dict] = []
    krop_id = 1
    for i, name in enumerate(image_names):
        tile_row, tile_col = divmod(i, n_cols)
        for det in detections[name]:
            if "mask" in det:
                feats = features_from_mask(det["mask"], gsd_cm_per_px)
            else:
                feats = features_from_polygon(det["polygon"], gsd_cm_per_px)
            weight = float(reg.predict(np.asarray([feature_vector(feats)]))[0])
            cls = weight_class(weight)
            image_h = det["image_hw"][0]
            lat, lon = pixel_to_latlon(det["cx"], det["cy"], tile_row, tile_col,
                                       gsd_cm_per_px, image_h)
            rows.append({
                "krop_id": krop_id,
                "image": name,
                "cx": round(det["cx"], 2),
                "cy": round(det["cy"], 2),
                "area_px": round(feats["area_cm2"] / gsd_cm_per_px ** 2, 2),
                "area_cm2": round(feats["area_cm2"], 2),
                "weight_g": round(weight, 2),
                "weight_class": cls,
                "lat": round(lat, 7),
                "lon": round(lon, 7),
            })
            krop_id += 1

    _write_csv(out_csv, rows)
    counts = {c: sum(1 for r in rows if r["weight_class"] == c)
              for c in ("light", "medium", "heavy")}
    return {"out_csv": out_csv, "detection_source": source,
            "num_detections": len(rows), "class_counts": counts}


def _write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = ["krop_id", "image", "cx", "cy", "area_px", "area_cm2",
              "weight_g", "weight_class", "lat", "lon"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the end-to-end krop pipeline.")
    p.add_argument("--coco", default="data/synthetic/annotations/instances.coco.json")
    p.add_argument("--weights-csv", default="data/synthetic/ground_truth_weights.csv")
    p.add_argument("--images", default="data/synthetic/images")
    p.add_argument("--out", default="results/field_detections.csv")
    p.add_argument("--gsd", type=float, default=0.25)
    p.add_argument("--model", default=None, help="trained YOLO best.pt (optional)")
    p.add_argument("--conf", type=float, default=0.25)
    return p


def main(argv=None) -> dict:
    args = _build_arg_parser().parse_args(argv)
    summary = run_pipeline(args.coco, args.weights_csv, args.images, args.out,
                           args.gsd, args.model, args.conf)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
