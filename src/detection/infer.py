"""YOLO11-seg inference wrapper.

Runs a trained YOLO-seg model over images, turns each predicted instance mask
into a lightweight detection record (centroid + pixel area, optionally cm²), and
writes a detection CSV that the regression and dashboard modules consume.

The mask→geometry helpers are pure NumPy and unit-testable without a model.
The Ultralytics import is lazy (only ``predict_masks`` needs it).
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np


def mask_area_centroid(mask: np.ndarray) -> Tuple[float, float, float]:
    """Return ``(area_px, cx, cy)`` for a binary/boolean mask.

    Empty masks return ``(0, nan, nan)``.
    """
    m = np.asarray(mask) > 0
    area = float(m.sum())
    if area == 0:
        return 0.0, float("nan"), float("nan")
    ys, xs = np.nonzero(m)
    return area, float(xs.mean()), float(ys.mean())


def predict_masks(
    model_path: str,
    image_paths: List[str],
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "cpu",
) -> Iterator[Tuple[str, List[np.ndarray]]]:
    """Yield ``(image_path, [mask, ...])`` for each image.

    Each mask is a boolean array at the original image resolution. Requires
    Ultralytics + a trained/base seg checkpoint.
    """
    from ultralytics import YOLO  # lazy import (heavy)

    model = YOLO(model_path)
    for image_path in image_paths:
        results = model.predict(image_path, conf=conf, imgsz=imgsz,
                                device=device, verbose=False)
        masks: List[np.ndarray] = []
        for res in results:
            if res.masks is None:
                continue
            # res.masks.data: (n, H, W) float tensor in mask resolution;
            # res.orig_shape gives the original (H, W).
            data = res.masks.data.cpu().numpy()
            oh, ow = res.orig_shape
            for m in data:
                if m.shape != (oh, ow):
                    m = _resize_mask(m, oh, ow)
                masks.append(m > 0.5)
        yield image_path, masks


def _resize_mask(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbour resize of a 2D mask to (out_h, out_w) without cv2."""
    in_h, in_w = mask.shape
    ys = (np.linspace(0, in_h - 1, out_h)).round().astype(int)
    xs = (np.linspace(0, in_w - 1, out_w)).round().astype(int)
    return mask[np.ix_(ys, xs)]


def run_inference(
    model_path: str,
    images_dir: str,
    out_csv: str,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "cpu",
    gsd_cm_per_px: Optional[float] = None,
) -> dict:
    """Run inference over ``images_dir`` and write a detection CSV.

    Columns: ``krop_id, image, cx, cy, area_px[, area_cm2]``.
    """
    image_paths = sorted(
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    rows: List[dict] = []
    krop_id = 1
    for image_path, masks in predict_masks(model_path, image_paths, conf, imgsz, device):
        for m in masks:
            area_px, cx, cy = mask_area_centroid(m)
            if area_px == 0:
                continue
            row = {
                "krop_id": krop_id,
                "image": os.path.basename(image_path),
                "cx": round(cx, 2),
                "cy": round(cy, 2),
                "area_px": round(area_px, 2),
            }
            if gsd_cm_per_px is not None:
                row["area_cm2"] = round(area_px * gsd_cm_per_px ** 2, 3)
            rows.append(row)
            krop_id += 1

    _write_csv(out_csv, rows)
    return {"out_csv": out_csv, "num_detections": len(rows)}


def _write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = ["krop_id", "image", "cx", "cy", "area_px", "area_cm2"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run YOLO11-seg inference -> detection CSV.")
    p.add_argument("--model", default="runs/segment/krop_seg/weights/best.pt")
    p.add_argument("--images", default="data/synthetic/images")
    p.add_argument("--out", default="results/detections.csv")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gsd", type=float, default=None, help="cm/px for area_cm2")
    return p


def main(argv=None) -> dict:
    args = _build_arg_parser().parse_args(argv)
    summary = run_inference(args.model, args.images, args.out, args.conf,
                            args.imgsz, args.device, args.gsd)
    import json
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
