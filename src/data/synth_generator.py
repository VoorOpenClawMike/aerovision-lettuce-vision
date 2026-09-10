"""Synthetic "drone orthomosaic" generator for lettuce-head (krop) vision.

This module fabricates a stand-in dataset for the SOIL-Data / Aerovision project
while no real annotated, weighed drone imagery is available. It produces:

1. RGB PNG "orthomosaic" tiles with randomly placed blob-shaped lettuce heads
   (varying size, colour and overlap) on a textured soil background.
2. A COCO-format instance-segmentation annotation file (``instances.coco.json``)
   whose ``segmentation`` field holds one polygon per head.
3. A ground-truth CSV (``ground_truth_weights.csv``) with a synthetic fresh
   weight per head, derived from its canopy area through a documented,
   noise-perturbed allometric relationship, plus the weight class.

The synthetic ground truth is intentionally *generated from a known model* so
that the downstream regression and sampling modules can be validated
end-to-end. See the README and RESULTS.md for how to swap in real data.

Design choices (documented so they can be revisited with real data):
- Heads are drawn as radially-perturbed polygons ("blobs"), not perfect
  circles, so segmentation features (compactness, hull ratio) are non-trivial.
- A nominal ground sampling distance ``NOMINAL_GSD_CM_PER_PX`` ties pixel area
  to physical cm², so weights land in a realistic 150-1000 g range.
- Fresh weight follows ``weight_g = WEIGHT_COEF * area_cm2 ** WEIGHT_POWER``
  times log-normal multiplicative noise (canopy-to-weight scatter). The
  power is sub-linear because projected canopy area saturates as heads grow.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# --------------------------------------------------------------------------- #
# Documented synthetic constants                                              #
# --------------------------------------------------------------------------- #
NOMINAL_GSD_CM_PER_PX = 0.25  # cm per pixel; chosen so a ~25 cm head ~= 50 px radius
WEIGHT_COEF = 2.2             # allometric coefficient (g / cm^(2*power))
WEIGHT_POWER = 0.85           # sub-linear canopy-area -> weight exponent
WEIGHT_NOISE_SIGMA = 0.10     # log-normal multiplicative noise (sigma in log space)

# Weight-class thresholds (grams). Consistent across the whole pipeline.
CLASS_LIGHT_MAX = 350.0       # < 350 g  -> "light"
CLASS_HEAVY_MIN = 500.0       # > 500 g  -> "heavy"; between -> "medium"
CLASS_NAMES = ("light", "medium", "heavy")


def weight_class(weight_g: float) -> str:
    """Map a fresh weight (g) to one of ``light`` / ``medium`` / ``heavy``."""
    if weight_g < CLASS_LIGHT_MAX:
        return "light"
    if weight_g > CLASS_HEAVY_MIN:
        return "heavy"
    return "medium"


# --------------------------------------------------------------------------- #
# Geometry helpers                                                            #
# --------------------------------------------------------------------------- #
def polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    """Shoelace polygon area (always non-negative), in the units of ``points``."""
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def make_blob_polygon(
    cx: float,
    cy: float,
    base_radius: float,
    rng: np.random.Generator,
    n_points: int = 24,
    jitter: float = 0.22,
) -> List[Tuple[float, float]]:
    """Return a closed blob polygon: a circle with smoothly perturbed radii.

    ``jitter`` is the relative std-dev of the per-vertex radius. The radial
    perturbation is low-pass filtered around the ring so the blob stays convex-ish
    and lettuce-like rather than spiky.
    """
    angles = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)
    raw = rng.normal(1.0, jitter, size=n_points)
    # circular smoothing (3-tap) so neighbouring radii correlate
    smoothed = (np.roll(raw, 1) + raw + np.roll(raw, -1)) / 3.0
    radii = np.clip(smoothed, 0.55, 1.55) * base_radius
    pts = [(float(cx + r * math.cos(a)), float(cy + r * math.sin(a)))
           for r, a in zip(radii, angles)]
    return pts


def area_px_to_cm2(area_px: float, gsd_cm_per_px: float = NOMINAL_GSD_CM_PER_PX) -> float:
    """Convert a pixel area to cm² given a ground sampling distance."""
    return area_px * (gsd_cm_per_px ** 2)


def weight_from_area_cm2(
    area_cm2: float,
    rng: np.random.Generator,
    coef: float = WEIGHT_COEF,
    power: float = WEIGHT_POWER,
    noise_sigma: float = WEIGHT_NOISE_SIGMA,
) -> float:
    """Synthetic fresh weight (g) from canopy area with log-normal noise."""
    base = coef * (area_cm2 ** power)
    noise = math.exp(rng.normal(0.0, noise_sigma)) if noise_sigma > 0 else 1.0
    return max(1.0, base * noise)


# --------------------------------------------------------------------------- #
# Data model                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Krop:
    polygon: List[Tuple[float, float]]
    cx: float
    cy: float
    area_px: float
    color: Tuple[int, int, int]

    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        x0, y0 = min(xs), min(ys)
        return (x0, y0, max(xs) - x0, max(ys) - y0)


@dataclass
class SynthConfig:
    num_images: int = 6
    width: int = 640
    height: int = 640
    heads_per_image: int = 25
    min_radius: float = 28.0
    max_radius: float = 66.0
    seed: int = 42
    gsd_cm_per_px: float = NOMINAL_GSD_CM_PER_PX
    out_dir: str = "data/synthetic"


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def _soil_background(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    """A brown, faintly row-striped, noisy soil background."""
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[..., 0] = 122  # R
    base[..., 1] = 96   # G
    base[..., 2] = 70   # B
    # planting-row stripes (vertical), subtle
    xs = np.arange(width)
    stripe = 10.0 * np.sin(2.0 * math.pi * xs / 46.0)
    base += stripe[None, :, None]
    # speckle noise
    base += rng.normal(0.0, 9.0, size=base.shape)
    base = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(base, mode="RGB")


def _lettuce_color(rng: np.random.Generator) -> Tuple[int, int, int]:
    """A green-ish head colour with natural variation (some yellow/blue-green)."""
    g = int(rng.integers(120, 190))
    r = int(np.clip(g - rng.integers(30, 80), 30, 200))
    b = int(np.clip(g - rng.integers(50, 110), 20, 160))
    return (r, g, b)


def render_image(cfg: SynthConfig, rng: np.random.Generator) -> Tuple[Image.Image, List[Krop]]:
    """Render one orthomosaic tile and return it with its head annotations."""
    img = _soil_background(cfg.width, cfg.height, rng)
    draw = ImageDraw.Draw(img)
    kroppen: List[Krop] = []

    for _ in range(cfg.heads_per_image):
        radius = float(rng.uniform(cfg.min_radius, cfg.max_radius))
        cx = float(rng.uniform(radius, cfg.width - radius))
        cy = float(rng.uniform(radius, cfg.height - radius))
        poly = make_blob_polygon(cx, cy, radius, rng)
        color = _lettuce_color(rng)

        draw.polygon(poly, fill=color, outline=tuple(max(0, c - 40) for c in color))
        # simple leafy highlight: a smaller, lighter blob near the centre
        hi = make_blob_polygon(cx, cy, radius * 0.5, rng, jitter=0.30)
        hi_color = tuple(min(255, c + 30) for c in color)
        draw.polygon(hi, fill=hi_color)

        area = polygon_area(poly)
        kroppen.append(Krop(polygon=poly, cx=cx, cy=cy, area_px=area, color=color))

    # gentle blur to soften polygon edges -> more photo-like
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return img, kroppen


# --------------------------------------------------------------------------- #
# COCO + CSV assembly                                                         #
# --------------------------------------------------------------------------- #
def _flatten_polygon(poly: Sequence[Tuple[float, float]]) -> List[float]:
    out: List[float] = []
    for x, y in poly:
        out.extend([round(float(x), 2), round(float(y), 2)])
    return out


def generate_dataset(cfg: SynthConfig) -> dict:
    """Generate the full synthetic dataset and write images, COCO json and CSV.

    Returns a summary dict with output paths and instance counts.
    """
    rng = np.random.default_rng(cfg.seed)
    images_dir = os.path.join(cfg.out_dir, "images")
    ann_dir = os.path.join(cfg.out_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    coco = {
        "info": {
            "description": "Synthetic drone orthomosaic lettuce heads (SOIL-Data / Aerovision)",
            "version": "1.0",
            "synthetic": True,
            "gsd_cm_per_px": cfg.gsd_cm_per_px,
        },
        "licenses": [],
        "images": [],
        "categories": [{"id": 1, "name": "krop", "supercategory": "lettuce"}],
        "annotations": [],
    }
    csv_rows: List[dict] = []
    ann_id = 1

    for image_id in range(cfg.num_images):
        img, kroppen = render_image(cfg, rng)
        file_name = f"orthomosaic_{image_id:03d}.png"
        img.save(os.path.join(images_dir, file_name))
        coco["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": cfg.width,
            "height": cfg.height,
        })

        for krop in kroppen:
            area_cm2 = area_px_to_cm2(krop.area_px, cfg.gsd_cm_per_px)
            weight_g = weight_from_area_cm2(area_cm2, rng)
            cls = weight_class(weight_g)
            bx, by, bw, bh = krop.bbox()

            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [_flatten_polygon(krop.polygon)],
                "area": round(krop.area_px, 2),
                "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                "iscrowd": 0,
            })
            csv_rows.append({
                "krop_id": ann_id,
                "image_id": image_id,
                "image": file_name,
                "cx": round(krop.cx, 2),
                "cy": round(krop.cy, 2),
                "area_px": round(krop.area_px, 2),
                "area_cm2": round(area_cm2, 3),
                "weight_g": round(weight_g, 2),
                "weight_class": cls,
            })
            ann_id += 1

    coco_path = os.path.join(ann_dir, "instances.coco.json")
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    csv_path = os.path.join(cfg.out_dir, "ground_truth_weights.csv")
    _write_csv(csv_path, csv_rows)

    return {
        "images_dir": images_dir,
        "coco_path": coco_path,
        "csv_path": csv_path,
        "num_images": cfg.num_images,
        "num_instances": len(csv_rows),
        "class_counts": {c: sum(1 for r in csv_rows if r["weight_class"] == c)
                         for c in CLASS_NAMES},
    }


def _write_csv(path: str, rows: List[dict]) -> None:
    import csv
    if not rows:
        # still write a header so downstream tools don't choke
        rows_fields = ["krop_id", "image_id", "image", "cx", "cy",
                       "area_px", "area_cm2", "weight_g", "weight_class"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(rows_fields)
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic lettuce drone dataset.")
    p.add_argument("--out", default="data/synthetic", help="output directory")
    p.add_argument("--num-images", type=int, default=6)
    p.add_argument("--heads-per-image", type=int, default=25)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gsd", type=float, default=NOMINAL_GSD_CM_PER_PX,
                   help="nominal ground sampling distance (cm/px)")
    return p


def main(argv: Sequence[str] | None = None) -> dict:
    args = _build_arg_parser().parse_args(argv)
    cfg = SynthConfig(
        num_images=args.num_images,
        width=args.width,
        height=args.height,
        heads_per_image=args.heads_per_image,
        seed=args.seed,
        gsd_cm_per_px=args.gsd,
        out_dir=args.out,
    )
    summary = generate_dataset(cfg)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
