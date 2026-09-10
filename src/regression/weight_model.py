"""Weight regression from segmentation-mask shape features (Module 4).

Pipeline: segmentation polygon/mask -> shape features -> Ridge regression ->
estimated fresh weight (g) -> weight class (<350 / 350-500 / >500 g).

Features (all derived from the instance geometry):
  * ``area_cm2``      : canopy area in cm² (needs a GSD; see calibration module).
  * ``perimeter_cm``  : polygon perimeter in cm.
  * ``compactness``   : 4π·area / perimeter²  (1.0 for a circle, lower for ragged).
  * ``hull_ratio``    : area / convex-hull-area (1.0 = convex, <1 = concave/lobed).
  * ``eq_diameter_cm``: diameter of the area-equivalent circle.

The model is trained on the synthetic ground truth (module 1). With real weighed
data, retrain on the same feature schema — nothing downstream changes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.data.synth_generator import CLASS_HEAVY_MIN, CLASS_LIGHT_MAX, weight_class

FEATURE_NAMES = ["area_cm2", "perimeter_cm", "compactness", "hull_ratio", "eq_diameter_cm"]


# --------------------------------------------------------------------------- #
# Geometry / feature extraction                                               #
# --------------------------------------------------------------------------- #
def _polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _polygon_perimeter(pts: np.ndarray) -> float:
    d = pts - np.roll(pts, -1, axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


def _convex_hull_area(pts: np.ndarray) -> float:
    """Convex-hull area; falls back to polygon area if scipy/hull unavailable."""
    if len(pts) < 3:
        return _polygon_area(pts)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        return float(hull.volume)  # 'volume' is area in 2D
    except Exception:
        return _polygon_area(pts)


def features_from_polygon(
    flat_or_pairs: Sequence[float] | Sequence[Tuple[float, float]],
    gsd_cm_per_px: float,
) -> Dict[str, float]:
    """Compute shape features from a polygon (flat ``[x1,y1,...]`` or pairs)."""
    arr = np.asarray(flat_or_pairs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if len(arr) < 3:
        raise ValueError("polygon needs >= 3 points")

    area_px = _polygon_area(arr)
    perim_px = _polygon_perimeter(arr)
    hull_px = _convex_hull_area(arr)

    area_cm2 = area_px * gsd_cm_per_px ** 2
    perimeter_cm = perim_px * gsd_cm_per_px
    compactness = (4 * math.pi * area_px / perim_px ** 2) if perim_px > 0 else 0.0
    hull_ratio = (area_px / hull_px) if hull_px > 0 else 1.0
    eq_diameter_cm = 2.0 * math.sqrt(area_cm2 / math.pi) if area_cm2 > 0 else 0.0

    return {
        "area_cm2": area_cm2,
        "perimeter_cm": perimeter_cm,
        "compactness": min(compactness, 1.0),
        "hull_ratio": min(hull_ratio, 1.0),
        "eq_diameter_cm": eq_diameter_cm,
    }


def features_from_mask(mask: np.ndarray, gsd_cm_per_px: float) -> Dict[str, float]:
    """Compute features from a binary mask by extracting its largest contour."""
    import cv2
    m = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("empty mask")
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return features_from_polygon(cnt, gsd_cm_per_px)


def feature_vector(feats: Dict[str, float]) -> List[float]:
    """Order a feature dict into the canonical feature vector."""
    return [feats[name] for name in FEATURE_NAMES]


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class RegressionReport:
    r2: float
    mae: float
    n_train: int
    n_test: int
    class_accuracy: float
    class_counts: Dict[str, int]


class WeightRegressor:
    """Ridge regression on standardised shape features -> fresh weight (g)."""

    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        self.alpha = alpha
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeightRegressor":
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("model not fitted")
        return np.clip(self.model.predict(X), 1.0, None)

    def predict_classes(self, X: np.ndarray) -> List[str]:
        return [weight_class(w) for w in self.predict(X)]

    def save(self, path: str) -> None:
        import pickle
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str) -> "WeightRegressor":
        import pickle
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.model = pickle.load(f)
        obj._fitted = True
        return obj


# --------------------------------------------------------------------------- #
# Training on the synthetic dataset                                           #
# --------------------------------------------------------------------------- #
def build_feature_table(
    coco_path: str,
    weights_csv: str,
    gsd_cm_per_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) from COCO polygons joined to the ground-truth weight CSV.

    Features come from the annotation polygons; targets from the weight CSV,
    joined on annotation id == krop_id.
    """
    import csv
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    weights: Dict[int, float] = {}
    with open(weights_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            weights[int(row["krop_id"])] = float(row["weight_g"])

    X: List[List[float]] = []
    y: List[float] = []
    for ann in coco["annotations"]:
        if ann["id"] not in weights:
            continue
        seg = ann["segmentation"][0]
        feats = features_from_polygon(seg, gsd_cm_per_px)
        X.append(feature_vector(feats))
        y.append(weights[ann["id"]])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def train_and_evaluate(
    coco_path: str,
    weights_csv: str,
    gsd_cm_per_px: float,
    alpha: float = 1.0,
    test_fraction: float = 0.25,
    seed: int = 0,
    model_out: str | None = None,
) -> Tuple[WeightRegressor, RegressionReport]:
    """Train the Ridge model and return it plus an evaluation report."""
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    X, y = build_feature_table(coco_path, weights_csv, gsd_cm_per_px)
    if len(X) < 4:
        raise ValueError("not enough samples to train/evaluate")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_fraction, random_state=seed)
    reg = WeightRegressor(alpha=alpha).fit(X_tr, y_tr)

    y_pred = reg.predict(X_te)
    r2 = float(r2_score(y_te, y_pred))
    mae = float(mean_absolute_error(y_te, y_pred))

    true_cls = [weight_class(w) for w in y_te]
    pred_cls = [weight_class(w) for w in y_pred]
    class_acc = float(np.mean([t == p for t, p in zip(true_cls, pred_cls)]))
    counts: Dict[str, int] = {}
    for c in pred_cls:
        counts[c] = counts.get(c, 0) + 1

    report = RegressionReport(
        r2=r2, mae=mae, n_train=len(X_tr), n_test=len(X_te),
        class_accuracy=class_acc, class_counts=counts,
    )
    if model_out:
        reg.save(model_out)
    return reg, report


def write_report_md(report: RegressionReport, path: str, gsd_cm_per_px: float) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = [
        "# Regressie-rapport — versgewichtschatting",
        "",
        "Ridge-regressie op vorm-features uit segmentatiemaskers → versgewicht (g).",
        "**Getraind op synthetische data** (module 1); hertrain op echte gewogen data.",
        "",
        f"- GSD gebruikt: **{gsd_cm_per_px} cm/px**",
        f"- Train/test split: **{report.n_train}** / **{report.n_test}**",
        "",
        "## Metrics (testset)",
        "",
        f"| Metric | Waarde |",
        f"|--------|--------|",
        f"| R² | {report.r2:.3f} |",
        f"| MAE (g) | {report.mae:.1f} |",
        f"| Klasse-accuraatheid (<350 / 350-500 / >500 g) | {report.class_accuracy:.1%} |",
        "",
        "## Voorspelde klasse-verdeling (testset)",
        "",
        f"| Klasse | Aantal |",
        f"|--------|--------|",
    ]
    for cls in ("light", "medium", "heavy"):
        lines.append(f"| {cls} | {report.class_counts.get(cls, 0)} |")
    lines += [
        "",
        f"Klassegrenzen: `light < {CLASS_LIGHT_MAX:.0f} g`, "
        f"`medium {CLASS_LIGHT_MAX:.0f}-{CLASS_HEAVY_MIN:.0f} g`, "
        f"`heavy > {CLASS_HEAVY_MIN:.0f} g`.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/evaluate the weight regressor.")
    p.add_argument("--coco", default="data/synthetic/annotations/instances.coco.json")
    p.add_argument("--weights-csv", default="data/synthetic/ground_truth_weights.csv")
    p.add_argument("--gsd", type=float, default=0.25)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--report", default="results/regression_report.md")
    p.add_argument("--model-out", default="results/weight_model.pkl")
    return p


def main(argv=None) -> RegressionReport:
    args = _build_arg_parser().parse_args(argv)
    _, report = train_and_evaluate(
        args.coco, args.weights_csv, args.gsd, alpha=args.alpha,
        model_out=args.model_out)
    write_report_md(report, args.report, args.gsd)
    print(json.dumps(report.__dict__, indent=2))
    print("Report written to", args.report)
    return report


if __name__ == "__main__":
    main()
