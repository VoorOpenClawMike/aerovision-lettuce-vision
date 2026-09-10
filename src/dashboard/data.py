"""Pure, testable data helpers for the Streamlit dashboard (Module 6).

Kept free of Streamlit so they can be unit-tested. The color scheme follows the
project "Bed & Bracket" accents:

    light  (< 350 g)     -> amber
    medium (350-500 g)   -> green
    heavy  (> 500 g)     -> rust red
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.data.synth_generator import CLASS_NAMES

# "Bed & Bracket" accent colours (hex).
CLASS_COLORS: Dict[str, str] = {
    "light": "#E0A81E",   # amber
    "medium": "#5C8A3A",  # green
    "heavy": "#A6431E",   # rust red
}
CLASS_LABELS: Dict[str, str] = {
    "light": "< 350 g",
    "medium": "350–500 g",
    "heavy": "> 500 g",
}

REQUIRED_COLUMNS = ["krop_id", "weight_class"]


def load_detections(csv_path: str) -> pd.DataFrame:
    """Load a detection CSV and validate the required columns are present."""
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"detection CSV missing columns: {missing}")
    return df


def class_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Count rows per weight class, always returning all three classes."""
    counts = {c: 0 for c in CLASS_NAMES}
    for cls, n in df["weight_class"].value_counts().items():
        if cls in counts:
            counts[cls] = int(n)
    return counts


def class_color(cls: str) -> str:
    """Return the hex colour for a weight class (grey for unknown)."""
    return CLASS_COLORS.get(cls, "#808080")


def summary_metrics(df: pd.DataFrame) -> Dict[str, object]:
    """High-level KPIs for the header row."""
    counts = class_counts(df)
    total = int(len(df))
    metrics: Dict[str, object] = {"total": total, "counts": counts}
    if "weight_g" in df.columns and total:
        metrics["mean_weight_g"] = round(float(df["weight_g"].mean()), 1)
        metrics["total_yield_kg"] = round(float(df["weight_g"].sum()) / 1000.0, 2)
    return metrics


def histogram_bins(df: pd.DataFrame, bins: int = 20) -> Dict[str, List[float]]:
    """Weight histogram data (counts + bin edges) for plotting."""
    if "weight_g" not in df.columns or df.empty:
        return {"counts": [], "edges": []}
    import numpy as np
    counts, edges = np.histogram(df["weight_g"].to_numpy(), bins=bins)
    return {"counts": counts.tolist(), "edges": edges.tolist()}


def build_field_map(df: pd.DataFrame, zoom_start: int = 18):
    """Build a folium map with class-coloured markers (requires lat/lon)."""
    import folium
    if "lat" not in df.columns or "lon" not in df.columns or df.empty:
        # centre on the synthetic field reference if no coordinates
        return folium.Map(location=[51.25, 5.95], zoom_start=zoom_start)

    center = [float(df["lat"].mean()), float(df["lon"].mean())]
    fmap = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")
    for _, row in df.iterrows():
        cls = row["weight_class"]
        popup = f"krop {row['krop_id']} — {CLASS_LABELS.get(cls, cls)}"
        if "weight_g" in df.columns:
            popup += f" ({row['weight_g']:.0f} g)"
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=5,
            color=class_color(cls),
            fill=True,
            fill_color=class_color(cls),
            fill_opacity=0.85,
            popup=popup,
        ).add_to(fmap)
    return fmap
