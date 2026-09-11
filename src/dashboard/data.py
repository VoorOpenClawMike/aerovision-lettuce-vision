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
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema
from pandera.errors import SchemaError, SchemaErrors

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

# Valid weight classes (kept in sync with the model / synth generator).
VALID_CLASSES = list(CLASS_NAMES)


class DetectionValidationError(ValueError):
    """Raised when a detection CSV fails schema validation.

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers keep
    working; carries a human-readable, multi-line message suitable for showing
    directly to a user (e.g. via ``st.error``).
    """


# pandera schema for a detection CSV. ``strict=False`` so bookkeeping columns
# the dashboard ignores (e.g. ``image``, ``area_px``) pass through untouched.
# Required columns must be present; optional columns are only validated when the
# CSV actually contains them.
DETECTION_SCHEMA = DataFrameSchema(
    {
        # Unique identifier per lettuce head. Coerced to string so a CSV with
        # integer ids still validates; duplicates are rejected.
        "krop_id": Column(str, unique=True, coerce=True, nullable=False),
        "weight_class": Column(
            str,
            Check.isin(VALID_CLASSES),
            coerce=True,
            nullable=False,
        ),
        # Optional numeric columns with sane physical ranges.
        "cx": Column(float, Check.ge(0), required=False, coerce=True, nullable=False),
        "cy": Column(float, Check.ge(0), required=False, coerce=True, nullable=False),
        "area_cm2": Column(
            float, Check.gt(0), required=False, coerce=True, nullable=False
        ),
        "weight_g": Column(
            float,
            Check.in_range(0, 2000),
            required=False,
            coerce=True,
            nullable=False,
        ),
        # When lat/lon are present they must be real, non-NaN coordinates.
        "lat": Column(
            float,
            Check.in_range(-90, 90),
            required=False,
            coerce=True,
            nullable=False,
        ),
        "lon": Column(
            float,
            Check.in_range(-180, 180),
            required=False,
            coerce=True,
            nullable=False,
        ),
    },
    strict=False,
    coerce=True,
)


def _format_schema_errors(exc: SchemaError | SchemaErrors) -> str:
    """Turn a pandera error into a short, readable, user-facing message."""
    cases = getattr(exc, "failure_cases", None)
    if cases is None or getattr(cases, "empty", True):
        return f"Ongeldige detectie-CSV: {exc}"

    lines: List[str] = []
    for _, row in cases.head(10).iterrows():
        column = row.get("column")
        check = row.get("check")
        value = row.get("failure_case")
        index = row.get("index")
        where = f" (rij {index})" if index is not None and pd.notna(index) else ""
        col = f"kolom '{column}'" if column is not None and pd.notna(column) else "schema"
        lines.append(f"  • {col}: {check} — waarde: {value!r}{where}")

    n = len(cases)
    more = f"\n  … en nog {n - 10} probleem(en)" if n > 10 else ""
    return (
        f"Ongeldige detectie-CSV — {n} probleem(en) gevonden:\n"
        + "\n".join(lines)
        + more
    )


def validate_detections(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a detection DataFrame against ``DETECTION_SCHEMA``.

    Returns the validated (coerced) frame on success. On any schema violation
    raises :class:`DetectionValidationError` with a clear, multi-line message
    instead of letting data be silently dropped or the app crash.
    """
    try:
        return DETECTION_SCHEMA.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise DetectionValidationError(_format_schema_errors(exc)) from exc


def load_detections(csv_path: str) -> pd.DataFrame:
    """Load a detection CSV and validate it against the pandera schema."""
    df = pd.read_csv(csv_path)
    return validate_detections(df)


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
