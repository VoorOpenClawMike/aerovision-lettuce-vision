"""Streamlit dashboard for the krop-sla weight-class field report (Module 6).

Loads a detection CSV (``krop_id, cx, cy, area_cm2, weight_class[, weight_g,
lat, lon]``) and shows:
  (a) counts per weight class,
  (b) a colour-coded field map (folium): amber <350 g, green 350-500 g,
      rust-red >500 g (Bed & Bracket scheme),
  (c) a weight histogram, optionally split per class.

Run:
    streamlit run src/dashboard/app.py
By default it loads ``results/field_detections.csv`` (produced by src.pipeline);
you can also upload a CSV in the sidebar.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import os

import pandas as pd
import streamlit as st

from src.dashboard.data import (
    CLASS_COLORS,
    CLASS_LABELS,
    build_field_map,
    class_counts,
    histogram_bins,
    load_detections,
    summary_metrics,
)

DEFAULT_CSV = "results/field_detections.csv"


def _read_source() -> pd.DataFrame | None:
    st.sidebar.header("Databron")
    uploaded = st.sidebar.file_uploader("Upload detectie-CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.sidebar.success("CSV geladen (upload).")
        return df
    if os.path.exists(DEFAULT_CSV):
        st.sidebar.info(f"Standaardbestand: `{DEFAULT_CSV}`")
        return load_detections(DEFAULT_CSV)
    st.sidebar.warning(
        f"Geen CSV gevonden op `{DEFAULT_CSV}`.\n\n"
        "Genereer eerst data en draai de pijplijn:\n"
        "`python -m src.data.synth_generator && python -m src.pipeline`")
    return None


def main() -> None:
    st.set_page_config(page_title="Kroppen sla — veldrapport", layout="wide")
    st.title("🥬 Kroppen sla — gewichtsklasse-veldrapport")
    st.caption("Aerovision / SOIL-Data · drone-vision pijplijn (demo op synthetische data)")

    df = _read_source()
    if df is None or df.empty:
        st.stop()

    metrics = summary_metrics(df)
    counts = metrics["counts"]

    # (a) counts per class
    st.subheader("Aantallen per gewichtsklasse")
    cols = st.columns(len(CLASS_LABELS) + 1)
    cols[0].metric("Totaal kroppen", metrics["total"])
    for col, cls in zip(cols[1:], CLASS_LABELS):
        col.metric(f"{cls} ({CLASS_LABELS[cls]})", counts.get(cls, 0))
    if "mean_weight_g" in metrics:
        st.write(
            f"Gemiddeld gewicht: **{metrics['mean_weight_g']} g** · "
            f"Geschatte totale opbrengst: **{metrics['total_yield_kg']} kg**")

    left, right = st.columns([3, 2])

    # (b) colour-coded field map
    with left:
        st.subheader("Veldkaart (kleur = gewichtsklasse)")
        legend = " · ".join(
            f"{CLASS_LABELS[c]}: {c}" for c in CLASS_LABELS)
        st.caption(f"Amber {CLASS_LABELS['light']} · Groen "
                   f"{CLASS_LABELS['medium']} · Roestrood {CLASS_LABELS['heavy']}")
        try:
            from streamlit_folium import st_folium
            fmap = build_field_map(df)
            st_folium(fmap, width=700, height=500, returned_objects=[])
        except Exception as exc:  # pragma: no cover - UI fallback
            st.warning(f"Kaart kon niet worden geladen ({exc}). "
                       "Toon coördinaten als tabel:")
            if {"lat", "lon"}.issubset(df.columns):
                st.map(df.rename(columns={"lat": "latitude", "lon": "longitude"}))

    # (c) histogram per class
    with right:
        st.subheader("Gewichtsverdeling")
        if "weight_g" in df.columns:
            st.bar_chart(
                _histogram_frame(df),
                color=[CLASS_COLORS[c] for c in CLASS_LABELS],
            )
        else:
            st.info("Geen `weight_g`-kolom; histogram niet beschikbaar.")

        st.subheader("Klasse-aandeel")
        share_df = pd.DataFrame(
            {"aantal": [counts[c] for c in CLASS_LABELS]},
            index=list(CLASS_LABELS.keys()))
        st.bar_chart(share_df)


def _histogram_frame(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """Build a per-class stacked histogram frame indexed by weight-bin centre."""
    import numpy as np
    hb = histogram_bins(df, bins=bins)
    edges = np.asarray(hb["edges"])
    if edges.size == 0:
        return pd.DataFrame()
    centers = (edges[:-1] + edges[1:]) / 2.0
    out = {}
    for cls in CLASS_LABELS:
        sub = df.loc[df["weight_class"] == cls, "weight_g"].to_numpy()
        c, _ = np.histogram(sub, bins=edges)
        out[cls] = c
    return pd.DataFrame(out, index=np.round(centers).astype(int))


if __name__ == "__main__":
    main()
