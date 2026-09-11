"""Schema-validation tests for the dashboard data loader (Module 6).

Covers the five detection CSVs we exercise the dashboard with:

  1. happy_full.csv        — every column present and valid                -> OK
  2. minimal_required.csv  — only the two required columns                -> OK
  3. no_coords.csv         — required + weight_g, no lat/lon               -> OK
  4. missing_column.csv    — required ``weight_class`` column absent       -> error
  5. edge_cases_fouten.csv — duplicate id, bad class, out-of-range weight,
                             NaN coordinate                               -> error

Before the pandera schema, cases 4 and 5 either raised an opaque crash further
downstream or silently dropped/ignored bad rows. They must now surface a clear
``DetectionValidationError`` from ``load_detections`` instead.
"""
import pandas as pd
import pytest

from src.dashboard.data import (
    DetectionValidationError,
    load_detections,
    validate_detections,
)


# --- CSV fixtures (written verbatim so the test data is easy to read) --------

HAPPY_FULL = """\
krop_id,image,cx,cy,area_px,area_cm2,weight_g,weight_class,lat,lon
1,orthomosaic_000.png,104.65,484.55,4100.44,256.28,249.02,light,51.2500035,5.9500038
2,orthomosaic_000.png,158.89,201.57,10810.93,675.68,564.49,heavy,51.2500098,5.9500057
3,orthomosaic_000.png,320.10,410.22,7200.10,410.00,430.00,medium,51.2500120,5.9500090
"""

MINIMAL_REQUIRED = """\
krop_id,weight_class
1,light
2,medium
3,heavy
"""

NO_COORDS = """\
krop_id,weight_g,weight_class
1,249.0,light
2,430.0,medium
3,564.0,heavy
"""

# Missing the required ``weight_class`` column entirely.
MISSING_COLUMN = """\
krop_id,cx,cy,area_cm2
1,104.65,484.55,256.28
2,158.89,201.57,675.68
"""

# Four distinct problems in one file:
#   - krop_id "1" appears twice (not unique)
#   - weight_class "giant" is not a valid class
#   - weight_g 5000 is outside [0, 2000]
#   - lat is NaN (blank) on the last row
EDGE_CASES_FOUTEN = """\
krop_id,weight_g,weight_class,lat,lon
1,249.0,light,51.25,5.95
1,430.0,medium,51.25,5.95
3,5000.0,giant,51.25,5.95
4,300.0,light,,5.95
"""


def _write(tmp_path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


# --- valid CSVs --------------------------------------------------------------

def test_happy_full_csv_loads(tmp_path):
    df = load_detections(_write(tmp_path, "happy_full.csv", HAPPY_FULL))
    assert len(df) == 3
    # krop_id is coerced to string by the schema
    assert df["krop_id"].tolist() == ["1", "2", "3"]
    assert set(df["weight_class"]) == {"light", "medium", "heavy"}


def test_minimal_required_csv_loads(tmp_path):
    df = load_detections(_write(tmp_path, "minimal_required.csv", MINIMAL_REQUIRED))
    assert len(df) == 3
    assert list(df.columns) == ["krop_id", "weight_class"]


def test_no_coords_csv_loads(tmp_path):
    df = load_detections(_write(tmp_path, "no_coords.csv", NO_COORDS))
    assert len(df) == 3
    assert "lat" not in df.columns and "lon" not in df.columns
    assert df["weight_g"].max() <= 2000


# --- invalid CSVs: must raise a clear error, not crash / lose data -----------

def test_missing_column_csv_raises(tmp_path):
    path = _write(tmp_path, "missing_column.csv", MISSING_COLUMN)
    with pytest.raises(DetectionValidationError) as exc:
        load_detections(path)
    assert "weight_class" in str(exc.value)


def test_edge_cases_fouten_csv_raises_clear_error(tmp_path):
    path = _write(tmp_path, "edge_cases_fouten.csv", EDGE_CASES_FOUTEN)
    with pytest.raises(DetectionValidationError) as exc:
        load_detections(path)
    msg = str(exc.value)
    # the message should name the offending columns rather than silently
    # dropping the bad rows
    assert "weight_class" in msg  # invalid class "giant"
    assert "weight_g" in msg      # out-of-range 5000
    assert ("krop_id" in msg or "lat" in msg)  # duplicate id and/or NaN coord


def test_validate_detections_accepts_dataframe():
    """Uploaded frames validate through the same path as file loads."""
    good = pd.DataFrame({"krop_id": [10, 11], "weight_class": ["light", "heavy"]})
    out = validate_detections(good)
    assert out["krop_id"].tolist() == ["10", "11"]

    bad = pd.DataFrame({"krop_id": [1, 1], "weight_class": ["light", "light"]})
    with pytest.raises(DetectionValidationError):
        validate_detections(bad)
