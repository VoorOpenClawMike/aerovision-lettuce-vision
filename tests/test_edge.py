"""Unit tests for the edge export module (Module 7)."""
import os

import pytest

from src.edge.export_tensorrt import (
    TENSORRT_INSTRUCTIONS,
    export_onnx,
    write_tensorrt_readme,
)


def test_export_onnx_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        export_onnx("does/not/exist_best.pt")


def test_tensorrt_instructions_mention_jetson_and_int8():
    text = TENSORRT_INSTRUCTIONS.lower()
    assert "jetson" in text
    assert "int8" in text
    assert "jetpack" in text
    # must warn that synthetic data is not representative for calibration
    assert "synthetic" in text


def test_write_tensorrt_readme(tmp_path):
    path = str(tmp_path / "edge" / "TENSORRT.md")
    written = write_tensorrt_readme(path)
    assert os.path.exists(written)
    content = open(written).read()
    assert "TensorRT INT8" in content
    assert "trtexec" in content


def test_export_onnx_from_trained_model_if_available():
    """If a trained checkpoint exists, ONNX export should succeed."""
    import glob
    candidates = glob.glob("runs/**/weights/best.pt", recursive=True)
    if not candidates:
        pytest.skip("no trained model available in this environment")
    out = export_onnx(candidates[0], imgsz=640)
    assert out.endswith(".onnx") and os.path.exists(out)
