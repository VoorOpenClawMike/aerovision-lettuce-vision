"""Edge export: YOLO11-seg -> ONNX, with a documented path to TensorRT INT8.

This container can export to **ONNX** (portable, CPU/GPU). The final
**TensorRT INT8** engine can only be built *on the target Jetson device* with a
matching JetPack/TensorRT — see :data:`TENSORRT_INSTRUCTIONS` and the README.

Usage:
    python -m src.edge.export_tensorrt --model runs/segment/.../best.pt --imgsz 640
"""
from __future__ import annotations

import argparse
import os

DEFAULT_IMGSZ = 640

# Step-by-step for the Jetson-only TensorRT INT8 conversion (NOT runnable here).
TENSORRT_INSTRUCTIONS = """\
TensorRT INT8 conversion — RUN ON THE JETSON, NOT IN THIS CONTAINER
===================================================================
Prerequisites (on the Jetson, e.g. Orin Nano/NX with JetPack 6.x):
  * JetPack with CUDA, cuDNN and TensorRT installed (`dpkg -l | grep nvinfer`).
  * `pip install ultralytics` (matching the training version) + onnx.
  * A small representative calibration set (~100-500 real orthomosaic crops) —
    INT8 needs real data to calibrate activation ranges; synthetic data will
    NOT give production-representative quantisation.

Option A — via Ultralytics (recommended, one call):
    from ultralytics import YOLO
    model = YOLO("best.pt")
    model.export(
        format="engine",      # TensorRT engine
        int8=True,            # INT8 quantisation
        data="data.yaml",     # calibration data (REAL crops)
        imgsz=640,
        device=0,             # the Jetson GPU
        workspace=4,          # GB
    )
    # -> best.engine

Option B — explicit ONNX -> trtexec:
    yolo export model=best.pt format=onnx opset=12 imgsz=640
    /usr/src/tensorrt/bin/trtexec \\
        --onnx=best.onnx --int8 --saveEngine=best.engine \\
        --calib=<calibration_cache> --workspace=4096

Validate after conversion:
  * Compare mAP of the INT8 engine vs the FP32/ONNX model on a held-out set;
    accept only if the mAP drop is within tolerance (typ. < 1-2 %).
  * Benchmark latency/throughput with `trtexec --loadEngine=best.engine`.

TODO (blocked here): building/validating the .engine requires the Jetson +
JetPack + real calibration data, none of which exist in this container.
"""


def export_onnx(
    model_path: str,
    imgsz: int = DEFAULT_IMGSZ,
    opset: int = 12,
    simplify: bool = True,
    dynamic: bool = False,
) -> str:
    """Export a trained YOLO model to ONNX; returns the .onnx path.

    Raises ``FileNotFoundError`` if the checkpoint is missing.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model checkpoint not found: {model_path}\n"
            "Train first: python -m src.detection.train")
    from ultralytics import YOLO  # lazy import (heavy)

    model = YOLO(model_path)
    onnx_path = model.export(
        format="onnx", imgsz=imgsz, opset=opset,
        simplify=simplify, dynamic=dynamic,
    )
    return str(onnx_path)


def write_tensorrt_readme(path: str = "src/edge/TENSORRT.md") -> str:
    """Write the TensorRT INT8 instructions to a markdown file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# TensorRT INT8 edge deployment\n\n```\n")
        f.write(TENSORRT_INSTRUCTIONS)
        f.write("\n```\n")
    return path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export YOLO11-seg to ONNX for edge.")
    p.add_argument("--model", default="runs/segment/krop_seg/weights/best.pt")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--no-simplify", action="store_true")
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--write-trt-readme", action="store_true",
                   help="also write src/edge/TENSORRT.md")
    return p


def main(argv=None) -> str:
    args = _build_arg_parser().parse_args(argv)
    if args.write_trt_readme:
        print("Wrote", write_tensorrt_readme())
    onnx_path = export_onnx(
        args.model, imgsz=args.imgsz, opset=args.opset,
        simplify=not args.no_simplify, dynamic=args.dynamic)
    print("ONNX export complete:", onnx_path)
    print("\nNext (on Jetson only):")
    print(TENSORRT_INSTRUCTIONS)
    return onnx_path


if __name__ == "__main__":
    main()
