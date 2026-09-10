"""YOLO11-seg training wrapper (Ultralytics).

Thin, configurable wrapper around ``ultralytics.YOLO(...).train(...)``. Defaults
are tuned so the whole thing runs **end-to-end on CPU in a few minutes** on the
synthetic dataset (module 1). For production on real imagery, scale up on a GPU
(see the README / ``build_train_config`` docstring).

The heavy Ultralytics import is done lazily inside ``train_model`` so that the
pure config-building helpers (and their unit tests) don't require it.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict

# Default base checkpoint: nano seg model — smallest, CPU-friendly.
DEFAULT_MODEL = "yolo11n-seg.pt"
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640


def build_train_config(
    data_yaml: str,
    model: str = DEFAULT_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = 8,
    device: str = "cpu",
    project: str = "runs/segment",
    name: str = "krop_seg",
    seed: int = 0,
) -> Dict[str, Any]:
    """Return the kwargs dict handed to ``YOLO.train``.

    Kept pure (no side effects, no Ultralytics import) so it is unit-testable
    and so callers can inspect/override the config before training.

    Scaling to production:
      * Real data + GPU: ``model='yolo11m-seg.pt'`` or ``l/x``, ``device=0``,
        ``epochs=100-300``, larger ``batch`` and ``imgsz`` (e.g. 1024) for the
        high-resolution orthomosaics. Consider ``patience`` for early stopping.
      * Colab: same call; set ``device=0`` and raise ``batch``.
    """
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if imgsz <= 0 or imgsz % 32 != 0:
        raise ValueError("imgsz must be a positive multiple of 32")
    return {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": project,
        "name": name,
        "seed": seed,
        "exist_ok": True,
        "verbose": True,
    }


def train_model(
    data_yaml: str,
    model: str = DEFAULT_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = 8,
    device: str = "cpu",
    project: str = "runs/segment",
    name: str = "krop_seg",
    seed: int = 0,
):
    """Train a YOLO11-seg model and return the Ultralytics results object.

    The best weights are written to ``<project>/<name>/weights/best.pt``.
    """
    from ultralytics import YOLO  # lazy import (heavy)

    cfg = build_train_config(data_yaml, model, epochs, imgsz, batch,
                             device, project, name, seed)
    yolo = YOLO(model)
    results = yolo.train(**cfg)
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train YOLO11-seg on the krop dataset.")
    p.add_argument("--data", default="data/yolo_seg/data.yaml")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--name", default="krop_seg")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    results = train_model(
        args.data, args.model, args.epochs, args.imgsz,
        args.batch, args.device, name=args.name,
    )
    print("Training complete. Results dir:", getattr(results, "save_dir", "?"))
    return results


if __name__ == "__main__":
    main()
