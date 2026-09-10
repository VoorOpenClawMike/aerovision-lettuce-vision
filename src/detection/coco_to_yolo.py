"""Convert a COCO instance-segmentation dataset to the YOLO-seg layout.

YOLO segmentation training expects, per image, a ``.txt`` label file where each
line is::

    <class_id> x1 y1 x2 y2 ... xn yn

with all polygon coordinates **normalised** to [0, 1] by image width/height.
Alongside the labels we emit a ``data.yaml`` describing the split and class names.

Output layout (Ultralytics-compatible)::

    <out>/
      data.yaml
      images/train/*.png   images/val/*.png
      labels/train/*.txt   labels/val/*.txt

This converter is deliberately dependency-light (stdlib + shutil) so it is fully
unit-testable without Ultralytics or a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List, Sequence, Tuple


def normalize_polygon(
    flat_xy: Sequence[float], width: int, height: int
) -> List[float]:
    """Normalise a flat ``[x1, y1, x2, y2, ...]`` polygon to [0, 1], clamped."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0")
    out: List[float] = []
    for i in range(0, len(flat_xy) - 1, 2):
        x = min(max(flat_xy[i] / width, 0.0), 1.0)
        y = min(max(flat_xy[i + 1] / height, 0.0), 1.0)
        out.extend([x, y])
    return out


def _split_indices(n: int, val_fraction: float) -> Tuple[List[int], List[int]]:
    """Deterministic split: last ``val_fraction`` of the (sorted) images -> val.

    Deterministic (no RNG) so conversions are reproducible. At least one image
    goes to each split when ``n >= 2``.
    """
    if n <= 0:
        return [], []
    n_val = max(1, round(n * val_fraction)) if n >= 2 else 0
    n_val = min(n_val, n - 1) if n >= 2 else 0
    train = list(range(0, n - n_val))
    val = list(range(n - n_val, n))
    return train, val


def convert_coco_to_yolo_seg(
    coco_path: str,
    images_dir: str,
    out_dir: str,
    val_fraction: float = 0.2,
    class_remap: Dict[int, int] | None = None,
) -> dict:
    """Convert a COCO json + image folder into a YOLO-seg dataset directory.

    ``class_remap`` maps COCO category ids -> contiguous YOLO class ids
    (default: sort category ids and map to 0..k-1).

    Returns a summary dict (paths, counts, data.yaml path).
    """
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = sorted(coco["categories"], key=lambda c: c["id"])
    if class_remap is None:
        class_remap = {c["id"]: i for i, c in enumerate(categories)}
    names = {class_remap[c["id"]]: c["name"] for c in categories}

    images = sorted(coco["images"], key=lambda im: im["id"])
    # group annotations by image id
    anns_by_image: Dict[int, List[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    train_idx, val_idx = _split_indices(len(images), val_fraction)
    split_of = {}
    for i in train_idx:
        split_of[i] = "train"
    for i in val_idx:
        split_of[i] = "val"

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    counts = {"train": 0, "val": 0}
    label_lines_total = 0

    for pos, im in enumerate(images):
        split = split_of.get(pos, "train")
        file_name = im["file_name"]
        width, height = im["width"], im["height"]
        stem = os.path.splitext(os.path.basename(file_name))[0]

        # copy image (if present on disk)
        src_img = os.path.join(images_dir, file_name)
        if os.path.exists(src_img):
            shutil.copy2(src_img, os.path.join(out_dir, "images", split,
                                               os.path.basename(file_name)))

        # write label file
        lines: List[str] = []
        for ann in anns_by_image.get(im["id"], []):
            cls = class_remap.get(ann["category_id"])
            if cls is None:
                continue
            for seg in ann.get("segmentation", []):
                if not isinstance(seg, list) or len(seg) < 6:
                    continue  # skip RLE / degenerate polygons
                norm = normalize_polygon(seg, width, height)
                lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in norm))

        label_path = os.path.join(out_dir, "labels", split, stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        counts[split] += 1
        label_lines_total += len(lines)

    data_yaml = os.path.join(out_dir, "data.yaml")
    _write_data_yaml(data_yaml, out_dir, names)

    return {
        "out_dir": os.path.abspath(out_dir),
        "data_yaml": data_yaml,
        "num_train": counts["train"],
        "num_val": counts["val"],
        "num_instances": label_lines_total,
        "names": names,
    }


def _write_data_yaml(path: str, out_dir: str, names: Dict[int, str]) -> None:
    """Write a minimal YOLO ``data.yaml`` (no PyYAML dependency)."""
    lines = [
        f"path: {os.path.abspath(out_dir)}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx in sorted(names):
        lines.append(f"  {idx}: {names[idx]}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert COCO-seg to YOLO-seg dataset.")
    p.add_argument("--coco", default="data/synthetic/annotations/instances.coco.json")
    p.add_argument("--images", default="data/synthetic/images")
    p.add_argument("--out", default="data/yolo_seg")
    p.add_argument("--val-fraction", type=float, default=0.2)
    return p


def main(argv=None) -> dict:
    args = _build_arg_parser().parse_args(argv)
    summary = convert_coco_to_yolo_seg(args.coco, args.images, args.out, args.val_fraction)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
