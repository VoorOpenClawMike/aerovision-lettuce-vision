#!/usr/bin/env bash
# End-to-end demo of the SOIL-Data / Aerovision krop-sla pipeline on SYNTHETIC data.
# Each step is documented in README.md. Safe to re-run (idempotent-ish).
set -euo pipefail

export PATH="$PATH:$HOME/.local/bin"
PY="${PYTHON:-python3}"

echo "==> [1/6] Generate synthetic dataset (images + COCO + weight CSV)"
$PY -m src.data.synth_generator --num-images 16 --heads-per-image 30 --seed 42

echo "==> [2/6] Convert COCO -> YOLO-seg dataset"
$PY -m src.detection.coco_to_yolo \
    --coco data/synthetic/annotations/instances.coco.json \
    --images data/synthetic/images --out data/yolo_seg

echo "==> [3/6] Train YOLO11-seg (CPU demo; scale up on GPU for production)"
# Short run for the demo; the documented production default is --epochs 50+ on GPU.
$PY -m src.detection.train --data data/yolo_seg/data.yaml \
    --epochs "${EPOCHS:-50}" --imgsz 640 --device "${DEVICE:-cpu}" --name krop_seg || \
    echo "   (training skipped/failed — pipeline still runs via ground-truth fallback)"

echo "==> [4/6] Train + evaluate the weight regressor (writes results/regression_report.md)"
$PY -m src.regression.weight_model --gsd 0.25

echo "==> [5/6] Run end-to-end pipeline -> results/field_detections.csv"
MODEL="$(find runs -name best.pt 2>/dev/null | head -1 || true)"
if [ -n "${MODEL}" ]; then
    $PY -m src.pipeline --gsd 0.25 --model "${MODEL}"
else
    $PY -m src.pipeline --gsd 0.25
fi

echo "==> [6/6] Sampling advice for a 5000-head field (95% conf, 5% margin)"
$PY -m src.sampling.stats --population 5000 --confidence 0.95 --margin 0.05 \
    --labels-csv results/field_detections.csv || true

echo ""
echo "Done. Launch the dashboard with:"
echo "    streamlit run src/dashboard/app.py"
