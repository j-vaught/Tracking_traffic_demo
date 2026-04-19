#!/usr/bin/env bash
#
# download_weights.sh — fetch pretrained weights for all four models.
# Weights live under PTZ_demo/weights/ and are gitignored.
#
# Usage:  bash scripts/download_weights.sh [yolo|rfdetr|boosttrack|lorat|all]

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$ROOT/weights"
mkdir -p "$W"

TARGET="${1:-all}"

dl_yolo26x() {
  echo "[yolo26x] downloading via ultralytics..."
  "$ROOT/.venv-detectors/bin/python" - <<'PY'
from ultralytics import YOLO
import shutil, os
m = YOLO("yolo26x.pt")
src = "yolo26x.pt"
dst = os.path.join(os.environ["W"], "yolo26x.pt")
if os.path.abspath(src) != os.path.abspath(dst):
    shutil.move(src, dst)
print("->", dst)
PY
}

dl_rfdetr_2xl() {
  echo "[rf-detr-2xl] downloading via rfdetr..."
  "$ROOT/.venv-detectors/bin/python" - <<'PY'
from rfdetr import RFDETR2XLarge
m = RFDETR2XLarge()
print("RF-DETR 2XLarge loaded; weights cached under ~/.cache")
PY
}

dl_boosttrack() {
  echo "[boosttrack] weights are hosted on Google Drive (Deep OC-SORT)."
  echo "  -> https://drive.google.com/drive/folders/15hZcR4bW_Z9hEaXXjeWhQl_jwRKllauG"
  echo "  Download and place under trackers/boosttrack/external/weights/"
}

dl_lorat() {
  echo "[lorat] checkpoints are hosted on Google Drive."
  echo "  -> https://drive.google.com/drive/folders/1FvViP0MCSiAu2FSrNjg7XEORn74yOBdD"
  echo "  Largest: LoRAT-g-378 (or LoRATv2 equivalent per repo README)."
  echo "  Place under weights/lorat/"
  mkdir -p "$W/lorat"
}

export W

case "$TARGET" in
  yolo)       dl_yolo26x ;;
  rfdetr)     dl_rfdetr_2xl ;;
  boosttrack) dl_boosttrack ;;
  lorat)      dl_lorat ;;
  all)
    dl_yolo26x
    dl_rfdetr_2xl
    dl_boosttrack
    dl_lorat
    ;;
  *) echo "unknown target: $TARGET"; exit 1 ;;
esac
