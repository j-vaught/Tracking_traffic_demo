#!/usr/bin/env bash
#
# download_weights.sh — fetch all pretrained weights for the four models.
# Weights live under weights/ or trackers/boosttrack/external/weights/ and are gitignored.
#
# Usage:  bash scripts/download_weights.sh [yolo|rfdetr|boosttrack|lorat|turbojpeg|all]

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
W="$ROOT/weights"
mkdir -p "$W"

TARGET="${1:-all}"
GDOWN="$ROOT/.venv-boosttrack/bin/gdown"

log() { printf '\n\033[1;36m[weights]\033[0m %s\n' "$*"; }

ensure_gdown_v5() {
    if [ ! -x "$GDOWN" ]; then
        echo "gdown not found in .venv-boosttrack — run scripts/setup.sh first." >&2
        exit 1
    fi
    ver="$($GDOWN --version 2>&1 | awk '{print $2}')"
    major="${ver%%.*}"
    if [ "${major:-0}" -lt 5 ]; then
        log "upgrading gdown to >=5 (current: $ver)"
        VIRTUAL_ENV="$ROOT/.venv-boosttrack" uv pip install -U 'gdown>=5' >/dev/null
    fi
}

dl_yolo26x() {
    log "yolo26x via ultralytics"
    "$ROOT/.venv-detectors/bin/python" - <<PY
from ultralytics import YOLO
import shutil, os
m = YOLO("yolo26x.pt")
src = "yolo26x.pt"
dst = os.path.join("$W", "yolo26x.pt")
if os.path.abspath(src) != os.path.abspath(dst):
    shutil.move(src, dst)
print("->", dst)
PY
}

dl_rfdetr_2xl() {
    log "rf-detr 2XLarge via rfdetr[plus] (~484 MB, cached under ~/.cache/rfdetr)"
    "$ROOT/.venv-detectors/bin/python" -c "from rfdetr import RFDETR2XLarge; RFDETR2XLarge()"
}

dl_boosttrack() {
    ensure_gdown_v5
    log "BoostTrack (Deep OC-SORT) weights via gdown folder"
    mkdir -p "$ROOT/trackers/boosttrack/external/weights"
    (cd "$ROOT/trackers/boosttrack/external/weights" && \
        "$GDOWN" --folder https://drive.google.com/drive/folders/15hZcR4bW_Z9hEaXXjeWhQl_jwRKllauG -O .)
}

dl_lorat() {
    ensure_gdown_v5
    log "LoRAT checkpoints via gdown folder (~2 GB total, 12 safetensors .bin files)"
    mkdir -p "$W/lorat"
    (cd "$W/lorat" && \
        "$GDOWN" --folder https://drive.google.com/drive/folders/1FvViP0MCSiAu2FSrNjg7XEORn74yOBdD -O .)
}

dl_turbojpeg() {
    log "libjpeg-turbo 3.1.0 into vendor/ (no sudo; required by PyTurboJPEG in LoRAT)"
    mkdir -p "$ROOT/vendor/deb" "$ROOT/vendor/prefix"
    local deb="$ROOT/vendor/deb/libjpeg-turbo-official_3.1.0_amd64.deb"
    if [ ! -f "$deb" ]; then
        curl -sSL -o "$deb" \
            https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.0/libjpeg-turbo-official_3.1.0_amd64.deb
    fi
    dpkg-deb -x "$deb" "$ROOT/vendor/prefix"
    echo "  shared lib: $ROOT/vendor/prefix/opt/libjpeg-turbo/lib64/libturbojpeg.so.0"
    echo "  set LD_LIBRARY_PATH=\$ROOT/vendor/prefix/opt/libjpeg-turbo/lib64 before running LoRAT"
}

case "$TARGET" in
    yolo)       dl_yolo26x ;;
    rfdetr)     dl_rfdetr_2xl ;;
    boosttrack) dl_boosttrack ;;
    lorat)      dl_lorat ;;
    turbojpeg)  dl_turbojpeg ;;
    all)
        dl_yolo26x
        dl_rfdetr_2xl
        dl_turbojpeg
        dl_lorat
        dl_boosttrack
        ;;
    *) echo "unknown target: $TARGET"; exit 1 ;;
esac
