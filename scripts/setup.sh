#!/usr/bin/env bash
#
# setup.sh — one-shot setup for Tracking_traffic_demo.
# Clones upstream tracker repos, creates the three venvs, installs all deps.
#
# Requires: uv (https://github.com/astral-sh/uv), git, libturbojpeg.
# Host CUDA driver should support >= 12.1 (tested on driver 550.x / CUDA 12.4).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log() { printf '\n\033[1;36m[setup]\033[0m %s\n' "$*"; }

# ------------------------------------------------------------------
# 1. Clone upstream tracker repos (pinned via shallow clone of default branch).
# ------------------------------------------------------------------
log "cloning trackers"
mkdir -p trackers
[ -d trackers/boosttrack ] || git clone --depth 1 https://github.com/vukasin-stanojevic/BoostTrack.git trackers/boosttrack
[ -d trackers/lorat ]      || git clone --depth 1 https://github.com/LitingLin/LoRAT.git trackers/lorat

# Apply our local overrides.
cp configs/boosttrack/requirements-pip.txt trackers/boosttrack/requirements-pip.txt
cp configs/boosttrack/README.md            trackers/boosttrack/README-setup.md

# ------------------------------------------------------------------
# 2. Detectors venv: YOLO26 + RF-DETR 2XL.
# ------------------------------------------------------------------
log "creating .venv-detectors (py3.11 + torch 2.6 + cu124)"
uv venv --python 3.11 .venv-detectors
VIRTUAL_ENV="$ROOT/.venv-detectors" uv pip install ultralytics 'rfdetr[plus]'
VIRTUAL_ENV="$ROOT/.venv-detectors" uv pip install \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ------------------------------------------------------------------
# 3. BoostTrack venv: py3.8 + torch 2.1 + cu121 (matches upstream env).
# ------------------------------------------------------------------
log "creating .venv-boosttrack (py3.8 + torch 2.1 + cu121)"
uv venv --python 3.8 .venv-boosttrack
VIRTUAL_ENV="$ROOT/.venv-boosttrack" uv pip install numpy==1.24.3 cython setuptools wheel
VIRTUAL_ENV="$ROOT/.venv-boosttrack" uv pip install \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121
VIRTUAL_ENV="$ROOT/.venv-boosttrack" uv pip install \
    -r trackers/boosttrack/requirements-pip.txt

# ------------------------------------------------------------------
# 4. LoRAT venv: py3.11 + torch 2.6 + cu124.
# ------------------------------------------------------------------
log "creating .venv-lorat (py3.11 + torch 2.6 + cu124)"
uv venv --python 3.11 .venv-lorat
VIRTUAL_ENV="$ROOT/.venv-lorat" uv pip install -r trackers/lorat/requirements.txt
VIRTUAL_ENV="$ROOT/.venv-lorat" uv pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ------------------------------------------------------------------
# 5. Sanity check.
# ------------------------------------------------------------------
log "verifying CUDA in each venv"
for v in detectors boosttrack lorat; do
    "$ROOT/.venv-$v/bin/python" -c "
import torch
print(f'  .venv-$v: torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')
"
done

log "done. Next: bash scripts/download_weights.sh all"
