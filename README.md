# Tracking Traffic Demo

Comparison playground for four current-gen (2025–2026) detection and tracking models on a single PTZ traffic clip.

**Author:** J.C. Vaught
**Source video:** `data/IMG_2327.MOV` (gitignored, ~1.5 GB)
**Hardware:** NVIDIA RTX 6000 Ada × 4, CUDA 12.4 driver

## Models

| Role | Model | Variant | Reference | Released |
|---|---|---|---|---|
| Detector | **YOLO26** | `yolo26x` (57.5 COCO mAP, 55.7M params) | [Ultralytics](https://docs.ultralytics.com/models/yolo26/) | 2026-01-14 |
| Detector | **RF-DETR** | `RFDETR2XLarge` (60.1 COCO AP50:95, SOTA real-time DETR) | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | 2026-04-10 (v1.6.4) |
| Tracker (MOT) | **BoostTrack++** | [paper](https://arxiv.org/abs/2408.13003) | [vukasin-stanojevic/BoostTrack](https://github.com/vukasin-stanojevic/BoostTrack) | 2025-08 |
| Tracker (SOT) | **LoRATv2** | `g-378` (largest variant) — NeurIPS 2025 Spotlight | [LitingLin/LoRAT](https://github.com/LitingLin/LoRAT) | 2025-12-05 |

Earlier versions considered and rejected in favor of the above: YOLOv11x, YOLOv12/v13, RT-DETR-X, RT-DETRv2, original BoostTrack / BoostTrack+, original LoRAT.

## Layout

```
PTZ_demo/
├── data/                     # source video (gitignored)
├── weights/                  # pretrained checkpoints (gitignored)
├── detectors/                # detector runners (our code)
├── trackers/                 # BoostTrack + LoRAT clones (gitignored, populated by setup.sh)
├── configs/
│   └── boosttrack/           # our overrides (pip requirements, setup notes)
├── scripts/
│   ├── setup.sh              # one-shot env + clone + install
│   └── download_weights.sh
├── .venv-detectors/          # YOLO26 + RF-DETR (py3.11, torch 2.6+cu124)
├── .venv-boosttrack/         # BoostTrack++ (py3.8, torch 2.1+cu121)
└── .venv-lorat/              # LoRAT / LoRATv2 (py3.11, torch 2.6+cu124)
```

Each model gets its own venv because BoostTrack pins Python 3.8 and old numpy/onnx versions, while LoRAT and the detectors prefer a modern stack. Keeping them separate avoids dependency conflicts.

## Setup

### System packages

```bash
sudo apt install -y libturbojpeg g++ cmake
```

Also requires [`uv`](https://github.com/astral-sh/uv) (>= 0.4) and an NVIDIA driver supporting CUDA >= 12.1 (tested on 550.x / CUDA 12.4).

### One-shot install

```bash
bash scripts/setup.sh
```

This clones `BoostTrack` and `LoRAT` into `trackers/` (both gitignored), creates the three venvs via `uv`, pins torch wheels against matching CUDA indices, and runs a CUDA sanity check per venv.

### Download weights

```bash
bash scripts/download_weights.sh all
```

YOLO26x and RF-DETR 2XL download automatically through their Python libraries. BoostTrack and LoRAT weights are hosted on Google Drive — the script prints the folder links.

## Quick sanity checks

```bash
# YOLO26x on one frame
.venv-detectors/bin/python -c "from ultralytics import YOLO; \
    YOLO('weights/yolo26x.pt').predict('data/IMG_2327.MOV', save=True, stream=False, vid_stride=60)"

# RF-DETR 2XL on one frame
.venv-detectors/bin/python -c "from rfdetr import RFDETR2XLarge; \
    print(RFDETR2XLarge().predict('data/IMG_2327.MOV'))"
```

## Licensing

This project's own code is MIT. **Upstream component licenses vary**:

- **YOLO26 / Ultralytics** — AGPL-3.0 (plus a commercial license tier).
- **RF-DETR L and smaller** — Apache 2.0.
- **RF-DETR XL / 2XL** — [Roboflow PML 1.0](https://github.com/roboflow/rf-detr). Permissive for research / non-commercial + most commercial use, but it's not OSI-standard Apache. **Weights are not committed to this repo**; the `[plus]` extra fetches them at runtime, so the repo itself stays clean, but any downstream user who runs `download_weights.sh` inherits the PML terms.
- **BoostTrack** — MIT.
- **LoRAT** — MIT.

If you need a fully Apache/MIT runtime, swap `RFDETR2XLarge` → `RFDETRLarge` in the detector runner.

## References

- Ultralytics YOLO26 docs — https://docs.ultralytics.com/models/yolo26/
- RF-DETR paper — https://arxiv.org/abs/2511.09554
- BoostTrack++ paper — https://arxiv.org/abs/2408.13003
- LoRAT (ECCV 2024) paper — https://arxiv.org/abs/2403.05231
- LoRATv2 (NeurIPS 2025 Spotlight) — https://neurips.cc/virtual/2025/loc/san-diego/poster/115907
