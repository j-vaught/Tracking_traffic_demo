"""
LoRAT-per-object tracking with detector-validated "lock" events.

For each frame F of the input video:
  1. Advance all alive LoRAT tracks in one batched forward.
  2. For each track, find best-matching YOLO+RF-DETR detection by IoU.
     If match IoU >= --reanchor-iou, re-anchor track's box to the detection
     (so the search region for F+1 is detector-precise).
  3. Lock event: a track becomes "locked" the first frame a matching
     detection's conf >= --lock-conf AND IoU with track >= --lock-iou.
  4. Spawn a new LoRAT track for any detection with conf >= --spawn-conf
     that didn't match any existing alive track.
  5. Kill tracks whose LoRAT self-score < --death-score for --death-window
     consecutive frames without a detection match.

Dumps tracks.json with per-track history; rendering is a separate step.

Usage:
    python trackers/run_lorat_lock.py \
        --video outputs/640_input.mp4 \
        --yolo outputs/640_yolo_color.json \
        --detr outputs/640_detr_color.json \
        --out outputs/lorat_tracks.json \
        --device cuda:0 --batch 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lorat_wrapper import BatchedLoRAT
from lorat_multigpu import MultiGPULoRAT


# ---------------------------------------------------------------------------
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Single-pair IoU on [x1, y1, x2, y2] floats."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(tracks: list[np.ndarray], dets: list[np.ndarray]) -> np.ndarray:
    if not tracks or not dets:
        return np.zeros((len(tracks), len(dets)), dtype=np.float32)
    T = np.stack(tracks)        # (N, 4)
    D = np.stack(dets)          # (M, 4)
    x1 = np.maximum(T[:, None, 0], D[None, :, 0])
    y1 = np.maximum(T[:, None, 1], D[None, :, 1])
    x2 = np.minimum(T[:, None, 2], D[None, :, 2])
    y2 = np.minimum(T[:, None, 3], D[None, :, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    area_t = ((T[:, 2] - T[:, 0]) * (T[:, 3] - T[:, 1]))[:, None]
    area_d = ((D[:, 2] - D[:, 0]) * (D[:, 3] - D[:, 1]))[None, :]
    union = area_t + area_d - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def load_detections(path: str, source_tag: str) -> dict[int, list[dict]]:
    """Returns {frame_idx: [{xyxy, conf, cls, source}, ...]}."""
    with open(path) as f:
        data = json.load(f)
    by_frame: dict[int, list[dict]] = {}
    for fr in data:
        fidx = fr["frame"]
        lst = by_frame.setdefault(fidx, [])
        for b in fr["boxes"]:
            lst.append({
                "xyxy": np.asarray(b["xyxy"], dtype=np.float64),
                "conf": float(b["conf"]),
                "cls": int(b["cls"]),
                "source": source_tag,
            })
    return by_frame


class TrackState:
    __slots__ = ("tid", "born", "lock", "died", "hist",
                 "last_det_frame", "low_score_streak")

    def __init__(self, tid: int, born: int):
        self.tid = tid
        self.born = born
        self.lock: int | None = None
        self.died: int | None = None
        self.hist: dict[int, dict] = {}     # {frame: {box, score, det_match}}
        self.last_det_frame = born
        self.low_score_streak = 0


def nms_xyxy(dets: list[dict], iou_thr: float = 0.5) -> list[dict]:
    """Confidence-ordered NMS, returns dedup'd detections across sources."""
    if not dets:
        return []
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["conf"])
    keep_idx = []
    suppressed = np.zeros(len(dets), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep_idx.append(i)
        for j in order:
            if j == i or suppressed[j]:
                continue
            if iou_xyxy(dets[i]["xyxy"], dets[j]["xyxy"]) >= iou_thr:
                suppressed[j] = True
    return [dets[i] for i in keep_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--detr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="weights/lorat/large.bin")
    ap.add_argument("--variant", default="large-224",
                    choices=["base-224", "base-378", "large-224", "large-378",
                             "giant-224", "giant-378"])
    ap.add_argument("--device", default="cuda:0",
                    help="Single-GPU device; ignored if --gpus is set")
    ap.add_argument("--gpus", default=None,
                    help="Comma-separated GPU ids for multi-GPU track split "
                         "(e.g. '0,1,2,3'). Overrides --device when set.")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-tracks", type=int, default=80,
                    help="Cap on concurrent alive LoRAT tracks")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--spawn-conf", type=float, default=0.60,
                    help="Detection confidence needed to spawn a new track")
    ap.add_argument("--lock-conf", type=float, default=0.75)
    ap.add_argument("--lock-iou", type=float, default=0.75)
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--reanchor-iou", type=float, default=0.5)
    ap.add_argument("--death-score", type=float, default=0.25)
    ap.add_argument("--death-window", type=int, default=30)
    ap.add_argument("--nms-iou", type=float, default=0.5,
                    help="IoU threshold for dedup'ing YOLO vs RF-DETR detections")
    args = ap.parse_args()

    # Detections live in COCO-80 convention for YOLO, COCO-91 for RF-DETR;
    # both JSONs already have 'cls' stored in each model's native id space,
    # so the spawn filter is applied via 'name' which is consistent.
    ALLOWED_NAMES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
    def det_allowed(d: dict) -> bool:
        # JSON 'name' is reliable across both detectors (we fixed the COCO-91 map).
        return True  # handled upstream via --classes in run_fast.py

    yolo_by_frame = load_detections(args.yolo, "yolo")
    detr_by_frame = load_detections(args.detr, "detr")

    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip()]
        engine = MultiGPULoRAT(weights=args.weights, variant=args.variant,
                               gpus=gpu_ids, dtype=torch.float16,
                               max_batch_per_gpu=args.batch)
        print(f"[lorat] multi-GPU on {gpu_ids}")
    else:
        engine = BatchedLoRAT(weights=args.weights, variant=args.variant,
                              device=args.device, dtype=torch.float16,
                              max_batch=args.batch)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames is not None:
        total = min(total, args.max_frames)

    tracks: dict[int, TrackState] = {}
    t0 = time.time()
    for fidx in range(total):
        ok, frame = cap.read()
        if not ok:
            break

        # --- 1. Advance all alive LoRAT tracks (batched) ---
        preds = engine.track(frame) if tracks else {}

        # --- 2+3. Merge YOLO + RF-DETR detections with NMS, then match ---
        merged = yolo_by_frame.get(fidx, []) + detr_by_frame.get(fidx, [])
        frame_dets = nms_xyxy(merged, args.nms_iou)
        det_boxes = [d["xyxy"] for d in frame_dets]
        alive_ids = list(preds.keys())
        track_boxes = [preds[tid][0] for tid in alive_ids]
        M = iou_matrix(track_boxes, det_boxes)
        det_claimed = np.zeros(len(frame_dets), dtype=bool)

        for i, tid in enumerate(alive_ids):
            tr = tracks[tid]
            box, score = preds[tid]
            det_match = None
            det_match_iou = 0.0
            if M.shape[1] > 0:
                j = int(M[i].argmax())
                iou = float(M[i, j])
                if iou >= args.match_iou and not det_claimed[j]:
                    det_match = frame_dets[j]
                    det_match_iou = iou
                    det_claimed[j] = True

            if det_match is not None:
                tr.last_det_frame = fidx
                tr.low_score_streak = 0
                if det_match_iou >= args.reanchor_iou:
                    if isinstance(engine, MultiGPULoRAT):
                        engine.set_last_box(tid, det_match["xyxy"])
                    else:
                        engine._tracks[tid].last_box = det_match["xyxy"].astype(np.float64)
                    box = det_match["xyxy"].copy()
                if (tr.lock is None
                        and det_match["conf"] >= args.lock_conf
                        and det_match_iou >= args.lock_iou):
                    tr.lock = fidx
            else:
                if score < args.death_score:
                    tr.low_score_streak += 1
                else:
                    tr.low_score_streak = 0

            tr.hist[fidx] = {
                "box": [float(x) for x in box],
                "score": float(score),
                "det_conf": float(det_match["conf"]) if det_match else 0.0,
                "det_iou": det_match_iou,
            }

        # --- 5. Kill LoRAT trackers past their death window (but KEEP
        # their TrackState so we can render their history later).
        to_kill = [tid for tid in alive_ids
                   if tracks[tid].low_score_streak >= args.death_window
                   and fidx - tracks[tid].last_det_frame >= args.death_window]
        for tid in to_kill:
            tracks[tid].died = fidx
            engine.kill(tid)

        # --- 4. Spawn new LoRAT tracks for unclaimed high-conf detections ---
        for j, d in enumerate(frame_dets):
            if det_claimed[j]:
                continue
            if d["conf"] < args.spawn_conf:
                continue
            if engine.n_alive() >= args.max_tracks:
                break
            try:
                tid = engine.init(frame, d["xyxy"])
            except ValueError:
                continue
            tr = TrackState(tid, fidx)
            tr.hist[fidx] = {
                "box": [float(x) for x in d["xyxy"]],
                "score": 1.0,
                "det_conf": d["conf"],
                "det_iou": 1.0,
            }
            if d["conf"] >= args.lock_conf:
                tr.lock = fidx
            tracks[tid] = tr

        if fidx and fidx % 200 == 0:
            dt = time.time() - t0
            print(f"[lorat] {fidx}/{total}  {fidx/dt:.1f} fps  "
                  f"alive={engine.n_alive()}  total_tracks={len(tracks)}  "
                  f"locked={sum(1 for t in tracks.values() if t.lock is not None)}")

    cap.release()

    # Write tracks.json
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "total_frames": total,
        "per_track": {
            str(tr.tid): {
                "born": tr.born,
                "lock": tr.lock,
                "died": tr.died,
                "hist": tr.hist,
            } for tr in tracks.values()
        },
    }
    with open(out, "w") as f:
        json.dump(payload, f)
    dt = time.time() - t0
    print(f"[lorat] done in {dt:.1f}s  ({total/dt:.2f} fps)  "
          f"{len(tracks)} tracks, {sum(1 for t in tracks.values() if t.lock is not None)} locked")


if __name__ == "__main__":
    main()
