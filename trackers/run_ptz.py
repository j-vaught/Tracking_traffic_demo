"""
Simulated PTZ pipeline: zoom into main-view low-confidence regions using the
1080p BT.709 SDR rescan of the 4K source, re-run YOLO26x on each zoom crop,
and spawn a LoRAT track for any zoom detection that clears the verify bar.
Verified tracks render as green boxes on the main 640x640 view.

Inputs:
    outputs/640_input.mp4            (main 640x640 SDR view, already exists)
    data/IMG_2327_1080sdr.mp4        (tone-mapped 1920x1080 for zoom crops)
    outputs/640_yolo_color.json      (main YOLO detections at conf >= 0.10)
    outputs/640_detr_color.json      (main RF-DETR detections at conf >= 0.10)
    weights/yolo26x.pt               (for zoom-view re-inference)
    weights/lorat/large.bin          (for tracking verified objects)

Outputs:
    outputs/640_ptz_verified.mp4     (main view + green verified boxes)
    outputs/ptz_tracks.json          (per-track history)

Pipeline per frame F:
  1. Advance all alive LoRAT tracks on the main view (batched).
  2. Combine YOLO + RF-DETR main-view detections, NMS-dedup them.
  3. Pick zoom targets: detections with low_conf_lo <= conf < low_conf_hi
     that do NOT overlap any alive verified track (IoU <= 0.3).
     Cap at --max-zooms by lowest confidence first (weakest detections get
     priority, "those are the ones we most want to verify").
  4. For each target: expand bbox by --zoom-context, map to 1920x1080 coords,
     crop, resize to 640x640, batched-YOLO inference.
  5. For each zoom detection with conf >= --verify-conf: project coords back
     to the main 640x640 view, dedup against alive tracks (IoU > 0.5), and if
     new spawn a LoRAT track marked "verified" immediately.
  6. Kill tracks whose LoRAT score < --death-score for --death-window frames.
  7. Record per-frame box history for all alive tracks.

At the end of the run, write an annotated video drawing green boxes on every
verified track for every frame they were alive (post-hoc so lines are smooth).
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
from lorat_wrapper import BatchedLoRAT  # noqa: E402


MAIN_W = MAIN_H = 640
Z1080_W, Z1080_H = 1920, 1080
SCALE_X = Z1080_W / MAIN_W      # 3.0
SCALE_Y = Z1080_H / MAIN_H      # 1.6875

GREEN = (0, 255, 0)             # BGR
DIM_RED = (0, 0, 150)           # for low-conf targets pre-verification


def iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    u = area_a + area_b - inter
    return inter / u if u > 0 else 0.0


def nms(dets, iou_thr=0.5):
    if not dets:
        return []
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["conf"])
    keep, supp = [], np.zeros(len(dets), dtype=bool)
    for i in order:
        if supp[i]:
            continue
        keep.append(i)
        for j in order:
            if j == i or supp[j]:
                continue
            if iou_xyxy(dets[i]["xyxy"], dets[j]["xyxy"]) >= iou_thr:
                supp[j] = True
    return [dets[i] for i in keep]


def load_detections(path):
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
            })
    return by_frame


def expand_box(xyxy, factor, img_w, img_h):
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * factor
    h = (y2 - y1) * factor
    return (
        max(0.0, cx - w / 2),
        max(0.0, cy - h / 2),
        min(img_w - 1.0, cx + w / 2),
        min(img_h - 1.0, cy + h / 2),
    )


class TrackState:
    __slots__ = ("tid", "born", "first_verified", "died", "hist",
                 "last_det_frame", "low_score_streak")

    def __init__(self, tid, born, init_box):
        self.tid = tid
        self.born = born
        self.first_verified = born  # all PTZ tracks are born verified
        self.died: int | None = None
        self.hist = {born: {"box": [float(x) for x in init_box]}}
        self.last_det_frame = born
        self.low_score_streak = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-video", default="outputs/640_input.mp4")
    ap.add_argument("--zoom-video", default="data/IMG_2327_1080sdr.mp4")
    ap.add_argument("--yolo-json", default="outputs/640_yolo_color.json")
    ap.add_argument("--detr-json", default="outputs/640_detr_color.json")
    ap.add_argument("--yolo-weights", default="weights/yolo26x.pt")
    ap.add_argument("--lorat-weights", default="weights/lorat/large.bin")
    ap.add_argument("--lorat-variant", default="large-224")
    ap.add_argument("--out-video", default="outputs/640_ptz_verified.mp4")
    ap.add_argument("--out-tracks", default="outputs/ptz_tracks.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-zooms", type=int, default=4,
                    help="Max zoom inferences per frame")
    ap.add_argument("--zoom-context", type=float, default=1.8,
                    help="Expand low-conf bbox by this factor before cropping")
    ap.add_argument("--low-conf-lo", type=float, default=0.10)
    ap.add_argument("--low-conf-hi", type=float, default=0.60)
    ap.add_argument("--verify-conf", type=float, default=0.75,
                    help="Zoom-YOLO confidence needed to accept a detection")
    ap.add_argument("--track-overlap-iou", type=float, default=0.5,
                    help="IoU above which a verified detection is considered "
                         "to duplicate an existing track and is dropped")
    ap.add_argument("--death-score", type=float, default=0.25)
    ap.add_argument("--death-window", type=int, default=30)
    ap.add_argument("--classes", type=int, nargs="*",
                    default=[0, 1, 2, 3, 5, 7])
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    # --- Load detectors ----------------------------------------------------
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)
    lorat = BatchedLoRAT(weights=args.lorat_weights, variant=args.lorat_variant,
                         device=args.device, dtype=torch.float16,
                         max_batch=args.max_zooms * 2 + 16)

    # --- Load main-view detections ----------------------------------------
    yolo_by_frame = load_detections(args.yolo_json)
    detr_by_frame = load_detections(args.detr_json)

    # --- Open videos -------------------------------------------------------
    main_cap = cv2.VideoCapture(args.main_video)
    zoom_cap = cv2.VideoCapture(args.zoom_video)
    if not main_cap.isOpened():
        raise SystemExit(f"cannot open {args.main_video}")
    if not zoom_cap.isOpened():
        raise SystemExit(f"cannot open {args.zoom_video}")
    total = int(main_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames is not None:
        total = min(total, args.max_frames)
    fps = main_cap.get(cv2.CAP_PROP_FPS) or 30.0

    Path(args.out_video).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (MAIN_W, MAIN_H))

    tracks: dict[int, TrackState] = {}
    stats = {"zooms_attempted": 0, "zooms_verified": 0,
             "spawned_tracks": 0, "verified_dets": 0}
    t0 = time.time()

    for fidx in range(total):
        ok_m, frame_main = main_cap.read()
        ok_z, frame_zoom = zoom_cap.read()
        if not ok_m or not ok_z:
            break

        # --- 1. Advance LoRAT tracks on main view (batched) ----------------
        preds = lorat.track(frame_main) if tracks else {}
        alive_ids = list(preds.keys())

        # --- 2. Main detections + NMS + target pick ------------------------
        merged = yolo_by_frame.get(fidx, []) + detr_by_frame.get(fidx, [])
        merged = nms(merged, 0.5)

        track_boxes = [preds[tid][0] for tid in alive_ids]
        candidates = []
        for d in merged:
            if not (args.low_conf_lo <= d["conf"] < args.low_conf_hi):
                continue
            # Skip if already covered by an alive track
            if any(iou_xyxy(d["xyxy"], tb) > 0.3 for tb in track_boxes):
                continue
            candidates.append(d)
        # Lowest-conf first: weakest detections get zoom priority
        candidates.sort(key=lambda x: x["conf"])
        targets = candidates[:args.max_zooms]

        # --- 3. Crop zoom regions + batched YOLO --------------------------
        verified_boxes = []
        if targets:
            crops, metas = [], []
            for t in targets:
                # Expand in main-view coords, then map to 1080p coords
                ex = expand_box(t["xyxy"], args.zoom_context, MAIN_W, MAIN_H)
                x1, y1, x2, y2 = ex
                cx1 = int(round(x1 * SCALE_X))
                cy1 = int(round(y1 * SCALE_Y))
                cx2 = int(round(x2 * SCALE_X))
                cy2 = int(round(y2 * SCALE_Y))
                if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                    continue
                crop = frame_zoom[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                crops.append(cv2.resize(crop, (640, 640)))
                metas.append({
                    "crop_1080": (cx1, cy1, cx2, cy2),
                    "src": t,
                })
            if crops:
                stats["zooms_attempted"] += len(crops)
                results = yolo.predict(crops, conf=args.verify_conf,
                                       classes=args.classes, half=True,
                                       imgsz=640, verbose=False)
                for r, meta in zip(results, metas):
                    if r.boxes is None or len(r.boxes) == 0:
                        continue
                    cx1, cy1, cx2, cy2 = meta["crop_1080"]
                    cw = cx2 - cx1
                    ch = cy2 - cy1
                    for b in r.boxes:
                        zx1, zy1, zx2, zy2 = [float(x) for x in b.xyxy[0].tolist()]
                        # Project zoom view (640x640) -> 1080 frame -> main 640
                        px1 = (cx1 + (zx1 / 640) * cw) / SCALE_X
                        py1 = (cy1 + (zy1 / 640) * ch) / SCALE_Y
                        px2 = (cx1 + (zx2 / 640) * cw) / SCALE_X
                        py2 = (cy1 + (zy2 / 640) * ch) / SCALE_Y
                        conf = float(b.conf.item())
                        verified_boxes.append({
                            "xyxy": [px1, py1, px2, py2],
                            "zoom_conf": conf,
                        })
                stats["verified_dets"] += len(verified_boxes)

        # --- 4. Spawn LoRAT tracks for new verified detections ------------
        for vb in verified_boxes:
            # Dedup vs alive tracks
            if any(iou_xyxy(vb["xyxy"], tb) > args.track_overlap_iou
                   for tb in track_boxes):
                continue
            xyxy = np.asarray(vb["xyxy"], dtype=np.float64)
            if xyxy[2] - xyxy[0] < 4 or xyxy[3] - xyxy[1] < 4:
                continue
            try:
                tid = lorat.init(frame_main, xyxy)
            except ValueError:
                continue
            tracks[tid] = TrackState(tid, fidx, xyxy)
            track_boxes.append(xyxy)
            stats["spawned_tracks"] += 1
            stats["zooms_verified"] += 1

        # --- 5. Record hist + LoRAT re-anchor from main detections -------
        for tid, (box, score) in preds.items():
            tr = tracks[tid]
            # Try to re-anchor if a high-IoU main-view detection exists
            best_iou, best_det = 0.0, None
            for d in merged:
                iou = iou_xyxy(box, d["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_det = d
            if best_det is not None and best_iou >= 0.5:
                tr.last_det_frame = fidx
                tr.low_score_streak = 0
                if best_iou >= 0.7:
                    lorat._tracks[tid].last_box = best_det["xyxy"].astype(np.float64)
                    box = best_det["xyxy"].copy()
            else:
                if score < args.death_score:
                    tr.low_score_streak += 1
                else:
                    tr.low_score_streak = 0
            tr.hist[fidx] = {"box": [float(x) for x in box], "score": float(score)}

        # --- 6. Kill dead tracks ------------------------------------------
        to_kill = [tid for tid in alive_ids
                   if tracks[tid].low_score_streak >= args.death_window
                   and fidx - tracks[tid].last_det_frame >= args.death_window]
        for tid in to_kill:
            tracks[tid].died = fidx
            lorat.kill(tid)

        # --- 7. Render frame with verified green boxes --------------------
        for tid, tr in tracks.items():
            entry = tr.hist.get(fidx)
            if entry is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in entry["box"]]
            cv2.rectangle(frame_main, (x1, y1), (x2, y2), GREEN, 2)
        writer.write(frame_main)

        if fidx and fidx % 200 == 0:
            dt = time.time() - t0
            print(f"[ptz] {fidx}/{total}  {fidx/dt:.1f} fps  "
                  f"alive={lorat.n_alive()}  tracks={len(tracks)}  "
                  f"zooms={stats['zooms_attempted']}  "
                  f"verified={stats['zooms_verified']}",
                  flush=True)

    main_cap.release()
    zoom_cap.release()
    writer.release()

    # --- Dump tracks.json ----------------------------------------------------
    payload = {
        "config": vars(args),
        "total_frames": total,
        "stats": stats,
        "per_track": {
            str(tr.tid): {
                "born": tr.born,
                "first_verified": tr.first_verified,
                "died": tr.died,
                "hist": tr.hist,
            } for tr in tracks.values()
        },
    }
    with open(args.out_tracks, "w") as f:
        json.dump(payload, f)
    dt = time.time() - t0
    print(f"[ptz] done in {dt:.1f}s  ({total/dt:.2f} fps)  "
          f"{len(tracks)} tracks, {stats['zooms_attempted']} zooms attempted, "
          f"{stats['zooms_verified']} spawned verified",
          flush=True)


if __name__ == "__main__":
    main()
