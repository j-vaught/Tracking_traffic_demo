"""
Parallel main-view tracking + PTZ verification, with two synchronized outputs.

Per frame:
  1. Advance all LoRAT trackers on the main 640x640 view (batched).
  2. Associate the frame's pre-computed YOLO detections (from
     640_yolo_color.json) to alive LoRAT trackers via IoU. Re-anchor high-IoU
     matches. Spawn a new LoRAT track for any unassigned detection with
     conf >= --spawn-conf. Kill tracks whose LoRAT score stays low without a
     detection match.
  3. PTZ planner: proximity-picked unverified LoRAT track is the target.
     Compute its crop from the 1920x1080 SDR source, EMA-smooth the camera
     state, run YOLO26x on the viewport, and verify the target (plus any
     in-transit tracks) via 5-consecutive ≥ --verify-conf detections.
  4. Render two videos this frame:
       main_with_verify.mp4 (640x640): every alive LoRAT track's predicted
           box, colored by (purple if verified else by last YOLO conf).
       ptz_view.mp4 (1280x720): the viewport crop + green/yellow/red YOLO
           boxes of whatever the PTZ currently sees.

Usage:
    python trackers/run_parallel_verify.py \
        --main-video outputs/640_input.mp4 \
        --zoom-video data/IMG_2327_1080sdr.mp4 \
        --yolo-json outputs/640_yolo_color.json \
        --out-main outputs/main_with_verify.mp4 \
        --out-ptz  outputs/ptz_view.mp4
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
SRC_W, SRC_H = 1920, 1080
SCALE_X = SRC_W / MAIN_W
SCALE_Y = SRC_H / MAIN_H

PURPLE = (255, 64, 192)            # BGR — distinct from YOLO colors
BRACKETS = (                       # same as the rest of the pipeline
    (0.75, (0, 255, 0)),
    (0.35, (0, 255, 255)),
    (0.10, (0, 0, 255)),
)


def color_for_conf(s):
    for t, c in BRACKETS:
        if s >= t:
            return c
    return None


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    au = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    bu = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    u = au + bu - inter
    return inter / u if u > 0 else 0.0


def iou_matrix(A, B):
    if not A or not B:
        return np.zeros((len(A), len(B)), dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    x1 = np.maximum(A[:, None, 0], B[None, :, 0])
    y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2])
    y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    aa = ((A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]))[:, None]
    ab = ((B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]))[None, :]
    u = aa + ab - inter
    return np.where(u > 0, inter / u, 0.0)


def load_detections(path):
    with open(path) as f:
        data = json.load(f)
    by_frame = {}
    for fr in data:
        lst = by_frame.setdefault(fr["frame"], [])
        for b in fr["boxes"]:
            lst.append({
                "xyxy": np.asarray(b["xyxy"], dtype=np.float64),
                "conf": float(b["conf"]),
                "cls": int(b["cls"]),
            })
    return by_frame


def expand_and_center(box, factor, out_aspect):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    w = (box[2] - box[0]) * factor
    h = (box[3] - box[1]) * factor
    # Enforce aspect of output
    if w / max(h, 1e-6) < out_aspect:
        w = h * out_aspect
    else:
        h = w / out_aspect
    # Minimum crop so tiny boxes don't pixelate past recognition
    min_w = 320.0
    if w < min_w:
        w = min_w
        h = w / out_aspect
    return cx, cy, w, h


def wide_target(out_aspect):
    if abs(out_aspect - SRC_W / SRC_H) < 0.01:
        return SRC_W / 2, SRC_H / 2, SRC_W, SRC_H
    h = SRC_H
    w = h * out_aspect
    if w > SRC_W:
        w = SRC_W
        h = w / out_aspect
    return SRC_W / 2, SRC_H / 2, w, h


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def project_box_to_yolo(box_main, crop_rect_1080, yolo_size=640):
    cx1, cy1, cx2, cy2 = crop_rect_1080
    cw = cx2 - cx1; ch = cy2 - cy1
    if cw <= 0 or ch <= 0:
        return None
    bx1 = box_main[0] * SCALE_X; by1 = box_main[1] * SCALE_Y
    bx2 = box_main[2] * SCALE_X; by2 = box_main[3] * SCALE_Y
    ix1 = max(bx1, cx1); iy1 = max(by1, cy1)
    ix2 = min(bx2, cx2); iy2 = min(by2, cy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    sx = yolo_size / cw; sy = yolo_size / ch
    return ((ix1 - cx1) * sx, (iy1 - cy1) * sy,
            (ix2 - cx1) * sx, (iy2 - cy1) * sy)


class MainTrack:
    __slots__ = ("tid", "lorat_tid", "born", "died", "verified", "last_box",
                 "last_conf", "last_det_frame", "low_score_streak")

    def __init__(self, tid, lorat_tid, born, init_box, init_conf):
        self.tid = tid
        self.lorat_tid = lorat_tid
        self.born = born
        self.died = None
        self.verified = False
        self.last_box = list(init_box)
        self.last_conf = init_conf
        self.last_det_frame = born
        self.low_score_streak = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-video", default="outputs/640_input.mp4")
    ap.add_argument("--zoom-video", default="data/IMG_2327_1080sdr.mp4")
    ap.add_argument("--yolo-json", default="outputs/640_yolo_color.json")
    ap.add_argument("--yolo-weights", default="weights/yolo26x.pt")
    ap.add_argument("--lorat-weights", default="weights/lorat/large.bin")
    ap.add_argument("--lorat-variant", default="large-224")
    ap.add_argument("--out-main", default="outputs/main_with_verify.mp4")
    ap.add_argument("--out-ptz",  default="outputs/ptz_view.mp4")
    ap.add_argument("--out-tracks", default="outputs/parallel_tracks.json")
    ap.add_argument("--out-fps", type=float, default=120.0)
    ap.add_argument("--device", default="cuda:0")
    # Tracker behavior
    ap.add_argument("--spawn-conf", type=float, default=0.35)
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--reanchor-iou", type=float, default=0.5)
    ap.add_argument("--max-tracks", type=int, default=80)
    ap.add_argument("--death-score", type=float, default=0.25)
    ap.add_argument("--death-window", type=int, default=30)
    # PTZ viewport + verify
    ap.add_argument("--out-w", type=int, default=1280)
    ap.add_argument("--out-h", type=int, default=720)
    ap.add_argument("--tau", type=float, default=0.04)
    ap.add_argument("--zoom-context", type=float, default=2.2)
    ap.add_argument("--verify-consec", type=int, default=5)
    ap.add_argument("--verify-conf", type=float, default=0.85)
    ap.add_argument("--max-hold", type=int, default=180)
    ap.add_argument("--in-transit-iou", type=float, default=0.3)
    ap.add_argument("--classes", type=int, nargs="*", default=[0, 1, 2, 3, 5, 7])
    # Run length
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    out_aspect = args.out_w / args.out_h

    # Pre-computed main-view YOLO dets
    yolo_by_frame = load_detections(args.yolo_json)

    # Models
    from ultralytics import YOLO
    ptz_yolo = YOLO(args.yolo_weights)
    lorat = BatchedLoRAT(weights=args.lorat_weights,
                         variant=args.lorat_variant,
                         device=args.device, dtype=torch.float16,
                         max_batch=args.max_tracks)

    # Videos
    main_cap = cv2.VideoCapture(args.main_video)
    zoom_cap = cv2.VideoCapture(args.zoom_video)
    if not main_cap.isOpened() or not zoom_cap.isOpened():
        raise SystemExit("cannot open input video(s)")
    total = int(main_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)
    if args.out_fps <= 0:
        raise SystemExit("--out-fps must be positive")
    out_fps = float(args.out_fps)

    Path(args.out_main).parent.mkdir(parents=True, exist_ok=True)
    writer_main = cv2.VideoWriter(args.out_main,
                                  cv2.VideoWriter_fourcc(*"mp4v"),
                                  out_fps, (MAIN_W, MAIN_H))
    writer_ptz = cv2.VideoWriter(args.out_ptz,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 out_fps, (args.out_w, args.out_h))

    # State
    tracks: dict[int, MainTrack] = {}
    ptz_state = list(wide_target(out_aspect))
    current_target: int | None = None
    frames_on_current = 0
    consec: dict[int, int] = {}
    # Tracks the PTZ gave up on (timed out). Never re-picked as primary.
    skip_primary: set[int] = set()
    stats = {"spawned": 0, "verified_primary": 0,
             "verified_transit": 0, "timeout": 0, "lost": 0}

    t0 = time.time()
    for fidx in range(total):
        okm, fmain = main_cap.read()
        okz, fzoom = zoom_cap.read()
        if not okm or not okz:
            break

        # --- 1. Advance LoRAT ---------------------------------------------
        preds = lorat.track(fmain) if tracks else {}
        # Map global tid -> lorat local tid is identity (we stored lorat_tid=tid
        # for simplicity since we only have one engine).

        # --- 2. Associate detections to tracks ---------------------------
        dets = yolo_by_frame.get(fidx, [])
        alive_tids = [t for t in tracks if tracks[t].died is None]
        alive_boxes = [preds[t][0].tolist() if t in preds else tracks[t].last_box
                       for t in alive_tids]
        det_boxes = [list(d["xyxy"]) for d in dets]
        M = iou_matrix(alive_boxes, det_boxes)
        det_claimed = np.zeros(len(dets), dtype=bool)
        # Greedy one-to-one matching by max IoU >= match_iou
        for _ in range(min(len(alive_tids), len(dets))):
            if M.size == 0:
                break
            i, j = np.unravel_index(M.argmax(), M.shape)
            if M[i, j] < args.match_iou:
                break
            tid = alive_tids[i]
            d = dets[j]
            tr = tracks[tid]
            tr.last_det_frame = fidx
            tr.low_score_streak = 0
            tr.last_conf = d["conf"]
            if M[i, j] >= args.reanchor_iou:
                lorat._tracks[tr.lorat_tid].last_box = d["xyxy"].astype(np.float64)
                tr.last_box = list(d["xyxy"])
            det_claimed[j] = True
            M[i, :] = -1
            M[:, j] = -1

        # For tracks that weren't matched, roll LoRAT-predicted box forward
        for tid in alive_tids:
            if tid in preds:
                tracks[tid].last_box = list(preds[tid][0])
                score = preds[tid][1]
                if score < args.death_score and tracks[tid].last_det_frame != fidx:
                    tracks[tid].low_score_streak += 1

        # --- 3. Spawn new tracks for unclaimed high-conf dets ------------
        spawn_order = sorted(
            [j for j, claimed in enumerate(det_claimed) if not claimed
             and dets[j]["conf"] >= args.spawn_conf],
            key=lambda j: -dets[j]["conf"])
        for j in spawn_order:
            if len([t for t in tracks if tracks[t].died is None]) >= args.max_tracks:
                break
            d = dets[j]
            # Don't spawn if we'd duplicate an alive track
            bad = False
            for tid in alive_tids:
                if tracks[tid].died is not None:
                    continue
                if iou_xyxy(d["xyxy"], tracks[tid].last_box) > 0.4:
                    bad = True
                    break
            if bad:
                continue
            try:
                ltid = lorat.init(fmain, d["xyxy"])
            except ValueError:
                continue
            mt = MainTrack(tid=ltid, lorat_tid=ltid, born=fidx,
                           init_box=list(d["xyxy"]), init_conf=d["conf"])
            tracks[ltid] = mt
            alive_tids.append(ltid)
            stats["spawned"] += 1

        # --- 4. Kill tracks ----------------------------------------------
        for tid in list(alive_tids):
            tr = tracks[tid]
            if tr.died is not None:
                continue
            if (tr.low_score_streak >= args.death_window
                    and fidx - tr.last_det_frame >= args.death_window):
                tr.died = fidx
                lorat.kill(tr.lorat_tid)
                if not tr.verified:
                    stats["lost"] += 1
                if tid == current_target:
                    current_target = None
                    frames_on_current = 0

        # --- 5. PTZ planner: choose target -------------------------------
        unverified_alive = [t for t, tr in tracks.items()
                            if tr.died is None and not tr.verified]
        # Drop current if verified or died
        if current_target is not None:
            tr = tracks.get(current_target)
            if tr is None or tr.died is not None or tr.verified:
                current_target = None
                frames_on_current = 0
        if current_target is None and unverified_alive:
            # Proximity to current camera, excluding tracks we've already
            # given up on (timed out once — it's unlikely YOLO will suddenly
            # verify at 0.85 what we couldn't in 1.5s).
            eligible = [t for t in unverified_alive if t not in skip_primary]
            if eligible:
                cam_cx, cam_cy = ptz_state[0], ptz_state[1]
                def key_prox(tid):
                    b = tracks[tid].last_box
                    bx = (b[0] + b[2]) / 2 * SCALE_X
                    by = (b[1] + b[3]) / 2 * SCALE_Y
                    return (bx - cam_cx) ** 2 + (by - cam_cy) ** 2
                current_target = min(eligible, key=key_prox)
                frames_on_current = 0
                consec.pop(current_target, None)

        # --- 6. Compute PTZ target rect and EMA-smooth --------------------
        if current_target is not None:
            b = tracks[current_target].last_box
            b_1080 = [b[0] * SCALE_X, b[1] * SCALE_Y,
                      b[2] * SCALE_X, b[3] * SCALE_Y]
            tgt = expand_and_center(b_1080, args.zoom_context, out_aspect)
        else:
            tgt = wide_target(out_aspect)
        for i in range(4):
            ptz_state[i] = (1 - args.tau) * ptz_state[i] + args.tau * tgt[i]
        cx, cy, cw, ch = ptz_state
        x1 = int(round(clamp(cx - cw / 2, 0, SRC_W - cw)))
        y1 = int(round(clamp(cy - ch / 2, 0, SRC_H - ch)))
        x2 = int(round(min(SRC_W, x1 + cw)))
        y2 = int(round(min(SRC_H, y1 + ch)))
        if x2 <= x1 or y2 <= y1:
            ptz_out = cv2.resize(fzoom, (args.out_w, args.out_h))
        else:
            ptz_out = cv2.resize(fzoom[y1:y2, x1:x2], (args.out_w, args.out_h),
                                 interpolation=cv2.INTER_LANCZOS4)

        # --- 7. Run YOLO on the PTZ viewport -----------------------------
        crop_rect = (x1, y1, x2, y2)
        max_conf_ptz = 0.0
        all_dets_yolo = []
        crop_yolo = ptz_out if (args.out_w, args.out_h) == (640, 640) else \
            cv2.resize(ptz_out, (640, 640))
        r = ptz_yolo.predict(crop_yolo, classes=args.classes,
                             conf=0.10, imgsz=640, half=True, verbose=False)[0]
        if r.boxes is not None and len(r.boxes) > 0:
            max_conf_ptz = float(r.boxes.conf.max().item())
            for b in r.boxes:
                c = float(b.conf.item())
                all_dets_yolo.append(
                    (tuple(float(x) for x in b.xyxy[0].tolist()), c))

        # --- 8. Verify: current target + in-transit ----------------------
        good_dets = [(xy, c) for (xy, c) in all_dets_yolo
                     if c >= args.verify_conf]
        transit_hits = []
        for tid, tr in tracks.items():
            if tr.died is not None or tr.verified:
                continue
            proj = project_box_to_yolo(tr.last_box, crop_rect, 640)
            if proj is None:
                consec[tid] = 0
                continue
            best = 0.0
            for db, _ in good_dets:
                iou = iou_xyxy(proj, db)
                if iou > best:
                    best = iou
            if best >= args.in_transit_iou:
                consec[tid] = consec.get(tid, 0) + 1
                if consec[tid] >= args.verify_consec:
                    tr.verified = True
                    if tid == current_target:
                        stats["verified_primary"] += 1
                        current_target = None
                        frames_on_current = 0
                    else:
                        stats["verified_transit"] += 1
                        transit_hits.append(tid)
            else:
                consec[tid] = 0

        # Timeout primary target; blacklist so we don't loop back.
        if current_target is not None:
            if frames_on_current >= args.max_hold:
                stats["timeout"] += 1
                skip_primary.add(current_target)
                current_target = None
                frames_on_current = 0

        # --- 9. Render main view -----------------------------------------
        for tid, tr in tracks.items():
            if tr.died is not None:
                continue
            box = tr.last_box
            x1m, y1m, x2m, y2m = [int(round(v)) for v in box]
            if tr.verified:
                col = PURPLE
                thk = 3
            else:
                col = color_for_conf(tr.last_conf)
                thk = 2
                if col is None:
                    continue
            cv2.rectangle(fmain, (x1m, y1m), (x2m, y2m), col, thk)
        writer_main.write(fmain)

        # --- 10. Render PTZ view ------------------------------------------
        if all_dets_yolo:
            sx = args.out_w / 640; sy = args.out_h / 640
            for (dx1, dy1, dx2, dy2), dc in all_dets_yolo:
                col = color_for_conf(dc)
                if col is None:
                    continue
                cv2.rectangle(ptz_out,
                              (int(round(dx1 * sx)), int(round(dy1 * sy))),
                              (int(round(dx2 * sx)), int(round(dy2 * sy))),
                              col, 2)
        # Overlay
        n_ver = sum(1 for t in tracks.values() if t.verified)
        tag = f"PTZ f={fidx}  verified={n_ver}/{len(tracks)}  "
        tag += f"track#{current_target}" if current_target is not None else "SCANNING"
        if current_target is not None:
            c = consec.get(current_target, 0)
            tag += f"  consec={c}/{args.verify_consec}"
            tag += f"  conf={max_conf_ptz:.2f}"
        if transit_hits:
            tag += f"  +transit:{','.join(str(t) for t in transit_hits)}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(ptz_out, (8, 8), (8 + tw + 10, 8 + th + 10), (0, 0, 0), -1)
        cv2.putText(ptz_out, tag, (14, 8 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        writer_ptz.write(ptz_out)

        frames_on_current += 1

        if fidx and fidx % 200 == 0:
            dt = time.time() - t0
            alive = sum(1 for t in tracks.values() if t.died is None)
            ver = sum(1 for t in tracks.values() if t.verified)
            print(f"[parallel] {fidx}/{total}  {fidx/dt:.1f} fps  "
                  f"alive={alive}  tracks={len(tracks)}  verified={ver}  "
                  f"cur={current_target}", flush=True)

    main_cap.release(); zoom_cap.release()
    writer_main.release(); writer_ptz.release()

    # Dump tracks.json
    payload = {
        "config": vars(args),
        "total_frames": total,
        "stats": stats,
        "per_track": {
            str(t.tid): {
                "born": t.born, "died": t.died,
                "verified": t.verified, "last_conf": t.last_conf,
            } for t in tracks.values()
        },
    }
    with open(args.out_tracks, "w") as f:
        json.dump(payload, f)

    dt = time.time() - t0
    print(f"[parallel] done in {dt:.1f}s ({total/dt:.2f} fps)  stats={stats}",
          flush=True)


if __name__ == "__main__":
    main()
