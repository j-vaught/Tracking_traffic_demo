"""
Render a "cinematic" simulated-PTZ view: a single 1280x720 (16:9) viewport
that pans/zooms to follow the verified tracks from ptz_tracks.json, cropping
from the 1920x1080 SDR source. Smoothed EMA on center + zoom gives a realistic
mechanical-PTZ feel (no instant teleports).

Behavior per frame:
  1. Find all tracks alive at this frame.
  2. Pick one to follow: the one whose life will last longest from here.
     Sticking with a choice lowers jitter; committing to what stays visible.
  3. Compute target (cx, cy, crop_w, crop_h) in 1080p coords to frame that
     track at ~30% of viewport area (configurable).
  4. EMA-smooth toward the target: camera inertia.
  5. Crop from 1080p, resize to out_wh, overlay small track-id tag.
  6. If no track alive: target is the full 1080p frame (aspect-fit).

Usage:
    python trackers/render_ptz_viewport.py \
        --tracks outputs/ptz_tracks.json \
        --source data/IMG_2327_1080sdr.mp4 \
        --out outputs/640_ptz_viewport.mp4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


SRC_W, SRC_H = 1920, 1080
# Main detector space the tracks.json boxes live in
MAIN_W = MAIN_H = 640
SCALE_X = SRC_W / MAIN_W          # 3.0
SCALE_Y = SRC_H / MAIN_H          # 1.6875


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def target_for_track(track_entry, fidx, out_aspect, box_fill=0.35):
    """Return (cx, cy, crop_w, crop_h) in 1080p coords framing the given
    track's current box. crop has aspect == out_aspect."""
    box_main = track_entry.get("box")
    if box_main is None:
        return None
    x1 = box_main[0] * SCALE_X
    y1 = box_main[1] * SCALE_Y
    x2 = box_main[2] * SCALE_X
    y2 = box_main[3] * SCALE_Y
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = max(x2 - x1, 8.0)
    bh = max(y2 - y1, 8.0)
    # Want box to occupy sqrt(box_fill) of the viewport linearly (box_fill of area).
    frac = np.sqrt(box_fill)
    # Try to fit both width and height, so whichever is larger governs zoom.
    crop_w = bw / frac
    crop_h = bh / frac
    # Enforce aspect
    if crop_w / crop_h < out_aspect:
        crop_w = crop_h * out_aspect
    else:
        crop_h = crop_w / out_aspect
    # Floor: never zoom closer than a reasonable minimum (so tiny boxes don't pixelate).
    min_w = 320.0
    min_h = min_w / out_aspect
    if crop_w < min_w:
        crop_w, crop_h = min_w, min_h
    # Ceiling: never zoom further than full source frame.
    if crop_w > SRC_W:
        crop_w = SRC_W
        crop_h = crop_w / out_aspect
    if crop_h > SRC_H:
        crop_h = SRC_H
        crop_w = crop_h * out_aspect
    return cx, cy, crop_w, crop_h


def wide_target(out_aspect):
    """Target = aspect-fit of full 1080p frame."""
    # Full 1080p is 16:9. 16:9 out_aspect matches exactly.
    if abs(out_aspect - SRC_W / SRC_H) < 0.01:
        return SRC_W / 2, SRC_H / 2, SRC_W, SRC_H
    # Otherwise pillarbox (keep height max).
    crop_h = SRC_H
    crop_w = crop_h * out_aspect
    if crop_w > SRC_W:
        crop_w = SRC_W
        crop_h = crop_w / out_aspect
    return SRC_W / 2, SRC_H / 2, crop_w, crop_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-w", type=int, default=1280)
    ap.add_argument("--out-h", type=int, default=720)
    ap.add_argument("--tau", type=float, default=0.04,
                    help="EMA factor per frame for smoothing (smaller = slower PTZ)")
    ap.add_argument("--box-fill", type=float, default=0.25,
                    help="Target box area fraction of viewport (0..1)")
    ap.add_argument("--switch-cooldown", type=int, default=60,
                    help="Minimum frames to hold on a track before switching")
    ap.add_argument("--planner", choices=["sticky", "coverage", "verify"],
                    default="verify",
                    help="sticky: follow the longest-lived track; "
                         "coverage: visit each track for fixed min-hold; "
                         "verify: hold until N consecutive YOLO frames clear "
                         "conf-bar OR max-hold hits")
    ap.add_argument("--min-hold", type=int, default=120,
                    help="Coverage planner only: frames to linger")
    ap.add_argument("--max-hold", type=int, default=180,
                    help="Verify planner: give up on a track after this many "
                         "frames if we haven't hit the consecutive threshold "
                         "(120 fps source -> 180 frames ~= 1.5 s)")
    ap.add_argument("--verify-consec", type=int, default=5,
                    help="Verify planner: require N consecutive frames above "
                         "the conf threshold")
    ap.add_argument("--verify-conf", type=float, default=0.85,
                    help="Verify planner: YOLO conf threshold")
    ap.add_argument("--yolo-weights", default="weights/yolo26x.pt")
    ap.add_argument("--classes", type=int, nargs="*",
                    default=[0, 1, 2, 3, 5, 7])
    ap.add_argument("--overlay", action="store_true", default=True,
                    help="Draw a small track-id tag on the viewport")
    ap.add_argument("--no-overlay", dest="overlay", action="store_false")
    args = ap.parse_args()

    out_aspect = args.out_w / args.out_h

    with open(args.tracks) as f:
        payload = json.load(f)
    per_track = payload["per_track"]
    total = payload["total_frames"]

    # Precompute per-frame active tracks + each track's "frames remaining"
    # using the last frame of its hist (for those without 'died').
    last_seen: dict[str, int] = {}
    for tid, tr in per_track.items():
        frames = [int(k) for k in tr["hist"].keys()]
        last_seen[tid] = max(frames) if frames else tr.get("born", 0)
        if tr.get("died") is None:
            tr["died"] = last_seen[tid]  # treat as dead after last recorded

    by_frame_active: dict[int, list[str]] = {}
    for tid, tr in per_track.items():
        born = tr["born"]
        died = tr["died"] if tr["died"] is not None else total
        for f in range(born, died + 1):
            by_frame_active.setdefault(f, []).append(tid)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (args.out_w, args.out_h))

    # Start wide
    state = list(wide_target(out_aspect))  # [cx, cy, crop_w, crop_h]
    current_target: str | None = None
    frames_on_current = 0
    consec_good = 0
    visited: set[str] = set()
    verify_reason: dict[str, str] = {}   # tid -> "verified" / "timeout" / "lost"

    # YOLO for the verify planner only.
    yolo = None
    if args.planner == "verify":
        from ultralytics import YOLO
        yolo = YOLO(args.yolo_weights)

    t0 = time.time()
    for fidx in range(total):
        ok, frame = cap.read()
        if not ok:
            break

        active = by_frame_active.get(fidx, [])
        # Drop current if the track is no longer alive this frame.
        if current_target is not None and current_target not in active:
            visited.add(current_target)
            verify_reason.setdefault(current_target, "lost")
            current_target = None
            frames_on_current = 0
            consec_good = 0

        if args.planner in ("coverage", "verify"):
            done = False
            if current_target is not None:
                if args.planner == "coverage":
                    done = frames_on_current >= args.min_hold
                    reason = "visited"
                else:  # verify
                    if consec_good >= args.verify_consec:
                        done = True
                        reason = "verified"
                    elif frames_on_current >= args.max_hold:
                        done = True
                        reason = "timeout"
                if done:
                    visited.add(current_target)
                    verify_reason[current_target] = reason
                    current_target = None
                    frames_on_current = 0
                    consec_good = 0
            if current_target is None and active:
                unvisited = [t for t in active if t not in visited]
                if unvisited:
                    def rem(tid: str) -> int:
                        return per_track[tid]["died"] - fidx
                    current_target = max(unvisited, key=rem)
                    frames_on_current = 0
                    consec_good = 0
        else:  # sticky
            if active:
                if current_target in active and frames_on_current < args.switch_cooldown:
                    pass
                else:
                    def rem(tid: str) -> int:
                        return per_track[tid]["died"] - fidx
                    current_target = max(active, key=rem)
                    frames_on_current = 0

        if current_target is not None:
            entry = per_track[current_target]["hist"].get(str(fidx))
            if entry is None:
                # Track hist missing this frame; fall back to wide
                tgt = wide_target(out_aspect)
            else:
                tgt = target_for_track(entry, fidx, out_aspect, args.box_fill)
                if tgt is None:
                    tgt = wide_target(out_aspect)
        else:
            tgt = wide_target(out_aspect)

        # EMA toward target
        for i in range(4):
            state[i] = (1 - args.tau) * state[i] + args.tau * tgt[i]
        cx, cy, cw, ch = state
        # Clamp crop to image
        x1 = int(round(clamp(cx - cw / 2, 0, SRC_W - cw)))
        y1 = int(round(clamp(cy - ch / 2, 0, SRC_H - ch)))
        x2 = int(round(min(SRC_W, x1 + cw)))
        y2 = int(round(min(SRC_H, y1 + ch)))
        if x2 <= x1 or y2 <= y1:
            out = cv2.resize(frame, (args.out_w, args.out_h))
        else:
            crop = frame[y1:y2, x1:x2]
            out = cv2.resize(crop, (args.out_w, args.out_h),
                             interpolation=cv2.INTER_LANCZOS4)

        # --- Verify planner: run YOLO on the displayed viewport crop ---
        max_conf = 0.0
        if yolo is not None and current_target is not None:
            # Use a downsized copy if out_w is big — YOLO wants 640x640 anyway.
            crop_yolo = out if (args.out_w, args.out_h) == (640, 640) else \
                cv2.resize(out, (640, 640))
            r = yolo.predict(crop_yolo, classes=args.classes,
                             conf=0.30, imgsz=640, half=True, verbose=False)[0]
            if r.boxes is not None and len(r.boxes) > 0:
                max_conf = float(r.boxes.conf.max().item())
            if max_conf >= args.verify_conf:
                consec_good += 1
            else:
                consec_good = 0

        if args.overlay:
            label = f"PTZ  f={fidx}  visited={len(visited)}/{len(per_track)}  "
            if current_target is not None:
                label += f"track#{current_target}"
                if args.planner == "verify":
                    label += f"  consec={consec_good}/{args.verify_consec}"
                    label += f"  conf={max_conf:.2f}"
            else:
                label += "SCANNING"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (8, 8), (8 + tw + 10, 8 + th + 10), (0, 0, 0), -1)
            cv2.putText(out, label, (14, 8 + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(out)
        frames_on_current += 1

        if fidx and fidx % 500 == 0:
            dt = time.time() - t0
            print(f"[ptz-view] {fidx}/{total}  {fidx/dt:.1f} fps  "
                  f"cur={current_target}  state=({cx:.0f},{cy:.0f},"
                  f"{cw:.0f}x{ch:.0f})", flush=True)

    cap.release()
    writer.release()
    # Summary of visit reasons
    from collections import Counter
    reasons = Counter(verify_reason.values())
    print(f"visits: {dict(reasons)}")
    print(f"wrote {args.out} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
