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


# Same brackets the detector color videos use: green/yellow/red by conf.
CONF_BRACKETS = (
    (0.75, (0, 255, 0)),    # green
    (0.35, (0, 255, 255)),  # yellow
    (0.10, (0, 0, 255)),    # red
)


def color_for_conf(score):
    for threshold, color in CONF_BRACKETS:
        if score >= threshold:
            return color
    return None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    au = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    bu = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    u = au + bu - inter
    return inter / u if u > 0 else 0.0


def project_track_to_yolo(track_box_640, crop_rect_1080, yolo_size=640):
    """Map a box in main-640 coords -> 1080p -> viewport crop -> 640x640 YOLO.
    Returns (x1, y1, x2, y2) in YOLO-input coords, or None if the 1080p box
    doesn't intersect the crop rect at all."""
    cx1, cy1, cx2, cy2 = crop_rect_1080
    cw = cx2 - cx1
    ch = cy2 - cy1
    if cw <= 0 or ch <= 0:
        return None
    # 640-main -> 1080p (SCALE_X=3.0, SCALE_Y=1.6875)
    bx1 = track_box_640[0] * SCALE_X
    by1 = track_box_640[1] * SCALE_Y
    bx2 = track_box_640[2] * SCALE_X
    by2 = track_box_640[3] * SCALE_Y
    # Intersect with crop rect
    ix1 = max(bx1, cx1); iy1 = max(by1, cy1)
    ix2 = min(bx2, cx2); iy2 = min(by2, cy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    # Map to YOLO input coords
    sx = yolo_size / cw
    sy = yolo_size / ch
    return (
        (ix1 - cx1) * sx,
        (iy1 - cy1) * sy,
        (ix2 - cx1) * sx,
        (iy2 - cy1) * sy,
    )


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
    ap.add_argument("--picker", choices=["proximity", "lifespan"],
                    default="proximity",
                    help="How to pick next target: closest to camera (sweeps "
                         "through clusters) or longest-remaining life")
    ap.add_argument("--in-transit-iou", type=float, default=0.3,
                    help="IoU between a YOLO det in the viewport and an "
                         "unvisited track's projected box to count as a hit")
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
    visited: set[str] = set()
    verify_reason: dict[str, str] = {}   # tid -> "verified" / "timeout" / "lost" / "transit"
    # Per-track running consecutive-hit counter (reset on miss).
    consec: dict[str, int] = {}

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
                    if consec.get(current_target, 0) >= args.verify_consec:
                        done = True
                        reason = "verified"
                    elif frames_on_current >= args.max_hold:
                        done = True
                        reason = "timeout"
                if done:
                    visited.add(current_target)
                    verify_reason.setdefault(current_target, reason)
                    current_target = None
                    frames_on_current = 0
            if current_target is None and active:
                unvisited = [t for t in active if t not in visited]
                if unvisited:
                    if args.picker == "proximity":
                        # Distance from current camera center to track center
                        # in 1080p coords.
                        cam_cx = state[0]
                        cam_cy = state[1]
                        def key_prox(tid: str):
                            entry = per_track[tid]["hist"].get(str(fidx))
                            if entry is None:
                                return (float("inf"), 0)
                            x1, y1, x2, y2 = entry["box"]
                            bx = (x1 + x2) / 2 * SCALE_X
                            by = (y1 + y2) / 2 * SCALE_Y
                            d = ((cam_cx - bx) ** 2 + (cam_cy - by) ** 2) ** 0.5
                            rem = per_track[tid]["died"] - fidx
                            return (d, -rem)
                        current_target = min(unvisited, key=key_prox)
                    else:
                        def rem(tid: str) -> int:
                            return per_track[tid]["died"] - fidx
                        current_target = max(unvisited, key=rem)
                    frames_on_current = 0
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
        # Always run so we can show color-coded boxes on the viewport too.
        max_conf = 0.0
        transit_hits: list[str] = []
        all_dets_yolo = []   # list of (xyxy_640, conf) — all dets at conf>=0.10
        if yolo is not None:
            crop_yolo = out if (args.out_w, args.out_h) == (640, 640) else \
                cv2.resize(out, (640, 640))
            # Lowered floor to 0.10 so we can draw red-bracket boxes too.
            r = yolo.predict(crop_yolo, classes=args.classes,
                             conf=0.10, imgsz=640, half=True, verbose=False)[0]
            if r.boxes is not None and len(r.boxes) > 0:
                max_conf = float(r.boxes.conf.max().item())
                for b in r.boxes:
                    c = float(b.conf.item())
                    all_dets_yolo.append(
                        (tuple(float(x) for x in b.xyxy[0].tolist()), c))
            # Verify threshold = only boxes above verify_conf feed in-transit.
            dets = [(xy, c) for (xy, c) in all_dets_yolo if c >= args.verify_conf]
            # For every unvisited alive track (including current target),
            # see if it projects into the current viewport and matches a
            # high-conf detection there.
            crop_rect_1080 = (x1, y1, x2, y2)
            for tid in active:
                if tid in visited:
                    continue
                entry = per_track[tid]["hist"].get(str(fidx))
                if entry is None:
                    continue
                proj = project_track_to_yolo(entry["box"], crop_rect_1080)
                if proj is None:
                    consec[tid] = 0
                    continue
                # Match: highest-IoU det above threshold.
                best_iou = 0.0
                for db, _ in dets:
                    iou = iou_xyxy(proj, db)
                    if iou > best_iou:
                        best_iou = iou
                if best_iou >= args.in_transit_iou:
                    consec[tid] = consec.get(tid, 0) + 1
                    if tid != current_target and consec[tid] >= args.verify_consec:
                        # Opportunistic verify mid-transit.
                        visited.add(tid)
                        verify_reason.setdefault(tid, "transit")
                        transit_hits.append(tid)
                else:
                    consec[tid] = 0

        # Draw YOLO detections on the viewport, colored by conf bracket.
        if all_dets_yolo:
            sx = args.out_w / 640
            sy = args.out_h / 640
            for (dx1, dy1, dx2, dy2), dc in all_dets_yolo:
                col = color_for_conf(dc)
                if col is None:
                    continue
                cv2.rectangle(out,
                              (int(round(dx1 * sx)), int(round(dy1 * sy))),
                              (int(round(dx2 * sx)), int(round(dy2 * sy))),
                              col, 2)

        if args.overlay:
            label = f"PTZ  f={fidx}  visited={len(visited)}/{len(per_track)}  "
            if current_target is not None:
                label += f"track#{current_target}"
                if args.planner == "verify":
                    c = consec.get(current_target, 0)
                    label += f"  consec={c}/{args.verify_consec}"
                    label += f"  conf={max_conf:.2f}"
                if transit_hits:
                    label += f"  +transit:{','.join(transit_hits)}"
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
