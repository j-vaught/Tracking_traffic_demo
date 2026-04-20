"""
Render two LoRAT-lock videos from tracks.json + the 640x640 source video.

  forward  : blue box from frame >= track.lock onward (no box before lock)
  backprop : blue box for EVERY frame a track was alive, if it ever locked
             (back-propagates the lock label through the track's history,
              revealing frames where detectors failed to label the object)

Usage:
    python trackers/render_lorat_videos.py \
        --video outputs/640_input.mp4 \
        --tracks outputs/lorat_tracks.json \
        --forward-out outputs/640_lorat_forward.mp4 \
        --backprop-out outputs/640_lorat_backprop.mp4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

BLUE = (255, 80, 0)   # BGR — vivid blue
THICK = 2


def draw_box(frame, box, color=BLUE, thickness=THICK):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def render(video_in, video_out, per_frame_tracks):
    """per_frame_tracks: {frame_idx: [box, box, ...]}"""
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_in}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    t0 = time.time()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for box in per_frame_tracks.get(i, ()):
            draw_box(frame, box)
        writer.write(frame)
        i += 1
        if i % 2000 == 0:
            dt = time.time() - t0
            print(f"  {Path(video_out).name}: {i}/{nfr}  {i/dt:.0f} fps")
    cap.release()
    writer.release()
    print(f"wrote {i} frames -> {video_out} in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--forward-out", required=True)
    ap.add_argument("--backprop-out", required=True)
    args = ap.parse_args()

    with open(args.tracks) as f:
        data = json.load(f)
    per_track = data["per_track"]
    total = data["total_frames"]

    forward_per_frame: dict[int, list[list[float]]] = {}
    backprop_per_frame: dict[int, list[list[float]]] = {}

    for tid, tr in per_track.items():
        lock = tr["lock"]
        if lock is None:
            continue  # never locked -> neither video draws it
        for frame_str, entry in tr["hist"].items():
            fidx = int(frame_str)
            box = entry["box"]
            backprop_per_frame.setdefault(fidx, []).append(box)
            if fidx >= lock:
                forward_per_frame.setdefault(fidx, []).append(box)

    print(f"tracks: {len(per_track)} total, "
          f"{sum(1 for t in per_track.values() if t['lock'] is not None)} locked")
    print(f"forward box frames: {len(forward_per_frame)}")
    print(f"backprop box frames: {len(backprop_per_frame)}")

    Path(args.forward_out).parent.mkdir(parents=True, exist_ok=True)
    render(args.video, args.forward_out, forward_per_frame)
    render(args.video, args.backprop_out, backprop_per_frame)


if __name__ == "__main__":
    main()
