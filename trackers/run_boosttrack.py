"""
Run BoostTrack++ on a video using external detections (YOLO26x or RF-DETR JSON).

Bypasses BoostTrack's bundled YOLOX/Deep-OC-SORT detector pipeline entirely.
No MOT17/20 detector weights needed.

Usage:
    python trackers/run_boosttrack.py \
        --video data/IMG_2327.MOV \
        --detections outputs/yolo26/run/detections.json \
        --out outputs/boosttrack/yolo26 \
        --classes 2 3 5 7   # optional COCO filter (car/moto/bus/truck)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# make trackers/boosttrack importable regardless of cwd. BoostTrack's
# tracker.embedding references `fast_reid.*` and `external.adaptors.*` as if
# the repo root + external/ are both on sys.path (as in its conda setup).
REPO = Path(__file__).resolve().parent / "boosttrack"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external"))

from default_settings import GeneralSettings, BoostTrackSettings  # noqa: E402
from tracker.boost_track import BoostTrack  # noqa: E402


def load_detections(path):
    with open(path) as f:
        data = json.load(f)
    return {d["frame"]: d["boxes"] for d in data}


def detections_for_frame(per_frame, frame_idx, class_filter):
    boxes = per_frame.get(frame_idx, [])
    rows = []
    for b in boxes:
        cls = b.get("cls")
        if class_filter is not None and cls not in class_filter:
            continue
        x1, y1, x2, y2 = b["xyxy"]
        rows.append([x1, y1, x2, y2, b["conf"]])
    if not rows:
        return np.empty((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--detections", required=True, help="detections JSON from YOLO26 or RF-DETR runner")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="COCO class IDs to keep (default: all). Traffic: 2 3 5 7")
    ap.add_argument("--det-thresh", type=float, default=0.35)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Stop after N frames; default: len(detections JSON)")
    ap.add_argument("--ecc", dest="use_ecc", action="store_true", default=True,
                    help="Camera motion compensation (on by default — important for PTZ)")
    ap.add_argument("--no-ecc", dest="use_ecc", action="store_false")
    ap.add_argument("--save-video", dest="save_video", action="store_true", default=True)
    ap.add_argument("--no-save-video", dest="save_video", action="store_false")
    args = ap.parse_args()

    # Resolve user paths BEFORE chdir so relative paths still work.
    args.video = str(Path(args.video).resolve())
    args.detections = str(Path(args.detections).resolve())
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # BoostTrack's ECC cache + config assume cwd == repo root.
    import os as _os
    _os.chdir(REPO)

    # --- BoostTrack config: bypass embedder, keep ECC for PTZ ---
    GeneralSettings.values["use_embedding"] = False
    GeneralSettings.values["use_ecc"] = args.use_ecc
    GeneralSettings.values["det_thresh"] = args.det_thresh
    GeneralSettings.values["dataset"] = "mot17"   # any MOT key works; only affects dlo_boost_coef
    BoostTrackSettings.values["s_sim_corr"] = True  # upstream bug-fix flag

    per_frame = load_detections(args.detections)
    max_frames = args.max_frames if args.max_frames is not None else (max(per_frame) + 1 if per_frame else 0)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(str(out / "tracked.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))

    tracker = BoostTrack(video_name="PTZ_demo")
    class_filter = set(args.classes) if args.classes else None

    tracks_log = []
    idx = 0
    rng = np.random.default_rng(42)
    id_to_color = {}

    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detections_for_frame(per_frame, idx, class_filter)
        # BoostTrack expects (img_tensor [1,C,H,W] in input-space, img_numpy [H,W,C]).
        # We keep detections in original coords by matching both shapes -> scale = 1.
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        outputs = tracker.update(dets, img_tensor, frame, tag="PTZ_demo")

        # BoostTrack returns [x1, y1, x2, y2, id, conf] per row.
        frame_tracks = []
        for row in outputs:
            x1, y1, x2, y2, tid, conf = row[:6]
            tid = int(tid)
            frame_tracks.append({"id": tid, "conf": float(conf),
                                 "xyxy": [float(x1), float(y1), float(x2), float(y2)]})
            if writer is not None:
                color = id_to_color.setdefault(tid, tuple(int(c) for c in rng.integers(64, 255, 3)))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"#{tid}", (int(x1), max(0, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        tracks_log.append({"frame": idx, "tracks": frame_tracks})
        if writer is not None:
            writer.write(frame)
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    with open(out / "tracks.json", "w") as f:
        json.dump(tracks_log, f)
    print(f"wrote {idx} frames of tracks to {out}/")


if __name__ == "__main__":
    main()
