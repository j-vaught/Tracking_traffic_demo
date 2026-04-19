"""
Multi-GPU orchestrator: splits frame range across N GPUs, runs one subprocess
per GPU, then ffmpeg-concats the chunk MP4s + merges chunk JSONs.

Usage:
    python scripts/detect_mgpu.py --detector yolo \
        --video outputs/640_input.mp4 --out outputs/640_yolo.mp4 \
        --gpus 0,1,2,3 --batch 8 --engine weights/yolo26x_b8_fp16.engine
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2


REPO = Path(__file__).resolve().parent.parent


def nframes(video):
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def split_ranges(total, k):
    # Give first (total % k) workers one extra frame, for even work distribution.
    base, rem = divmod(total, k)
    ranges, cur = [], 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        ranges.append((cur, cur + size))
        cur += size
    return ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=["yolo", "rfdetr"], required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1,2,3",
                    help="Comma-separated GPU indices")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--classes", type=int, nargs="*", default=[0, 2])
    ap.add_argument("--engine", default=None)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--parts-dir", default=None,
                    help="Where to write chunk MP4s (default: outputs/parts_<timestamp>)")
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    video = str(Path(args.video).resolve())
    out = str(Path(args.out).resolve())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    parts_dir = Path(args.parts_dir) if args.parts_dir else \
        Path(out).parent / f"parts_{int(time.time())}"
    parts_dir.mkdir(parents=True, exist_ok=True)

    total = nframes(video)
    ranges = split_ranges(total, len(gpus))
    print(f"[mgpu] {args.detector} | {total} frames | {len(gpus)} GPUs | "
          f"batch={args.batch} fp16={args.fp16}")
    for g, (s, e) in zip(gpus, ranges):
        print(f"  gpu {g}: frames [{s}, {e}) ({e-s} frames)")

    # Spawn workers in parallel
    procs = []
    t0 = time.time()
    for g, (s, e) in zip(gpus, ranges):
        chunk_out = str(parts_dir / f"chunk_{g}.mp4")
        cmd = [
            str(REPO / ".venv-detectors" / "bin" / "python"),
            str(REPO / "detectors" / "run_fast.py"),
            "--detector", args.detector,
            "--video", video,
            "--out", chunk_out,
            "--start", str(s),
            "--end", str(e),
            "--batch", str(args.batch),
            "--conf", str(args.conf),
            "--gpu-tag", str(g),
        ]
        if args.classes:
            cmd += ["--classes", *map(str, args.classes)]
        if not args.fp16:
            cmd.append("--no-fp16")
        if args.engine:
            cmd += ["--engine", args.engine]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": g}
        procs.append((g, subprocess.Popen(cmd, env=env)))

    # Wait for all
    exit_codes = []
    for g, p in procs:
        rc = p.wait()
        print(f"  gpu {g} exit code {rc}")
        exit_codes.append(rc)
    if any(rc != 0 for rc in exit_codes):
        print("ERROR: one or more workers failed", file=sys.stderr)
        sys.exit(1)

    # Concat MP4s via ffmpeg (same codec/params → -c copy)
    listing = parts_dir / "concat.txt"
    with open(listing, "w") as f:
        for g, _ in procs:
            f.write(f"file '{parts_dir / f'chunk_{g}.mp4'}'\n")
    concat_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                  "-f", "concat", "-safe", "0", "-i", str(listing),
                  "-c", "copy", out]
    subprocess.check_call(concat_cmd)

    # Merge chunk JSONs into one ordered list
    merged = []
    for g, _ in procs:
        chunk_json = parts_dir / f"chunk_{g}.json"
        if chunk_json.exists():
            with open(chunk_json) as f:
                merged.extend(json.load(f))
    merged.sort(key=lambda d: d["frame"])
    with open(Path(out).with_suffix(".json"), "w") as f:
        json.dump(merged, f)

    if not args.keep_parts:
        shutil.rmtree(parts_dir)

    dt = time.time() - t0
    print(f"[mgpu] done in {dt:.1f}s ({total/dt:.1f} fps effective). "
          f"Output: {out}")


if __name__ == "__main__":
    main()
