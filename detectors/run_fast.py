"""
Fast detector runner: FP16 + TensorRT/torch.compile + batching, single GPU.

Meant to be invoked by scripts/detect_mgpu.py as one process per GPU.
Processes a contiguous frame range [--start, --end) and writes a chunk MP4.

Usage:
    CUDA_VISIBLE_DEVICES=0 python detectors/run_fast.py \
        --detector yolo --video outputs/640_input.mp4 \
        --out outputs/parts/640_yolo_0.mp4 --start 0 --end 3828 \
        --engine weights/yolo26x_b8_fp16.engine --batch 8
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

# YOLO / Ultralytics: 0-based contiguous 80 classes.
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
              5: "bus", 6: "train", 7: "truck"}

# RF-DETR returns raw COCO-91 IDs (1-based, with gaps). Only the classes we
# care about for a traffic demo are listed here; extend as needed.
COCO_91_NAMES = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
                 5: "airplane", 6: "bus", 7: "train", 8: "truck",
                 9: "boat", 10: "traffic light", 11: "fire hydrant"}

# Translate a COCO-80 id (what the user passes via --classes, YOLO convention)
# to its COCO-91 id (what RF-DETR returns). Identity for classes 0..10; beyond
# that requires the full table if you start filtering other classes.
COCO80_TO_91 = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9,
                9: 10, 10: 11}


def color_for_class(cid):
    rng = np.random.default_rng(cid * 1337 + 7)
    return tuple(int(c) for c in rng.integers(64, 255, size=3))


# BGR. Bracket thresholds are inclusive of the lower edge.
# Adjust these in one place if the brackets change later.
CONF_BRACKETS = (
    (0.75, (0, 255, 0)),    # green  : score >= 0.75
    (0.35, (0, 255, 255)),  # yellow : 0.35 <= score < 0.75
    (0.10, (0, 0, 255)),    # red    : 0.10 <= score < 0.35
)


def color_for_conf(score):
    for threshold, color in CONF_BRACKETS:
        if score >= threshold:
            return color
    return None  # below the lowest bracket -> don't draw


def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (int(x1), int(y1) - th - 4),
                  (int(x1) + tw + 2, int(y1)), color, -1)
    cv2.putText(frame, label, (int(x1) + 1, int(y1) - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_box_only(frame, x1, y1, x2, y2, color):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


def iter_frames(video, start, end):
    # CAP_PROP_POS_FRAMES is unreliable on some mp4 codecs; grab-skip is exact.
    cap = cv2.VideoCapture(video)
    for _ in range(start):
        if not cap.grab():
            cap.release()
            return
    i = start
    while i < end:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()


def run_yolo(args):
    from ultralytics import YOLO
    weights = args.engine if args.engine else "weights/yolo26x.pt"
    model = YOLO(weights)

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    records = []
    batch_frames, batch_indices = [], []
    classes = args.classes
    t0 = time.time()

    def flush():
        if not batch_frames:
            return
        # Pad to batch size so a compiled/engine model doesn't barf on stragglers.
        pad = args.batch - len(batch_frames)
        padded = batch_frames + ([batch_frames[-1]] * pad) if pad > 0 else batch_frames
        results = model.predict(padded, classes=classes, conf=args.conf,
                                imgsz=w, half=args.fp16, verbose=False)
        results = results[:len(batch_frames)]
        for idx, frame, r in zip(batch_indices, batch_frames, results):
            boxes = []
            if r.boxes is not None:
                for b in r.boxes:
                    cid = int(b.cls.item())
                    score = float(b.conf.item())
                    x1, y1, x2, y2 = [float(x) for x in b.xyxy[0].tolist()]
                    name = COCO_NAMES.get(cid, str(cid))
                    if args.color_conf:
                        color = color_for_conf(score)
                        if color is not None:
                            draw_box_only(frame, x1, y1, x2, y2, color)
                    else:
                        draw_box(frame, x1, y1, x2, y2, f"{name} {score:.2f}",
                                 color_for_class(cid))
                    boxes.append({"cls": cid, "name": name, "conf": score,
                                  "xyxy": [x1, y1, x2, y2]})
            records.append({"frame": idx, "boxes": boxes})
            writer.write(frame)
        batch_frames.clear()
        batch_indices.clear()

    for i, frame in iter_frames(args.video, args.start, args.end):
        batch_frames.append(frame)
        batch_indices.append(i)
        if len(batch_frames) == args.batch:
            flush()
    flush()
    writer.release()
    dt = time.time() - t0
    n = args.end - args.start
    print(f"yolo gpu={args.gpu_tag} start={args.start} end={args.end} "
          f"wrote {n} frames in {dt:.1f}s ({n/dt:.1f} fps)")
    return records


def run_rfdetr(args):
    import torch
    from rfdetr import RFDETR2XLarge
    dtype = torch.float16 if args.fp16 else torch.float32
    model = RFDETR2XLarge()
    # torch.compile + FP16 batched inference
    model.optimize_for_inference(compile=True, batch_size=args.batch, dtype=dtype)

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    records = []
    # Translate user's YOLO-convention (COCO-80) class IDs -> RF-DETR's COCO-91.
    classes_set = {COCO80_TO_91.get(c, c) for c in args.classes} if args.classes else None
    batch_frames, batch_indices = [], []
    t0 = time.time()

    def flush():
        if not batch_frames:
            return
        # RF-DETR is compiled for exactly batch_size; pad the last short batch.
        pad = args.batch - len(batch_frames)
        padded = batch_frames + ([batch_frames[-1]] * pad) if pad > 0 else batch_frames
        batch_dets = model.predict(padded, threshold=args.conf)
        if not isinstance(batch_dets, list):
            batch_dets = [batch_dets]
        batch_dets = batch_dets[:len(batch_frames)]
        for idx, frame, dets in zip(batch_indices, batch_frames, batch_dets):
            boxes = []
            if dets.xyxy is not None and len(dets.xyxy) > 0:
                for j in range(len(dets.xyxy)):
                    cid = int(dets.class_id[j])
                    if classes_set is not None and cid not in classes_set:
                        continue
                    score = float(dets.confidence[j])
                    x1, y1, x2, y2 = [float(x) for x in dets.xyxy[j]]
                    # RF-DETR uses COCO-91 IDs; name via the matching table.
                    name = COCO_91_NAMES.get(cid, str(cid))
                    if args.color_conf:
                        color = color_for_conf(score)
                        if color is not None:
                            draw_box_only(frame, x1, y1, x2, y2, color)
                    else:
                        draw_box(frame, x1, y1, x2, y2,
                                 f"{name} {score:.2f}", color_for_class(cid))
                    boxes.append({"cls": cid, "name": name, "conf": score,
                                  "xyxy": [x1, y1, x2, y2]})
            records.append({"frame": idx, "boxes": boxes})
            writer.write(frame)
        batch_frames.clear()
        batch_indices.clear()

    for i, frame in iter_frames(args.video, args.start, args.end):
        batch_frames.append(frame)
        batch_indices.append(i)
        if len(batch_frames) == args.batch:
            flush()
    flush()
    writer.release()
    dt = time.time() - t0
    n = args.end - args.start
    print(f"rfdetr gpu={args.gpu_tag} start={args.start} end={args.end} "
          f"wrote {n} frames in {dt:.1f}s ({n/dt:.1f} fps)")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=["yolo", "rfdetr"], required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--classes", type=int, nargs="*", default=[0, 1, 2, 3, 5, 7],
                    help="COCO class IDs. Default: person+all road vehicles "
                         "(0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck)")
    ap.add_argument("--color-conf", action="store_true",
                    help="Draw boxes only (no label), color by confidence brackets")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--engine", default=None,
                    help="Path to YOLO TensorRT .engine (yolo backend only)")
    ap.add_argument("--gpu-tag", default="?",
                    help="Tag for log lines (which GPU index this proc is on)")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.detector == "yolo":
        records = run_yolo(args)
    else:
        records = run_rfdetr(args)
    with open(Path(args.out).with_suffix(".json"), "w") as f:
        json.dump(records, f)


if __name__ == "__main__":
    main()
