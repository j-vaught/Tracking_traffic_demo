"""
Run YOLO26x and/or RF-DETR 2XLarge on a pre-resized video and write
annotated MP4s filtered to traffic classes.

Usage:
    python detectors/run_640_experiment.py --detector yolo   \
        --video outputs/640_input.mp4 --out outputs/640_yolo.mp4
    python detectors/run_640_experiment.py --detector rfdetr \
        --video outputs/640_input.mp4 --out outputs/640_detr.mp4

Classes default to person(0) + car(2). COCO traffic set: 0 2 3 5 7.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 6: "train", 7: "truck",
}


def color_for_class(cid: int):
    # Stable deterministic color by class id.
    rng = np.random.default_rng(cid * 1337 + 7)
    return tuple(int(c) for c in rng.integers(64, 255, size=3))


def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (int(x1), int(y1) - th - 4), (int(x1) + tw + 2, int(y1)), color, -1)
    cv2.putText(frame, label, (int(x1) + 1, int(y1) - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def run_yolo(video, out, classes, conf):
    from ultralytics import YOLO
    model = YOLO("weights/yolo26x.pt")
    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    records = []
    t0 = time.time()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, classes=classes, conf=conf, verbose=False)[0]
        boxes = []
        if results.boxes is not None:
            for b in results.boxes:
                cid = int(b.cls.item())
                score = float(b.conf.item())
                x1, y1, x2, y2 = [float(x) for x in b.xyxy[0].tolist()]
                name = COCO_NAMES.get(cid, str(cid))
                draw_box(frame, x1, y1, x2, y2, f"{name} {score:.2f}", color_for_class(cid))
                boxes.append({"cls": cid, "name": name, "conf": score, "xyxy": [x1, y1, x2, y2]})
        records.append({"frame": i, "boxes": boxes})
        writer.write(frame)
        i += 1
        if i % 500 == 0:
            elapsed = time.time() - t0
            print(f"  yolo26x: {i}/{nframes} frames ({i / elapsed:.1f} fps)")

    cap.release()
    writer.release()
    print(f"yolo26x: wrote {i} frames to {out} in {time.time() - t0:.1f}s")
    return records


def run_rfdetr(video, out, classes, conf):
    from rfdetr import RFDETR2XLarge
    model = RFDETR2XLarge()
    model.optimize_for_inference()

    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    classes_set = set(classes) if classes else None
    records = []
    t0 = time.time()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = model.predict(frame, threshold=conf)
        # supervision.Detections has .xyxy (N,4), .confidence (N,), .class_id (N,)
        boxes = []
        if dets.xyxy is not None:
            for j in range(len(dets.xyxy)):
                cid = int(dets.class_id[j])
                if classes_set is not None and cid not in classes_set:
                    continue
                score = float(dets.confidence[j])
                x1, y1, x2, y2 = [float(x) for x in dets.xyxy[j]]
                name = COCO_NAMES.get(cid, str(cid))
                draw_box(frame, x1, y1, x2, y2, f"{name} {score:.2f}", color_for_class(cid))
                boxes.append({"cls": cid, "name": name, "conf": score,
                              "xyxy": [x1, y1, x2, y2]})
        records.append({"frame": i, "boxes": boxes})
        writer.write(frame)
        i += 1
        if i % 500 == 0:
            elapsed = time.time() - t0
            print(f"  rfdetr2xl: {i}/{nframes} frames ({i / elapsed:.1f} fps)")

    cap.release()
    writer.release()
    print(f"rfdetr2xl: wrote {i} frames to {out} in {time.time() - t0:.1f}s")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=["yolo", "rfdetr"], required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", type=int, nargs="*", default=[0, 2],
                    help="COCO class IDs (default: 0 person, 2 car). "
                         "Traffic set: 0 2 3 5 7")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.detector == "yolo":
        records = run_yolo(args.video, args.out, args.classes, args.conf)
    else:
        records = run_rfdetr(args.video, args.out, args.classes, args.conf)

    json_out = Path(args.out).with_suffix(".json")
    with open(json_out, "w") as f:
        json.dump(records, f)
    print(f"detections JSON -> {json_out}")


if __name__ == "__main__":
    main()
