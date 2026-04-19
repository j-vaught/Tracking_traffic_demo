"""Run RF-DETR (2XL by default) on the demo video."""
import argparse, json, os
from pathlib import Path

import cv2
from rfdetr import RFDETR2XL, RFDETRLarge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/IMG_2327.MOV")
    ap.add_argument("--out", default="outputs/rfdetr")
    ap.add_argument("--variant", choices=["2xl", "large"], default="2xl")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--vid-stride", type=int, default=1)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model = (RFDETR2XL if args.variant == "2xl" else RFDETRLarge)()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"failed to open {args.video}")

    detections = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.vid_stride == 0:
            dets = model.predict(frame, threshold=args.threshold)
            detections.append({
                "frame": i,
                "boxes": [
                    {
                        "cls": int(d.class_id),
                        "conf": float(d.confidence),
                        "xyxy": list(map(float, d.xyxy)),
                    }
                    for d in dets
                ],
            })
        i += 1
    cap.release()

    with open(os.path.join(args.out, "detections.json"), "w") as f:
        json.dump(detections, f)
    print(f"wrote {len(detections)} frames to {args.out}/detections.json")


if __name__ == "__main__":
    main()
