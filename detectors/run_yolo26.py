"""Run YOLO26x on the demo video and dump an annotated MP4 + JSON detections."""
import argparse, json, os
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/IMG_2327.MOV")
    ap.add_argument("--weights", default="weights/yolo26x.pt")
    ap.add_argument("--out", default="outputs/yolo26")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--vid-stride", type=int, default=1)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    results = model.predict(
        source=args.video,
        conf=args.conf,
        vid_stride=args.vid_stride,
        save=True,
        project=args.out,
        name="run",
        stream=True,
    )

    detections = []
    for i, r in enumerate(results):
        frame = []
        if r.boxes is not None:
            for b in r.boxes:
                frame.append({
                    "cls": int(b.cls.item()),
                    "name": model.names[int(b.cls.item())],
                    "conf": float(b.conf.item()),
                    "xyxy": [float(x) for x in b.xyxy[0].tolist()],
                })
        detections.append({"frame": i, "boxes": frame})

    with open(os.path.join(args.out, "detections.json"), "w") as f:
        json.dump(detections, f)
    print(f"wrote {len(detections)} frames of detections to {args.out}/")


if __name__ == "__main__":
    main()
