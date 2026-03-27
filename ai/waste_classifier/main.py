#!/usr/bin/env python3
"""
CLI entry: load image → YOLOv8 detect → waste classifier → JSON stdout.

Optional: draw bounding boxes, run on webcam (real-time with stride).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from classifier import to_contract_dict
from detector import draw_boxes
from image_loader import load_image, load_image_from_array
from pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Waste type classification from images (YOLOv8 + rules).")
    p.add_argument("image", nargs="?", help="Path to input image (JPEG/PNG/…)")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights (e.g. yolov8n.pt)")
    p.add_argument("--annotate", metavar="OUT", help="Write image with bounding boxes to OUT")
    p.add_argument("--boxes", action="store_true", help="Include per-detection bboxes in JSON")
    p.add_argument("--webcam", action="store_true", help="Classify from default webcam (press q to quit)")
    p.add_argument("--webcam-interval", type=float, default=0.5, help="Seconds between inferences")
    args = p.parse_args(argv)

    if args.webcam:
        return _run_webcam(args)

    if not args.image:
        p.error("Provide an image path or use --webcam")

    loaded = load_image(args.image)
    result, bgr_model, dets = run_pipeline(
        loaded.bgr,
        quality_factor=loaded.quality_factor,
        conf_threshold=args.conf,
        model_name=args.model,
        include_boxes=args.boxes or bool(args.annotate),
    )

    if args.annotate:
        vis = draw_boxes(bgr_model, dets, labels=True)
        out_path = Path(args.annotate)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)

    payload = to_contract_dict(result, include_boxes=args.boxes or bool(args.annotate))
    print(json.dumps(payload, indent=2))
    return 0


def _run_webcam(args) -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam 0", file=sys.stderr)
        return 1

    window = "Waste classifier (q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    last_print = 0.0
    last_vis = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            now = time.time()
            if now - last_print >= args.webcam_interval:
                loaded = load_image_from_array(frame, path="webcam")
                result, bgr_model, dets = run_pipeline(
                    loaded.bgr,
                    quality_factor=loaded.quality_factor,
                    conf_threshold=args.conf,
                    model_name=args.model,
                    include_boxes=True,
                )
                vis = draw_boxes(bgr_model, dets, labels=True)
                payload = to_contract_dict(result, include_boxes=True)
                if vis.shape[:2] != (h, w):
                    vis = cv2.resize(vis, (w, h), interpolation=cv2.INTER_LINEAR)
                line = f"{payload['classification']} ({payload['confidence']:.2f})"
                cv2.putText(
                    vis,
                    line,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
                last_vis = vis
                print(json.dumps(payload))
                sys.stdout.flush()
                last_print = now

            display = last_vis if last_vis is not None else frame
            cv2.imshow(window, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
