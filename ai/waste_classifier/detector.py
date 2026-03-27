"""
Object detection using Ultralytics YOLOv8 (COCO pre-trained).

Returns normalized detection dicts with bounding boxes for optional visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, List, Sequence

import numpy as np

# Lazy singleton avoids reloading weights on every request
_model_lock = Lock()
_model_instance: Any = None
_model_name: str | None = None


@dataclass
class Detection:
    """Single detection after NMS."""

    label: str
    confidence: float
    # xyxy in pixel coords of the image passed to the model (possibly resized)
    bbox_xyxy: tuple[float, float, float, float]


def _get_yolo(model_name: str = "yolov8n.pt"):
    global _model_instance, _model_name
    with _model_lock:
        if _model_instance is None or _model_name != model_name:
            from ultralytics import YOLO  # import here to speed CLI --help

            root = Path(__file__).resolve().parent
            weights = root / model_name
            if weights.is_file():
                _model_instance = YOLO(str(weights))
            else:
                _model_instance = YOLO(model_name)  # downloads to ultralytics cache
            _model_name = model_name
    return _model_instance


def detect_objects(
    bgr: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    model_name: str = "yolov8n.pt",
    imgsz: int = 640,
) -> List[Detection]:
    """
    Run YOLOv8 inference on a BGR image.

    conf_threshold: minimum confidence for a box to be kept.
    """
    model = _get_yolo(model_name)
    # stream=False -> single Result
    results = model.predict(
        source=bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
    )
    if not results:
        return []
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    names = r0.names
    out: List[Detection] = []
    boxes = r0.boxes
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy().tolist()
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        label = names.get(cls_id, f"class_{cls_id}")
        out.append(
            Detection(
                label=label,
                confidence=conf,
                bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
            )
        )
    return out


def detections_to_json(dets: Sequence[Detection]) -> List[dict]:
    """Serialize detections for API / JSON output."""
    return [
        {
            "label": d.label,
            "confidence": round(d.confidence, 4),
            "bbox": {
                "x1": round(d.bbox_xyxy[0], 2),
                "y1": round(d.bbox_xyxy[1], 2),
                "x2": round(d.bbox_xyxy[2], 2),
                "y2": round(d.bbox_xyxy[3], 2),
            },
        }
        for d in dets
    ]


def draw_boxes(bgr: np.ndarray, detections: Sequence[Detection], labels: bool = True) -> np.ndarray:
    """
    Draw bounding boxes on a copy of the image (BGR).
    Used for CLI --annotate and optional API export.
    """
    import cv2

    vis = bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(round(x)) for x in d.bbox_xyxy]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        if labels:
            text = f"{d.label} {d.confidence:.2f}"
            cv2.putText(
                vis,
                text,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                1,
                cv2.LINE_AA,
            )
    return vis
