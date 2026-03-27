"""
Single-pass inference: resize → YOLO → rule-based waste classification.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import cv2

from classifier import ClassificationResult, classify_detections
from detector import Detection, detect_objects, detections_to_json
from image_loader import resize_max_side
from learned_classifier import predict_waste_class_from_bgr


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    ix1 = max(0, min(int(round(x1)), w - 1))
    iy1 = max(0, min(int(round(y1)), h - 1))
    ix2 = max(0, min(int(round(x2)), w - 1))
    iy2 = max(0, min(int(round(y2)), h - 1))
    if ix2 < ix1:
        ix1, ix2 = ix2, ix1
    if iy2 < iy1:
        iy1, iy2 = iy2, iy1
    return ix1, iy1, ix2, iy2


def _infer_bin_color_override(
    bgr: np.ndarray,
    dets: List[Detection],
    *,
    min_area_px: int = 32 * 32,
) -> tuple[str | None, float, str | None]:
    """
    Heuristic: if a (detected) trash can region is dominantly green/blue, override stream.

    Returns (override_classification, color_confidence, extra_reasoning).
    """
    if bgr is None or bgr.size == 0:
        return None, 0.0, None

    h, w = bgr.shape[:2]

    # Prefer the highest-confidence "trash can" detection.
    trash_dets = [d for d in dets if str(d.label).strip().lower() in {"trash can", "trashcan"}]
    # Only run color override when a trash can is actually detected.
    # Using full-image fallback can produce false overrides on UI screenshots
    # or scenes where non-bin colored regions dominate.
    roi = None
    if trash_dets:
        best = max(trash_dets, key=lambda d: float(d.confidence))
        x1, y1, x2, y2 = _clip_xyxy(*best.bbox_xyxy, w=w, h=h)
        if (x2 - x1) * (y2 - y1) >= min_area_px:
            roi = bgr[y1:y2, x1:x2]
    else:
        return None, 0.0, None

    if roi is None or roi.size == 0:
        return None, 0.0, None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # HSV thresholds are intentionally broad to handle lighting variance.
    # We compute dominance among "colored" pixels (green+blue), not the whole ROI,
    # because bins often occupy a minority of the crop and backgrounds are neutral.
    green_mask = cv2.inRange(hsv, (30, 25, 25), (95, 255, 255))
    blue_mask = cv2.inRange(hsv, (85, 25, 25), (150, 255, 255))

    total = float(hsv.shape[0] * hsv.shape[1])
    if total <= 0:
        return None, 0.0, None

    g = float(np.count_nonzero(green_mask))
    b = float(np.count_nonzero(blue_mask))
    colored = g + b
    colored_frac = colored / total
    if colored <= 0:
        return None, 0.0, None

    green_share = g / colored
    blue_share = b / colored

    # Require: enough colored pixels, and a clear winner among colored pixels.
    # These thresholds are designed for real photos with bags/shadows.
    if colored_frac < 0.08:
        return None, max(g / total, b / total), None

    if green_share >= 0.65 and (green_share - blue_share) >= 0.25:
        conf = min(1.0, 0.4 + 0.6 * green_share) * min(1.0, 0.6 + colored_frac)
        return (
            "Biodegradable",
            float(conf),
            f"Trashcan color heuristic: green-dominant (green_share={green_share:.2f}, colored_frac={colored_frac:.2f}).",
        )

    if blue_share >= 0.65 and (blue_share - green_share) >= 0.25:
        conf = min(1.0, 0.4 + 0.6 * blue_share) * min(1.0, 0.6 + colored_frac)
        return (
            "Non-biodegradable",
            float(conf),
            f"Trashcan color heuristic: blue-dominant (blue_share={blue_share:.2f}, colored_frac={colored_frac:.2f}).",
        )

    return None, max(g / total, b / total), None


def _infer_organic_scene_hint(bgr: np.ndarray) -> tuple[bool, float, str]:
    """
    Heuristic for food/organic-heavy piles (brown+green dominance).
    Helps avoid overconfident non-bio predictions on mixed/organic scenes.
    """
    if bgr is None or bgr.size == 0:
        return False, 0.0, ""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    total = float(hsv.shape[0] * hsv.shape[1])
    if total <= 0:
        return False, 0.0, ""

    # Brown-ish decomposed organic tones
    brown_mask = cv2.inRange(hsv, (8, 35, 25), (28, 255, 255))
    # Green-ish organic tones
    green_mask = cv2.inRange(hsv, (30, 25, 25), (95, 255, 255))

    brown_ratio = float(np.count_nonzero(brown_mask)) / total
    green_ratio = float(np.count_nonzero(green_mask)) / total
    organic_ratio = brown_ratio + green_ratio

    is_organic = organic_ratio >= 0.24 and brown_ratio >= 0.10
    if not is_organic:
        return False, organic_ratio, ""
    reason = (
        f"Organic-scene heuristic: brown/green dominance detected "
        f"(brown={brown_ratio:.2f}, green={green_ratio:.2f})."
    )
    return True, organic_ratio, reason


def run_pipeline(
    bgr: np.ndarray,
    *,
    quality_factor: float,
    conf_threshold: float,
    model_name: str,
    include_boxes: bool,
    max_side: int = 1280,
) -> Tuple[ClassificationResult, np.ndarray, List[Detection]]:
    """One YOLO forward pass; returns result tensor image (possibly resized), and raw detections."""
    bgr_in, _ = resize_max_side(bgr, max_side=max_side)
    dets = detect_objects(
        bgr_in,
        conf_threshold=conf_threshold,
        model_name=model_name,
    )
    result = classify_detections(dets, quality_factor=quality_factor)

    # Learned model (if checkpoint exists): can improve on cluttered scenes.
    learned_pred = predict_waste_class_from_bgr(bgr_in)
    if learned_pred:
        learned_cls, learned_conf, learned_reason = learned_pred
        # If heuristic output is unknown/mixed with low confidence, trust learned model.
        # Otherwise, blend conservatively and keep stable behavior.
        if (result.classification in {"Unknown", "Mixed"} and float(result.confidence) < 0.6) or (
            learned_cls == result.classification and learned_cls != "Unknown"
        ):
            det_max = max((float(d.confidence) for d in dets), default=0.0)
            det_count = len(dets)
            # Conservative confidence merge to avoid extreme overconfidence
            # on weak/noisy detections.
            merged_conf = 0.65 * float(result.confidence) + 0.35 * (float(learned_conf) * float(quality_factor))
            if det_count <= 1 and det_max < 0.45:
                merged_conf = min(merged_conf, 0.72)
            result = ClassificationResult(
                classification=learned_cls,
                confidence=round(min(0.95, max(0.0, merged_conf)), 4),
                detected_objects=result.detected_objects,
                reasoning=f"{learned_reason}. {result.reasoning}",
                detections_detail=result.detections_detail,
            )

    # Optional override: map bin color → stream.
    override, color_conf, extra_reason = _infer_bin_color_override(bgr_in, dets)
    if override:
        merged_conf = max(float(result.confidence), float(color_conf) * float(quality_factor))
        reasoning = result.reasoning
        if extra_reason:
            reasoning = f"{extra_reason} {reasoning}"
        result = ClassificationResult(
            classification=override,
            confidence=round(float(min(1.0, merged_conf)), 4),
            detected_objects=result.detected_objects,
            reasoning=reasoning,
            detections_detail=result.detections_detail,
        )

    # If scene looks strongly organic, avoid forcing non-bio on uncertain outputs.
    organic_hit, organic_strength, organic_reason = _infer_organic_scene_hint(bgr_in)
    if organic_hit and result.classification == "Non-biodegradable" and float(result.confidence) < 0.75:
        boosted = min(0.95, max(float(result.confidence), 0.45 + organic_strength * 0.8))
        result = ClassificationResult(
            classification="Biodegradable",
            confidence=round(float(boosted), 4),
            detected_objects=result.detected_objects,
            reasoning=f"{organic_reason} {result.reasoning}",
            detections_detail=result.detections_detail,
        )
    detail = detections_to_json(dets) if include_boxes else None
    full = ClassificationResult(
        classification=result.classification,
        confidence=result.confidence,
        detected_objects=result.detected_objects,
        reasoning=result.reasoning,
        detections_detail=detail,
    )
    return full, bgr_in, dets
