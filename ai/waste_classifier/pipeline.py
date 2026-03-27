"""
Keras-first waste classification pipeline.

This repo used to include a YOLO + rule-based pipeline, but the user has removed
the old model and wants to rely on the provided Keras model instead.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2

from keras_classifier import predict_from_bgr as keras_predict_from_bgr


def resize_max_side(bgr: np.ndarray, *, max_side: int = 1280) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = float(max_side) / float(m)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    nh, nw = max(1, nh), max(1, nw)
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def _infer_bin_color_override(
    bgr: np.ndarray,
) -> tuple[str | None, float, str | None]:
    """
    Heuristic: if the image is dominantly green/blue, infer bin stream.

    Returns (override_classification, color_confidence, extra_reasoning).
    """
    if bgr is None or bgr.size == 0:
        return None, 0.0, None

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # HSV thresholds are intentionally broad to handle lighting variance.
    # We compute dominance among "colored" pixels (green+blue), not the whole image,
    # because backgrounds are often neutral.
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
    if colored_frac < 0.06:
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


def classify_bgr(
    bgr: np.ndarray,
    *,
    quality_factor: float = 1.0,
    max_side: int = 1280,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Contract-compatible classifier used by the FastAPI `/classify` endpoint.

    Returns (payload_dict, image_used).
    """
    bgr_in = resize_max_side(bgr, max_side=max_side)

    override, override_conf, override_reason = _infer_bin_color_override(bgr_in)
    keras_pred = keras_predict_from_bgr(bgr_in)

    classification = "Unknown"
    confidence = 0.0
    reasoning_parts: list[str] = []
    raw: Dict[str, Any] = {}

    if keras_pred:
        k_cls, k_conf, k_reason = keras_pred
        raw["keras"] = {"mapped": k_cls, "confidence": float(k_conf), "reason": k_reason}
        if k_cls != "Unknown":
            classification = k_cls
            confidence = float(min(0.95, max(0.0, float(k_conf) * float(quality_factor))))
            reasoning_parts.append(f"{k_reason}. (Primary Keras model)")
    else:
        raw["keras"] = {"available": False}
        reasoning_parts.append("Keras model not available (missing model files or TensorFlow).")

    # Apply explicit bin color override last (user requirement).
    if override:
        merged = max(float(confidence), float(override_conf) * float(quality_factor))
        classification = override
        confidence = float(min(0.99, merged))
        if override_reason:
            reasoning_parts.insert(0, override_reason)

    # Organic hint: only when still unknown or weak non-bio.
    organic_hit, organic_strength, organic_reason = _infer_organic_scene_hint(bgr_in)
    if organic_hit and (classification == "Unknown" or (classification == "Non-biodegradable" and confidence < 0.70)):
        classification = "Biodegradable"
        confidence = float(min(0.95, max(confidence, 0.45 + organic_strength * 0.8)))
        reasoning_parts.insert(0, organic_reason)

    payload = {
        "classification": classification,
        "confidence": round(float(confidence), 4),
        "detected_objects": [],
        "reasoning": " ".join([p for p in reasoning_parts if p]),
        "detections_detail": None,
        "provider": "keras",
        "raw": raw or None,
    }
    return payload, bgr_in
