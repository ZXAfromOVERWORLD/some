"""
Load and validate images for the waste classification pipeline.

Uses OpenCV for I/O and optional quality heuristics (blur / low detail)
to scale down confidence when the input is hard to interpret.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class LoadedImage:
    """BGR image array plus metadata used by downstream stages."""

    bgr: np.ndarray
    path: str | None
    blur_score: float  # Laplacian variance; higher = sharper
    quality_factor: float  # 0..1 multiplier for confidence adjustment


def laplacian_blur_score(bgr: np.ndarray) -> float:
    """
    Variance of the Laplacian — common proxy for image sharpness.
    Very low values suggest blur or heavy compression artifacts.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def blur_score_to_quality_factor(blur_score: float) -> float:
    """
    Map blur score to a soft multiplier in [min_mult, 1.0].
    Tunable thresholds — YOLO often still works on moderate blur.
    """
    # Empirical: < 80 often noticeably soft; > 300 usually crisp
    if blur_score < 30:
        return 0.55
    if blur_score < 80:
        return 0.72
    if blur_score < 150:
        return 0.88
    return 1.0


def load_image(path: str | Path) -> LoadedImage:
    """Read an image from disk; raises FileNotFoundError / ValueError on failure."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if not path.is_file():
        raise FileNotFoundError(str(path))

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"OpenCV could not decode image: {path}")

    score = laplacian_blur_score(bgr)
    qf = blur_score_to_quality_factor(score)
    return LoadedImage(bgr=bgr, path=str(path), blur_score=score, quality_factor=qf)


def load_image_from_bytes(data: bytes) -> LoadedImage:
    """Decode image from raw bytes (e.g. multipart upload)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image bytes")
    score = laplacian_blur_score(bgr)
    qf = blur_score_to_quality_factor(score)
    return LoadedImage(bgr=bgr, path=None, blur_score=score, quality_factor=qf)


def load_image_from_array(bgr: np.ndarray, path: str | None = None) -> LoadedImage:
    """Wrap an existing BGR ndarray (e.g. webcam frame)."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected BGR image with shape (H, W, 3)")
    score = laplacian_blur_score(bgr)
    qf = blur_score_to_quality_factor(score)
    return LoadedImage(bgr=bgr, path=path, blur_score=score, quality_factor=qf)


def resize_max_side(bgr: np.ndarray, max_side: int = 1280) -> Tuple[np.ndarray, float]:
    """
    Optionally downscale for faster inference. Returns (resized_bgr, scale).
    scale is the factor applied to original dimensions (new/old).
    """
    h, w = bgr.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return bgr, 1.0
    scale = max_side / float(side)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale
