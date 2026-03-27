"""
Keras/TensorFlow image classifier for waste stream prediction.

Expected files (relative to this directory by default):
- keras_model/model.keras
- keras_model/classes.json

Environment variables:
- WASTE_KERAS_MODEL_PATH
- WASTE_KERAS_CLASSES_PATH
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def _to_contract_label_from_class_name(name: str) -> str:
    n = str(name or "").strip().lower()

    # Organic-like → Biodegradable
    if n in {"organic", "compost", "food", "biodegradable"}:
        return "Biodegradable"

    # Recyclable-like → Non-biodegradable
    if n in {
        "recyclable",
        "recycle",
        "plastic",
        "metal",
        "glass",
        "paper",
        "cardboard",
        "non-biodegradable",
        "non biodegradable",
        "recyclables",
    }:
        return "Non-biodegradable"

    # Mixed stream
    if n in {"mixed"}:
        return "Mixed"
    if n in {"trash", "other", "landfill"}:
        return "Mixed"

    return "Unknown"


class _KerasBundle:
    def __init__(self, model, class_names: list[str]):
        self.model = model
        self.class_names = class_names


@lru_cache(maxsize=1)
def _load_bundle() -> Optional[_KerasBundle]:
    # Lazy import so TF is only required at runtime for this service.
    try:
        from tensorflow import keras  # type: ignore
    except Exception:
        return None

    root = Path(__file__).resolve().parent
    model_path_raw = os.getenv("WASTE_KERAS_MODEL_PATH", "keras_model/model.keras")
    classes_path_raw = os.getenv("WASTE_KERAS_CLASSES_PATH", "keras_model/classes.json")

    model_path = Path(model_path_raw)
    classes_path = Path(classes_path_raw)
    if not model_path.is_absolute():
        model_path = root / model_path
    if not classes_path.is_absolute():
        classes_path = root / classes_path

    if not model_path.exists() or not classes_path.is_file():
        return None

    class_names = json.loads(classes_path.read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or not class_names:
        return None

    model = keras.models.load_model(str(model_path), compile=False)
    return _KerasBundle(model=model, class_names=[str(x) for x in class_names])


def predict_from_bgr(bgr: np.ndarray) -> Optional[Tuple[str, float, str]]:
    bundle = _load_bundle()
    if bundle is None:
        return None

    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)
    x = x.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    probs = bundle.model.predict(x, verbose=0)
    probs = np.array(probs).reshape(-1)
    if probs.size != len(bundle.class_names):
        return None

    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    raw = bundle.class_names[idx]
    mapped = _to_contract_label_from_class_name(raw)
    reason = f"Keras predicted '{raw}'"
    return mapped, conf, reason

