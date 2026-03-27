"""
Optional learned image classifier inference.

If a trained checkpoint exists, this module predicts:
  Biodegradable / Non-biodegradable / Mixed / Unknown
Otherwise callers can safely ignore it and fall back to heuristics.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import models, transforms


def _norm_label(label: str) -> str:
    return str(label or "").strip().lower().replace("_", " ").replace("-", " ")


def _to_contract_label(label: str) -> str:
    n = _norm_label(label)
    if n in {"biodegradable", "bio", "organic", "food", "paper"}:
        return "Biodegradable"
    if n in {
        "non biodegradable",
        "nonbiodegradable",
        "non bio",
        "plastic",
        "metal",
        "glass",
        "ewaste",
        "e waste",
    }:
        return "Non-biodegradable"
    if n in {"mixed", "mixed waste"}:
        return "Mixed"
    return "Unknown"


def _family_for_contract(label: str) -> str:
    if label == "Biodegradable":
        return "bio"
    if label == "Non-biodegradable":
        return "nonbio"
    if label == "Mixed":
        return "mixed"
    return "unknown"


class _ClassifierBundle:
    def __init__(self, model: torch.nn.Module, class_names: list[str], device: torch.device):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.pre = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


@lru_cache(maxsize=1)
def _load_bundle() -> Optional[_ClassifierBundle]:
    raw_ckpt = os.getenv("WASTE_CLASSIFIER_CKPT", "models/waste_cls_best.pt")
    ckpt_path = Path(raw_ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).resolve().parent / ckpt_path
    if not ckpt_path.is_file():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(str(ckpt_path), map_location=device)
    class_names = blob.get("class_names") or []
    if not class_names:
        return None

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(class_names))
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return _ClassifierBundle(model=model, class_names=class_names, device=device)


def predict_waste_class_from_bgr(bgr: np.ndarray) -> Optional[Tuple[str, float, str]]:
    bundle = _load_bundle()
    if bundle is None:
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = bundle.pre(rgb).unsqueeze(0).to(bundle.device)

    with torch.no_grad():
        logits = bundle.model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    raw_label = bundle.class_names[idx]
    mapped = _to_contract_label(raw_label)

    # Mixed-scene detection: if the model is uncertain, prefer Mixed
    # instead of forcing a single material class.
    order = np.argsort(-probs)
    top1_i = int(order[0])
    top2_i = int(order[1]) if len(order) > 1 else top1_i
    top1_p = float(probs[top1_i])
    top2_p = float(probs[top2_i])
    top1_label = _to_contract_label(bundle.class_names[top1_i])
    top2_label = _to_contract_label(bundle.class_names[top2_i])
    fam1 = _family_for_contract(top1_label)
    fam2 = _family_for_contract(top2_label)

    is_cross_family = {fam1, fam2} == {"bio", "nonbio"}
    is_close = abs(top1_p - top2_p) <= 0.18
    has_signal = top1_p >= 0.30 and top2_p >= 0.20
    if is_cross_family and is_close and has_signal:
        mix_conf = min(0.95, top1_p + 0.15)
        reason = (
            f"Learned classifier suggests mixed scene (top classes span bio/non-bio: "
            f"{bundle.class_names[top1_i]}={top1_p:.2f}, {bundle.class_names[top2_i]}={top2_p:.2f})"
        )
        return "Mixed", float(mix_conf), reason

    # Additional uncertainty rule: if top-1 is not very strong and top-2 is close,
    # treat as Mixed even when both classes map to same family.
    if top1_p < 0.75 and top2_p >= 0.18 and abs(top1_p - top2_p) <= 0.22:
        mix_conf = min(0.9, top1_p + 0.10)
        reason = (
            f"Learned classifier uncertain between classes "
            f"({bundle.class_names[top1_i]}={top1_p:.2f}, {bundle.class_names[top2_i]}={top2_p:.2f}); "
            "using Mixed fallback."
        )
        return "Mixed", float(mix_conf), reason

    reason = f"Learned classifier predicted '{raw_label}'"
    return mapped, confidence, reason

