"""
Map COCO-style detector labels to waste categories.

Biodegradable: food waste, paper, organic matter (per project spec).
Non-biodegradable: plastic, glass, metal, e-waste, synthetics.
Unknown / ambiguous labels reduce aggregated confidence but still appear in output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Set, Tuple

from detector import Detection


class WasteKind(str, Enum):
    BIODEGRADABLE = "biodegradable"
    NON_BIODEGRADABLE = "non_biodegradable"
    UNKNOWN = "unknown"


# Normalized keys: lowercased COCO class names from YOLOv8
BIODEGRADABLE_LABELS: Set[str] = {
    # Food / organic
    "banana",
    "apple",
    "orange",
    "broccoli",
    "carrot",
    "sandwich",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "bowl",  # often food context; ambiguous but commonly organic waste stream
    "potted plant",
    # Paper / wood / organic-adjacent
    "book",
    # Animals not typical in bins but organic if ever detected
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}

NON_BIODEGRADABLE_LABELS: Set[str] = {
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "refrigerator",
    "sink",
    "toilet",
    "hair drier",
    "toothbrush",
    "scissors",
    "clock",
    "backpack",
    "handbag",
    "suitcase",
    "umbrella",
    "frisbee",
    "skateboard",
    "sports ball",
    "skis",
    "snowboard",
    "kite",
    "baseball bat",
    "baseball glove",
    "tennis racket",
    "surfboard",
    "vase",  # often ceramic/glass
    # Vehicles / infrastructure — not biodegradable
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
}


def normalize_label(label: str) -> str:
    return label.strip().lower()


def kind_for_label(label: str) -> WasteKind:
    key = normalize_label(label)
    if key in BIODEGRADABLE_LABELS:
        return WasteKind.BIODEGRADABLE
    if key in NON_BIODEGRADABLE_LABELS:
        return WasteKind.NON_BIODEGRADABLE
    return WasteKind.UNKNOWN


@dataclass
class ClassificationResult:
    classification: str  # Biodegradable | Non-biodegradable | Mixed | Unknown
    confidence: float
    detected_objects: List[str]
    reasoning: str
    # Optional diagnostics
    detections_detail: List[Dict] | None = None


def _unique_labels(detections: Sequence[Detection]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for d in detections:
        if d.label not in seen:
            seen.add(d.label)
            ordered.append(d.label)
    return ordered


def classify_detections(
    detections: Sequence[Detection],
    quality_factor: float = 1.0,
) -> ClassificationResult:
    """
    Aggregate detector output into the contract JSON fields.

    quality_factor: 0..1 from image_loader blur heuristic — scales confidence down.
    """
    if not detections:
        return ClassificationResult(
            classification="Unknown",
            confidence=0.0,
            detected_objects=[],
            reasoning="No trash detected",
            detections_detail=None,
        )

    kinds_present: Set[WasteKind] = set()
    bio_labels: List[str] = []
    nonbio_labels: List[str] = []
    unknown_labels: List[str] = []

    confs: List[float] = []
    unknown_confs: List[float] = []

    for d in detections:
        confs.append(d.confidence)
        k = kind_for_label(d.label)
        kinds_present.add(k)
        if k == WasteKind.BIODEGRADABLE:
            bio_labels.append(d.label)
        elif k == WasteKind.NON_BIODEGRADABLE:
            nonbio_labels.append(d.label)
        else:
            unknown_labels.append(d.label)
            unknown_confs.append(d.confidence)

    has_bio = bool(bio_labels)
    has_non = bool(nonbio_labels)
    has_unknown = bool(unknown_labels)

    detected_objects = _unique_labels(detections)

    # Mean detection confidence, then apply blur and unknown penalties
    base_conf = sum(confs) / len(confs)
    unknown_ratio = (len(unknown_confs) / len(confs)) if confs else 0.0
    # Softer penalty: many unknowns should pull confidence down
    unknown_penalty = 1.0 - 0.35 * unknown_ratio
    final_conf = max(0.0, min(1.0, base_conf * quality_factor * unknown_penalty))

    if has_bio and has_non:
        return ClassificationResult(
            classification="Mixed",
            confidence=round(final_conf, 4),
            detected_objects=detected_objects,
            reasoning=(
                "Both biodegradable items (e.g. food or paper-related) and "
                "non-biodegradable items (e.g. plastic, metal, or glass objects) were detected."
            ),
        )

    if has_bio and not has_non:
        return ClassificationResult(
            classification="Biodegradable",
            confidence=round(final_conf, 4),
            detected_objects=detected_objects,
            reasoning=(
                "Detected objects map mainly to organic or paper-like waste; "
                "no strong non-biodegradable materials identified."
            ),
        )

    if has_non and not has_bio:
        return ClassificationResult(
            classification="Non-biodegradable",
            confidence=round(final_conf, 4),
            detected_objects=detected_objects,
            reasoning=(
                "Detected objects align with plastics, glass, metal, or other "
                "synthetic / inorganic materials."
            ),
        )

    # Only unknown-mapped labels (e.g. person, chair)
    unk_only_conf = max(0.0, min(1.0, base_conf * quality_factor * 0.5))
    return ClassificationResult(
        classification="Mixed",
        confidence=round(unk_only_conf, 4),
        detected_objects=detected_objects,
        reasoning=(
            "Detected objects could not be confidently mapped to biodegradable vs "
            "non-biodegradable waste categories; treat as uncertain / mixed stream."
        ),
    )


def to_contract_dict(result: ClassificationResult, include_boxes: bool = False) -> Dict:
    """Exact API shape requested by the product spec."""
    out = {
        "classification": result.classification,
        "confidence": float(result.confidence),
        "detected_objects": result.detected_objects,
        "reasoning": result.reasoning,
    }
    if include_boxes and result.detections_detail is not None:
        out["detections"] = result.detections_detail
    return out
