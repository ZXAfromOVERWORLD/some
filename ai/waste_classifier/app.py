from __future__ import annotations

import io
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline import classify_bgr


app = FastAPI(title="Waste Classifier (Keras)", version="2.0.0")

# Keep CORS permissive by default; restrict via env at deployment/proxy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


def _read_upload_to_bgr(upload: UploadFile) -> np.ndarray:
    content = upload.file.read()
    data = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


@app.post("/classify")
def classify(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Multipart field name must be `file` (matches backend proxy).
    """
    bgr = _read_upload_to_bgr(file)
    payload, _ = classify_bgr(bgr, quality_factor=1.0)
    return payload

