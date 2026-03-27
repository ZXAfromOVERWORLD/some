"""
FastAPI service: upload an image, receive classification JSON.

Optional query param `include_boxes=1` adds detection payloads with bboxes.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from classifier import to_contract_dict
from detector import draw_boxes
from image_loader import load_image_from_bytes
from pipeline import run_pipeline

app = FastAPI(title="Waste Classifier", version="1.0.0")

_default_allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]
_env_origins = os.getenv("WASTE_CLASSIFIER_ALLOWED_ORIGINS") or os.getenv("CORS_ALLOWED_ORIGINS")
_parsed_allowed_origins = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    if _env_origins
    else _default_allowed_origins
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parsed_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONF = 0.25


def _run_on_bytes(data: bytes, include_boxes: bool) -> tuple[Dict[str, Any], np.ndarray | None]:
    loaded = load_image_from_bytes(data)
    full, bgr_in, dets = run_pipeline(
        loaded.bgr,
        quality_factor=loaded.quality_factor,
        conf_threshold=DEFAULT_CONF,
        model_name=DEFAULT_MODEL,
        include_boxes=include_boxes,
    )
    body = to_contract_dict(full, include_boxes=include_boxes)
    annotated = draw_boxes(bgr_in, dets) if include_boxes else None
    return body, annotated


@app.get("/health")
def health():
    return {"status": "ok"}


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Waste classifier</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 42rem; margin: 2rem auto; padding: 0 1rem; }
    pre { background: #111; color: #eee; padding: 1rem; overflow: auto; border-radius: 8px; }
    input, button { margin: 0.5rem 0; }
    label { display: block; margin-top: 1rem; }
  </style>
</head>
<body>
  <h1>Waste classifier</h1>
  <p>Upload a photo of trash or a bin. Results use YOLOv8 (COCO) plus waste-category rules.</p>
  <form id="f">
    <label>Image <input type="file" id="file" accept="image/*" required /></label>
    <label><input type="checkbox" id="boxes"/> Include bounding boxes (JSON)</label>
    <button type="submit">Classify</button>
  </form>
  <h2>Response</h2>
  <pre id="out">(submit an image)</pre>
  <script>
    document.getElementById('f').onsubmit = async (e) => {
      e.preventDefault();
      const file = document.getElementById('file').files[0];
      const boxes = document.getElementById('boxes').checked;
      const fd = new FormData();
      fd.append('file', file);
      const q = boxes ? '?include_boxes=1' : '';
      document.getElementById('out').textContent = 'Loading…';
      const r = await fetch('/classify' + q, { method: 'POST', body: fd });
      const text = await r.text();
      try { document.getElementById('out').textContent = JSON.stringify(JSON.parse(text), null, 2); }
      catch { document.getElementById('out').textContent = text; }
    };
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return HTMLResponse(INDEX_HTML)


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    include_boxes: int = 0,
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image upload")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        body, _ = _run_on_bytes(data, include_boxes=bool(include_boxes))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return JSONResponse(body)


@app.post("/classify_annotated")
async def classify_annotated(file: UploadFile = File(...)):
    """Returns JPEG with bounding boxes (bonus)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image upload")
    data = await file.read()
    try:
        body, vis = _run_on_bytes(data, include_boxes=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if vis is None:
        raise HTTPException(status_code=500, detail="annotation failed")
    ok, buf = cv2.imencode(".jpg", vis)
    if not ok:
        raise HTTPException(status_code=500, detail="encode failed")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


def run():
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
