# Waste classification agent

AI-assisted **image → waste stream** classification using **YOLOv8** (COCO pre-trained) for object detection and a **rule layer** that maps labels to biodegradable, non-biodegradable, or mixed. OpenCV handles I/O and blur heuristics for confidence scaling.

## Architecture

| Module | Role |
|--------|------|
| `image_loader.py` | Load JPEG/PNG/BMP/WebP; Laplacian blur score → quality factor |
| `detector.py` | YOLOv8 inference, bounding boxes, optional drawing |
| `classifier.py` | COCO label → waste category; Mixed if both present; Unknown if no detections |
| `learned_classifier.py` | Optional fine-tuned image classifier (loads checkpoint if available) |
| `pipeline.py` | Single-pass resize → detect → classify |
| `main.py` | CLI: file, `--annotate`, `--webcam` |
| `api.py` | FastAPI: `/classify`, `/classify_annotated`, simple HTML UI at `/` |
| `train_classifier.py` | Train a transferable classifier from folder datasets |

## Setup

Python 3.10+ recommended.

```bash
cd waste_classifier
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Smaller CPU-only PyTorch (avoids multi‑GB CUDA wheels on some platforms):** install Torch first, then the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python numpy fastapi "uvicorn[standard]" python-multipart
```

On first run, Ultralytics downloads `yolov8n.pt` (~6 MB). For NVIDIA GPU, use the CUDA build from [pytorch.org](https://pytorch.org/) instead of the CPU index.

## CLI

```bash
# JSON to stdout (exact contract fields + optional detections)
python main.py path/to/image.jpg
python main.py path/to/image.jpg --boxes

# Annotated image (bonus)
python main.py path/to/image.jpg --annotate out/vis.jpg --boxes

# Webcam (q to quit); prints JSON lines on each interval
python main.py --webcam --webcam-interval 0.5
```

## Train a better model (Kaggle/manual datasets)

You can train an improved classifier and the API will auto-use it when the checkpoint exists.

Expected dataset structure:

- `dataset/train/<class>/...` and `dataset/val/<class>/...`, or
- `dataset/<class>/...` (script does a train/val split automatically)

Example:

```bash
python train_classifier.py \
  --dataset data/trash_type_image_dataset \
  --epochs 10 \
  --batch-size 32 \
  --out models/waste_cls_best.pt
```

At inference time, `learned_classifier.py` loads:

- `WASTE_CLASSIFIER_CKPT` env var, or
- default `models/waste_cls_best.pt`

If checkpoint is missing, service falls back to the original YOLO+rules+color pipeline.

## HTTP API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

- `GET /` — minimal upload form
- `POST /classify` — multipart file field `file`; optional query `include_boxes=1`
- `POST /classify_annotated` — returns JPEG with boxes
- `GET /health`

Example:

```bash
curl -s -X POST "http://127.0.0.1:8000/classify?include_boxes=1" \
  -F "file=@samples/bottle.jpg"
```

## Sample images

Run `python scripts/fetch_samples.py` to download a few small test images into `samples/` (requires network). You can also use any JPEG/PNG of trash or bins.

## Output JSON shape

```json
{
  "classification": "Biodegradable | Non-biodegradable | Mixed | Unknown",
  "confidence": 0.0,
  "detected_objects": ["list", "of", "labels"],
  "reasoning": "Short explanation"
}
```

With `--boxes` or `include_boxes=1`, an extra `"detections"` array includes `bbox` per item.

## Notes

- COCO classes are a **proxy** for waste; real production systems benefit from custom waste datasets and fine-tuned detectors.
- For best accuracy, fine-tune with your own local photos and periodically retrain with corrected labels.
- **Unknown** is returned when **no** objects pass the confidence threshold (e.g. empty frame or extremely faint subjects).
- Inference time depends on hardware; `yolov8n` targets sub-second CPU inference on many machines at 640px.

## License

Example project code you can adapt for your use case.
