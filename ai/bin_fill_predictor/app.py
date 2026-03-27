from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


class PredictRequest(BaseModel):
    user_id: str
    bin_id: str
    completion_timestamps_ms: List[int]
    interval_deltas_ms: List[int]
    last_completion_ms: int


class PredictResponse(BaseModel):
    predicted_interval_ms: float
    predicted_fill_ms: int
    confidence: float
    reasoning: str


app = FastAPI(title="Bin Fill Predictor", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


def _safe_interval_model(intervals_ms: List[int]) -> tuple[float, float, str]:
    """
    Lightweight time-series predictor:
    - Fit a linear trend to recent interval durations.
    - Predict next interval as intercept + slope * next_index.
    - Confidence decreases as residual variance increases.
    """
    intervals = np.array([float(x) for x in intervals_ms if x is not None], dtype=np.float64)
    intervals = intervals[np.isfinite(intervals)]

    if len(intervals) < 1:
        return 0.0, 0.0, "Not enough interval history"

    # If only one interval exists, use it directly.
    if len(intervals) == 1:
        predicted = float(intervals[0])
        confidence = 0.5
        return predicted, confidence, "Single interval observed; low confidence"

    # Fit y = a + b*x where x = 0..n-1
    x = np.arange(len(intervals), dtype=np.float64)
    y = intervals

    # Least squares: [1, x] @ [a, b]
    A = np.vstack([np.ones_like(x), x]).T
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    a = float(coeffs[0])
    b = float(coeffs[1])

    next_x = float(len(intervals))
    predicted = a + b * next_x
    predicted = max(0.0, predicted)  # don't predict negative intervals

    # Residual variance for confidence
    y_hat = a + b * x
    resid = y - y_hat
    var = float(np.var(resid))  # >= 0

    # Normalize variance relative to scale
    scale = max(1.0, float(np.mean(y)) ** 2)
    confidence = 1.0 / (1.0 + var / scale)
    confidence = float(max(0.0, min(1.0, confidence)))

    reasoning = f"Linear trend on last {len(intervals)} intervals; variance={var:.2f}"
    return predicted, confidence, reasoning


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    completion_ts = [int(x) for x in req.completion_timestamps_ms if x is not None]
    if len(completion_ts) < 3:
        # We want at least 2 previous completed pickups => at least 3 timestamps
        # (because interval_deltas_ms length = timestamps length - 1).
        # If your backend sends smaller sequences, we still try but reduce confidence.
        completion_ts = sorted(completion_ts)

    completion_ts = sorted(completion_ts)
    intervals = [int(x) for x in req.interval_deltas_ms if x is not None]
    intervals = [x for x in intervals if x > 0]

    predicted_interval_ms, confidence, reasoning = _safe_interval_model(intervals)

    last_completion_ms = int(req.last_completion_ms)
    predicted_fill_ms = int(last_completion_ms + predicted_interval_ms)

    return PredictResponse(
        predicted_interval_ms=predicted_interval_ms,
        predicted_fill_ms=predicted_fill_ms,
        confidence=confidence,
        reasoning=reasoning,
    )

