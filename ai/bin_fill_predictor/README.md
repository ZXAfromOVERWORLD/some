# Bin Fill Predictor (AI Service)

This service predicts the next time a trash bin is expected to reach “filled” based on
historical completion timestamps.

## Run

From `ai/bin_fill_predictor/`:

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8010
```

## Endpoints

`GET /health` returns `{status: "ok"}`

`POST /predict` accepts JSON:

```json
{
  "user_id": "string",
  "bin_id": "string",
  "completion_timestamps_ms": [1690000000000, 1690007200000, ...],
  "interval_deltas_ms": [7200000, ...],
  "last_completion_ms": 1690007200000
}
```

Response:

```json
{
  "predicted_interval_ms": 7200000.0,
  "predicted_fill_ms": 1690014400000,
  "confidence": 0.0,
  "reasoning": "..."
}
```

