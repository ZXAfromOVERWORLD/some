const express = require('express');
const multer = require('multer');

const { protect } = require('../middleware/authMiddleware');

const router = express.Router();

// In-memory upload so we can forward bytes to the FastAPI classifier.
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
});

async function fetchJsonWithRetry(url, options, { timeoutMs, retries, retryOnStatus }) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const resp = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timer);

      if (resp.ok) {
        return await resp.json();
      }

      if (retryOnStatus.has(resp.status) && attempt < retries) {
        lastErr = new Error(`Retryable status ${resp.status}`);
        continue;
      }

      const text = await resp.text().catch(() => '');
      const err = new Error(`AI classify failed with ${resp.status}`);
      err.details = text;
      err.status = resp.status;
      throw err;
    } catch (err) {
      clearTimeout(timer);
      lastErr = err;
      if (attempt < retries && (err?.name === 'AbortError' || retryOnStatus.has(err?.status))) {
        continue;
      }
      throw err;
    }
  }
  throw lastErr;
}

async function classifyViaAiService(file) {
  const aiUrl = process.env.AI_CLASSIFIER_URL || 'http://localhost:8000';

  const blob = new Blob([file.buffer], { type: file.mimetype || 'application/octet-stream' });
  const form = new FormData();
  form.append('file', blob, file.originalname || 'upload');

  const timeoutMs = Number(process.env.AI_CLASSIFIER_TIMEOUT_MS || 8000);
  const retries = Number(process.env.AI_CLASSIFIER_RETRIES || 2);
  const retryOnStatus = new Set([408, 429, 500, 502, 503, 504]);

  return await fetchJsonWithRetry(
    `${aiUrl}/classify`,
    { method: 'POST', body: form },
    { timeoutMs, retries, retryOnStatus }
  );
}

// POST /api/ai/classify
// Expects multipart/form-data with field named: `image`
router.post('/classify', protect, upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }

    const result = await classifyViaAiService(req.file);
    return res.json(result);
  } catch (err) {
    const status = err.status || 500;
    return res.status(status).json({
      message: err.message || 'AI classification failed',
      details: err.details,
    });
  }
});

module.exports = router;

