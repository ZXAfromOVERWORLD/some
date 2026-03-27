const Request = require('../models/Request');
const Bin = require('../models/Bin');

async function fetchJsonWithRetry(url, options, { timeoutMs, retries, retryOnStatus }) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const resp = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timer);

      if (resp.ok) return await resp.json();

      if (retryOnStatus.has(resp.status) && attempt < retries) {
        lastErr = new Error(`Retryable status ${resp.status}`);
        continue;
      }

      const text = await resp.text().catch(() => '');
      const err = new Error(`AI predictor failed with ${resp.status}`);
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

const haversineKm = (lat1, lon1, lat2, lon2) => {
  const R = 6371; // Earth radius in km
  const dLat = (lat2 - lat1) * (Math.PI / 180);
  const dLon = (lon2 - lon1) * (Math.PI / 180);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * (Math.PI / 180)) * Math.cos(lat2 * (Math.PI / 180)) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
};

const bruteForceNearestBin = async (lat, lng) => {
  if (lat === undefined || lng === undefined || Number.isNaN(Number(lat)) || Number.isNaN(Number(lng))) {
    return null;
  }

  const bins = await Bin.find({}, { location: 1 }).lean();
  let best = null;
  let bestKm = Infinity;

  for (const b of bins) {
    const bLat = b.location?.lat;
    const bLng = b.location?.lng;
    if (typeof bLat !== 'number' || typeof bLng !== 'number') continue;
    const km = haversineKm(Number(lat), Number(lng), bLat, bLng);
    if (km < bestKm) {
      bestKm = km;
      best = b;
    }
  }

  // If there's no nearby bin, we don't attach one.
  if (!best || bestKm > 5) return null;
  return best;
};

const findNearestBin = async (lat, lng) => {
  if (lat === undefined || lng === undefined || Number.isNaN(Number(lat)) || Number.isNaN(Number(lng))) {
    return null;
  }

  const maxDistanceMeters = Number(process.env.BIN_NEAR_MAX_DISTANCE_M || 5000);
  const coords = [Number(lng), Number(lat)]; // [lng, lat]

  // Fast path: geo query using $nearSphere (+ 2dsphere index).
  try {
    const geoBin = await Bin.findOne({
      geoLocation: {
        $nearSphere: {
          $geometry: { type: 'Point', coordinates: coords },
          $maxDistance: maxDistanceMeters,
        },
      },
    }).lean();

    if (geoBin) return geoBin;
  } catch (err) {
    // If index/geoLocation isn't available yet, fall back to brute force.
  }

  return await bruteForceNearestBin(lat, lng);
};

const maybeCreateNextPredictionRequest = async ({
  citizenUserId,
  bin,
  lastCompletedRequest,
  io,
}) => {
  // Need at least 2 previous completed pickups for this citizen+bin.
  // Because `lastCompletedRequest` is already marked Completed when we call this function,
  // we need at least 3 total completed requests to have 2 intervals worth of history.
  const completedRequests = await Request.find({
    userId: citizenUserId,
    binId: bin._id,
    status: 'Completed',
  })
    .sort({ updatedAt: -1 })
    .limit(6);

  if (completedRequests.length < 3) return;

  // Use chronological order to compute recent intervals
  const times = completedRequests
    .slice()
    .reverse()
    .map((r) => new Date(r.updatedAt || r.createdAt).getTime())
    .filter((t) => Number.isFinite(t));

  if (times.length < 2) return;

  const predictNextFillViaAi = async () => {
    const predictorUrl = process.env.AI_FILL_PREDICTOR_URL || 'http://localhost:8010/predict';
    const minConfidence = Number(process.env.AI_FILL_MIN_CONFIDENCE || 0.25);
    const timeoutMs = Number(process.env.AI_PREDICTOR_TIMEOUT_MS || 6000);
    const retries = Number(process.env.AI_PREDICTOR_RETRIES || 2);
    const retryOnStatus = new Set([408, 429, 500, 502, 503, 504]);

    const lastCompletionMs = times[times.length - 1];
    const intervals = [];
    for (let i = 1; i < times.length; i++) intervals.push(times[i] - times[i - 1]);

    // Call the separate AI prediction service (timeout+retries; prediction failures are non-blocking)
    let data;
    try {
      data = await fetchJsonWithRetry(
        predictorUrl,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: citizenUserId,
            bin_id: bin._id,
            completion_timestamps_ms: times,
            interval_deltas_ms: intervals,
            last_completion_ms: lastCompletionMs,
          }),
        },
        { timeoutMs, retries, retryOnStatus }
      );
    } catch (err) {
      return null;
    }

    const predictedFillMs = Number(data.predicted_fill_ms);
    const confidence = Number(data.confidence);

    if (!Number.isFinite(predictedFillMs) || predictedFillMs <= 0) return null;
    if (!Number.isFinite(confidence) || confidence < minConfidence) return null;

    return {
      predictedFillMs,
      confidence,
      reasoning: data.reasoning || 'AI prediction',
      predictedIntervalMs: Number(data.predicted_interval_ms) || null,
    };
  };

  const prediction = await predictNextFillViaAi();
  if (!prediction) return;

  const { predictedFillMs } = prediction;

  // Dispatch lead time: send the truck a bit before the predicted fill moment.
  const LEAD_TIME_MS = 6 * 60 * 60 * 1000; // 6 hours
  const dispatchMs = predictedFillMs - LEAD_TIME_MS;
  const nowMs = Date.now();

  // Don't create predictions too far in the past (e.g. clock drift).
  if (dispatchMs < nowMs - 60 * 60 * 1000) return;

  const windowStartISO = new Date(dispatchMs - 24 * 60 * 60 * 1000).toISOString(); // -1 day
  const windowEndISO = new Date(dispatchMs + 24 * 60 * 60 * 1000).toISOString(); // +1 day

  // "Only": avoid creating multiple overlapping predicted dispatches.
  const existingPending = await Request.find({
    userId: citizenUserId,
    binId: bin._id,
    status: { $in: ['Pending', 'Assigned', 'In Progress'] },
    scheduledDate: { $gte: windowStartISO, $lte: windowEndISO },
  });

  if (existingPending.length > 0) return;

  const predictedReq = await Request.create({
    userId: citizenUserId,
    binId: bin._id,
    wasteType: lastCompletedRequest?.wasteType || 'General',
    location: bin.location?.address || `Bin ${bin._id}`,
    lat: bin.location?.lat,
    lng: bin.location?.lng,
    scheduledDate: new Date(dispatchMs).toISOString(),
    scheduledTime: '',
    notes: `${
      lastCompletedRequest?.notes ? `(${lastCompletedRequest.notes}) ` : ''
    }Auto-dispatch prediction (AI). Predicted fill: ${new Date(predictedFillMs).toISOString()} (confidence: ${prediction.confidence}). ${prediction.reasoning}`,
    status: 'Pending',
  });

  // Emit so UI refreshes immediately (admin dashboard / map).
  await predictedReq.populate('userId', 'name email');
  io.to('admin').emit('new_request', predictedReq);
  io.to('driver').emit('new_request', predictedReq);
};

// POST /api/request
const createRequest = async (req, res) => {
  try {
    const { wasteType, location, lat, lng, scheduledDate, scheduledTime, notes } = req.body;
    if (!location || !scheduledDate) {
      return res.status(400).json({ message: 'Location and scheduledDate are required' });
    }

    const nearestBin = await findNearestBin(lat, lng);
    const binId = nearestBin?._id || null;
    const request = await Request.create({
      userId: req.user._id,
      binId,
      wasteType: wasteType || 'General',
      location,
      lat,
      lng,
      scheduledDate,
      scheduledTime: scheduledTime || '',
      notes: notes || '',
    });

    // Populate user details before emitting so frontend has name
    await request.populate('userId', 'name email');
    const io = req.app.get('io');
    io.to('admin').emit('new_request', request);
    io.to('driver').emit('new_request', request);

    res.status(201).json(request);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// GET /api/request/user
const getUserRequests = async (req, res) => {
  try {
    const requests = await Request.find({ userId: req.user._id })
      .populate('assignedDriverId', 'name email')
      .sort({ createdAt: -1 });
    res.json(requests);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// PATCH /api/request/:id/status  (driver only)
const updateRequestStatus = async (req, res) => {
  try {
    const { status } = req.body;
    const validStatuses = ['Pending', 'Assigned', 'In Progress', 'Completed'];
    if (!validStatuses.includes(status)) {
      return res.status(400).json({ message: 'Invalid status' });
    }

    // Atomic update to prevent duplicate "Completed" transitions.
    const updateFilter = { _id: req.params.id };

    // Driver can only update their assigned requests.
    if (req.user.role === 'driver') {
      updateFilter.assignedDriverId = req.user._id;
    }

    // If marking completed, only allow transition if it isn't completed already.
    if (status === 'Completed') {
      updateFilter.status = { $ne: 'Completed' };
    }

    const updated = await Request.findOneAndUpdate(
      updateFilter,
      { status },
      { new: true }
    );

    if (!updated) {
      return res
        .status(status === 'Completed' ? 409 : 404)
        .json({
          message:
            status === 'Completed'
              ? 'Request already completed or not assigned'
              : 'Request not found',
        });
    }

    // When collection is completed, reset the associated bin and create a new prediction (if enough history).
    if (status === 'Completed') {
      let bin = null;
      if (updated.binId) {
        bin = await Bin.findById(updated.binId);
      }
      if (!bin && updated.lat !== undefined && updated.lng !== undefined) {
        const nearest = await findNearestBin(updated.lat, updated.lng);
        if (nearest?._id) bin = await Bin.findById(nearest._id);
      }

      if (bin) {
        bin.fillLevel = 0;
        bin.lastCollected = new Date();
        bin.status = 'Empty';
        await bin.save();

        await maybeCreateNextPredictionRequest({
          citizenUserId: updated.userId,
          bin,
          lastCompletedRequest: updated,
          io: req.app.get('io'),
        });
      }
    }

    await updated.populate('userId', 'name email');
    const io = req.app.get('io');
    io.to('admin').emit('task_updated', updated);
    io.to('driver').emit('task_updated', updated);
    if (updated.userId?._id) io.to(String(updated.userId._id)).emit('task_updated', updated);

    res.json(updated);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

module.exports = { createRequest, getUserRequests, updateRequestStatus };
