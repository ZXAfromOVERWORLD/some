const Complaint = require('../models/Complaint');
const Driver = require('../models/Driver');

const haversineKm = (lat1, lon1, lat2, lon2) => {
  const R = 6371;
  const dLat = (lat2 - lat1) * (Math.PI / 180);
  const dLon = (lon2 - lon1) * (Math.PI / 180);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * (Math.PI / 180)) *
      Math.cos(lat2 * (Math.PI / 180)) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
};

function allowedTruckTypesFromClassification(classification) {
  // Compatibility rules:
  // - Biodegradable: Biodegradable OR Mixed
  // - Non-biodegradable: Non-biodegradable OR Mixed
  // - Mixed: Mixed only
  // - Unknown/empty: allow any (don't block assignment if AI missing)
  const c = String(classification || '').toLowerCase();
  if (c === 'biodegradable') return ['Biodegradable', 'Mixed'];
  if (c === 'non-biodegradable') return ['Non-biodegradable', 'Mixed'];
  if (c === 'mixed') return ['Mixed'];
  return ['Mixed', 'Biodegradable', 'Non-biodegradable'];
}

async function maybeAutoAssignComplaint({ complaint, io }) {
  if (!complaint) return null;
  if (complaint.assignedDriverId) return complaint;
  if (!Number.isFinite(Number(complaint.lat)) || !Number.isFinite(Number(complaint.lng))) return complaint;

  const allowedTruckTypes = allowedTruckTypesFromClassification(complaint.ai?.classification);

  const activeDrivers = await Driver.find({
    shiftStatus: 'Active',
    truckType: { $in: allowedTruckTypes },
  })
    .populate('userId', 'name email role')
    .lean();

  if (!activeDrivers.length) return complaint;

  const cLat = Number(complaint.lat);
  const cLng = Number(complaint.lng);

  let best = null;
  let bestKm = Infinity;
  for (const d of activeDrivers) {
    const dLat = d.currentLocation?.lat;
    const dLng = d.currentLocation?.lng;
    if (typeof dLat !== 'number' || typeof dLng !== 'number') continue;
    const km = haversineKm(cLat, cLng, dLat, dLng);
    if (km < bestKm) {
      bestKm = km;
      best = d;
    }
  }

  const driverUserId = best?.userId?._id;
  if (!driverUserId) return complaint;

  const updated = await Complaint.findOneAndUpdate(
    { _id: complaint._id, assignedDriverId: null },
    { assignedDriverId: driverUserId, assignedAt: new Date(), status: 'In Progress' },
    { new: true }
  )
    .populate('userId', 'name email')
    .populate('assignedDriverId', 'name email');

  if (!updated) return complaint;

  io.to('admin').emit('complaint_updated', updated);
  io.to('driver').emit('complaint_assigned', updated);
  if (driverUserId) io.to(String(driverUserId)).emit('complaint_assigned', updated);
  if (updated.userId?._id) io.to(String(updated.userId._id)).emit('complaint_updated', updated);

  return updated;
}

// POST /api/complaint
const createComplaint = async (req, res) => {
  try {
    const { description, location, priority, imageUrl, ai, lat, lng } = req.body;
    if (!description || !location) {
      return res.status(400).json({ message: 'Description and location are required' });
    }

    const safeAi =
      ai && typeof ai === 'object'
        ? {
            classification: typeof ai.classification === 'string' ? ai.classification : '',
            confidence: Number.isFinite(Number(ai.confidence)) ? Number(ai.confidence) : null,
            provider: typeof ai.provider === 'string' ? ai.provider : 'waste_classifier',
            raw: ai.raw ?? null,
          }
        : undefined;

    const complaint = await Complaint.create({
      userId: req.user._id,
      description,
      location,
      ...(lat !== undefined ? { lat: Number(lat) } : {}),
      ...(lng !== undefined ? { lng: Number(lng) } : {}),
      priority: priority || 'Medium',
      imageUrl: imageUrl || '',
      ...(safeAi ? { ai: safeAi } : {}),
    });

    await complaint.populate('userId', 'name email');
    const io = req.app.get('io');
    io.to('admin').emit('new_complaint', complaint);

    // Auto-assign nearest compatible driver (non-blocking)
    const maybeAssigned = await maybeAutoAssignComplaint({ complaint, io });

    // Return the latest view (so UI/admin immediately sees assigned driver if any)
    if (maybeAssigned && maybeAssigned._id) {
      return res.status(201).json(maybeAssigned);
    }

    res.status(201).json(complaint);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// GET /api/complaint/user
const getUserComplaints = async (req, res) => {
  try {
    const complaints = await Complaint.find({ userId: req.user._id }).sort({ createdAt: -1 });
    res.json(complaints);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

module.exports = { createComplaint, getUserComplaints };
