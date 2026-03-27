const Bin = require('../models/Bin');

// GET /api/bins
const getBins = async (req, res) => {
  try {
    const bins = await Bin.find().sort({ createdAt: -1 });
    res.json(bins);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// POST /api/bins
const createBin = async (req, res) => {
  try {
    const { address, lat, lng, fillLevel, status } = req.body;
    if (!address || lat === undefined || lng === undefined) {
      return res.status(400).json({ message: 'Address, lat, and lng are required' });
    }
    const bin = await Bin.create({
      location: { address, lat: parseFloat(lat), lng: parseFloat(lng) },
      geoLocation: { coordinates: [parseFloat(lng), parseFloat(lat)] },
      fillLevel: fillLevel || 0,
      status: status || 'Active',
    });

    const io = req.app.get('io');
    io.to('admin').emit('new_bin', bin);
    io.emit('new_bin', bin);

    res.status(201).json(bin);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// POST /api/bins/register
// User self-registration of a new household bin from user portal.
const registerBinForUser = async (req, res) => {
  try {
    const { address, lat, lng, fillLevel } = req.body;
    if (!address || lat === undefined || lng === undefined) {
      return res.status(400).json({ message: 'Address, lat, and lng are required' });
    }

    const parsedLat = parseFloat(lat);
    const parsedLng = parseFloat(lng);
    if (!Number.isFinite(parsedLat) || !Number.isFinite(parsedLng)) {
      return res.status(400).json({ message: 'Invalid coordinates' });
    }

    const parsedFill = fillLevel !== undefined ? Number(fillLevel) : 0;
    const safeFill = Math.max(0, Math.min(100, Number.isFinite(parsedFill) ? parsedFill : 0));

    const bin = await Bin.create({
      location: { address, lat: parsedLat, lng: parsedLng },
      geoLocation: { coordinates: [parsedLng, parsedLat] },
      fillLevel: safeFill,
      status: 'Active',
      createdBy: req.user._id,
      source: 'user',
    });

    const io = req.app.get('io');
    io.to('admin').emit('new_bin', bin);
    io.to(String(req.user._id)).emit('new_bin', bin);
    io.emit('new_bin', bin);

    res.status(201).json(bin);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// PATCH /api/bins/:id
const updateBin = async (req, res) => {
  try {
    const { fillLevel, status, address, lat, lng } = req.body;
    const updateData = {};
    if (fillLevel !== undefined) updateData.fillLevel = fillLevel;
    if (status) updateData.status = status;
    if (address || lat !== undefined || lng !== undefined) {
      updateData['location.address'] = address;

      // For geoLocation updates we may need missing coordinate values.
      const existingBin = await Bin.findById(req.params.id).lean();
      const finalLat = lat !== undefined ? parseFloat(lat) : existingBin?.location?.lat;
      const finalLng = lng !== undefined ? parseFloat(lng) : existingBin?.location?.lng;

      if (lat !== undefined) updateData['location.lat'] = finalLat;
      if (lng !== undefined) updateData['location.lng'] = finalLng;

      if (finalLat !== undefined && finalLng !== undefined) {
        updateData['geoLocation.coordinates'] = [finalLng, finalLat];
      }
    }

    const bin = await Bin.findByIdAndUpdate(req.params.id, updateData, { new: true, runValidators: true });
    if (!bin) return res.status(404).json({ message: 'Bin not found' });

    req.app.get('io').to('admin').emit('bin_updated', bin);

    res.json(bin);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// DELETE /api/bins/:id
const deleteBin = async (req, res) => {
  try {
    const bin = await Bin.findByIdAndDelete(req.params.id);
    if (!bin) return res.status(404).json({ message: 'Bin not found' });

    req.app.get('io').to('admin').emit('delete_bin', req.params.id);

    res.json({ message: 'Bin removed' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

module.exports = { getBins, createBin, registerBinForUser, updateBin, deleteBin };
