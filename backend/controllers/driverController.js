const Request = require('../models/Request');
const Driver = require('../models/Driver');
const Complaint = require('../models/Complaint');

// GET /api/driver/tasks
const getDriverTasks = async (req, res) => {
  try {
    const tasks = await Request.find({ assignedDriverId: req.user._id })
      .populate('userId', 'name email')
      .sort({ createdAt: -1 });
    res.json(tasks);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// GET /api/driver/complaints
const getDriverComplaints = async (req, res) => {
  try {
    const complaints = await Complaint.find({ assignedDriverId: req.user._id })
      .populate('userId', 'name email')
      .sort({ createdAt: -1 });
    res.json(complaints);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// PATCH /api/driver/complaint/:id/resolve
const resolveAssignedComplaint = async (req, res) => {
  try {
    const complaintId = req.params.id;

    const updated = await Complaint.findOneAndUpdate(
      { _id: complaintId, assignedDriverId: req.user._id, status: { $ne: 'Resolved' } },
      { status: 'Resolved' },
      { new: true }
    )
      .populate('userId', 'name email')
      .populate('assignedDriverId', 'name email');

    if (!updated) {
      return res.status(404).json({ message: 'Complaint not found, not assigned to you, or already resolved.' });
    }

    const io = req.app.get('io');
    io.to('admin').emit('complaint_updated', updated);
    io.to('driver').emit('complaint_updated', updated);
    if (updated.assignedDriverId?._id) io.to(String(updated.assignedDriverId._id)).emit('complaint_updated', updated);
    if (updated.userId?._id) io.to(String(updated.userId._id)).emit('complaint_updated', updated);

    res.json(updated);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// PATCH /api/driver/shift
// Updates whether the driver is currently on shift (so admin UI counts are correct).
const updateDriverShiftStatus = async (req, res) => {
  try {
    const { shiftStatus } = req.body;
    const validStatuses = ['Active', 'Off-duty', 'On-break'];
    if (!shiftStatus || !validStatuses.includes(shiftStatus)) {
      return res.status(400).json({ message: 'Invalid shiftStatus' });
    }

    const updated = await Driver.findOneAndUpdate(
      { userId: req.user._id },
      { shiftStatus },
      { new: true }
    ).populate('userId', 'name email role');

    if (!updated) return res.status(404).json({ message: 'Driver profile not found' });

    req.app.get('io').to('admin').emit('driver_shift_updated', updated);
    res.json(updated);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// GET /api/driver/shift
const getDriverShiftStatus = async (req, res) => {
  try {
    const driver = await Driver.findOne({ userId: req.user._id }).select('shiftStatus truckType');
    if (!driver) return res.status(404).json({ message: 'Driver profile not found' });
    res.json({ shiftStatus: driver.shiftStatus, truckType: driver.truckType || '' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// PATCH /api/driver/truck-type
const updateDriverTruckType = async (req, res) => {
  try {
    const { truckType } = req.body;
    const valid = ['Mixed', 'Biodegradable', 'Non-biodegradable'];
    if (!truckType || !valid.includes(truckType)) {
      return res.status(400).json({ message: 'Invalid truckType' });
    }

    const updated = await Driver.findOneAndUpdate(
      { userId: req.user._id },
      { truckType },
      { new: true }
    ).populate('userId', 'name email role');

    if (!updated) return res.status(404).json({ message: 'Driver profile not found' });

    req.app.get('io').to('admin').emit('driver_truck_updated', updated);
    res.json(updated);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

module.exports = { getDriverTasks, getDriverComplaints, resolveAssignedComplaint, updateDriverShiftStatus, getDriverShiftStatus, updateDriverTruckType };
