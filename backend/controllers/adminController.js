const User = require('../models/User');
const Driver = require('../models/Driver');
const Request = require('../models/Request');
const Complaint = require('../models/Complaint');
const Bin = require('../models/Bin');

// GET /api/admin/stats
const getDashboardStats = async (req, res) => {
  try {
    const [totalBins, fullBins, totalDrivers, pendingComplaints, pendingRequests] = await Promise.all([
      Bin.countDocuments(),
      Bin.countDocuments({ fillLevel: { $gte: 80 } }),
      Driver.countDocuments({ shiftStatus: 'Active' }),
      Complaint.countDocuments({ status: { $ne: 'Resolved' } }),
      Request.countDocuments({ status: 'Pending' }),
    ]);

    // Weekly waste collection data (last 7 days request completions)
    const last7Days = [];
    const today = new Date();
    for (let i = 6; i >= 0; i--) {
      const d = new Date(today);
      d.setDate(today.getDate() - i);
      last7Days.push(d.toISOString().split('T')[0]); // YYYY-MM-DD
    }

    const weeklyDataRaw = await Request.aggregate([
      { $match: { status: 'Completed', updatedAt: { $gte: new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000) } } },
      {
        $group: {
          _id: { $dateToString: { format: "%Y-%m-%d", date: "$updatedAt" } },
          count: { $sum: 1 }
        }
      }
    ]);

    const weeklyData = last7Days.map(date => {
      const match = weeklyDataRaw.find(d => d._id === date);
      return { 
        date, 
        count: match ? match.count : 0 
      };
    });

    // Recent complaints
    const recentComplaints = await Complaint.find()
      .populate('userId', 'name email')
      .sort({ createdAt: -1 })
      .limit(5);

    res.json({
      stats: { totalBins, fullBins, activeDrivers: totalDrivers, alerts: pendingComplaints + pendingRequests },
      weeklyData,
      recentComplaints,
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// POST /api/admin/assign
const assignDriver = async (req, res) => {
  try {
    const { requestId, driverId } = req.body;
    if (!requestId || !driverId) {
      return res.status(400).json({ message: 'requestId and driverId are required' });
    }

    // Conditional update to prevent double assignment (auto-dispatch vs manual).
    let request = await Request.findOneAndUpdate(
      { _id: requestId, status: 'Pending', assignedDriverId: null },
      { assignedDriverId: driverId, status: 'Assigned' },
      { new: true }
    );

    if (!request) {
      return res.status(409).json({ message: 'Request already assigned or not pending' });
    }

    // Populate must be done sequentially to avoid chaining on a promise/thenable.
    await request.populate('assignedDriverId', 'name email');
    await request.populate('userId', 'name email');

    const io = req.app.get('io');
    io.to('admin').emit('task_assigned', request);
    io.to('driver').emit('task_assigned', request);
    if (request.userId?._id) io.to(String(request.userId._id)).emit('task_assigned', request);
    res.json(request);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// GET /api/admin/drivers
const getAllDrivers = async (req, res) => {
  try {
    const drivers = await Driver.find().populate('userId', 'name email role');
    res.json(drivers);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// DELETE /api/admin/driver/:id  (deletes Driver profile + User)
const deleteDriver = async (req, res) => {
  try {
    const driver = await Driver.findByIdAndDelete(req.params.id);
    if (!driver) return res.status(404).json({ message: 'Driver not found' });
    await User.findByIdAndDelete(driver.userId);
    res.json({ message: 'Driver removed' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// GET /api/admin/complaints
const getAllComplaints = async (req, res) => {
  try {
    const complaints = await Complaint.find()
      .populate('userId', 'name email')
      .populate('assignedDriverId', 'name email')
      .sort({ createdAt: -1 });
    res.json(complaints);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// PATCH /api/admin/complaint/:id
const updateComplaint = async (req, res) => {
  try {
    const { status, priority } = req.body;
    if (status !== undefined) {
      return res.status(403).json({ message: 'Only the assigned driver can resolve/advance complaint status.' });
    }
    const updateData = {};
    if (priority) updateData.priority = priority;

    const complaint = await Complaint.findByIdAndUpdate(req.params.id, updateData, { new: true }).populate('userId', 'name email');
    if (!complaint) return res.status(404).json({ message: 'Complaint not found' });

    req.app.get('io').to('admin').emit('complaint_updated', complaint);
    res.json(complaint);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// DELETE /api/admin/complaint/:id
const deleteComplaint = async (req, res) => {
  try {
    const complaint = await Complaint.findByIdAndDelete(req.params.id);
    if (!complaint) return res.status(404).json({ message: 'Complaint not found' });
    res.json({ message: 'Complaint deleted' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// GET /api/admin/requests
const getAllRequests = async (req, res) => {
  try {
    const requests = await Request.find()
      .populate('userId', 'name email')
      .populate('assignedDriverId', 'name email')
      .sort({ createdAt: -1 });
    res.json(requests);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

module.exports = {
  getDashboardStats,
  assignDriver,
  getAllDrivers,
  deleteDriver,
  getAllComplaints,
  updateComplaint,
  deleteComplaint,
  getAllRequests,
};
