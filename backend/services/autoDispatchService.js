const Request = require('../models/Request');
const Driver = require('../models/Driver');

// Auto-dispatch: assign a driver to due pending pickup requests.
function startAutoDispatchService(io, { intervalMs = 60 * 1000, windowMs = 10 * 60 * 1000 } = {}) {
  const AUTO_ASSIGN_INTERVAL_MS = intervalMs;
  const AUTO_ASSIGN_WINDOW_MS = windowMs;

  return setInterval(async () => {
    try {
      const nowISO = new Date().toISOString();
      const dueISO = new Date(Date.now() + AUTO_ASSIGN_WINDOW_MS).toISOString();
      const pastISO = new Date(Date.now() - AUTO_ASSIGN_WINDOW_MS).toISOString();

      const dueRequests = await Request.find({
        status: 'Pending',
        assignedDriverId: null,
        scheduledDate: { $lte: dueISO, $gte: pastISO },
      }).lean();

      if (!dueRequests.length) return;

      // Pick the first available active driver (simple strategy).
      const driver = await Driver.findOne({ shiftStatus: 'Active' }).populate('userId', 'name email role');
      const driverUserId = driver?.userId?._id;
      if (!driverUserId) return;

      for (const r of dueRequests) {
        // Conditional update avoids races with manual assignment.
        const updated = await Request.findOneAndUpdate(
          { _id: r._id, status: 'Pending', assignedDriverId: null },
          { assignedDriverId: driverUserId, status: 'Assigned' },
          { new: true }
        );

        if (!updated) continue;

        await updated.populate('assignedDriverId', 'name email');
        await updated.populate('userId', 'name email');

        io.to('admin').emit('task_assigned', updated);
        io.to('driver').emit('task_assigned', updated);
        if (updated.userId?._id) io.to(String(updated.userId._id)).emit('task_assigned', updated);
      }
    } catch (err) {
      console.error('Auto-dispatch error:', err.message);
    }
  }, AUTO_ASSIGN_INTERVAL_MS);
}

module.exports = { startAutoDispatchService };

