const express = require('express');
const router = express.Router();
const { getDriverTasks, getDriverComplaints, resolveAssignedComplaint, updateDriverShiftStatus, getDriverShiftStatus, updateDriverTruckType } = require('../controllers/driverController');
const { updateRequestStatus } = require('../controllers/requestController');
const { protect } = require('../middleware/authMiddleware');
const { requireRole } = require('../middleware/roleMiddleware');

router.get('/tasks', protect, requireRole('driver'), getDriverTasks);
router.get('/complaints', protect, requireRole('driver'), getDriverComplaints);
router.patch('/complaint/:id/resolve', protect, requireRole('driver'), resolveAssignedComplaint);
router.patch('/shift', protect, requireRole('driver'), updateDriverShiftStatus);
router.get('/shift', protect, requireRole('driver'), getDriverShiftStatus);
router.patch('/truck-type', protect, requireRole('driver'), updateDriverTruckType);
router.patch('/request/:id/status', protect, requireRole('driver', 'admin'), updateRequestStatus);

module.exports = router;
