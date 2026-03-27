const express = require('express');
const router = express.Router();
const {
  getDashboardStats,
  assignDriver,
  getAllDrivers,
  deleteDriver,
  getAllComplaints,
  updateComplaint,
  deleteComplaint,
  getAllRequests,
} = require('../controllers/adminController');
const { protect } = require('../middleware/authMiddleware');
const { requireRole } = require('../middleware/roleMiddleware');

const adminOnly = [protect, requireRole('admin')];

router.get('/stats', ...adminOnly, getDashboardStats);
router.post('/assign', ...adminOnly, assignDriver);
router.get('/drivers', ...adminOnly, getAllDrivers);
router.delete('/driver/:id', ...adminOnly, deleteDriver);
router.get('/complaints', ...adminOnly, getAllComplaints);
router.patch('/complaint/:id', ...adminOnly, updateComplaint);
router.delete('/complaint/:id', ...adminOnly, deleteComplaint);
router.get('/requests', ...adminOnly, getAllRequests);

module.exports = router;
