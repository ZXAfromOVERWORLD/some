const express = require('express');
const router = express.Router();
const { createComplaint, getUserComplaints } = require('../controllers/complaintController');
const { protect } = require('../middleware/authMiddleware');
const { requireRole } = require('../middleware/roleMiddleware');

router.post('/', protect, requireRole('user'), createComplaint);
router.get('/user', protect, requireRole('user'), getUserComplaints);

module.exports = router;
