const express = require('express');
const router = express.Router();
const { createRequest, getUserRequests, updateRequestStatus } = require('../controllers/requestController');
const { protect } = require('../middleware/authMiddleware');
const { requireRole } = require('../middleware/roleMiddleware');

router.post('/', protect, requireRole('user'), createRequest);
router.get('/user', protect, requireRole('user'), getUserRequests);
router.patch('/:id/status', protect, requireRole('driver', 'admin'), updateRequestStatus);

module.exports = router;
