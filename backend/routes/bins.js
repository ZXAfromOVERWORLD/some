const express = require('express');
const router = express.Router();
const { getBins, createBin, registerBinForUser, updateBin, deleteBin } = require('../controllers/binController');
const { protect } = require('../middleware/authMiddleware');
const { requireRole } = require('../middleware/roleMiddleware');

router.get('/', protect, getBins);
router.post('/register', protect, requireRole('user'), registerBinForUser);
router.post('/', protect, requireRole('admin'), createBin);
router.patch('/:id', protect, requireRole('admin'), updateBin);
router.delete('/:id', protect, requireRole('admin'), deleteBin);

module.exports = router;
