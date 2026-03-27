const express = require('express');
const router = express.Router();
const { upload } = require('../config/cloudinary');
const { protect } = require('../middleware/authMiddleware');

// POST /api/upload
// Expects a multipart/form-data request with a field named 'image'
router.post('/', protect, upload.single('image'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }
    // multer-storage-cloudinary attaches `path` which is the URL.
    // Local disk fallback attaches `filename`.
    const imageUrl =
      req.file.path ||
      (req.file.filename ? `${req.protocol}://${req.get('host')}/uploads/${req.file.filename}` : '');
    res.json({ imageUrl });
  } catch (error) {
    res.status(500).json({ message: 'Image upload failed', error: error.message });
  }
});

module.exports = router;
