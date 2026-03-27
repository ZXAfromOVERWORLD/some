const cloudinary = require('cloudinary');
const multerCloudinaryStorage = require('multer-storage-cloudinary');
const multer = require('multer');
const path = require('path');

cloudinary.config({
  cloud_name: String(process.env.CLOUDINARY_CLOUD_NAME || '').trim(),
  api_key: String(process.env.CLOUDINARY_API_KEY || '').trim(),
  api_secret: String(process.env.CLOUDINARY_API_SECRET || '').trim(),
});

// Create Cloudinary-backed uploads.
// If Cloudinary is misconfigured, fall back to memory storage so the backend never crashes.
let storage;
try {
  const hasCreds =
    process.env.CLOUDINARY_CLOUD_NAME &&
    process.env.CLOUDINARY_API_KEY &&
    process.env.CLOUDINARY_API_SECRET;

  if (!hasCreds) {
    throw new Error('Missing Cloudinary env credentials');
  }

  // `multer-storage-cloudinary` exports a function that creates the storage.
  storage = multerCloudinaryStorage({
    // multer-storage-cloudinary expects the full cloudinary module (with `.v2`).
    cloudinary,
    params: {
      folder: 'smart-waste-complaints',
      allowedFormats: ['jpg', 'png', 'jpeg', 'webp'],
    },
  });
} catch (err) {
  console.error('Cloudinary upload disabled (fallback to local disk):', err.message);
  storage = multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, path.join(__dirname, '..', 'uploads'));
    },
    filename: (req, file, cb) => {
      const safeOriginal = String(file.originalname || 'upload')
        .replace(/[^a-zA-Z0-9._-]/g, '_')
        .slice(-80);
      cb(null, `${Date.now()}_${safeOriginal}`);
    },
  });
}

const upload = multer({ storage });

module.exports = { cloudinary, upload };
