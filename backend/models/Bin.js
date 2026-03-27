const mongoose = require('mongoose');

const binSchema = new mongoose.Schema({
  location: {
    address: { type: String, required: true },
    lat: { type: Number, required: true },
    lng: { type: Number, required: true },
  },
  // GeoJSON point for efficient nearest-bin lookup.
  // Coordinates are in [lng, lat] order (GeoJSON spec).
  geoLocation: {
    type: { type: String, enum: ['Point'], default: 'Point' },
    coordinates: { type: [Number], required: true, index: '2dsphere' }, // [lng, lat]
  },
  fillLevel: {
    type: Number,
    default: 0,
    min: 0,
    max: 100,
  },
  status: {
    type: String,
    enum: ['Active', 'Full', 'Empty', 'Maintenance'],
    default: 'Active',
  },
  lastCollected: {
    type: Date,
    default: Date.now,
  },
  createdBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    default: null,
  },
  source: {
    type: String,
    enum: ['admin', 'user'],
    default: 'admin',
  },
}, { timestamps: true });

binSchema.index({ geoLocation: '2dsphere' });

module.exports = mongoose.model('Bin', binSchema);
