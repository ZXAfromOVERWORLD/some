const mongoose = require('mongoose');

const driverSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    unique: true,
  },
  currentLocation: {
    lat: { type: Number, default: 28.6139 },
    lng: { type: Number, default: 77.2090 },
  },
  shiftStatus: {
    type: String,
    enum: ['Active', 'Off-duty', 'On-break'],
    default: 'Active',
  },
  truckType: {
    type: String,
    // Allow empty string so driver profiles can exist before truck type is set.
    enum: ['Mixed', 'Biodegradable', 'Non-biodegradable', ''],
    default: '',
  },
  vehicleNumber: {
    type: String,
    default: '',
  },
}, { timestamps: true });

module.exports = mongoose.model('Driver', driverSchema);
