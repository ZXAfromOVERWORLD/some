const mongoose = require('mongoose');

const requestSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
  },
  wasteType: {
    type: String,
    enum: ['General', 'Recyclable', 'Organic', 'Hazardous', 'Electronic'],
    default: 'General',
  },
  location: {
    type: String,
    required: [true, 'Location is required'],
  },
  lat: {
    type: Number,
  },
  lng: {
    type: Number,
  },
  scheduledDate: {
    type: String,
    required: true,
  },
  scheduledTime: {
    type: String,
    default: '',
  },
  status: {
    type: String,
    enum: ['Pending', 'Assigned', 'In Progress', 'Completed'],
    default: 'Pending',
  },
  assignedDriverId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    default: null,
  },
  // Nearest bin associated with this pickup request (used for predictions)
  binId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Bin',
    default: null,
  },
  notes: {
    type: String,
    default: '',
  },
}, { timestamps: true });

module.exports = mongoose.model('Request', requestSchema);
