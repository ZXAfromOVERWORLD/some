const mongoose = require('mongoose');

const complaintSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
  },
  assignedDriverId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    default: null,
  },
  assignedAt: {
    type: Date,
    default: null,
  },
  description: {
    type: String,
    required: [true, 'Description is required'],
    trim: true,
  },
  location: {
    type: String,
    required: [true, 'Location is required'],
  },
  lat: {
    type: Number,
    default: null,
  },
  lng: {
    type: Number,
    default: null,
  },
  priority: {
    type: String,
    enum: ['Low', 'Medium', 'High'],
    default: 'Medium',
  },
  status: {
    type: String,
    enum: ['Pending', 'In Progress', 'Resolved'],
    default: 'Pending',
  },
  imageUrl: {
    type: String,
    default: '',
  },
  ai: {
    classification: { type: String, default: '' }, // e.g. Biodegradable / Non-biodegradable / Mixed / Unknown
    confidence: { type: Number, default: null },
    provider: { type: String, default: 'waste_classifier' },
    raw: { type: mongoose.Schema.Types.Mixed, default: null },
  },
}, { timestamps: true });

module.exports = mongoose.model('Complaint', complaintSchema);
