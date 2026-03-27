require('dotenv').config();
const express = require('express');
const cors = require('cors');
const http = require('http');
const path = require('path');
const { Server } = require('socket.io');
const rateLimit = require('express-rate-limit');
const connectDB = require('./config/db');
const { startAutoDispatchService } = require('./services/autoDispatchService');

// Route imports
const authRoutes = require('./routes/auth');
const binRoutes = require('./routes/bins');
const requestRoutes = require('./routes/request');
const complaintRoutes = require('./routes/complaint');
const driverRoutes = require('./routes/driver');
const adminRoutes = require('./routes/admin');
const uploadRoutes = require('./routes/upload');
const aiRoutes = require('./routes/ai');

const app = express();
const server = http.createServer(app);

// Setup Socket.IO Server
const io = new Server(server, {
  cors: {
    origin: ['http://localhost:5173', 'http://localhost:3000'],
    methods: ['GET', 'POST'],
    credentials: true
  }
});

// Attach io to the Express app so routers/controllers can emit events
app.set('io', io);

// Handle new socket connections
io.on('connection', (socket) => {
  console.log(`📡 New client connected: ${socket.id}`);

  // Optional: Client joins specific role-rooms based on data sent, or just generic rooms
  socket.on('join_role_room', (roleName) => {
    socket.join(roleName);
    console.log(`🔌 Socket ${socket.id} joined room -> ${roleName}`);
  });

  // Per-user room (used for targeted notifications)
  socket.on('join_user_room', (userRoomId) => {
    if (!userRoomId) return;
    socket.join(String(userRoomId));
    console.log(`🔌 Socket ${socket.id} joined user room -> ${userRoomId}`);
  });
  
  // Custom driver location relay
  socket.on('driver_emit_location', (data) => {
    // data = { driverId, lat, lng, vehicleNumber }
    io.to('admin').emit('driver_location_update', data);

    // Persist driver location so "nearest driver" logic works reliably.
    // `driverId` here is the driver's User id (from AuthContext).
    try {
      const Driver = require('./models/Driver');
      if (data?.driverId && Number.isFinite(Number(data?.lat)) && Number.isFinite(Number(data?.lng))) {
        Driver.findOneAndUpdate(
          { userId: data.driverId },
          {
            currentLocation: { lat: Number(data.lat), lng: Number(data.lng) },
            ...(data.vehicleNumber ? { vehicleNumber: String(data.vehicleNumber) } : {}),
          },
          { new: false }
        ).catch(() => {});
      }
    } catch {
      // ignore
    }
  });

  socket.on('disconnect', () => {
    console.log(`🔴 Client disconnected: ${socket.id}`);
  });
});

// Middleware
app.use(cors({
  origin: ['http://localhost:5173', 'http://localhost:3000'],
  credentials: true,
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve locally stored uploads (when Cloudinary is unavailable)
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Rate limiting (basic abuse protection for write/AI endpoints)
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 20, // limit each IP
  standardHeaders: true,
  legacyHeaders: false,
  message: { message: 'Too many auth requests, please try again later.' },
});

const aiLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 10,
  standardHeaders: true,
  legacyHeaders: false,
  message: { message: 'Too many AI requests, please try again later.' },
});

const writeLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 60,
  standardHeaders: true,
  legacyHeaders: false,
  message: { message: 'Too many requests, please try again later.' },
});

// Apply limiters to route groups
app.use('/api/auth', authLimiter);
app.use('/api/ai', aiLimiter);
app.use('/api/upload', writeLimiter);
app.use('/api/complaint', writeLimiter);
app.use('/api/request', writeLimiter);
app.use('/api/admin', writeLimiter);
app.use('/api/driver', writeLimiter);

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/bins', binRoutes);
app.use('/api/request', requestRoutes);
app.use('/api/complaint', complaintRoutes);
app.use('/api/driver', driverRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api/upload', uploadRoutes);
app.use('/api/ai', aiRoutes);

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Smart Waste API is running' });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ message: `Route ${req.originalUrl} not found` });
});

// Global error handler
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(err.status || 500).json({ message: err.message || 'Internal Server Error' });
});

const PORT = process.env.PORT || 5000;

connectDB().then(() => {
  // Use `server.listen` instead of `app.listen` to allow socket.io to intercept http upgrade requests
  server.listen(PORT, () => {
    console.log(`🚀 API + Socket.io Server running on port ${PORT}`);
    // Seed admin user on first run
    seedAdmin();

    // Auto-assign predicted pickup requests when they're due.
    startAutoDispatchService(io);

    // Keep bin fill levels moving over time (used by predictions / UI alerts).
    const enableBinFillSimulation = String(process.env.ENABLE_BIN_FILL_SIMULATION || '').toLowerCase() === 'true';
    if (enableBinFillSimulation) {
      startBinFillSimulation();
    } else {
      console.log('Bin fill simulation disabled (set ENABLE_BIN_FILL_SIMULATION=true to enable).');
    }
  });
});

// Seed admin user if not exists
async function seedAdmin() {
  try {
    const User = require('./models/User');
    const enabled = String(process.env.SEED_ADMIN_ENABLED || '').toLowerCase() === 'true';
    if (!enabled) return;

    const seedEmail = process.env.SEED_ADMIN_EMAIL;
    const seedPassword = process.env.SEED_ADMIN_PASSWORD;

    if (!seedEmail || !seedPassword) {
      console.log('Admin seeding enabled but SEED_ADMIN_EMAIL/SEED_ADMIN_PASSWORD not set.');
      return;
    }

    const existing = await User.findOne({ email: seedEmail });
    if (!existing) {
      await User.create({
        name: 'Admin',
        email: seedEmail,
        password: seedPassword,
        role: 'admin',
      });
      console.log(`✅ Admin user seeded: ${seedEmail}`);
    }
  } catch (err) {
    console.error('Seed error:', err.message);
  }
}

// Auto-dispatch: assign a driver to due pending pickup requests.
// This is what makes the prediction actionable without requiring manual admin assignment.
function startAutoDispatch() {
  // Backwards-compatible wrapper; main startup uses startAutoDispatchService(io).
  return startAutoDispatchService(io);
}

function startBinFillSimulation() {
  const Request = require('./models/Request');
  const Bin = require('./models/Bin');

  const SIM_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes
  const DEFAULT_CYCLE_MS = 7 * 24 * 60 * 60 * 1000; // 7 days fallback

  const computeAvgIntervalMsForBin = async (binId) => {
    const completed = await Request.find({ binId, status: 'Completed' })
      .sort({ updatedAt: -1 })
      .limit(8)
      .select({ updatedAt: 1, createdAt: 1 })
      .lean();

    if (completed.length < 2) return DEFAULT_CYCLE_MS;

    const times = completed
      .slice()
      .reverse()
      .map((r) => new Date(r.updatedAt || r.createdAt).getTime())
      .filter((t) => Number.isFinite(t));

    if (times.length < 2) return DEFAULT_CYCLE_MS;

    const intervals = [];
    for (let i = 1; i < times.length; i++) intervals.push(times[i] - times[i - 1]);
    const lastIntervals = intervals.slice(-3);
    return lastIntervals.reduce((a, b) => a + b, 0) / lastIntervals.length || DEFAULT_CYCLE_MS;
  };

  setInterval(async () => {
    try {
      const bins = await Bin.find().select({ lastCollected: 1, fillLevel: 1, status: 1 }).lean();
      if (!bins.length) return;

      for (const b of bins) {
        const binId = b._id;
        const avgIntervalMs = await computeAvgIntervalMsForBin(binId);
        if (!Number.isFinite(avgIntervalMs) || avgIntervalMs <= 0) continue;

        const elapsedMs = Date.now() - new Date(b.lastCollected).getTime();
        const progress = Math.max(0, Math.min(1, elapsedMs / avgIntervalMs));
        const nextFill = Math.max(0, Math.min(100, Math.round(progress * 100)));

        const nextStatus = nextFill === 0 ? 'Empty' : nextFill >= 80 ? 'Full' : 'Active';

        if (nextFill === b.fillLevel && nextStatus === b.status) continue;

        await Bin.findByIdAndUpdate(
          binId,
          { fillLevel: nextFill, status: nextStatus },
          { new: true }
        );

        // Notify admin dashboards/maps so they can update bin state.
        io.to('admin').emit('bin_updated', { _id: binId, fillLevel: nextFill, status: nextStatus });
      }
    } catch (err) {
      console.error('Bin fill simulation error:', err.message);
    }
  }, SIM_INTERVAL_MS);
}
