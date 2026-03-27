const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');
const path = require('path');
const fs = require('fs');

const connectDB = require('../config/db');
const User = require('../models/User');
const Driver = require('../models/Driver');
const Bin = require('../models/Bin');
const Request = require('../models/Request');
const { updateRequestStatus } = require('../controllers/requestController');
const { startAutoDispatchService } = require('../services/autoDispatchService');

jest.setTimeout(60000);

let mongo;

function makeRes() {
  const res = {};
  res.status = jest.fn(() => res);
  res.json = jest.fn(() => res);
  return res;
}

function makeReq({ params, body, user, ioStub }) {
  return {
    params,
    body,
    user,
    app: { get: () => ioStub },
  };
}

describe('Prediction + auto-dispatch workflow', () => {
  beforeAll(async () => {
    // mongodb-memory-server downloads binaries to a cache dir by default (~/.cache),
    // which might be unwritable in this environment. Force it into the repo workspace.
    const downloadDir = path.join(__dirname, '..', '.mongodb-memory-server-cache');
    fs.mkdirSync(downloadDir, { recursive: true });
    process.env.MONGOMS_DOWNLOAD_DIR = downloadDir;

    mongo = await MongoMemoryServer.create();
    process.env.MONGO_URI = mongo.getUri();
    await connectDB();
  });

  afterAll(async () => {
    await mongoose.disconnect();
    if (mongo) await mongo.stop();
  });

  beforeEach(async () => {
    await Request.deleteMany({});
    await Driver.deleteMany({});
    await User.deleteMany({});
    await Bin.deleteMany({});

    jest.restoreAllMocks();

    // Mock AI predictor: return an "about 10 minutes from now" dispatch.
    global.fetch = jest.fn(async () => {
      const predictedFillMs =
        Date.now() + 6 * 60 * 60 * 1000 + 10 * 60 * 1000; // predicted fill = now + lead + 10m
      return {
        ok: true,
        status: 200,
        json: async () => ({
          predicted_fill_ms: predictedFillMs,
          predicted_interval_ms: 60 * 60 * 1000,
          confidence: 0.9,
          reasoning: 'mock',
        }),
      };
    });
  });

  test('creates an AI predicted Pending request when completed history is sufficient', async () => {
    const emitFn = jest.fn();
    const ioStub = { to: () => ({ emit: emitFn }) };

    const citizen = await User.create({
      name: 'Citizen',
      email: 'citizen@test.com',
      password: 'password123',
      role: 'user',
    });

    const driverUser = await User.create({
      name: 'Driver',
      email: 'driver@test.com',
      password: 'password123',
      role: 'driver',
    });
    await Driver.create({ userId: driverUser._id, shiftStatus: 'Active' });

    const bin = await Bin.create({
      location: { address: 'Bin A', lat: 12.34, lng: 56.78 },
      geoLocation: { type: 'Point', coordinates: [56.78, 12.34] },
      fillLevel: 80,
      status: 'Active',
    });

    // Two previous completed pickups + the one we're about to complete => at least 3 total completed.
    await Request.create({
      userId: citizen._id,
      binId: bin._id,
      wasteType: 'General',
      location: 'Bin A',
      lat: 12.34,
      lng: 56.78,
      scheduledDate: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
      status: 'Completed',
      notes: 'prev1',
    });
    await Request.create({
      userId: citizen._id,
      binId: bin._id,
      wasteType: 'General',
      location: 'Bin A',
      lat: 12.34,
      lng: 56.78,
      scheduledDate: new Date(Date.now() - 90 * 60 * 1000).toISOString(),
      status: 'Completed',
      notes: 'prev2',
    });

    const current = await Request.create({
      userId: citizen._id,
      binId: bin._id,
      wasteType: 'General',
      location: 'Bin A',
      lat: 12.34,
      lng: 56.78,
      scheduledDate: new Date(Date.now() + 30 * 60 * 1000).toISOString(),
      status: 'Assigned',
      assignedDriverId: driverUser._id,
      notes: 'current',
    });

    const req = makeReq({
      params: { id: current._id.toString() },
      body: { status: 'Completed' },
      user: { role: 'driver', _id: driverUser._id },
      ioStub,
    });
    const res = makeRes();

    await updateRequestStatus(req, res);

    // AI predicted request should exist and be Pending.
    const predicted = await Request.findOne({
      userId: citizen._id,
      binId: bin._id,
      status: 'Pending',
      notes: { $regex: /Auto-dispatch prediction \(AI\)/ },
    });

    expect(predicted).toBeTruthy();
    expect(predicted.scheduledDate).toBeTruthy();
    expect(global.fetch).toHaveBeenCalled();
  });

  test('does NOT create an AI predicted Pending request when completed history is insufficient', async () => {
    const emitFn = jest.fn();
    const ioStub = { to: () => ({ emit: emitFn }) };

    const citizen = await User.create({
      name: 'Citizen',
      email: 'citizen2@test.com',
      password: 'password123',
      role: 'user',
    });
    const driverUser = await User.create({
      name: 'Driver',
      email: 'driver2@test.com',
      password: 'password123',
      role: 'driver',
    });
    await Driver.create({ userId: driverUser._id, shiftStatus: 'Active' });

    const bin = await Bin.create({
      location: { address: 'Bin B', lat: 1.23, lng: 4.56 },
      geoLocation: { type: 'Point', coordinates: [4.56, 1.23] },
      fillLevel: 60,
      status: 'Active',
    });

    // Only one previous completed pickup + the one we're about to complete => 2 total completed.
    await Request.create({
      userId: citizen._id,
      binId: bin._id,
      wasteType: 'General',
      location: 'Bin B',
      lat: 1.23,
      lng: 4.56,
      scheduledDate: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
      status: 'Completed',
      notes: 'prev1',
    });

    const current = await Request.create({
      userId: citizen._id,
      binId: bin._id,
      wasteType: 'General',
      location: 'Bin B',
      lat: 1.23,
      lng: 4.56,
      scheduledDate: new Date(Date.now() + 30 * 60 * 1000).toISOString(),
      status: 'Assigned',
      assignedDriverId: driverUser._id,
      notes: 'current',
    });

    const req = makeReq({
      params: { id: current._id.toString() },
      body: { status: 'Completed' },
      user: { role: 'driver', _id: driverUser._id },
      ioStub,
    });
    const res = makeRes();

    await updateRequestStatus(req, res);

    const predicted = await Request.findOne({
      userId: citizen._id,
      binId: bin._id,
      status: 'Pending',
      notes: { $regex: /Auto-dispatch prediction \(AI\)/ },
    });

    expect(predicted).toBeNull();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  test('auto-dispatch assigns a due Pending request to an Active driver', async () => {
    const emitFn = jest.fn();
    const ioStub = { to: () => ({ emit: emitFn }) };

    const citizen = await User.create({
      name: 'Citizen3',
      email: 'citizen3@test.com',
      password: 'password123',
      role: 'user',
    });

    const driverUser = await User.create({
      name: 'Driver3',
      email: 'driver3@test.com',
      password: 'password123',
      role: 'driver',
    });
    await Driver.create({ userId: driverUser._id, shiftStatus: 'Active' });

    const bin = await Bin.create({
      location: { address: 'Bin C', lat: 9.87, lng: 6.54 },
      geoLocation: { type: 'Point', coordinates: [6.54, 9.87] },
      fillLevel: 40,
      status: 'Active',
    });

    const due = await Request.create({
      userId: citizen._id,
      binId: bin._id,
      wasteType: 'General',
      location: 'Bin C',
      lat: 9.87,
      lng: 6.54,
      scheduledDate: new Date(Date.now() + 2 * 60 * 1000).toISOString(), // due soon
      status: 'Pending',
      assignedDriverId: null,
      notes: 'due',
    });

    const intervalHandle = startAutoDispatchService(ioStub, { intervalMs: 50, windowMs: 10 * 60 * 1000 });

    // Wait for one auto-dispatch tick.
    await new Promise((r) => setTimeout(r, 120));

    clearInterval(intervalHandle);

    const updated = await Request.findById(due._id);
    expect(updated.status).toBe('Assigned');
    expect(updated.assignedDriverId.toString()).toBe(driverUser._id.toString());
  });
});

