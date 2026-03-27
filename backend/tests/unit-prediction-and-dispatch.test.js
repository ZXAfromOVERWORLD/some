jest.mock('../models/Request', () => ({
  find: jest.fn(),
  findOneAndUpdate: jest.fn(),
  create: jest.fn(),
}));

jest.mock('../models/Bin', () => ({
  findById: jest.fn(),
}));

jest.mock('../models/Driver', () => ({
  findOne: jest.fn(),
}));

const { startAutoDispatchService } = require('../services/autoDispatchService');
const { updateRequestStatus } = require('../controllers/requestController');

const Request = require('../models/Request');
const Bin = require('../models/Bin');
const Driver = require('../models/Driver');

jest.setTimeout(30000);

function makeRes() {
  const res = {};
  res.status = jest.fn(() => res);
  res.json = jest.fn(() => res);
  return res;
}

function makeReq({ id, nextStatus, user, ioStub }) {
  return {
    params: { id },
    body: { status: nextStatus },
    user,
    app: { get: () => ioStub },
  };
}

function makeIoStub() {
  const emitFn = jest.fn();
  return {
    emitFn,
    to: () => ({ emit: emitFn }),
  };
}

describe('Unit: prediction + auto-dispatch', () => {
  beforeEach(() => {
    jest.resetAllMocks();
    global.fetch = jest.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        predicted_fill_ms: Date.now() + 6 * 60 * 60 * 1000 + 10 * 60 * 1000,
        predicted_interval_ms: 60 * 60 * 1000,
        confidence: 0.9,
        reasoning: 'mock',
      }),
    }));
  });

  test('creates an AI predicted Pending request with sufficient completed history', async () => {
    const ioStub = makeIoStub();

    const citizenId = 'citizen1';
    const driverUserId = 'driver1';
    const binId = 'bin1';
    const currentRequestId = 'reqCurrent';

    const updatedDoc = {
      _id: currentRequestId,
      binId,
      userId: citizenId,
      wasteType: 'General',
      notes: 'current',
      async populate(field) {
        if (field === 'userId') this.userId = { _id: citizenId, name: 'Citizen', email: 'c@test.com' };
        return this;
      },
    };

    Request.findOneAndUpdate.mockResolvedValue(updatedDoc);

    Bin.findById.mockResolvedValue({
      _id: binId,
      location: { address: 'Bin A', lat: 12.34, lng: 56.78 },
      fillLevel: 80,
      status: 'Active',
      async save() {},
    });

    const completedRequests = [
      { _id: 'c1', updatedAt: new Date(Date.now() - 30 * 60 * 1000) },
      { _id: 'c2', updatedAt: new Date(Date.now() - 20 * 60 * 1000) },
      { _id: 'c3', updatedAt: new Date(Date.now() - 10 * 60 * 1000) },
    ];

    // Completed history query: Request.find(...).sort(...).limit(...)
    Request.find.mockImplementation((filter) => {
      if (filter && filter.status === 'Completed') {
        return {
          sort: () => ({
            limit: () => Promise.resolve(completedRequests),
          }),
        };
      }
      // existingPending query: Request.find({...})
      if (filter && filter.status && filter.status.$in) {
        return Promise.resolve([]);
      }
      return Promise.resolve([]);
    });

    const predictedReqDoc = {
      _id: 'pred1',
      userId: citizenId,
      async populate() {
        this.userId = { _id: citizenId, name: 'Citizen', email: 'c@test.com' };
        return this;
      },
    };
    Request.create.mockResolvedValue(predictedReqDoc);

    const req = makeReq({
      id: currentRequestId,
      nextStatus: 'Completed',
      user: { role: 'driver', _id: driverUserId },
      ioStub,
    });
    const res = makeRes();

    await updateRequestStatus(req, res);

    expect(Request.create).toHaveBeenCalledTimes(1);
    const created = Request.create.mock.calls[0][0];
    expect(created.userId).toBe(citizenId);
    expect(created.binId).toBe(binId);
    expect(created.status).toBe('Pending');

    expect(global.fetch).toHaveBeenCalled();
  });

  test('does not create a prediction when completed history is insufficient', async () => {
    const ioStub = makeIoStub();

    const citizenId = 'citizen2';
    const driverUserId = 'driver2';
    const binId = 'bin2';
    const currentRequestId = 'reqCurrent2';

    const updatedDoc = {
      _id: currentRequestId,
      binId,
      userId: citizenId,
      wasteType: 'General',
      notes: 'current',
      async populate(field) {
        if (field === 'userId') this.userId = { _id: citizenId, name: 'Citizen', email: 'c@test.com' };
        return this;
      },
    };
    Request.findOneAndUpdate.mockResolvedValue(updatedDoc);

    Bin.findById.mockResolvedValue({
      _id: binId,
      location: { address: 'Bin B', lat: 1.23, lng: 4.56 },
      fillLevel: 60,
      status: 'Active',
      async save() {},
    });

    const completedRequests = [
      { _id: 'p1', updatedAt: new Date(Date.now() - 20 * 60 * 1000) },
      { _id: 'p2', updatedAt: new Date(Date.now() - 10 * 60 * 1000) },
    ];

    Request.find.mockImplementation((filter) => {
      if (filter && filter.status === 'Completed') {
        return {
          sort: () => ({
            limit: () => Promise.resolve(completedRequests),
          }),
        };
      }
      return Promise.resolve([]);
    });

    const req = makeReq({
      id: currentRequestId,
      nextStatus: 'Completed',
      user: { role: 'driver', _id: driverUserId },
      ioStub,
    });
    const res = makeRes();

    await updateRequestStatus(req, res);

    expect(Request.create).not.toHaveBeenCalled();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  test('auto-dispatch assigns a due Pending request to an Active driver', async () => {
    const ioStub = makeIoStub();

    const dueRequests = [{ _id: 'due1' }];
    const driverUserId = 'driverUserX';

    Driver.findOne.mockReturnValue({
      populate: () => Promise.resolve({ userId: { _id: driverUserId } }),
    });

    const updatedDoc = {
      userId: { _id: 'citizen3' },
      assignedDriverId: { _id: driverUserId },
      async populate() {
        return this;
      },
    };

    Request.find.mockImplementation((filter) => {
      if (filter && filter.status === 'Pending' && filter.assignedDriverId === null) {
        return { lean: () => Promise.resolve(dueRequests) };
      }
      return { lean: () => Promise.resolve([]) };
    });

    Request.findOneAndUpdate.mockResolvedValue(updatedDoc);

    const intervalHandle = startAutoDispatchService(ioStub, { intervalMs: 25, windowMs: 10 * 60 * 1000 });
    await new Promise((r) => setTimeout(r, 80));
    clearInterval(intervalHandle);

    expect(Request.findOneAndUpdate).toHaveBeenCalled();
    const [calledFilter, calledUpdate] = Request.findOneAndUpdate.mock.calls[0];
    expect(calledFilter.status).toBe('Pending');
    expect(calledUpdate.status).toBe('Assigned');
    expect(calledUpdate.assignedDriverId).toBe(driverUserId);
  });
});

