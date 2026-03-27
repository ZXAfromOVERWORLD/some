import api from './api';

export interface CollectionRequest {
  _id: string;
  userId: string;
  wasteType: string;
  location: string;
  lat?: number;
  lng?: number;
  scheduledDate: string;
  scheduledTime: string;
  status: 'Pending' | 'Assigned' | 'In Progress' | 'Completed';
  assignedDriverId?: { _id: string; name: string; email: string } | null;
  notes: string;
  createdAt: string;
}

export const createRequest = async (data: {
  wasteType: string;
  location: string;
  scheduledDate: string;
  scheduledTime?: string;
  notes?: string;
}): Promise<CollectionRequest> => {
  const res = await api.post('/request', data);
  return res.data;
};

export const getMyRequests = async (): Promise<CollectionRequest[]> => {
  const res = await api.get('/request/user');
  return res.data;
};

export const updateRequestStatus = async (
  id: string,
  status: string
): Promise<CollectionRequest> => {
  const res = await api.patch(`/request/${id}/status`, { status });
  return res.data;
};
