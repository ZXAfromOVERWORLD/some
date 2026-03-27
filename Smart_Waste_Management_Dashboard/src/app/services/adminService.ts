import api from './api';
import type { CollectionRequest } from './requestService';
import type { Complaint } from './complaintService';

export interface DriverProfile {
  _id: string;
  userId: { _id: string; name: string; email: string };
  shiftStatus: 'Active' | 'Off-duty' | 'On-break';
  truckType?: 'Mixed' | 'Biodegradable' | 'Non-biodegradable' | '';
  currentLocation: { lat: number; lng: number };
  vehicleNumber: string;
}

export interface DashboardStats {
  stats: {
    totalBins: number;
    fullBins: number;
    activeDrivers: number;
    alerts: number;
  };
  weeklyData: { date: string; count: number }[];
  recentComplaints: Complaint[];
}

export const getDashboardStats = async (): Promise<DashboardStats> => {
  const res = await api.get('/admin/stats');
  return res.data;
};

export const assignDriver = async (
  requestId: string,
  driverId: string
): Promise<CollectionRequest> => {
  const res = await api.post('/admin/assign', { requestId, driverId });
  return res.data;
};

export const getAllDrivers = async (): Promise<DriverProfile[]> => {
  const res = await api.get('/admin/drivers');
  return res.data;
};

export const deleteDriver = async (id: string): Promise<void> => {
  await api.delete(`/admin/driver/${id}`);
};

export const getAllComplaints = async (): Promise<Complaint[]> => {
  const res = await api.get('/admin/complaints');
  return res.data;
};

export const updateComplaint = async (
  id: string,
  data: { status?: string; priority?: string }
): Promise<Complaint> => {
  const res = await api.patch(`/admin/complaint/${id}`, data);
  return res.data;
};

export const deleteComplaint = async (id: string): Promise<void> => {
  await api.delete(`/admin/complaint/${id}`);
};

export const getAllRequests = async (): Promise<CollectionRequest[]> => {
  const res = await api.get('/admin/requests');
  return res.data;
};
