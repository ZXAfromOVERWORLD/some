import api from './api';

export interface Bin {
  _id: string;
  location: { address: string; lat: number; lng: number };
  fillLevel: number;
  status: 'Active' | 'Full' | 'Empty' | 'Maintenance';
  source?: 'admin' | 'user';
  createdBy?: string;
  lastCollected: string;
  createdAt: string;
}

export const getBins = async (): Promise<Bin[]> => {
  const res = await api.get('/bins');
  return res.data;
};

export const createBin = async (data: {
  address: string; lat: number; lng: number; fillLevel?: number; status?: string;
}): Promise<Bin> => {
  const res = await api.post('/bins', data);
  return res.data;
};

export const registerUserBin = async (data: {
  address: string; lat: number; lng: number; fillLevel?: number;
}): Promise<Bin> => {
  const res = await api.post('/bins/register', data);
  return res.data;
};

export const updateBin = async (id: string, data: Partial<{
  address: string; lat: number; lng: number; fillLevel: number; status: string;
}>): Promise<Bin> => {
  const res = await api.patch(`/bins/${id}`, data);
  return res.data;
};

export const deleteBin = async (id: string): Promise<void> => {
  await api.delete(`/bins/${id}`);
};
