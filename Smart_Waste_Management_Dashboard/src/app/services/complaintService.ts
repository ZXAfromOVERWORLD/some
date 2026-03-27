import api from './api';

export interface Complaint {
  _id: string;
  userId: string | { _id: string; name: string; email: string };
  assignedDriverId?: null | { _id: string; name: string; email: string } | string;
  assignedAt?: string | null;
  description: string;
  location: string;
  lat?: number | null;
  lng?: number | null;
  priority: 'Low' | 'Medium' | 'High';
  status: 'Pending' | 'In Progress' | 'Resolved';
  imageUrl?: string;
  ai?: {
    classification?: string;
    confidence?: number | null;
    provider?: string;
    raw?: any;
  };
  createdAt: string;
}

export const createComplaint = async (data: {
  description: string;
  location: string;
  priority?: string;
  imageUrl?: string;
  lat?: number;
  lng?: number;
  ai?: {
    classification?: string;
    confidence?: number | null;
    provider?: string;
    raw?: any;
  };
}): Promise<Complaint> => {
  const res = await api.post('/complaint', data);
  return res.data;
};

export const getMyComplaints = async (): Promise<Complaint[]> => {
  const res = await api.get('/complaint/user');
  return res.data;
};
