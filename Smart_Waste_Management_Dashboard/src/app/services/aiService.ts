import api from './api';

export type WasteClassification =
  | 'Biodegradable'
  | 'Non-biodegradable'
  | 'Mixed'
  | 'Unknown';

export interface WasteClassificationResponse {
  classification: WasteClassification;
  confidence: number;
  detected_objects: string[];
  reasoning: string;
}

export const classifyWasteFromImage = async (file: File): Promise<WasteClassificationResponse> => {
  const formData = new FormData();
  formData.append('image', file);

  const res = await api.post('/ai/classify', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  return res.data as WasteClassificationResponse;
};

