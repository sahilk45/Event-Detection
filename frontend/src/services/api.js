import axios from 'axios';

// Get API URL from environment variable
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

export const audioAPI = {
  predictAudio: async (audioFile) => {
    const formData = new FormData();
    formData.append('file', audioFile);
    
    const response = await api.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000,
    });
    
    return response.data;
  },
  
  getModelInfo: async () => {
    const response = await api.get('/model-info');
    return response.data;
  },
  
  getSupportedFormats: async () => {
    const response = await api.get('/supported-formats');
    return response.data;
  },
};
