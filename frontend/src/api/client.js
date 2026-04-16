import axios from 'axios';

const API_BASE = 'http://localhost:8080';

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});

// Attach JWT token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 responses
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authApi = {
  login: (username, password) => api.post('/api/auth/login', { username, password }),
  register: (data) => api.post('/api/auth/register', data),
  me: () => api.get('/api/auth/me'),
};

// User Management API (admin)
export const userApi = {
  getAll: () => api.get('/api/users'),
  create: (data) => api.post('/api/users', data),
  update: (id, data) => api.put(`/api/users/${id}`, data),
  delete: (id) => api.delete(`/api/users/${id}`),
};

// ML Data API
export const dataApi = {
  upload: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/api/ml/data/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  getInfo: (useRaw = false) =>
    api.get('/api/ml/data/info', { params: useRaw ? { use_raw: true } : {} }),
  getPreview: (rows = 100) => api.get(`/api/ml/data/preview?rows=${rows}`),
  generateSample: (records = 100) => api.post(`/api/ml/data/generate-sample?records=${records}`),
  getStats: () => api.get('/api/ml/data/stats'),
  getDistribution: (column, processed = false, bins = 20) =>
    api.get(`/api/ml/data/distribution?column=${encodeURIComponent(column)}&processed=${processed}&bins=${bins}`),
  getCorrelation: (processed = false, method = 'pearson', threshold = 0.8) =>
    api.get(`/api/ml/data/correlation?processed=${processed}&method=${encodeURIComponent(method)}&threshold=${threshold}`),
  getJointDistribution: (columnX, columnY, processed = false, bins = 8) =>
    api.get(`/api/ml/data/joint-distribution?column_x=${encodeURIComponent(columnX)}&column_y=${encodeURIComponent(columnY)}&processed=${processed}&bins=${bins}`),
  getScatterMatrix: (data) => api.post('/api/ml/data/scatter-matrix', data),
  getGroupedAnalysis: (valueColumn, groupColumn, processed = false, topGroups = 10, maxPoints = 2500) =>
    api.get(`/api/ml/data/grouped-analysis?value_column=${encodeURIComponent(valueColumn)}&group_column=${encodeURIComponent(groupColumn)}&processed=${processed}&top_groups=${topGroups}&max_points=${maxPoints}`),
  getScatter2D: (columnX, columnY, processed = false, hueColumn = '', maxPoints = 2500, bins = 12) =>
    api.get(`/api/ml/data/scatter-2d?column_x=${encodeURIComponent(columnX)}&column_y=${encodeURIComponent(columnY)}&processed=${processed}&hue_column=${encodeURIComponent(hueColumn || '')}&max_points=${maxPoints}&bins=${bins}`),
  getCategoricalSummary: (processed = false) =>
    api.get(`/api/ml/data/categorical-summary?processed=${processed}`),
  removeCategorical: (data) =>
    api.post('/api/ml/data/remove-categorical', data),
  cleanInvalidNumbers: (data) => api.post('/api/ml/data/clean-invalid-numbers', data),
  setTarget: (target_column) => api.post('/api/ml/data/set-target', { target_column }),
  handleMissing: (data) => api.post('/api/ml/data/handle-missing', data),
  encode: (data) => api.post('/api/ml/data/encode', data),
  handleOutliers: (data) => api.post('/api/ml/data/outliers', data),
  transformSkewness: (data) => api.post('/api/ml/data/transform-skewness', data),
  binning: (data) => api.post('/api/ml/data/binning', data),
  woeAnalysis: (data) => api.post('/api/ml/data/woe-analysis', data),
  multicollinearity: (data) => api.post('/api/ml/data/multicollinearity', data),
  balance: (data) => api.post('/api/ml/data/balance', data),
  split: (data) => api.post('/api/ml/data/split', data),
  scale: (data) => api.post('/api/ml/data/scale', data),
  featureImportance: (data) => api.post('/api/ml/data/feature-importance', data),
  setSelectedFeatures: (columns) => api.post('/api/ml/data/selected-features', { columns }),
  getSessionInfo: () => api.get('/api/ml/data/session-info'),
};

// Model API
export const modelApi = {
  train: (data) => api.post('/api/ml/model/train', data),
  trainStacking: (data) => api.post('/api/ml/model/train-stacking', data),
  tune: (data) => api.post('/api/ml/model/tune', data),
  tuneStacking: (data) => api.post('/api/ml/model/tune-stacking', data),
  getTuningResults: () => api.get('/api/ml/model/tuning-results'),
  crossValidate: (data) => api.post('/api/ml/model/cross-validate', data),
  getHistory: () => api.get('/api/ml/model/history'),
  selectModel: (model_index) => api.post('/api/ml/model/select', { model_index }),
  approveModel: (data) => api.post('/api/ml/model/approve', data),
  getApprovals: () => api.get('/api/ml/model/approvals'),
  getCurrent: () => api.get('/api/ml/model/current'),
};

// Prediction API
export const predictApi = {
  single: (input_data) => api.post('/api/ml/predict/single', { input_data }),
  getFeatures: () => api.get('/api/ml/predict/features'),
};

// SHAP API
export const shapApi = {
  init: () => api.post('/api/ml/shap/init'),
  getGlobal: () => api.get('/api/ml/shap/global'),
  getLocal: (sample_idx) => api.post('/api/ml/shap/local', { sample_idx }),
};

// LLM API
export const llmApi = {
  analyzeEda: () => api.post('/api/ml/llm/analyze-eda'),
  getEdaSummary: () => api.get('/api/ml/llm/eda-summary'),
  analyzeShapGlobal: () => api.post('/api/ml/llm/analyze-shap-global'),
  analyzeShapLocal: (sample_idx) => api.post('/api/ml/llm/analyze-shap-local', { sample_idx }),
  chat: (question, conversation_history) =>
    api.post('/api/ml/llm/chat', { question, conversation_history }),
  getStatus: () => api.get('/api/ml/llm/status'),
};

export default api;
