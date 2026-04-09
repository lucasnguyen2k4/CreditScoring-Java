import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Sidebar from './components/Sidebar';
import ProtectedRoute from './components/ProtectedRoute';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import DataUploadPage from './pages/DataUploadPage';
import FeatureEngineeringPage from './pages/FeatureEngineeringPage';
import ModelTrainingPage from './pages/ModelTrainingPage';
import ModelApprovalPage from './pages/ModelApprovalPage';
import ShapExplanationPage from './pages/ShapExplanationPage';
import PredictionPage from './pages/PredictionPage';
import AdminSettingsPage from './pages/AdminSettingsPage';

function AppRoutes() {
  const { user } = useAuth();

  if (!user) {
    return (
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    );
  }

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <Routes>
          <Route path="/dashboard" element={
            <ProtectedRoute path="/dashboard"><DashboardPage /></ProtectedRoute>
          } />
          <Route path="/upload" element={
            <ProtectedRoute path="/upload"><DataUploadPage /></ProtectedRoute>
          } />
          <Route path="/feature-engineering" element={
            <ProtectedRoute path="/feature-engineering"><FeatureEngineeringPage /></ProtectedRoute>
          } />
          <Route path="/training" element={
            <ProtectedRoute path="/training"><ModelTrainingPage /></ProtectedRoute>
          } />
          <Route path="/model-approval" element={
            <ProtectedRoute path="/model-approval"><ModelApprovalPage /></ProtectedRoute>
          } />
          <Route path="/shap" element={
            <ProtectedRoute path="/shap"><ShapExplanationPage /></ProtectedRoute>
          } />
          <Route path="/prediction" element={
            <ProtectedRoute path="/prediction"><PredictionPage /></ProtectedRoute>
          } />
          <Route path="/admin" element={
            <ProtectedRoute path="/admin"><AdminSettingsPage /></ProtectedRoute>
          } />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}
