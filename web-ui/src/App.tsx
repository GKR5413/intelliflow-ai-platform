import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box } from '@mui/material';
import { useAuth } from './contexts/AuthContext.tsx';
import Layout from './components/Layout/Layout.tsx';
import LoginPage from './pages/Auth/LoginPage.tsx';
import RegisterPage from './pages/Auth/RegisterPage.tsx';
import DashboardPage from './pages/Dashboard/DashboardPage.tsx';
import TransactionsPage from './pages/Transactions/TransactionsPage.tsx';
import CreateTransactionPage from './pages/Transactions/CreateTransactionPage.tsx';
import FraudDetectionPage from './pages/FraudDetection/FraudDetectionPage.tsx';
import AnalyticsPage from './pages/Analytics/AnalyticsPage.tsx';
import NotificationsPage from './pages/Notifications/NotificationsPage.tsx';
import ProfilePage from './pages/Profile/ProfilePage.tsx';
import SettingsPage from './pages/Settings/SettingsPage.tsx';
import LoadingSpinner from './components/Common/LoadingSpinner.tsx';

const App: React.FC = () => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <Box 
        sx={{ 
          height: '100vh', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
        }}
      >
        <LoadingSpinner size="large" message="Loading IntelliFlow AI Platform..." />
      </Box>
    );
  }

  return (
    <Routes>
      {/* Public Routes */}
      <Route 
        path="/login" 
        element={!user ? <LoginPage /> : <Navigate to="/dashboard" replace />} 
      />
      <Route 
        path="/register" 
        element={!user ? <RegisterPage /> : <Navigate to="/dashboard" replace />} 
      />
      
      {/* Protected Routes */}
      <Route 
        path="/*" 
        element={user ? (
          <Layout>
            <Routes>
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/transactions" element={<TransactionsPage />} />
              <Route path="/transactions/create" element={<CreateTransactionPage />} />
              <Route path="/fraud-detection" element={<FraudDetectionPage />} />
              <Route path="/analytics" element={<AnalyticsPage />} />
              <Route path="/notifications" element={<NotificationsPage />} />
              <Route path="/profile" element={<ProfilePage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </Layout>
        ) : (
          <Navigate to="/login" replace />
        )} 
      />
    </Routes>
  );
};

export default App;
