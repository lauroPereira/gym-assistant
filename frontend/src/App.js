import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './contexts/AuthContext';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Dashboard from './pages/Dashboard';
import Login from './pages/Login';
import Register from './pages/Register';
import TrainingPlan from './pages/TrainingPlan';
import DietPlan from './pages/DietPlan';
import Habits from './pages/Habits';
import QualityOfLife from './pages/QualityOfLife';
import Profile from './pages/Profile';
import LoadingSpinner from './components/LoadingSpinner';

function App() {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <div className="App">
      <Routes>
        {/* Public routes */}
        <Route 
          path="/login" 
          element={!isAuthenticated ? <Login /> : <Navigate to="/dashboard" />} 
        />
        <Route 
          path="/register" 
          element={!isAuthenticated ? <Register /> : <Navigate to="/dashboard" />} 
        />
        
        {/* Protected routes */}
        <Route 
          path="/" 
          element={isAuthenticated ? <Layout /> : <Navigate to="/login" />}
        >
          <Route index element={<Navigate to="/dashboard" />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="agents/training" element={<AgentTraining />} />
          <Route path="agents/diet" element={<AgentDiet />} />
          <Route path="agents/habit" element={<AgentHabit />} />
          <Route path="agents/qol" element={<AgentQol />} />
          <Route path="orchestrator" element={<Orchestrator />} />
          <Route path="training" element={<TrainingPlan />} />
          <Route path="diet" element={<DietPlan />} />
          <Route path="habits" element={<Habits />} />
          <Route path="quality-of-life" element={<QualityOfLife />} />
          <Route path="profile" element={<Profile />} />
        </Route>
        
        {/* Catch all route */}
        <Route path="*" element={<Navigate to={isAuthenticated ? "/dashboard" : "/login"} />} />
      </Routes>
    </div>
  );
}

export default App;
