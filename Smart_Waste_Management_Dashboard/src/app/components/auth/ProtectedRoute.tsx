import { Navigate, useLocation } from 'react-router';
import { useAuth, UserRole } from '../../context/AuthContext';

interface ProtectedRouteProps {
  children: React.ReactNode;
  allowedRoles?: UserRole[];
}

export default function ProtectedRoute({ children, allowedRoles }: ProtectedRouteProps) {
  const { user } = useAuth();
  const location = useLocation();

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (allowedRoles && !allowedRoles.includes(user.role)) {
    // Redirect to appropriate home based on role
    if (user.role === 'admin') {
      return <Navigate to="/dashboard" replace />;
    } else if (user.role === 'driver') {
      return <Navigate to="/driver-map" replace />;
    } else {
      return <Navigate to="/user-complaints" replace />;
    }
  }

  // Handle root redirect based on role
  if (location.pathname === '/' && !allowedRoles) {
    if (user.role === 'admin') {
      return <Navigate to="/dashboard" replace />;
    } else if (user.role === 'driver') {
      return <Navigate to="/driver-map" replace />;
    } else {
      return <Navigate to="/user-complaints" replace />;
    }
  }

  return <>{children}</>;
}