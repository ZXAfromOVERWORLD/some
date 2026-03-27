import { Navigate } from 'react-router';
import { useAuth } from '../../context/AuthContext';

export default function RoleBasedRedirect() {
  const { user } = useAuth();

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (user.role === 'admin') {
    return <Navigate to="/dashboard" replace />;
  } else if (user.role === 'driver') {
    return <Navigate to="/driver-map" replace />;
  } else {
    return <Navigate to="/user-collection" replace />;
  }
}