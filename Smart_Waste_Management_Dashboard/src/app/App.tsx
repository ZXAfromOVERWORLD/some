import { RouterProvider } from 'react-router';
import { router } from './routes';
import { AuthProvider } from './context/AuthContext';
import { SocketProvider } from './context/SocketContext';
import { AnimatedBackground } from './components/AnimatedBackground';

export default function App() {
  return (
    <AuthProvider>
      <SocketProvider>
        <div className="relative min-h-screen">
          <AnimatedBackground />
          <div className="relative z-10">
            <RouterProvider router={router} />
          </div>
        </div>
      </SocketProvider>
    </AuthProvider>
  );
}