// c:\Users\Harshit\Downloads\Smart Waste Management Dashboard (Copy)\Smart_Waste_Management_Dashboard\src\app\context\SocketContext.tsx
import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { io, Socket } from 'socket.io-client';
import { useAuth } from './AuthContext';

interface SocketContextProps {
  socket: Socket | null;
  isConnected: boolean;
  toast: { message: string, type: 'info'|'success'|'warning'|'error', visible: boolean };
  hideToast: () => void;
  showToast: (message: string, type?: 'info'|'success'|'warning'|'error') => void;
}

const SocketContext = createContext<SocketContextProps | undefined>(undefined);

export function SocketProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [toast, setToast] = useState<{ message: string, type: 'info'|'success'|'warning'|'error', visible: boolean }>({ message: '', type: 'info', visible: false });

  const hideToast = () => setToast(prev => ({ ...prev, visible: false }));
  
  const showToast = (message: string, type: 'info'|'success'|'warning'|'error' = 'info') => {
    setToast({ message, type, visible: true });
    setTimeout(() => {
      setToast(prev => ({ ...prev, visible: false }));
    }, 5000);
  };

  useEffect(() => {
    const URL = import.meta.env.VITE_API_URL?.replace('/api', '') || 'http://localhost:5000';
    
    const socketInstance = io(URL, {
      withCredentials: true,
      transports: ['websocket', 'polling'] // Try websocket first
    });

    socketInstance.on('connect', () => {
      setIsConnected(true);
      console.log('Real-time connection established', socketInstance.id);
      
      // Join role based room
      if (user) {
        socketInstance.emit('join_role_room', user.role);
        socketInstance.emit('join_user_room', user.id);
      }
    });

    socketInstance.on('disconnect', () => {
      setIsConnected(false);
      console.log('Real-time connection lost');
    });

    // Global notifications
    socketInstance.on('new_request', () => {
      if (user?.role === 'admin' || user?.role === 'driver') showToast('New garbage pickup request received!', 'info');
    });

    socketInstance.on('task_assigned', (task) => {
       if (user?.role === 'driver' || user?.role === 'admin') showToast(`Task assigned to a driver!`, 'success');
       if (user?.role === 'user' && task.userId?._id === user.id) showToast('A driver has been assigned to your request!', 'success');
    });

    socketInstance.on('task_updated', (task) => {
       if (user?.role === 'user' && task.userId?._id === user.id) showToast(`Your pickup request is now ${task.status}!`, 'success');
       if (user?.role === 'admin') showToast(`A task has been marked as ${task.status}`, 'info');
    });

    socketInstance.on('new_complaint', () => {
      if (user?.role === 'admin') showToast('New Citizen Complaint filed!', 'warning');
    });

    setSocket(socketInstance);

    return () => {
      socketInstance.disconnect();
    };
  }, [user]);

  return (
    <SocketContext.Provider value={{ socket, isConnected, toast, hideToast, showToast }}>
      {children}
      {/* Global Realtime Toast Container */}
      {toast.visible && (
        <div className="fixed top-4 right-4 z-[9999] animate-fade-in-down">
          <div className={`px-6 py-4 rounded-xl shadow-2xl flex items-center gap-3 border backdrop-blur-md ${
             toast.type === 'success' ? 'bg-emerald-900/90 border-emerald-500 text-white' :
             toast.type === 'warning' ? 'bg-amber-500/90 border-amber-300 text-white' :
             toast.type === 'error' ? 'bg-red-900/90 border-red-500 text-white' :
             'bg-[#1F7A63]/90 border-[#4CAF50] text-white'
          }`}>
             <div className="font-medium text-sm">{toast.message}</div>
             <button onClick={hideToast} className="ml-4 opacity-70 hover:opacity-100 text-white shrink-0">✕</button>
          </div>
        </div>
      )}
    </SocketContext.Provider>
  );
}

export const useSocket = () => {
  const context = useContext(SocketContext);
  if (context === undefined) {
    throw new Error('useSocket must be used within a SocketProvider');
  }
  return context;
};
