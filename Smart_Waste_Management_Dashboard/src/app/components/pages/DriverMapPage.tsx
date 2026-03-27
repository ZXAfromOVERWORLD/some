import { useState, useEffect } from 'react';
import { Navigation, CheckCircle, Trash2, MapPin, User, LogOut, Route, Clock, Menu, X, Target, Loader2, AlertTriangle } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useNavigate } from 'react-router';
import { useSocket } from '../../context/SocketContext';
import { DriverMap } from '../DriverMap';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '../ui/alert-dialog';

import { getDriverTasks, getDriverComplaints, resolveAssignedComplaint, updateTaskStatus, updateDriverShiftStatus, getMyDriverMeta, updateTruckType } from '../../services/driverService';
import { getBins, updateBin } from '../../services/binService';
import { getRoadRoute } from '../../services/routingService';

interface CollectionPoint {
  id: string;
  type: 'bin' | 'user-request' | 'complaint';
  location: string;
  latitude: number;
  longitude: number;
  fillLevel?: number;
  priority: 'high' | 'medium' | 'low';
  status: 'pending' | 'completed';
  details: string;
}

export default function DriverMapPage() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [currentLocation, setCurrentLocation] = useState<[number, number]>([28.4595, 77.0266]);
  const [collectionPoints, setCollectionPoints] = useState<CollectionPoint[]>([]);

  const { socket } = useSocket();
  const [showLogoutDialog, setShowLogoutDialog] = useState(false);
  const [isShiftActive, setIsShiftActive] = useState(false);
  const [truckType, setTruckType] = useState<'Mixed' | 'Biodegradable' | 'Non-biodegradable' | ''>('');
  const [showTruckTypePrompt, setShowTruckTypePrompt] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [routeGeometry, setRouteGeometry] = useState<[number, number][]>([]);
  const [completedCount, setCompletedCount] = useState(0);

  // Keep backend `Driver.shiftStatus` in sync with what the driver sees as "on shift",
  // so the admin panel reflects the correct active/off-duty count.
  const handleToggleShift = async () => {
    const nextShiftStatus: 'Active' | 'Off-duty' = isShiftActive ? 'Off-duty' : 'Active';
    try {
      await updateDriverShiftStatus(nextShiftStatus);
      setIsShiftActive(!isShiftActive);
    } catch (err: any) {
      alert(err.response?.data?.message || 'Failed to update shift status');
    }
  };

  const handleSaveTruckType = async (next: 'Mixed' | 'Biodegradable' | 'Non-biodegradable') => {
    try {
      await updateTruckType(next);
      setTruckType(next);
      setShowTruckTypePrompt(false);
    } catch (err: any) {
      alert(err.response?.data?.message || 'Failed to update truck type');
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setIsLoading(true);
    try {
      const [binsResponse, tasksResponse, complaintsResponse, meta] = await Promise.all([
        getBins(),
        getDriverTasks(),
        getDriverComplaints(),
        getMyDriverMeta(),
      ]);

      const initialCompleted =
        tasksResponse.filter((t) => t.status?.toLowerCase() === 'completed').length +
        complaintsResponse.filter((c: any) => c.status?.toLowerCase() === 'resolved').length;
      setCompletedCount(initialCompleted);

      const binsPoints: CollectionPoint[] = binsResponse
        .filter((bin) => Number(bin.fillLevel) > 80)
        .map(bin => ({
          id: bin._id,
          type: 'bin',
          location: bin.location?.address || `Bin ${bin._id.substring(0, 4)}`,
          latitude: bin.location?.lat || 28.4595,
          longitude: bin.location?.lng || 77.0266,
          fillLevel: bin.fillLevel,
          priority: 'high',
          status: 'pending',
          details: `Bin requires collection (Fill Level: ${bin.fillLevel}%)`,
        }));

      const activeTasks = tasksResponse.filter(
        (task) => task.status.toLowerCase() !== 'completed'
      );

      const taskPoints: CollectionPoint[] = activeTasks.map(task => {
        let status: 'pending' | 'completed' = 'pending';
        if (task.status.toLowerCase() === 'completed') status = 'completed';

        return {
          id: task._id,
          type: 'user-request',
          location: task.location || 'Unknown Location',
          latitude: task.lat || 28.4595 + (Math.random() * 0.05 - 0.025), // Use real map coordinates if available
          longitude: task.lng || 77.0266 + (Math.random() * 0.05 - 0.025),
          priority: 'high',
          status: status,
          details: `User Request: ${task.notes || task.wasteType}`,
        };
      });

      const complaintPoints: CollectionPoint[] = complaintsResponse
        .filter((c: any) => c.status?.toLowerCase() !== 'resolved')
        .map((c: any) => {
        const pr: 'high' | 'medium' | 'low' =
          c.priority === 'High' ? 'high' : c.priority === 'Low' ? 'low' : 'medium';

        return {
          id: c._id,
          type: 'complaint',
          location: c.location || 'Unknown Location',
          latitude: c.lat || 28.4595 + (Math.random() * 0.05 - 0.025),
          longitude: c.lng || 77.0266 + (Math.random() * 0.05 - 0.025),
          priority: pr,
          status: 'pending',
          details: `Complaint: ${c.ai?.classification ? `${c.ai.classification} - ` : ''}${c.description}`,
        };
      });

      setCollectionPoints([...binsPoints, ...taskPoints, ...complaintPoints]);
      setIsShiftActive(meta.shiftStatus === 'Active');
      setTruckType((meta.truckType as any) || '');
      setShowTruckTypePrompt(!meta.truckType);
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to fetch collection points');
    } finally {
      setIsLoading(false);
    }
  };

  // Real-time location broadcast
  useEffect(() => {
    if (!socket || !user || !isShiftActive) return;

    const interval = setInterval(() => {
      setCurrentLocation(prev => {
        // Move truck slowly for demonstration
        const newLat = prev[0] + (Math.random() - 0.5) * 0.001;
        const newLng = prev[1] + (Math.random() - 0.5) * 0.001;
        
        socket.emit('driver_emit_location', {
          driverId: user.id,
          lat: newLat,
          lng: newLng
        });

        return [newLat, newLng];
      });
    }, 3000);

    return () => clearInterval(interval);
  }, [socket, user, isShiftActive]);

  // Real-time task updates
  useEffect(() => {
    if (!socket || !user) return;

    const onTaskAssigned = (task: any) => {
       // Only add if assigned to us
       if (task.assignedDriverId?._id === user.id || task.assignedDriverId === user.id) {
          if (task.status?.toLowerCase() === 'completed') return;
          setCollectionPoints(prev => {
            if (prev.find(p => p.id === task._id)) return prev;
            return [...prev, {
              id: task._id,
              type: 'user-request',
              location: task.location || 'Unknown Location',
              latitude: task.lat || 28.4595 + (Math.random() * 0.05 - 0.025),
              longitude: task.lng || 77.0266 + (Math.random() * 0.05 - 0.025),
              priority: 'high',
              status: 'pending',
              details: `User Request: ${task.notes || task.wasteType}`,
            }];
          });
       }
    };

    const onTaskUpdated = (task: any) => {
       if (task.assignedDriverId?._id === user.id || task.assignedDriverId === user.id) {
           if (task.status?.toLowerCase() === 'completed') {
             setCollectionPoints(prev => prev.filter(p => p.id !== task._id));
           } else {
             setCollectionPoints(prev => prev.map(p => p.id === task._id ? { ...p, status: 'pending' } : p));
           }
       }
    };

    socket.on('task_assigned', onTaskAssigned);
    socket.on('task_updated', onTaskUpdated);

    const onComplaintAssigned = (c: any) => {
      // Only add if assigned to us
      if (c.assignedDriverId?._id === user.id || c.assignedDriverId === user.id) {
        if (c.status?.toLowerCase() === 'resolved') return;
        setCollectionPoints((prev) => {
          if (prev.find((p) => p.id === c._id)) return prev;
          const pr: 'high' | 'medium' | 'low' =
            c.priority === 'High' ? 'high' : c.priority === 'Low' ? 'low' : 'medium';
          return [
            ...prev,
            {
              id: c._id,
              type: 'complaint',
              location: c.location || 'Unknown Location',
              latitude: c.lat || 28.4595 + (Math.random() * 0.05 - 0.025),
              longitude: c.lng || 77.0266 + (Math.random() * 0.05 - 0.025),
              priority: pr,
              status: 'pending',
              details: `Complaint: ${c.ai?.classification ? `${c.ai.classification} - ` : ''}${c.description}`,
            },
          ];
        });
      }
    };

    socket.on('complaint_assigned', onComplaintAssigned);
    const onComplaintUpdated = (c: any) => {
      if (!(c.assignedDriverId?._id === user.id || c.assignedDriverId === user.id)) return;
      if (c.status?.toLowerCase() === 'resolved') {
        setCollectionPoints((prev) => prev.filter((p) => p.id !== c._id));
      }
    };
    socket.on('complaint_updated', onComplaintUpdated);

    return () => {
       socket.off('task_assigned', onTaskAssigned);
       socket.off('task_updated', onTaskUpdated);
       socket.off('complaint_assigned', onComplaintAssigned);
       socket.off('complaint_updated', onComplaintUpdated);
    }
  }, [socket, user]);

  const handleLogout = () => {
    setShowLogoutDialog(true);
  };

  const confirmLogout = () => {
    logout();
    navigate('/login');
    setShowLogoutDialog(false);
  };

  const markAsCompleted = async (id: string, type: 'bin' | 'user-request' | 'complaint') => {
    try {
      if (type === 'user-request') {
        await updateTaskStatus(id, 'Completed');
      } else if (type === 'complaint') {
        await resolveAssignedComplaint(id);
      } else {
        await updateBin(id, { fillLevel: 0 });
      }

      setCollectionPoints(prev => prev.filter((point) => point.id !== id));
      setCompletedCount((prev) => prev + 1);
    } catch (err: any) {
      alert('Failed to mark as completed: ' + (err.response?.data?.message || err.message));
    }
  };

  // Calculate distance between two points (Haversine formula)
  const calculateDistance = (
    lat1: number,
    lon1: number,
    lat2: number,
    lon2: number
  ): number => {
    const R = 6371; // Earth's radius in km
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) *
        Math.cos((lat2 * Math.PI) / 180) *
        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  };

  // Nearest neighbor route optimization (priority-aware)
  const calculateOptimizedRoute = (): [number, number][] => {
    const pending = collectionPoints.filter((p) => p.status === 'pending');
    if (pending.length === 0) return [currentLocation];

    const route: [number, number][] = [currentLocation];
    let current = { latitude: currentLocation[0], longitude: currentLocation[1] };

    const bands: Array<'high' | 'medium' | 'low'> = ['high', 'medium', 'low'];
    for (const band of bands) {
      const bandPoints = pending.filter((p) => p.priority === band);
      while (bandPoints.length > 0) {
        let nearestIndex = 0;
        let shortestDistance = calculateDistance(
          current.latitude,
          current.longitude,
          bandPoints[0].latitude,
          bandPoints[0].longitude
        );

        for (let i = 1; i < bandPoints.length; i++) {
          const distance = calculateDistance(
            current.latitude,
            current.longitude,
            bandPoints[i].latitude,
            bandPoints[i].longitude
          );
          if (distance < shortestDistance) {
            shortestDistance = distance;
            nearestIndex = i;
          }
        }

        const nearest = bandPoints[nearestIndex];
        route.push([nearest.latitude, nearest.longitude]);
        current = nearest;
        bandPoints.splice(nearestIndex, 1);
      }
    }

    return route;
  };

  // Optimized route using nearest neighbor algorithm
  const optimizedRoute: [number, number][] = calculateOptimizedRoute();

  // Fetch a road-following polyline for the ordered stops
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        if (optimizedRoute.length < 2) {
          if (!cancelled) setRouteGeometry([]);
          return;
        }
        const r = await getRoadRoute(optimizedRoute);
        if (!cancelled) setRouteGeometry(r.geometry);
      } catch {
        // Fallback: keep straight-line route if routing fails
        if (!cancelled) setRouteGeometry([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [optimizedRoute]);

  const pendingPoints = collectionPoints.filter((p) => p.status === 'pending');

  // Sort pending points to match the optimized route order
  const orderedPendingPoints = optimizedRoute.slice(1).map(([lat, lng]) => {
    return pendingPoints.find(p => p.latitude === lat && p.longitude === lng);
  }).filter(Boolean) as CollectionPoint[];

  // Separate bins and user requests
  const pendingBins = orderedPendingPoints.filter(p => p.type === 'bin');
  const pendingUserRequests = orderedPendingPoints.filter(p => p.type === 'user-request');

  // Find nearest location
  const nearestLocation = orderedPendingPoints.length > 0 ? orderedPendingPoints[0] : null;
  const nearestDistance = nearestLocation 
    ? calculateDistance(currentLocation[0], currentLocation[1], nearestLocation.latitude, nearestLocation.longitude)
    : 0;

  const priorityConfig = {
    high: { color: 'from-rose-500 to-red-600', shadow: 'shadow-rose-500/20', bg: 'bg-rose-50', border: 'border-rose-200', text: 'text-rose-700' },
    medium: { color: 'from-amber-400 to-orange-500', shadow: 'shadow-amber-500/20', bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-700' },
    low: { color: 'from-blue-400 to-indigo-500', shadow: 'shadow-blue-500/20', bg: 'bg-blue-50', border: 'border-blue-200', text: 'text-blue-700' },
  };

  const renderCollectionPoint = (point: CollectionPoint, index: number) => {
    const config = priorityConfig[point.priority];
    return (
      <div
        key={point.id}
        className={`bg-white border-2 ${config.border} rounded-[1.25rem] p-4 sm:p-5 hover:shadow-lg transition-all duration-300 group`}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3 flex-1">
            <div className={`w-8 h-8 rounded-xl flex items-center justify-center bg-gradient-to-tr ${config.color} text-white shadow-md ${config.shadow} font-black text-sm shrink-0`}>
              {index + 1}
            </div>
            <div className="min-w-0 flex-1">
              <p className="font-bold text-slate-800 text-sm truncate">{point.location}</p>
              <div className="flex items-center gap-2 mt-1.5 flex-wrap">
                {point.type === 'bin' ? (
                  <span className="text-[10px] uppercase tracking-widest font-bold bg-slate-100 text-slate-600 px-2 py-1 rounded-md flex items-center gap-1 border border-slate-200">
                    <Trash2 className="w-3 h-3" /> Bin
                  </span>
                ) : (
                  <span className="text-[10px] uppercase tracking-widest font-bold bg-blue-50 text-blue-600 px-2 py-1 rounded-md flex items-center gap-1 border border-blue-100">
                    <MapPin className="w-3 h-3" /> User Request
                  </span>
                )}
                <span className={`text-[10px] uppercase tracking-widest font-bold px-2 py-1 rounded-md flex items-center gap-1 ${config.bg} ${config.text}`}>
                  {point.priority}
                </span>
              </div>
            </div>
          </div>
        </div>
        {point.fillLevel !== undefined && (
          <div className="mb-3 px-1">
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="text-slate-500 font-semibold">Fill Level</span>
              <span className="font-black text-slate-800">{point.fillLevel}%</span>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden border border-slate-200">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  point.fillLevel > 80 ? 'bg-gradient-to-r from-rose-500 to-red-500' : 'bg-gradient-to-r from-emerald-400 to-emerald-500'
                }`}
                style={{ width: `${point.fillLevel}%` }}
              />
            </div>
          </div>
        )}
        <p className="text-xs text-slate-500 mb-4 px-1 font-medium leading-relaxed truncate">{point.details}</p>
        <button
          onClick={() => markAsCompleted(point.id, point.type)}
          className="w-full bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-white py-2.5 rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-2 shadow-md shadow-emerald-500/20 hover:shadow-lg hover:shadow-emerald-500/30 group-hover:scale-[1.02]"
        >
          <CheckCircle className="w-4 h-4" /> Mark as Collected
        </button>
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-slate-50">
        <Loader2 className="w-10 h-10 animate-spin text-emerald-500" />
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-slate-50 font-sans text-slate-900 selection:bg-emerald-200 overflow-hidden relative">
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-emerald-100/40 rounded-full blur-[120px] pointer-events-none z-0" />
      <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-blue-100/40 rounded-full blur-[120px] pointer-events-none z-0" />

      <header className="bg-white/80 backdrop-blur-xl border-b border-slate-200 z-50 relative shadow-[0_4px_20px_rgb(0,0,0,0.02)]">
        <div className="px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-gradient-to-tr from-emerald-500 to-teal-400 rounded-[1.25rem] flex items-center justify-center shadow-lg shadow-emerald-500/25">
                <Navigation className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl sm:text-2xl font-black text-slate-800 tracking-tight leading-none">SmartRoute</h1>
                <p className="text-xs sm:text-sm font-bold text-emerald-600 uppercase tracking-wider mt-1 hidden sm:block">Driver Navigation System</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3 sm:gap-5">
              <div className="hidden sm:flex items-center gap-3 px-4 py-2 bg-slate-100/80 rounded-2xl border border-slate-200">
                <div className="w-9 h-9 bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl flex items-center justify-center shadow-md">
                  <User className="w-4 h-4 text-white" />
                </div>
                <div className="hidden lg:block pr-2">
                  <p className="text-sm font-bold text-slate-800 leading-tight">{user?.name || 'Driver Profile'}</p>
                  <p className={`text-[10px] font-bold uppercase tracking-wider mt-0.5 ${isShiftActive ? 'text-emerald-500' : 'text-slate-400'}`}>
                    {isShiftActive ? 'Active Shift' : 'Off Duty'}
                  </p>
                  <p className="text-[10px] font-bold uppercase tracking-wider mt-0.5 text-slate-500">
                    Truck: {truckType || 'Not set'}
                  </p>
                </div>
              </div>
              
              <button
                onClick={handleToggleShift}
                className={`hidden sm:flex items-center gap-2.5 px-4 py-2.5 rounded-2xl transition-all duration-300 font-bold border shadow-sm ${
                  isShiftActive 
                    ? 'bg-rose-50 border-rose-200 text-rose-600 hover:bg-rose-100 hover:border-rose-300' 
                    : 'bg-emerald-50 border-emerald-200 text-emerald-600 hover:bg-emerald-100 hover:border-emerald-300'
                }`}
              >
                <Clock className="w-4 h-4" />
                <span className="text-sm hidden lg:block">{isShiftActive ? 'End Shift' : 'Start Shift'}</span>
              </button>

              <button
                onClick={handleLogout}
                className="hidden sm:flex items-center gap-2.5 px-4 py-2.5 bg-white hover:bg-slate-100 rounded-2xl transition-all duration-300 text-slate-600 font-bold border border-slate-200 shadow-sm"
              >
                <LogOut className="w-4 h-4" />
              </button>
              
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="sm:hidden p-3 rounded-2xl bg-slate-100 hover:bg-slate-200 text-slate-700 transition-colors"
              >
                {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
              </button>
            </div>
          </div>
        </div>
      </header>

      {showTruckTypePrompt && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-lg rounded-2xl bg-white shadow-2xl border border-slate-200 overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-200">
              <h3 className="text-lg font-black text-slate-800">Select your truck type</h3>
              <p className="text-sm text-slate-600 mt-1">
                Choose the type of waste your truck is designed to carry.
              </p>
            </div>
            <div className="p-6 grid grid-cols-1 gap-3">
              <button
                onClick={() => handleSaveTruckType('Mixed')}
                className="w-full px-4 py-3 rounded-xl border border-slate-200 hover:bg-slate-50 text-left"
              >
                <div className="font-bold text-slate-800">Mixed trash</div>
                <div className="text-xs text-slate-500">Handles mixed waste pickups</div>
              </button>
              <button
                onClick={() => handleSaveTruckType('Biodegradable')}
                className="w-full px-4 py-3 rounded-xl border border-slate-200 hover:bg-slate-50 text-left"
              >
                <div className="font-bold text-slate-800">Biodegradable trash</div>
                <div className="text-xs text-slate-500">Organic / compostable waste</div>
              </button>
              <button
                onClick={() => handleSaveTruckType('Non-biodegradable')}
                className="w-full px-4 py-3 rounded-xl border border-slate-200 hover:bg-slate-50 text-left"
              >
                <div className="font-bold text-slate-800">Non-biodegradable trash</div>
                <div className="text-xs text-slate-500">Plastic / glass / metals etc.</div>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Mobile Sidebar Drawer */}
      {mobileMenuOpen && (
        <>
          {/* Backdrop/Overlay */}
          <div
            className="sm:hidden fixed inset-0 bg-black/50 z-40 transition-opacity"
            onClick={() => setMobileMenuOpen(false)}
          />

          {/* Sidebar Drawer */}
          <div className="sm:hidden fixed top-0 left-0 bottom-0 w-80 bg-white z-50 shadow-2xl transform transition-transform duration-300 ease-in-out overflow-y-auto">
            {/* Sidebar Header */}
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-[#F1C40F] to-[#F39C12] rounded-xl flex items-center justify-center shadow-lg">
                    <Navigation className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h2 className="text-lg font-bold text-gray-800">Driver Dashboard</h2>
                    <p className="text-xs text-gray-600">Collection Route</p>
                  </div>
                </div>
                <button
                  onClick={() => setMobileMenuOpen(false)}
                  className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <X className="h-5 w-5 text-gray-700" />
                </button>
              </div>

              {/* User Info in Sidebar */}
              {user && (
                <div className="flex items-center gap-3 p-4 bg-gradient-to-r from-[#F1C40F] to-[#F39C12] rounded-xl">
                  <div className="w-12 h-12 bg-white/20 backdrop-blur rounded-full flex items-center justify-center">
                    <User className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <p className="font-semibold text-white">{user?.name}</p>
                    <p className="text-xs text-white/80">Driver</p>
                  </div>
                </div>
              )}
            </div>

            {/* Route Stats in Sidebar */}
            <div className="p-6">
              <h3 className="text-sm font-semibold text-gray-500 uppercase mb-4">Today's Route</h3>
              <div className="space-y-3">
                <div className="bg-blue-50 p-4 rounded-xl border border-blue-200">
                  <div className="flex items-center gap-3">
                    <div className="bg-blue-500 p-2 rounded-lg">
                      <Target className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <p className="text-xs text-gray-600">Pending</p>
                      <p className="text-xl font-bold text-gray-800">{pendingPoints.length}</p>
                    </div>
                  </div>
                </div>
                <div className="bg-green-50 p-4 rounded-xl border border-green-200">
                  <div className="flex items-center gap-3">
                    <div className="bg-green-500 p-2 rounded-lg">
                      <CheckCircle className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <p className="text-xs text-gray-600">Completed</p>
                      <p className="text-xl font-bold text-gray-800">{completedCount}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Logout Button at Bottom */}
            {user && (
              <div className="absolute bottom-0 left-0 right-0 p-6 border-t border-gray-200 bg-white">
                <button
                  onClick={() => {
                    setMobileMenuOpen(false);
                    handleLogout();
                  }}
                  className="w-full flex items-center justify-center gap-3 px-4 py-3 bg-red-50 text-red-600 hover:bg-red-100 rounded-xl transition-all font-medium"
                >
                  <LogOut className="h-5 w-5" />
                  Logout
                </button>
              </div>
            )}
          </div>
        </>
      )}

      <div className="flex-1 flex overflow-hidden relative z-10 w-full bg-slate-100">
        <div className={`${sidebarOpen ? 'w-full sm:w-[420px]' : 'w-0'} bg-white/80 backdrop-blur-3xl border-r border-slate-200 overflow-y-auto transition-all duration-300 ${sidebarOpen ? '' : 'hidden'} custom-scrollbar shadow-[8px_0_30px_rgb(0,0,0,0.05)] z-20`}>
          <div className="p-6 border-b border-slate-100">
            {error && (
              <div className="mb-6 bg-red-50 text-red-600 p-3 rounded-xl border border-red-100 font-medium flex items-center gap-2 text-sm">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                {error}
              </div>
            )}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-[1.5rem] p-5 text-white shadow-xl shadow-slate-900/10 relative overflow-hidden group">
                <div className="absolute top-0 right-0 w-24 h-24 bg-white/10 rounded-full blur-2xl transform translate-x-1/2 -translate-y-1/2 group-hover:scale-150 transition-transform duration-700" />
                <div className="flex items-center gap-3 mb-2 relative z-10">
                  <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center backdrop-blur-md">
                    <Route className="w-4 h-4" />
                  </div>
                  <span className="text-xs font-bold uppercase tracking-widest text-slate-300">Pending</span>
                </div>
                <p className="text-3xl sm:text-4xl font-black relative z-10">{pendingPoints.length}</p>
              </div>
              <div className="bg-gradient-to-br from-emerald-500 to-teal-600 rounded-[1.5rem] p-5 text-white shadow-xl shadow-emerald-500/20 relative overflow-hidden group">
                <div className="absolute top-0 right-0 w-24 h-24 bg-white/20 rounded-full blur-2xl transform translate-x-1/2 -translate-y-1/2 group-hover:scale-150 transition-transform duration-700" />
                <div className="flex items-center gap-3 mb-2 relative z-10">
                  <div className="w-8 h-8 rounded-lg bg-white/20 flex items-center justify-center backdrop-blur-md">
                    <CheckCircle className="w-4 h-4" />
                  </div>
                  <span className="text-xs font-bold uppercase tracking-widest text-emerald-100">Completed</span>
                </div>
                <p className="text-3xl sm:text-4xl font-black relative z-10">{completedCount}</p>
              </div>
            </div>
          </div>

          <div className="p-6">
            {nearestLocation && (
              <div className="mb-8">
                <h3 className="font-black text-slate-800 text-lg mb-4 flex items-center gap-2">
                  <Target className="w-5 h-5 text-emerald-500" /> Next Stop
                </h3>
                <div className="bg-gradient-to-br from-emerald-500 to-teal-600 rounded-[1.5rem] p-5 sm:p-6 text-white shadow-xl shadow-emerald-500/20 relative overflow-hidden">
                  <div className="absolute -right-4 -bottom-4 opacity-10 blur-[2px]">
                    <MapPin className="w-32 h-32" />
                  </div>
                  <div className="relative z-10">
                    <p className="font-black text-lg sm:text-xl mb-3 leading-tight">{nearestLocation.location}</p>
                    <div className="flex items-center gap-2 flex-wrap mb-4">
                      {nearestLocation.type === 'bin' ? (
                        <span className="text-[10px] font-black uppercase tracking-widest bg-white/20 px-3 py-1.5 rounded-lg flex items-center gap-1.5 backdrop-blur-sm">
                          <Trash2 className="w-3.5 h-3.5" /> Bin
                        </span>
                      ) : (
                        <span className="text-[10px] font-black uppercase tracking-widest bg-white/20 px-3 py-1.5 rounded-lg flex items-center gap-1.5 backdrop-blur-sm">
                          <MapPin className="w-3.5 h-3.5" /> User Request
                        </span>
                      )}
                    </div>
                    <div className="flex items-center justify-between mt-4 pt-4 border-t border-white/20">
                      <div>
                        <span className="block text-sm font-semibold text-emerald-50 mb-0.5">Distance</span>
                        <span className="font-black text-xl">{nearestDistance.toFixed(2)} km</span>
                      </div>
                      <div className="text-right">
                        <span className="block text-sm font-semibold text-emerald-50 mb-0.5">Est. Time</span>
                        <span className="font-black text-xl">{Math.max(1, Math.round(nearestDistance * 3))} min</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {pendingUserRequests.length > 0 && (
              <div className="mb-8">
                <h3 className="font-black text-slate-800 text-lg mb-4 flex items-center gap-2">
                  <MapPin className="w-5 h-5 text-blue-500" /> User Requests <span className="bg-blue-100 text-blue-600 px-2 py-0.5 rounded-lg text-sm">{pendingUserRequests.length}</span>
                </h3>
                <div className="space-y-4">
                  {pendingUserRequests.map((point, index) => renderCollectionPoint(point, index))}
                </div>
              </div>
            )}

            {pendingBins.length > 0 && (
              <div className="mb-8">
                <h3 className="font-black text-slate-800 text-lg mb-4 flex items-center gap-2">
                  <Trash2 className="w-5 h-5 text-amber-500" /> Bins <span className="bg-amber-100 text-amber-600 px-2 py-0.5 rounded-lg text-sm">{pendingBins.length}</span>
                </h3>
                <div className="space-y-4">
                  {pendingBins.map((point, index) => renderCollectionPoint(point, pendingUserRequests.length + index))}
                </div>
              </div>
            )}

            
          </div>
        </div>

        {/* Map */}
        <div className="flex-1 relative">
          {/* Toggle Sidebar Button (Mobile) */}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="absolute top-4 left-4 z-[1000] bg-white rounded-lg shadow-lg p-2 sm:hidden"
          >
            <Menu className="w-5 h-5 text-gray-700" />
          </button>

          <DriverMap
            currentLocation={currentLocation}
            collectionPoints={collectionPoints}
            optimizedRoute={optimizedRoute}
            routeGeometry={routeGeometry}
            onMarkCompleted={(id) => markAsCompleted(id, collectionPoints.find(p => p.id === id)?.type || 'bin')}
          />

          {/* Legend */}
          <div className="absolute bottom-6 right-6 lg:left-8 lg:bottom-8 bg-white/90 backdrop-blur-md rounded-[1.5rem] shadow-xl p-5 z-[1000] max-w-[240px] border border-slate-100">
            <p className="font-black text-slate-800 mb-4 text-sm uppercase tracking-wider">Map Legend</p>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-4 h-4 rounded-full bg-gradient-to-tr from-rose-500 to-red-600 shadow-md shadow-rose-500/20 shrink-0" />
                <span className="text-xs font-bold text-slate-600">High Priority</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-4 h-4 rounded-full bg-gradient-to-tr from-amber-400 to-orange-500 shadow-md shadow-amber-500/20 shrink-0" />
                <span className="text-xs font-bold text-slate-600">Medium Priority</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-4 h-4 rounded-full bg-gradient-to-tr from-blue-400 to-indigo-500 shadow-md shadow-blue-500/20 shrink-0" />
                <span className="text-xs font-bold text-slate-600">Low Priority / User</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-5 h-5 rounded-full bg-gradient-to-tr from-emerald-500 to-teal-500 border-2 border-white shadow-md flex items-center justify-center shrink-0">
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" fill="white"/></svg>
                </div>
                <span className="text-xs font-bold text-slate-600">Your Truck</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-4 h-4 rounded-full bg-slate-400 shadow-inner shrink-0" />
                <span className="text-xs font-bold text-slate-600">Completed</span>
              </div>
              <div className="border-t border-slate-200 mt-3 pt-3">
                <div className="flex items-center gap-3">
                  <div className="w-6 h-1 bg-emerald-500 opacity-80 shrink-0" style={{ border: '1.5px dashed #10B981' }} />
                  <span className="text-xs font-bold text-slate-600">Route Path</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <AlertDialog open={showLogoutDialog} onOpenChange={setShowLogoutDialog}>
        <AlertDialogContent className="rounded-[2rem] border-0 shadow-2xl p-6 sm:p-8 max-w-md">
          <AlertDialogHeader>
            <div className="w-16 h-16 bg-rose-100 text-rose-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <LogOut className="w-8 h-8" />
            </div>
            <AlertDialogTitle className="text-center text-2xl font-bold text-slate-800">End Shift?</AlertDialogTitle>
            <AlertDialogDescription className="text-center text-slate-500 text-base mt-2">
              Are you sure you want to end your shift and logout from the driver dashboard?
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter className="mt-8 flex gap-3 sm:space-x-0">
            <AlertDialogCancel className="w-full border-2 border-slate-200 text-slate-600 font-bold hover:bg-slate-50 py-6 rounded-2xl">Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirmLogout} className="w-full bg-rose-500 hover:bg-rose-600 text-white font-bold py-6 rounded-2xl shadow-lg shadow-rose-500/30">End Shift</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background-color: #cbd5e1; border-radius: 20px; }
      `}</style>
    </div>
  );
}