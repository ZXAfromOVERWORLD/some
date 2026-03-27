import { useState, useEffect } from "react";
import { Trash2, Navigation, Filter, Truck, MapPin, User, Phone, Calendar, Loader2, AlertTriangle } from "lucide-react";
import LeafletMap from "../LeafletMap";
import { getBins } from "../../services/binService";
import { getAllDrivers, getAllRequests } from "../../services/adminService";
import { useSocket } from "../../context/SocketContext";
import { getRoadRoute } from "../../services/routingService";

interface Bin {
  id: string | number;
  lat: number;
  lng: number;
  fillLevel: number;
  status: "empty" | "medium" | "full";
  predictedOverflow: string;
  location: string;
}

interface UITruck {
  id: string | number;
  lat: number;
  lng: number;
  name: string;
  status: string;
}

interface UserRequest {
  id: string;
  userName: string;
  phone: string;
  address: string;
  lat: number;
  lng: number;
  wasteType: string;
  requestedDate: string;
  status: "pending" | "assigned" | "completed";
  priority: "high" | "medium" | "low";
  // Optional: set when this request is an AI auto-dispatch prediction.
  isAIPredicted?: boolean;
  aiConfidence?: number | null;
  predictedFill?: string | null;
}

export default function MapPage() {
  const [selectedBin, setSelectedBin] = useState<Bin | null>(null);
  const [showFullBinsOnly, setShowFullBinsOnly] = useState(false);
  const [showRoutes, setShowRoutes] = useState(true);

  const [bins, setBins] = useState<Bin[]>([]);
  const [trucks, setTrucks] = useState<UITruck[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  const [userRequests, setUserRequests] = useState<UserRequest[]>([]);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [routeOptimized, setRouteOptimized] = useState(false);
  const [roadRoute, setRoadRoute] = useState<[number, number][]>([]);

  // REALTIME SOCKET LINK
  const { socket } = useSocket();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [apiBins, apiDrivers, apiRequests] = await Promise.all([getBins(), getAllDrivers(), getAllRequests()]);
        
        // Map backend bin status to LeafletMap UI status
        const mappedBins: Bin[] = apiBins.map((b) => {
           let uiStatus: "empty" | "medium" | "full" = "empty";
           if (b.fillLevel > 70) uiStatus = "full";
           else if (b.fillLevel > 30) uiStatus = "medium";
           
           return {
             id: b._id,
             lat: b.location.lat || 0,
             lng: b.location.lng || 0,
             fillLevel: b.fillLevel,
             status: uiStatus,
             predictedOverflow: b.fillLevel > 70 ? "1-3 hours" : "12+ hours",
             location: b.location.address || "Unknown"
           };
        });

        const mappedTrucks: UITruck[] = apiDrivers.map(d => ({
           id: d._id,
           lat: d.currentLocation.lat || 28.6139,
           lng: d.currentLocation.lng || 77.2090,
           name: `${d.userId.name} (${d.vehicleNumber})`,
           status: d.shiftStatus
        }));

        const mappedRequests: UserRequest[] = apiRequests.map(r => ({
           id: r._id,
           userName: (r as any).userId?.name || "Citizen",
           phone: (r as any).userId?.email || "N/A",
           address: r.location,
           lat: r.lat || 28.6139 + (Math.random()-0.5)*0.1, // slightly offset fallback
           lng: r.lng || 77.2090 + (Math.random()-0.5)*0.1,
           wasteType: r.wasteType,
           requestedDate: new Date(r.createdAt || Date.now()).toLocaleString(),
           status: r.status.toLowerCase() as any,
           priority: r.wasteType === 'Hazardous' || r.wasteType === 'Electronic' ? 'high' : 'medium',
           // AI transparency: derived from server-side prediction notes (non-blocking).
           ...(parseAiPredictionFromNotes((r as any).notes) || {}),
        }));

        setBins(mappedBins);
        setTrucks(mappedTrucks);
        setUserRequests(mappedRequests);
      } catch (err: any) {
        setError("Failed to load map data from backend. " + (err.message || ""));
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  // Set up socket event listeners for real-time map pushes
  useEffect(() => {
    if (!socket) return;

    const onNewRequest = (r: any) => {
      const ai = parseAiPredictionFromNotes(r.notes);
      const mappedReq: UserRequest = {
         id: r._id,
         userName: r.userId?.name || "Citizen",
         phone: r.userId?.email || "N/A",
         address: r.location,
         lat: r.lat || 28.6139 + (Math.random()-0.5)*0.1,
         lng: r.lng || 77.2090 + (Math.random()-0.5)*0.1,
         wasteType: r.wasteType,
         requestedDate: new Date(r.createdAt || Date.now()).toLocaleString(),
         status: r.status?.toLowerCase() as any,
         priority: r.wasteType === 'Hazardous' || r.wasteType === 'Electronic' ? 'high' : 'medium',
         ...(ai || {}),
      };
      setUserRequests(prev => [mappedReq, ...prev]);
    };

    const onTaskUpdated = (r: any) => {
       setUserRequests(prev => prev.map(req => req.id === r._id ? { ...req, status: r.status?.toLowerCase() as any } : req));
    };

    const onTaskAssigned = (r: any) => {
       setUserRequests(prev => prev.map(req => req.id === r._id ? { ...req, status: r.status?.toLowerCase() as any } : req));
    };

    const onDriverLocationUpdate = (data: { driverId: string, lat: number, lng: number }) => {
       setTrucks(prev => prev.map(t => t.id === data.driverId ? { ...t, lat: data.lat, lng: data.lng } : t));
    };

    const onBinUpdated = (b: any) => {
       setBins(prev => prev.map(bin => bin.id === b._id ? { ...bin, fillLevel: b.fillLevel, status: b.fillLevel > 70 ? "full" : b.fillLevel > 30 ? "medium" : "empty" } : bin));
    };

    const onNewBin = (b: any) => {
       const mappedBin: Bin = {
         id: b._id,
         lat: b.location?.lat || 0,
         lng: b.location?.lng || 0,
         fillLevel: b.fillLevel || 0,
         status: "empty",
         predictedOverflow: "12+ hours",
         location: b.location?.address || "Unknown"
       };
       setBins(prev => [...prev, mappedBin]);
    };

    socket.on('new_request', onNewRequest);
    socket.on('task_updated', onTaskUpdated);
    socket.on('task_assigned', onTaskAssigned);
    socket.on('driver_location_update', onDriverLocationUpdate);
    socket.on('bin_updated', onBinUpdated);
    socket.on('new_bin', onNewBin);

    return () => {
      socket.off('new_request', onNewRequest);
      socket.off('task_updated', onTaskUpdated);
      socket.off('task_assigned', onTaskAssigned);
      socket.off('driver_location_update', onDriverLocationUpdate);
      socket.off('bin_updated', onBinUpdated);
      socket.off('new_bin', onNewBin);
    };
  }, [socket]);

  const filteredBins = showFullBinsOnly ? bins.filter((b) => b.status === "full") : bins;
  const activeUserRequests = userRequests.filter(
    (r) => r.status === "pending" || r.status === "assigned"
  );

  // Notes format for predictions (created server-side):
  // "Auto-dispatch prediction (AI). Predicted fill: <ISO> (confidence: <number>). <reasoning>"
  const parseAiPredictionFromNotes = (notes?: string) => {
    const text = notes || "";
    const isAIPredicted = text.includes("Auto-dispatch prediction (AI)");
    if (!isAIPredicted) return null;

    const confidenceMatch = text.match(/\(confidence:\s*([0-9.]+)\)/i);
    const confidence = confidenceMatch ? Number(confidenceMatch[1]) : null;

    const fillMatch = text.match(/Predicted fill:\s*([0-9T:\-.\+Z]+)/i);
    const predictedFill = fillMatch ? fillMatch[1] : null;

    return {
      isAIPredicted: true,
      aiConfidence: confidence,
      predictedFill,
    };
  };

  // Calculate distance between two points (Haversine formula)
  const calculateDistance = (lat1: number, lon1: number, lat2: number, lon2: number): number => {
    const R = 6371; // Earth's radius in km
    const dLat = (lat2 - lat1) * (Math.PI / 180);
    const dLon = (lon2 - lon1) * (Math.PI / 180);
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(lat1 * (Math.PI / 180)) * Math.cos(lat2 * (Math.PI / 180)) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  };

  // Nearest neighbor route optimization
  const calculateOptimizedRoute = (): [number, number][] => {
    // 1. Collect all points that need visiting
    const pointsToVisit: { lat: number, lng: number }[] = [];
    
    // Add full bins
    filteredBins.filter(b => b.status === "full").forEach(b => {
      pointsToVisit.push({ lat: b.lat, lng: b.lng });
    });

    // Add pending/assigned user requests
    activeUserRequests.forEach(req => {
      pointsToVisit.push({ lat: req.lat, lng: req.lng });
    });

    if (pointsToVisit.length === 0) return [];

    // 2. Start from the active truck if exists, else default center
    let currentLat = trucks.length > 0 ? trucks[0].lat : 28.6139;
    let currentLng = trucks.length > 0 ? trucks[0].lng : 77.2090;

    const optimizedRoute: [number, number][] = [[currentLat, currentLng]]; // starting point

    // 3. Keep finding the nearest unvisited point
    while (pointsToVisit.length > 0) {
      let nearestIdx = 0;
      let minDistance = Infinity;

      for (let i = 0; i < pointsToVisit.length; i++) {
        const d = calculateDistance(currentLat, currentLng, pointsToVisit[i].lat, pointsToVisit[i].lng);
        if (d < minDistance) {
          minDistance = d;
          nearestIdx = i;
        }
      }

      // Move to the nearest point
      const nextPoint = pointsToVisit[nearestIdx];
      optimizedRoute.push([nextPoint.lat, nextPoint.lng]);
      
      currentLat = nextPoint.lat;
      currentLng = nextPoint.lng;

      // Remove visited point
      pointsToVisit.splice(nearestIdx, 1);
    }

    return optimizedRoute;
  };

  // Compute optimized route only if toggled
  const combinedRoute = routeOptimized ? calculateOptimizedRoute() : [];
  const combinedRouteKey = combinedRoute.map(([lat, lng]) => `${lat},${lng}`).join("|");

  // Fetch a road-following route geometry from OSRM for the optimized stop order.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!routeOptimized || combinedRoute.length < 2) {
        if (!cancelled) setRoadRoute([]);
        return;
      }

      try {
        const res = await getRoadRoute(combinedRoute);
        if (!cancelled) setRoadRoute(res.geometry);
      } catch {
        // Fallback to straight line between stops if routing service is unavailable.
        if (!cancelled) setRoadRoute(combinedRoute);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [routeOptimized, combinedRouteKey]);

  // Single route only (no static demo lines).
  const routes =
    routeOptimized && roadRoute.length > 1
      ? [{ positions: roadRoute, color: "#EF4444", type: "optimized-route" }]
      : [];

  if (isLoading) {
    return (
      <div className="fixed inset-0 top-16 flex items-center justify-center bg-[#0B2F26]">
        <Loader2 className="h-10 w-10 animate-spin text-[#4CAF50]" />
        <span className="ml-3 text-xl text-white">Loading live map data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="fixed inset-0 top-16 flex flex-col items-center justify-center bg-[#0B2F26] text-white">
        <AlertTriangle className="h-16 w-16 text-red-500 mb-4" />
        <h2 className="text-2xl font-bold mb-2">Error Loading Map</h2>
        <p className="text-[#B0BEC5]">{error}</p>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 top-16 flex overflow-hidden">
      {/* Map Container */}
      <div className="flex-1 relative bg-[#0B2F26]">
        <LeafletMap
          bins={filteredBins}
          trucks={trucks}
          userRequests={activeUserRequests}
          showRoutes={showRoutes}
          routes={routes}
          onBinClick={setSelectedBin}
        />

        {/* Map Legend */}
        <div className="absolute bottom-6 left-6 dark-card-elevated rounded-xl p-4 shadow-lg border border-[#1F7A63]/30 z-[1000]">
          <h4 className="font-semibold text-white mb-3 text-sm">Legend</h4>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-[#4CAF50]" />
              <span className="text-xs text-[#B0BEC5]">Empty Bin (0-30%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-[#F1C40F]" />
              <span className="text-xs text-[#B0BEC5]">Medium Bin (31-70%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-[#E74C3C]" />
              <span className="text-xs text-[#B0BEC5]">Full Bin (71-100%)</span>
            </div>
            <div className="flex items-center gap-2">
              <Truck className="w-4 h-4 text-blue-400" />
              <span className="text-xs text-[#B0BEC5]">Active Truck</span>
            </div>
            <div className="flex items-center gap-2">
              <User className="w-4 h-4 text-[#9C27B0]" />
              <span className="text-xs text-[#B0BEC5]">User Request</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-0.5 bg-[#9C27B0]" style={{ borderTop: '2px dashed #9C27B0' }}></div>
              <span className="text-xs text-[#B0BEC5]">Optimized Route</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right Side Panel */}
      <div className="w-80 border-l border-[#1F7A63]/30 bg-[#0F3D2E]/90 backdrop-blur-md p-6 space-y-6 overflow-y-auto">
        <h3 className="text-lg font-semibold text-white">Map Controls</h3>

        {/* Filters */}
        <div className="space-y-4">
          <div className="flex items-center justify-between p-3 bg-[#0F3D2E]/50 rounded-lg border border-[#1F7A63]/20">
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-[#4CAF50]" />
              <span className="text-sm font-medium text-white">Show full bins only</span>
            </div>
            <button
              onClick={() => setShowFullBinsOnly(!showFullBinsOnly)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                showFullBinsOnly ? "bg-[#4CAF50]" : "bg-[#0F3D2E] border border-[#1F7A63]/30"
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  showFullBinsOnly ? "translate-x-6" : "translate-x-1"
                }`}
              />
            </button>
          </div>

          <div className="flex items-center justify-between p-3 bg-[#0F3D2E]/50 rounded-lg border border-[#1F7A63]/20">
            <div className="flex items-center gap-2">
              <Navigation className="h-4 w-4 text-[#4CAF50]" />
              <span className="text-sm font-medium text-white">Show routes</span>
            </div>
            <button
              onClick={() => setShowRoutes(!showRoutes)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                showRoutes ? "bg-[#4CAF50]" : "bg-[#0F3D2E] border border-[#1F7A63]/30"
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  showRoutes ? "translate-x-6" : "translate-x-1"
                }`}
              />
            </button>
          </div>
        </div>

        {/* Optimize Route Button */}
        <button 
          onClick={() => {
            setIsOptimizing(true);
            setTimeout(() => {
              setIsOptimizing(false);
              setRouteOptimized(!routeOptimized);
              setShowRoutes(true); // force show routes
            }, 1000);
          }}
          disabled={isOptimizing}
          className="w-full bg-gradient-to-r from-[#1F7A63] to-[#4CAF50] text-white py-3 rounded-lg font-bold hover:shadow-[0_0_20px_rgba(76,175,80,0.5)] transition-all shadow-md flex items-center justify-center gap-2 transform active:scale-95"
        >
          {isOptimizing ? (
             <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
             <Navigation className="w-5 h-5" />
          )}
          {routeOptimized ? "Clear Optimized Route" : "Optimize Total Route"}
        </button>

        {/* Selected Bin Details */}
        {selectedBin && (
          <div className="border border-[#1F7A63]/30 rounded-lg p-4 bg-[#0F3D2E]/30">
            <h4 className="font-semibold text-white mb-3 flex items-center gap-2">
              <MapPin className="h-4 w-4 text-[#4CAF50]" />
              Selected Bin Details
            </h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-[#B0BEC5]">Bin ID:</span>
                <span className="text-white font-medium truncate max-w-[120px]">#{selectedBin.id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#B0BEC5]">Location:</span>
                <span className="text-white font-medium text-right truncate max-w-[120px]">{selectedBin.location}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#B0BEC5]">Fill Level:</span>
                <span className="text-white font-medium">{selectedBin.fillLevel}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#B0BEC5]">Status:</span>
                <span
                  className={`font-medium ${
                    selectedBin.status === "full"
                      ? "text-[#E74C3C]"
                      : selectedBin.status === "medium"
                      ? "text-[#F1C40F]"
                      : "text-[#4CAF50]"
                  }`}
                >
                  {selectedBin.status.toUpperCase()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#B0BEC5]">Overflow:</span>
                <span className="text-white font-medium">{selectedBin.predictedOverflow}</span>
              </div>
            </div>
          </div>
        )}

        {/* Statistics */}
        <div className="space-y-3">
          <h4 className="font-semibold text-white text-sm">Quick Stats</h4>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-red-900/20 border border-[#E74C3C]/30 rounded-lg p-3 text-center">
              <div className="text-2xl font-semibold text-[#E74C3C]">
                {bins.filter((b) => b.status === "full").length}
              </div>
              <div className="text-xs text-[#B0BEC5] mt-1">Full Bins</div>
            </div>
            <div className="bg-green-900/20 border border-[#4CAF50]/30 rounded-lg p-3 text-center">
              <div className="text-2xl font-semibold text-[#4CAF50]">
                {bins.filter((b) => b.status === "empty").length}
              </div>
              <div className="text-xs text-[#B0BEC5] mt-1">Empty Bins</div>
            </div>
          </div>
        </div>

        {/* User Pickup Requests */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="font-semibold text-white text-sm flex items-center gap-2">
              <User className="h-4 w-4 text-[#9C27B0]" />
              User Pickup Requests
            </h4>
            <span className="text-xs text-[#B0BEC5] bg-[#9C27B0]/20 px-2 py-1 rounded-full border border-[#9C27B0]/30">
              {userRequests.filter(r => r.status === "pending" || r.status === "assigned").length} Active
            </span>
          </div>
          
          <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
            {userRequests.map((request) => (
              <div
                key={request.id}
                className={`p-3 rounded-lg border transition-all hover:shadow-lg cursor-pointer ${
                  request.status === "completed"
                    ? "bg-[#0F3D2E]/20 border-[#4CAF50]/20 opacity-60"
                    : request.priority === "high"
                    ? "bg-[#9C27B0]/10 border-[#9C27B0]/30"
                    : "bg-[#0F3D2E]/30 border-[#1F7A63]/30"
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h5 className="text-sm font-semibold text-white">{request.userName}</h5>
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full ${
                          request.status === "pending"
                            ? "bg-yellow-900/30 text-[#F1C40F] border border-[#F1C40F]/30"
                            : request.status === "assigned"
                            ? "bg-blue-900/30 text-[#42A5F5] border border-[#42A5F5]/30"
                            : "bg-green-900/30 text-[#4CAF50] border border-[#4CAF50]/30"
                        }`}
                      >
                        {request.status}
                      </span>
                      {request.isAIPredicted && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-[#9C27B0]/20 text-[#9C27B0] border border-[#9C27B0]/30">
                          AI{request.aiConfidence != null ? ` ${(request.aiConfidence * 100).toFixed(0)}%` : ''}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-1 text-xs text-[#B0BEC5] mb-1">
                      <MapPin className="h-3 w-3" />
                      <span className="line-clamp-1">{request.address}</span>
                    </div>
                    <div className="flex items-center gap-1 text-xs text-[#B0BEC5]">
                      <Phone className="h-3 w-3" />
                      <span>{request.phone}</span>
                    </div>
                  </div>
                  <div
                    className={`px-2 py-1 rounded text-xs font-medium ${
                      request.priority === "high"
                        ? "bg-[#E74C3C]/20 text-[#E74C3C]"
                        : request.priority === "medium"
                        ? "bg-[#F1C40F]/20 text-[#F1C40F]"
                        : "bg-[#4CAF50]/20 text-[#4CAF50]"
                    }`}
                  >
                    {request.priority}
                  </div>
                </div>
                
                <div className="flex items-center justify-between text-xs pt-2 border-t border-[#1F7A63]/20">
                  <span className="text-[#B0BEC5]">{request.wasteType}</span>
                  <div className="flex items-center gap-1 text-[#B0BEC5]">
                    <Calendar className="h-3 w-3" />
                    <span>{request.requestedDate}</span>
                  </div>
                </div>

                {request.isAIPredicted && request.predictedFill && (
                  <div className="text-xs text-[#9C27B0] pt-1">
                    Predicted fill: {new Date(request.predictedFill).toLocaleString()}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Route Info */}
          {routeOptimized && combinedRoute.length > 0 && (
            <div className="mt-3 p-4 bg-gradient-to-r from-[#9C27B0]/20 to-purple-900/10 border border-[#9C27B0]/50 rounded-xl shadow-lg border-l-4 border-l-[#9C27B0] animate-fade-in relative overflow-hidden">
              <div className="absolute -right-4 -top-4 opacity-10">
                 <Navigation className="w-24 h-24 text-white" />
              </div>
              <div className="flex items-center gap-2 mb-2 relative z-10">
                <Navigation className="h-5 w-5 text-[#E040FB]" />
                <span className="text-sm font-bold text-white tracking-widest uppercase">Smart Route Active</span>
              </div>
              <p className="text-xs text-[#E1BEE7] mb-3 leading-relaxed relative z-10 font-medium">
                The map is now displaying the most efficient path connecting all <strong>Full Bins</strong> and pending <strong>Citizen Requests</strong>.
              </p>
              <div className="bg-[#0F3D2E]/50 rounded-lg p-2 flex justify-around items-center text-xs relative z-10 border border-[#1F7A63]">
                  <div className="text-center">
                    <span className="block text-[#B0BEC5] mb-1">Pickups</span>
                    <span className="text-white font-black text-lg">{combinedRoute.length - 1}</span>
                  </div>
                  <div className="w-px h-8 bg-[#1F7A63]"></div>
                  <div className="text-center">
                    <span className="block text-[#B0BEC5] mb-1">Est. Time</span>
                    <span className="text-emerald-400 font-black text-lg">{Math.max(15, (combinedRoute.length - 1) * 8)}m</span>
                  </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}