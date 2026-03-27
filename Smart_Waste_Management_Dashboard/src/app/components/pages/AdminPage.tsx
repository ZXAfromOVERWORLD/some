import { useState, useEffect, useRef } from "react";
import { Plus, Trash2, Truck, MapPin, Users, Settings as SettingsIcon, Loader2, AlertTriangle, Search } from "lucide-react";
import { getBins, createBin, deleteBin, Bin } from "../../services/binService";
import { getAllDrivers, getAllComplaints, updateComplaint, getAllRequests, assignDriver, DriverProfile } from "../../services/adminService";
import type { Complaint } from "../../services/complaintService";
import type { CollectionRequest } from "../../services/requestService";
import { format } from "date-fns";
import LocationPickerMap from "../LocationPickerMap";
import { useSocket } from "../../context/SocketContext";

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState<"bins" | "trucks" | "complaints">("bins");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  // Data states
  const [bins, setBins] = useState<Bin[]>([]);
  const [drivers, setDrivers] = useState<DriverProfile[]>([]);
  const [complaints, setComplaints] = useState<Complaint[]>([]);
  const [selectedComplaint, setSelectedComplaint] = useState<Complaint | null>(null);
  const [requests, setRequests] = useState<CollectionRequest[]>([]);

  // Form states
  const [newBinLocation, setNewBinLocation] = useState("");
  const [newBinCoords, setNewBinCoords] = useState("");
  
  const [selectedDriverId, setSelectedDriverId] = useState("");
  const [selectedRequestId, setSelectedRequestId] = useState("");
  
  // Search state
  const [suggestions, setSuggestions] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [externalLatLgn, setExternalLatLgn] = useState<[number, number] | null>(null);

  const searchTimeoutRef = useRef<any>(null);
  const { socket } = useSocket();

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setIsLoading(true);
    try {
      const [binsData, driversData, complaintsData, requestsData] = await Promise.all([
        getBins(),
        getAllDrivers(),
        getAllComplaints(),
        getAllRequests()
      ]);
      setBins(binsData);
      setDrivers(driversData);
      setComplaints(complaintsData);
      setRequests(requestsData);
    } catch (err: any) {
      setError(err.message || "Failed to load admin data");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (!socket) return;
    
    // Listen for realtime events and update list structures natively
    const h1 = (bin: any) => setBins(p => [bin, ...p]);
    const h2 = (bin: any) => setBins(p => p.map(b => b._id === bin._id ? bin : b));
    const h3 = (id: string) => setBins(p => p.filter(b => b._id !== id));
    const h4 = (req: any) => setRequests(p => [req, ...p]);
    const h5 = (req: any) => setRequests(p => p.map(r => r._id === req._id ? req : r));
    const h6 = (comp: any) => setComplaints(p => [comp, ...p]);
    const h7 = (comp: any) => setComplaints(p => p.map(c => c._id === comp._id ? comp : c));
    const h8 = (driver: any) =>
      setDrivers((prev) => {
        if (!prev?.length) return prev;
        const updated = prev.map((d) => (d._id === driver._id ? driver : d));
        // If driver wasn't already in list, append.
        if (!prev.some((d) => d._id === driver._id)) return [driver, ...prev];
        return updated;
      });
    const h9 = h8;

    socket.on('new_bin', h1);
    socket.on('bin_updated', h2);
    socket.on('delete_bin', h3);
    socket.on('new_request', h4);
    socket.on('task_assigned', h5);
    socket.on('task_updated', h5);
    socket.on('new_complaint', h6);
    socket.on('complaint_updated', h7);
    socket.on('driver_shift_updated', h8);
    socket.on('driver_truck_updated', h9);

    return () => {
      socket.off('new_bin', h1);
      socket.off('bin_updated', h2);
      socket.off('delete_bin', h3);
      socket.off('new_request', h4);
      socket.off('task_assigned', h5);
      socket.off('task_updated', h5);
      socket.off('new_complaint', h6);
      socket.off('complaint_updated', h7);
      socket.off('driver_shift_updated', h8);
      socket.off('driver_truck_updated', h9);
    };
  }, [socket]);

  const handleAddBin = async () => {
    if (!newBinLocation || !newBinCoords) return;
    try {
      const [lat, lng] = newBinCoords.split(',').map(s => parseFloat(s.trim()));
      if (isNaN(lat) || isNaN(lng)) {
        alert("Invalid coordinates format. Use: lat, lng");
        return;
      }
      const bin = await createBin({ address: newBinLocation, lat, lng });
      setBins([...bins, bin]);
      setNewBinLocation("");
      setNewBinCoords("");
      setExternalLatLgn(null);
    } catch (err: any) {
      alert(err.response?.data?.message || "Failed to add bin");
    }
  };

  const handleSearch = async (query: string) => {
    setNewBinLocation(query);

    if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current);

    if (query.trim().length > 2) {
      setIsSearching(true);
      setShowSuggestions(true);
      
      searchTimeoutRef.current = setTimeout(async () => {
        try {
          const response = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5`);
          const data = await response.json();
          setSuggestions(data);
        } catch (error) {
          console.error("Search failed", error);
        } finally {
          setIsSearching(false);
        }
      }, 500);
    } else {
      setSuggestions([]);
      setShowSuggestions(false);
      setIsSearching(false);
    }
  };

  const selectSuggestion = (result: any) => {
    const lat = parseFloat(result.lat);
    const lng = parseFloat(result.lon);
    
    setNewBinLocation(result.display_name.split(',')[0]); // first part only to keep it clean
    setNewBinCoords(`${lat.toFixed(6)}, ${lng.toFixed(6)}`);
    setExternalLatLgn([lat, lng]);
    
    setShowSuggestions(false);
    setSuggestions([]);
  };

  const handleMapLocationSelect = async (lat: number, lng: number) => {
    setNewBinCoords(`${lat.toFixed(6)}, ${lng.toFixed(6)}`);
    try {
      const response = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}`);
      const data = await response.json();
      if (data && data.display_name) {
        // Construct a clean, short address string
        const parts = data.display_name.split(',');
        const shortAddress = parts.slice(0, 2).join(',').trim();
        setNewBinLocation(shortAddress);
      }
    } catch (error) {
      console.error("Reverse geocoding failed", error);
    }
  };

  const handleDeleteBin = async (id: string) => {
    if (!confirm("Are you sure you want to delete this bin?")) return;
    try {
      await deleteBin(id);
      setBins(bins.filter(b => b._id !== id));
    } catch (err: any) {
      alert("Failed to delete bin");
    }
  };

  const handleAssignDriver = async () => {
    if (!selectedDriverId || !selectedRequestId) return;
    try {
      await assignDriver(selectedRequestId, selectedDriverId);
      alert("Driver assigned successfully!");
      // Refresh requests to update status
      const updatedReqs = await getAllRequests();
      setRequests(updatedReqs);
      setSelectedDriverId("");
      setSelectedRequestId("");
    } catch (err: any) {
      alert(err.response?.data?.message || "Failed to assign driver");
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-[80vh] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2 text-lg text-muted-foreground">Loading admin panel...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-center text-destructive">
        <AlertTriangle className="mx-auto h-12 w-12 mb-4" />
        <p>{error}</p>
        <button onClick={fetchData} className="mt-4 px-4 py-2 bg-primary text-white rounded">Retry</button>
      </div>
    );
  }

  const parseAiPredictionFromNotes = (notes?: string) => {
    const text = notes || "";
    const isAIPredicted = text.includes("Auto-dispatch prediction (AI)");
    if (!isAIPredicted) return null;

    const confidenceMatch = text.match(/\(confidence:\s*([0-9.]+)\)/i);
    const confidence = confidenceMatch ? Number(confidenceMatch[1]) : null;

    return { isAIPredicted: true, confidence };
  };

  const pendingRequests = requests.filter(r => r.status === 'Pending');

  return (
    <div className="p-6 space-y-6 bg-gray-50">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-gray-800">Admin Control Panel</h2>
          <p className="text-gray-600 mt-1">Manage bins, trucks, and system operations</p>
        </div>
        <div className="flex items-center gap-2 cursor-pointer hover:bg-gray-200 p-2 rounded transition">
          <SettingsIcon className="h-5 w-5 text-blue-600" />
          <span className="text-sm text-gray-600">System Settings</span>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm flex items-center gap-3">
          <div className="bg-blue-500 p-3 rounded-lg"><Trash2 className="h-5 w-5 text-white" /></div>
          <div>
            <p className="text-sm text-gray-500">Total Bins</p>
            <p className="text-2xl font-semibold text-gray-800">{bins.length}</p>
          </div>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm flex items-center gap-3">
          <div className="bg-[#10B981] p-3 rounded-lg"><Truck className="h-5 w-5 text-white" /></div>
          <div>
            <p className="text-sm text-gray-500">Active Trucks/Drivers</p>
            <p className="text-2xl font-semibold text-gray-800">{drivers.filter(d => d.shiftStatus === 'Active').length}</p>
          </div>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm flex items-center gap-3">
          <div className="bg-purple-500 p-3 rounded-lg"><Users className="h-5 w-5 text-white" /></div>
          <div>
            <p className="text-sm text-gray-500">Total Drivers</p>
            <p className="text-2xl font-semibold text-gray-800">{drivers.length}</p>
          </div>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm flex items-center gap-3">
          <div className="bg-[#EF4444] p-3 rounded-lg"><MapPin className="h-5 w-5 text-white" /></div>
          <div>
            <p className="text-sm text-gray-500">Complaints</p>
            <p className="text-2xl font-semibold text-gray-800">{complaints.length}</p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="border-b border-gray-200 flex">
          {(["bins", "trucks", "complaints"] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`flex-1 px-6 py-4 font-medium transition-colors capitalize ${
                activeTab === tab ? "bg-[#1F7A63] text-white" : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              {tab === "bins" ? "Manage Bins" : tab === "trucks" ? "Assign Trucks" : "View Complaints"}
            </button>
          ))}
        </div>

        <div className="p-6">
          {/* Bins Tab */}
          {activeTab === "bins" && (
            <div className="space-y-6">
              <div className="bg-gray-100 rounded-lg p-6 border border-gray-200">
                <h3 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
                  <Plus className="h-5 w-5 text-[#4CAF50]" />
                  Add New Bin
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Search className="h-4 w-4 text-gray-400" />
                      </div>
                      <input
                        type="text"
                        placeholder="Location Address or Landmark"
                        value={newBinLocation}
                        onChange={(e) => handleSearch(e.target.value)}
                        onFocus={() => { if(suggestions.length > 0) setShowSuggestions(true); }}
                        className="w-full pl-10 pr-10 py-3 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#4CAF50] outline-none shadow-sm"
                      />
                      {isSearching && (
                        <div className="absolute inset-y-0 right-0 pr-3 flex items-center bg-transparent pointer-events-none">
                          <Loader2 className="h-4 w-4 text-[#1F7A63] animate-spin" />
                        </div>
                      )}

                      {/* Dropdown Suggestions */}
                      {showSuggestions && suggestions.length > 0 && (
                        <div className="absolute top-full mt-1 left-0 right-0 bg-white border border-gray-200 rounded-lg shadow-xl z-50 max-h-60 overflow-y-auto">
                          {suggestions.map((result) => (
                            <div 
                              key={result.place_id}
                              onClick={() => selectSuggestion(result)}
                              className="px-4 py-3 hover:bg-gray-50 border-b border-gray-100 cursor-pointer flex items-start gap-3 last:border-0"
                            >
                              <MapPin className="w-4 h-4 text-[#1F7A63] mt-0.5 shrink-0" />
                              <p className="text-sm text-gray-700 leading-tight">
                                {result.display_name}
                              </p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                    
                    <div className="bg-white p-4 rounded-lg border border-gray-200 text-sm shadow-sm">
                      <p className="font-semibold text-gray-700 mb-1">Selected Map Coordinates:</p>
                      <p className="text-[#1F7A63] font-mono font-medium text-base">
                        {newBinCoords ? newBinCoords : "Click pointer on map"}
                      </p>
                    </div>
                    <button 
                      onClick={handleAddBin} 
                      disabled={!newBinLocation || !newBinCoords}
                      className="w-full bg-[#1F7A63] text-white py-3 rounded-lg font-bold hover:bg-[#4CAF50] transition-colors shadow-md disabled:opacity-50 flex items-center justify-center gap-2"
                    >
                      <Plus className="h-5 w-5" /> Add Bin at Location
                    </button>
                  </div>
                  <div className="bg-white p-2 rounded-xl border border-gray-300 shadow-sm relative z-10">
                    <LocationPickerMap 
                      onLocationSelect={handleMapLocationSelect}
                      initialLocation={[28.6139, 77.2090]}
                      selectedLocation={externalLatLgn}
                    />
                    <p className="text-xs text-gray-500 mt-2 text-center flex items-center justify-center gap-1">
                      <MapPin className="h-3 w-3" /> Click anywhere on map to drop the pin
                    </p>
                  </div>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">ID</th>
                      <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Location</th>
                      <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Status</th>
                      <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Fill Level</th>
                      <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Created</th>
                      <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bins.map((bin) => (
                      <tr key={bin._id} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="py-3 px-4 text-sm text-gray-500 truncate max-w-[100px]">{bin._id}</td>
                        <td className="py-3 px-4 text-sm text-gray-800">{bin.location.address}</td>
                        <td className="py-3 px-4">
                          <span className={`px-3 py-1 rounded-full text-xs font-medium ${bin.status === "Active" ? "bg-green-100 text-green-700" : bin.status === "Full" ? "bg-red-100 text-red-700" : "bg-amber-100 text-amber-700"}`}>
                            {bin.status}
                          </span>
                        </td>
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            <div className="w-24 bg-gray-200 rounded-full h-2">
                              <div className="h-full rounded-full" style={{ width: `${bin.fillLevel}%`, backgroundColor: bin.fillLevel > 70 ? "#EF4444" : bin.fillLevel > 30 ? "#F59E0B" : "#10B981" }} />
                            </div>
                            <span className="text-sm text-gray-700">{bin.fillLevel}%</span>
                          </div>
                        </td>
                        <td className="py-3 px-4 text-sm text-gray-500">{format(new Date(bin.createdAt), "MMM d, yyyy")}</td>
                        <td className="py-3 px-4">
                          <button onClick={() => handleDeleteBin(bin._id)} className="text-red-500 hover:text-red-700 transition">Remove</button>
                        </td>
                      </tr>
                    ))}
                    {bins.length === 0 && (
                      <tr><td colSpan={6} className="text-center py-4 text-gray-500">No bins found.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Trucks Tab */}
          {activeTab === "trucks" && (
            <div className="space-y-6">
              <div className="bg-gray-100 rounded-lg p-6 border border-gray-200">
                <h3 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
                  <Truck className="h-5 w-5 text-[#4CAF50]" />
                  Assign Driver to Collection Request
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 border-b pb-6 mb-6">
                  <select
                    value={selectedDriverId}
                    onChange={(e) => setSelectedDriverId(e.target.value)}
                    className="px-4 py-2 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#4CAF50] outline-none"
                  >
                    <option value="">Select Driver</option>
                    {drivers.map(d => (
                      <option key={d._id} value={d.userId._id}>
                        {d.userId.name} (Vehicle: {d.vehicleNumber || "N/A"}{d.truckType ? `, Truck: ${d.truckType}` : ""})
                      </option>
                    ))}
                  </select>
                  <select
                    value={selectedRequestId}
                    onChange={(e) => setSelectedRequestId(e.target.value)}
                    className="px-4 py-2 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#4CAF50] outline-none"
                  >
                    <option value="">Select Pending Request</option>
                    {pendingRequests.map(r => {
                      const ai = parseAiPredictionFromNotes(r.notes);
                      const aiLabel =
                        ai?.confidence != null ? ` (AI ${(ai.confidence * 100).toFixed(0)}%)` : ai ? ' (AI predicted)' : '';

                      return (
                        <option key={r._id} value={r._id}>
                          {r.location} - {format(new Date(r.scheduledDate), "MMM d")}
                          {aiLabel}
                        </option>
                      );
                    })}
                  </select>
                  <button onClick={handleAssignDriver} disabled={!selectedDriverId || !selectedRequestId} className="bg-[#1F7A63] text-white py-2 rounded-lg font-medium hover:bg-[#4CAF50] transition-colors disabled:opacity-50">
                    Assign Driver
                  </button>
                </div>
              </div>

              <div className="overflow-x-auto">
                <h3 className="font-semibold text-gray-800 mb-4">Driver Roster</h3>
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Vehicle #</th>
                      <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Driver Name</th>
                      <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Email</th>
                      <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Shift Status</th>
                      <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Truck Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {drivers.map((driver) => (
                      <tr key={driver._id} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="py-3 px-4 text-sm font-medium text-gray-800">{driver.vehicleNumber}</td>
                        <td className="py-3 px-4 text-sm text-gray-800">{driver.userId?.name || "Unknown"}</td>
                        <td className="py-3 px-4 text-sm text-gray-500">{driver.userId?.email || "Unknown"}</td>
                        <td className="py-3 px-4">
                          <span className={`px-3 py-1 rounded-full text-xs font-medium ${driver.shiftStatus === "Active" ? "bg-green-100 text-green-700" : "bg-gray-200 text-gray-700"}`}>
                            {driver.shiftStatus}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-sm text-gray-700">
                          {driver.truckType || <span className="text-gray-400">Not set</span>}
                        </td>
                      </tr>
                    ))}
                    {drivers.length === 0 && (
                      <tr><td colSpan={5} className="text-center py-4 text-gray-500">No drivers registered.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Complaints Tab */}
          {activeTab === "complaints" && (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">ID</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Location</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Issue</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Assigned Driver</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Priority</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Status</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Date</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {complaints.map((complaint) => (
                    <tr
                      key={complaint._id}
                      className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer"
                      onClick={() => setSelectedComplaint(complaint)}
                      title="Click to view full complaint"
                    >
                      <td className="py-3 px-4 text-sm text-gray-500 truncate max-w-[80px]">{complaint._id}</td>
                      <td className="py-3 px-4 text-sm text-gray-800">{complaint.location}</td>
                      <td className="py-3 px-4 text-sm text-gray-800 truncate max-w-[200px]">{complaint.description}</td>
                      <td className="py-3 px-4 text-sm text-gray-700">
                        {(complaint as any).assignedDriverId?.name || <span className="text-gray-400">Unassigned</span>}
                      </td>
                      <td className="py-3 px-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${complaint.priority === "High" ? "bg-red-100 text-red-700" : complaint.priority === "Medium" ? "bg-amber-100 text-amber-700" : "bg-blue-100 text-blue-700"}`}>
                          {complaint.priority}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${complaint.status === "Resolved" ? "bg-green-100 text-green-700" : complaint.status === "In Progress" ? "bg-blue-100 text-blue-700" : "bg-amber-100 text-amber-700"}`}>
                          {complaint.status}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-500">{format(new Date(complaint.createdAt), "MMM d, yyyy")}</td>
                      <td className="py-3 px-4">
                        <span className="text-gray-500 text-sm">View</span>
                      </td>
                    </tr>
                  ))}
                  {complaints.length === 0 && (
                    <tr><td colSpan={8} className="text-center py-4 text-gray-500">No complaints reported.</td></tr>
                  )}
                </tbody>
              </table>

              {/* Complaint details modal */}
              {selectedComplaint && (
                <div
                  className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 p-4"
                  onClick={() => setSelectedComplaint(null)}
                >
                  <div
                    className="w-full max-w-3xl rounded-xl bg-white shadow-2xl border border-gray-200 overflow-hidden"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
                      <div>
                        <h3 className="text-lg font-semibold text-gray-800">Complaint Details</h3>
                        <p className="text-sm text-gray-500 truncate max-w-[70vw]">{selectedComplaint._id}</p>
                      </div>
                      <button
                        onClick={() => setSelectedComplaint(null)}
                        className="px-3 py-2 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium"
                      >
                        Close
                      </button>
                    </div>

                    <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-3">
                        <div>
                          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Citizen</p>
                          <p className="text-sm text-gray-800">
                            {(selectedComplaint as any).userId?.name || "Unknown"}{" "}
                            <span className="text-gray-500">({(selectedComplaint as any).userId?.email || "N/A"})</span>
                          </p>
                        </div>

                        <div>
                          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Assigned Driver</p>
                          <p className="text-sm text-gray-800">
                            {(selectedComplaint as any).assignedDriverId?.name
                              ? `${(selectedComplaint as any).assignedDriverId.name} (${(selectedComplaint as any).assignedDriverId.email})`
                              : "Not assigned yet"}
                          </p>
                        </div>

                        <div>
                          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Location</p>
                          <p className="text-sm text-gray-800 whitespace-pre-wrap">{selectedComplaint.location}</p>
                        </div>

                        <div>
                          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Description</p>
                          <p className="text-sm text-gray-800 whitespace-pre-wrap">{selectedComplaint.description}</p>
                        </div>

                        <div className="flex items-center gap-2 flex-wrap">
                          <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                            selectedComplaint.priority === "High" ? "bg-red-100 text-red-700" :
                            selectedComplaint.priority === "Medium" ? "bg-amber-100 text-amber-700" :
                            "bg-blue-100 text-blue-700"
                          }`}>
                            Priority: {selectedComplaint.priority}
                          </span>
                          <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                            selectedComplaint.status === "Resolved" ? "bg-green-100 text-green-700" :
                            selectedComplaint.status === "In Progress" ? "bg-blue-100 text-blue-700" :
                            "bg-amber-100 text-amber-700"
                          }`}>
                            Status: {selectedComplaint.status}
                          </span>
                          {(selectedComplaint as any).ai?.classification && (
                            <span className="px-3 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                              AI: {(selectedComplaint as any).ai.classification}
                              {Number.isFinite((selectedComplaint as any).ai?.confidence)
                                ? ` (${Math.round((selectedComplaint as any).ai.confidence * 100)}%)`
                                : ''}
                            </span>
                          )}
                        </div>

                        <div>
                          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Created</p>
                          <p className="text-sm text-gray-800">{format(new Date(selectedComplaint.createdAt), "PPPp")}</p>
                        </div>

                        <div className="mt-2 text-xs text-gray-500">
                          Complaints can only be resolved by the assigned driver after collection.
                        </div>
                      </div>

                      <div className="space-y-3">
                        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">AI Summary</p>
                        {(selectedComplaint as any).ai?.classification ? (
                          <div className="rounded-lg border border-purple-200 bg-purple-50 p-4">
                            <div className="text-sm font-semibold text-purple-900">
                              {(selectedComplaint as any).ai.classification}
                              {Number.isFinite((selectedComplaint as any).ai?.confidence)
                                ? ` (${Math.round((selectedComplaint as any).ai.confidence * 100)}%)`
                                : ''}
                            </div>
                            <div className="text-xs text-purple-700 mt-1">
                              Provider: {(selectedComplaint as any).ai?.provider || 'waste_classifier'}
                            </div>
                          </div>
                        ) : (
                          <div className="text-sm text-gray-500 bg-gray-50 border border-gray-200 rounded-lg p-4">
                            No AI result stored for this complaint.
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}