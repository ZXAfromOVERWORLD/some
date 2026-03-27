import { useState, useRef, useCallback, useEffect } from 'react';
import { MapPin, Send, CheckCircle2, Clock, Trash2, Calendar, ArrowUpDown, Package, Recycle, HardHat, Home, Loader2, Sparkles, Search, Target, AlertTriangle, Camera, X } from 'lucide-react';
import { ComplaintMap } from '../ComplaintMap';
import { createRequest, getMyRequests, CollectionRequest as ApiCollectionRequest } from '../../services/requestService';
import { classifyWasteFromImage, WasteClassificationResponse } from '../../services/aiService';
import { registerUserBin } from '../../services/binService';

interface UICollectionRequest {
  id: string;
  location: string;
  latitude: number;
  longitude: number;
  description: string;
  category: 'household' | 'plastic' | 'ewaste' | 'construction' | string;
  status: 'pending' | 'in-progress' | 'completed' | string;
  date: string;
  pickupDate?: string;
  pickupTime?: string;
}

interface SearchResult {
  display_name: string;
  lat: string;
  lon: string;
}

const wasteCategories = [
  { id: 'household', label: 'Household', icon: Home, color: 'from-emerald-400 to-teal-500', shadow: 'shadow-emerald-500/20', bg: 'bg-emerald-50', border: 'border-emerald-200' },
  { id: 'plastic', label: 'Recyclables', icon: Recycle, color: 'from-blue-400 to-indigo-500', shadow: 'shadow-blue-500/20', bg: 'bg-blue-50', border: 'border-blue-200' },
  { id: 'ewaste', label: 'E-Waste', icon: Sparkles, color: 'from-purple-400 to-fuchsia-500', shadow: 'shadow-purple-500/20', bg: 'bg-purple-50', border: 'border-purple-200' },
  { id: 'construction', label: 'Heavy/Bulk', icon: HardHat, color: 'from-orange-400 to-rose-500', shadow: 'shadow-orange-500/20', bg: 'bg-orange-50', border: 'border-orange-200' },
];

export default function GarbageCollectionPage() {
  const [markerPosition, setMarkerPosition] = useState<[number, number] | null>(null);
  const [description, setDescription] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [pickupDate, setPickupDate] = useState('');
  const [pickupTime, setPickupTime] = useState('');
  const [sortBy, setSortBy] = useState<'date' | 'status'>('date');
  const [filterStatus, setFilterStatus] = useState<'all' | 'pending' | 'in-progress' | 'completed' | string>('all');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isRegisteringBin, setIsRegisteringBin] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [initialBinFill, setInitialBinFill] = useState('0');

  // Location search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedLocationName, setSelectedLocationName] = useState('');
  const searchTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const photoInputRef = useRef<HTMLInputElement>(null);

  const [requests, setRequests] = useState<UICollectionRequest[]>([]);

  // Optional photo classification via AI agent
  const [photoFile, setPhotoFile] = useState<File | null>(null);
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);
  const [isAiClassifying, setIsAiClassifying] = useState(false);
  const [aiWaste, setAiWaste] = useState<WasteClassificationResponse | null>(null);

  useEffect(() => {
    fetchRequests();
  }, []);

  const fetchRequests = async () => {
    setIsLoading(true);
    try {
      const data = await getMyRequests();
      const mapped = data.map(mapApiToUI);
      setRequests(mapped);
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to load requests');
    } finally {
      setIsLoading(false);
    }
  };

  const mapApiToUI = (req: ApiCollectionRequest): UICollectionRequest => {
    let uiStatus = req.status.toLowerCase();
    if (uiStatus === 'assigned') uiStatus = 'pending'; // map to simpler UI status
    if (uiStatus === 'in progress') uiStatus = 'in-progress';
    
    // Reverse map backend enum to frontend ID
    const reverseMap: Record<string, string> = {
      'General': 'household',
      'Recyclable': 'plastic',
      'Electronic': 'ewaste',
      'Hazardous': 'construction'
    };

    return {
      id: req._id,
      location: req.location || 'Unknown location',
      latitude: 0,
      longitude: 0,
      description: req.notes || 'No description',
      category: reverseMap[req.wasteType] || 'household',
      status: uiStatus,
      date: req.createdAt,
      pickupDate: req.scheduledDate,
      pickupTime: req.scheduledTime,
    };
  };

  const handleLocateMe = () => {
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setMarkerPosition([pos.coords.latitude, pos.coords.longitude]);
          setSearchQuery(`Current Location (${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)})`);
          setSelectedLocationName(`Current Location (${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)})`);
        },
        () => {
          alert('Unable to retrieve your location');
        }
      );
    }
  };

  // Location search using Nominatim (OpenStreetMap geocoding)
  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query);
    if (searchTimeout.current) clearTimeout(searchTimeout.current);

    if (query.length < 3) {
      setSearchResults([]);
      return;
    }

    searchTimeout.current = setTimeout(async () => {
      setIsSearching(true);
      try {
        const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5`);
        const data: SearchResult[] = await res.json();
        setSearchResults(data);
      } catch {
        setSearchResults([]);
      } finally {
        setIsSearching(false);
      }
    }, 400);
  }, []);

  const selectSearchResult = (result: SearchResult) => {
    const lat = parseFloat(result.lat);
    const lon = parseFloat(result.lon);
    setMarkerPosition([lat, lon]);
    setSelectedLocationName(result.display_name.split(',').slice(0, 3).join(', '));
    setSearchQuery(result.display_name.split(',').slice(0, 3).join(', '));
    setSearchResults([]);
  };

  const classifyToWasteCategory = (ai: WasteClassificationResponse): string => {
    const labels = (ai.detected_objects || []).map((o) => o.toLowerCase());
    const electronicHints = [
      'tv',
      'laptop',
      'mouse',
      'remote',
      'keyboard',
      'cell phone',
      'microwave',
      'oven',
      'toaster',
      'refrigerator',
    ];

    if (ai.classification === 'Non-biodegradable') {
      const isElectronic = labels.some((l) => electronicHints.includes(l));
      return isElectronic ? 'ewaste' : 'plastic';
    }

    if (ai.classification === 'Biodegradable') return 'household';
    if (ai.classification === 'Mixed') return 'household';
    return 'household'; // Unknown fallback
  };

  const handlePhotoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setPhotoFile(file);
    const reader = new FileReader();
    reader.onloadend = () => setPhotoPreview(reader.result as string);
    reader.readAsDataURL(file);
  };

  const removePhoto = () => {
    setPhotoFile(null);
    setPhotoPreview(null);
    setAiWaste(null);
    if (photoInputRef.current) photoInputRef.current.value = '';
  };

  useEffect(() => {
    if (!photoFile) {
      setAiWaste(null);
      return;
    }

    let cancelled = false;
    setIsAiClassifying(true);

    classifyWasteFromImage(photoFile)
      .then((ai) => {
        if (cancelled) return;
        setAiWaste(ai);
        const nextCategory = classifyToWasteCategory(ai);
        if (nextCategory) setSelectedCategory(nextCategory);
      })
      .catch(() => {
        if (cancelled) return;
        setAiWaste(null);
      })
      .finally(() => {
        if (cancelled) return;
        setIsAiClassifying(false);
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [photoFile]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!markerPosition || !description || !selectedCategory) return;

    setIsSubmitting(true);
    try {
      const locName = selectedLocationName || `Lat: ${markerPosition[0].toFixed(4)}, Lng: ${markerPosition[1].toFixed(4)}`;
      
      const apiWasteTypeMap: Record<string, string> = {
        'household': 'General',
        'plastic': 'Recyclable',
        'ewaste': 'Electronic',
        'construction': 'Hazardous'
      };

      const newReq = await createRequest({
        wasteType: apiWasteTypeMap[selectedCategory] || 'General',
        location: locName,
        lat: markerPosition[0],
        lng: markerPosition[1],
        scheduledDate: pickupDate || new Date().toISOString(),
        scheduledTime: pickupTime,
        notes: description,
      } as any);

      const updated = [mapApiToUI(newReq), ...requests];
      setRequests(updated);
      
      setDescription('');
      setSelectedCategory('');
      setPickupDate('');
      setPickupTime('');
      setMarkerPosition(null);
      setSearchQuery('');
      setSelectedLocationName('');
    } catch (err: any) {
      alert(err.response?.data?.message || 'Failed to submit request');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRegisterBin = async () => {
    if (!markerPosition) {
      alert('Please select a location on the map first');
      return;
    }
    const locName =
      selectedLocationName ||
      searchQuery ||
      `Lat: ${markerPosition[0].toFixed(4)}, Lng: ${markerPosition[1].toFixed(4)}`;

    const fill = Number(initialBinFill);
    if (!Number.isFinite(fill) || fill < 0 || fill > 100) {
      alert('Initial fill level must be between 0 and 100');
      return;
    }

    setIsRegisteringBin(true);
    try {
      await registerUserBin({
        address: locName,
        lat: markerPosition[0],
        lng: markerPosition[1],
        fillLevel: fill,
      });
      alert('Bin registered successfully');
    } catch (err: any) {
      alert(err.response?.data?.message || 'Failed to register bin');
    } finally {
      setIsRegisteringBin(false);
    }
  };

  const getStatusConfig = (s: string) => {
    if (s === 'completed') return { color: 'bg-emerald-100 text-emerald-700', icon: CheckCircle2 };
    if (s === 'in-progress') return { color: 'bg-blue-100 text-blue-700', icon: Loader2 };
    return { color: 'bg-amber-100 text-amber-700', icon: Clock };
  };

  const filtered = requests
    .filter(r => filterStatus === 'all' || r.status === filterStatus)
    .sort((a, b) => {
      if (sortBy === 'date') return new Date(b.date).getTime() - new Date(a.date).getTime();
      return a.status.localeCompare(b.status);
    });

  const isFormValid = markerPosition && description && selectedCategory;

  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  const minDate = tomorrow.toISOString().split('T')[0];

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-[60vh]">
        <Loader2 className="w-10 h-10 animate-spin text-emerald-500" />
      </div>
    );
  }

  return (
    <div className="w-full max-w-[1400px] mx-auto animate-fade-in pb-8">
      {/* Header */}
      <div className="mb-8 pl-2">
        <h1 className="text-3xl md:text-4xl font-black text-slate-800 tracking-tight mb-2">Schedule Pickup</h1>
        <p className="text-slate-500 font-medium text-lg">Request a special garbage collection at your convenience.</p>
      </div>

      {/* Summary Tracking Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm flex items-center gap-4">
          <div className="w-12 h-12 bg-slate-50 text-slate-500 rounded-xl flex items-center justify-center font-bold text-lg border border-slate-100">
            {requests.length}
          </div>
          <div>
            <p className="font-bold text-slate-800 text-sm">Total Requests</p>
            <p className="text-xs text-slate-500 font-medium">All time</p>
          </div>
        </div>
        
        <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm flex items-center gap-4">
          <div className="w-12 h-12 bg-amber-50 text-amber-600 rounded-xl flex items-center justify-center shadow-sm shadow-amber-500/10">
            <Clock className="w-6 h-6" />
          </div>
          <div>
            <p className="font-bold text-slate-800 text-sm">Pending</p>
            <p className="font-black text-xl text-amber-600">{requests.filter(r => r.status === 'pending').length}</p>
          </div>
        </div>

        <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm flex items-center gap-4">
          <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-xl flex items-center justify-center shadow-sm shadow-blue-500/10">
            <Loader2 className="w-6 h-6" />
          </div>
          <div>
            <p className="font-bold text-slate-800 text-sm">In Progress</p>
            <p className="font-black text-xl text-blue-600">{requests.filter(r => r.status === 'in-progress').length}</p>
          </div>
        </div>

        <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm flex items-center gap-4">
          <div className="w-12 h-12 bg-emerald-50 text-emerald-600 rounded-xl flex items-center justify-center shadow-sm shadow-emerald-500/10">
            <CheckCircle2 className="w-6 h-6" />
          </div>
          <div>
            <p className="font-bold text-slate-800 text-sm">Completed</p>
            <p className="font-black text-xl text-emerald-600">{requests.filter(r => r.status === 'completed').length}</p>
          </div>
        </div>
      </div>

      {error && (
        <div className="mb-8 bg-red-50 text-red-600 p-4 rounded-xl border border-red-100 font-medium flex items-center gap-2">
          <AlertTriangle className="w-5 h-5" />
          {error}
        </div>
      )}

      <div className="space-y-8">
        {/* Form Section - Full Width */}
        <div>
          <div className="bg-white rounded-[2.5rem] p-6 sm:p-10 shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-slate-100">
            <div className="flex items-center gap-4 mb-8">
              <div className="w-14 h-14 bg-gradient-to-tr from-emerald-500 to-teal-400 rounded-2xl flex items-center justify-center shadow-lg shadow-emerald-500/20">
                <Package className="w-7 h-7 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-slate-800">New Request</h2>
                <p className="text-slate-500 font-medium">Select category and location</p>
              </div>
            </div>
            
            <form onSubmit={handleSubmit} className="space-y-8">
              {/* Category Grid */}
              <div>
                <label className="block text-sm font-bold text-slate-700 uppercase tracking-wider mb-4">Waste Category</label>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  {wasteCategories.map((cat) => {
                    const Icon = cat.icon;
                    const isSelected = selectedCategory === cat.id;
                    return (
                      <button
                        key={cat.id}
                        type="button"
                        onClick={() => setSelectedCategory(cat.id)}
                        className={`relative p-5 rounded-[1.5rem] border-2 transition-all duration-300 flex flex-col items-center justify-center text-center group ${
                          isSelected ? `${cat.bg} ${cat.border} shadow-lg ${cat.shadow} scale-[1.02]` : 'bg-white border-slate-100 hover:border-slate-200 hover:bg-slate-50 hover:shadow-md'
                        }`}
                      >
                        <div className={`w-14 h-14 rounded-2xl flex items-center justify-center mb-3 transition-transform duration-300 ${
                          isSelected ? `bg-gradient-to-tr ${cat.color} text-white shadow-md ${cat.shadow}` : 'bg-slate-100 text-slate-400 group-hover:scale-110 group-hover:text-slate-600'
                        }`}>
                          <Icon className="w-7 h-7" />
                        </div>
                        <span className={`font-bold text-sm ${isSelected ? 'text-slate-800' : 'text-slate-500'}`}>{cat.label}</span>
                        {isSelected && (
                          <div className="absolute top-3 right-3">
                            <CheckCircle2 className={`w-5 h-5 ${cat.color.split(' ')[1].replace('to-', 'text-')}`} />
                          </div>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Location Search + Map & Date Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div>
                  <label className="flex items-center gap-2 text-sm font-bold text-slate-700 uppercase tracking-wider mb-3">
                    <MapPin className="w-4 h-4 text-emerald-500" /> Location
                  </label>
                  
                  {/* Search Bar */}
                  <div className="mb-3 flex gap-2">
                    <div className="relative flex-1">
                      <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => handleSearch(e.target.value)}
                        placeholder="Search location..."
                        className="w-full pl-12 pr-5 py-3.5 bg-slate-50 border-2 border-slate-100 rounded-[1.25rem] focus:bg-white focus:outline-none focus:border-emerald-400 focus:ring-4 focus:ring-emerald-400/10 text-slate-800 font-medium placeholder:text-slate-400 transition-all text-sm"
                      />
                      {isSearching && (
                        <Loader2 className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-emerald-400 animate-spin" />
                      )}
                      
                      {/* Search Results Dropdown */}
                      {searchResults.length > 0 && (
                        <div className="absolute z-50 top-full left-0 right-0 mt-2 bg-white rounded-2xl shadow-2xl border border-slate-100 overflow-hidden">
                          {searchResults.map((result, i) => (
                            <button
                              key={i}
                              type="button"
                              onClick={() => selectSearchResult(result)}
                              className="w-full text-left px-4 py-3 hover:bg-slate-50 flex items-start gap-2.5 transition-colors border-b border-slate-50 last:border-0"
                            >
                              <MapPin className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                              <span className="text-xs font-medium text-slate-700 leading-snug">{result.display_name}</span>
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={handleLocateMe}
                      className="bg-slate-50 border-2 border-slate-100 text-slate-600 hover:border-emerald-300 hover:text-emerald-500 rounded-[1.25rem] px-5 flex items-center justify-center transition-all group"
                      title="Auto detect location"
                    >
                      <Target className="w-6 h-6 group-hover:scale-110 transition-transform" />
                    </button>
                  </div>
                    
                    <div className="bg-slate-50 border-2 border-slate-100 rounded-[1.5rem] overflow-hidden h-[200px] relative group">
                    <ComplaintMap position={markerPosition} setPosition={(pos) => {
                      setMarkerPosition(pos);
                      setSelectedLocationName('');
                      setSearchQuery('');
                      setSearchResults([]);
                    }} compact />
                    {!markerPosition && (
                      <div className="absolute inset-0 pointer-events-none flex items-center justify-center bg-white/20 backdrop-blur-[1px]">
                        <div className="bg-white/90 px-4 py-2 rounded-xl shadow-lg border border-slate-200 text-sm font-bold text-slate-700">
                          Search or click map
                        </div>
                      </div>
                    )}
                  </div>
                  {markerPosition && (
                    <div className="mt-2 flex items-center gap-2 text-xs font-bold text-emerald-600 bg-emerald-50 px-3 py-2 rounded-xl border border-emerald-100">
                      <CheckCircle2 className="w-4 h-4 shrink-0" />
                      <span className="truncate">{selectedLocationName || `${markerPosition[0].toFixed(4)}, ${markerPosition[1].toFixed(4)}`}</span>
                    </div>
                  )}
                  <div className="mt-4 p-4 rounded-2xl border border-emerald-100 bg-emerald-50/60">
                    <p className="text-xs font-bold text-emerald-700 uppercase tracking-wider mb-3">
                      Register Household Bin
                    </p>
                    <div className="flex items-center gap-3">
                      <input
                        type="number"
                        min={0}
                        max={100}
                        value={initialBinFill}
                        onChange={(e) => setInitialBinFill(e.target.value)}
                        className="w-32 px-3 py-2 rounded-lg border border-emerald-200 bg-white text-sm font-medium text-slate-700"
                        placeholder="Fill %"
                      />
                      <button
                        type="button"
                        onClick={handleRegisterBin}
                        disabled={!markerPosition || isRegisteringBin}
                        className="px-4 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-bold disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {isRegisteringBin ? 'Registering...' : 'Register Bin'}
                      </button>
                    </div>
                    <p className="text-[11px] text-emerald-700/80 mt-2">
                      Choose a map point and register this as your home bin.
                    </p>
                  </div>
                </div>

                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-bold text-slate-700 uppercase tracking-wider mb-3">Scheduling</label>
                    <div className="space-y-4">
                      <div className="relative">
                        <Calendar className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                        <input
                          type="date"
                          value={pickupDate}
                          onChange={(e) => setPickupDate(e.target.value)}
                          min={minDate}
                          className="w-full pl-12 pr-5 py-4 bg-slate-50 border-2 border-slate-100 rounded-[1.25rem] focus:bg-white focus:outline-none focus:border-emerald-400 focus:ring-4 focus:ring-emerald-400/10 text-slate-800 font-semibold transition-all cursor-pointer"
                        />
                      </div>
                      <div className="relative">
                        <Clock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                        <input
                          type="time"
                          value={pickupTime}
                          onChange={(e) => setPickupTime(e.target.value)}
                          className="w-full pl-12 pr-5 py-4 bg-slate-50 border-2 border-slate-100 rounded-[1.25rem] focus:bg-white focus:outline-none focus:border-emerald-400 focus:ring-4 focus:ring-emerald-400/10 text-slate-800 font-semibold transition-all cursor-pointer"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Description */}
              <div>
                <label className="block text-sm font-bold text-slate-700 uppercase tracking-wider mb-3">Pickup Instructions</label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="E.g., 3 large black bags, left by the side gate..."
                  rows={3}
                  className="w-full px-5 py-4 bg-slate-50 border-2 border-slate-100 rounded-[1.25rem] focus:bg-white focus:outline-none focus:border-emerald-400 focus:ring-4 focus:ring-emerald-400/10 text-slate-800 font-medium placeholder:text-slate-400 transition-all resize-none"
                  required
                />
              </div>

              {/* Trash photo upload (AI auto-selects waste category) */}
              <div className="pt-6 border-t border-slate-100">
                <label className="block text-sm font-bold text-slate-700 uppercase tracking-wider mb-3 flex items-center gap-2">
                  <Camera className="w-4 h-4 text-emerald-500" />
                  Trash Photo (optional)
                </label>

                {photoPreview ? (
                  <div className="relative rounded-[1.25rem] overflow-hidden border-2 border-slate-100">
                    <img src={photoPreview} alt="Trash preview" className="h-48 w-full object-cover" />
                    <button
                      type="button"
                      onClick={removePhoto}
                      className="absolute top-3 right-3 bg-white/90 backdrop-blur-md p-2 rounded-full shadow-lg text-slate-600 hover:bg-white"
                      title="Remove photo"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => photoInputRef.current?.click()}
                    className="w-full py-6 bg-slate-50 border-2 border-slate-100 rounded-[1.25rem] hover:border-emerald-300 transition-all flex flex-col items-center justify-center gap-2"
                  >
                    <Camera className="w-6 h-6 text-emerald-500" />
                    <span className="text-sm font-bold text-slate-700">Upload a photo to classify</span>
                  </button>
                )}

                <input
                  ref={photoInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handlePhotoChange}
                  className="hidden"
                />

                {isAiClassifying && (
                  <p className="mt-3 text-xs font-semibold text-emerald-700">
                    Classifying image...
                  </p>
                )}

                {aiWaste && !isAiClassifying && (
                  <p className="mt-3 text-xs font-semibold text-emerald-700">
                    AI: {aiWaste.classification} ({Math.round(aiWaste.confidence * 100)}% confidence)
                  </p>
                )}
              </div>

              <div className="pt-4 border-t border-slate-100">
                <button
                  type="submit"
                  disabled={!isFormValid || isSubmitting}
                  className={`w-full py-5 rounded-[1.25rem] font-bold text-lg transition-all flex items-center justify-center gap-2 ${
                    isFormValid && !isSubmitting
                      ? 'bg-gradient-to-r from-emerald-500 to-teal-500 text-white shadow-lg shadow-emerald-500/25 hover:shadow-xl hover:scale-[1.01]'
                      : 'bg-slate-100 text-slate-400 cursor-not-allowed'
                  }`}
                >
                  {isSubmitting ? <Loader2 className="w-6 h-6 animate-spin" /> : <><Send className="w-6 h-6" /> Schedule Request</>}
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* History Section - Full Width Below Form */}
        <div className="bg-white rounded-[2.5rem] p-6 sm:p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-slate-100">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
              <Trash2 className="w-6 h-6 text-slate-400" /> Request History
            </h2>
            <div className="flex items-center gap-2">
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value as any)}
                className="px-4 py-2 bg-slate-50 border border-slate-100 rounded-xl text-sm font-semibold text-slate-700 outline-none focus:border-emerald-300 appearance-none cursor-pointer"
              >
                <option value="all">All</option>
                <option value="pending">Pending</option>
                <option value="in-progress">In Progress</option>
                <option value="completed">Completed</option>
              </select>
              <button onClick={() => setSortBy(sortBy === 'date' ? 'status' : 'date')} className="p-2.5 bg-slate-50 border border-slate-100 rounded-xl text-slate-600 hover:bg-slate-100 transition-colors">
                <ArrowUpDown className="w-4 h-4" />
              </button>
            </div>
          </div>

          {filtered.length === 0 ? (
            <div className="py-16 flex flex-col items-center justify-center text-slate-400">
              <Package className="w-16 h-16 opacity-20 mb-4" />
              <p className="font-semibold">No requests found</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {filtered.map(r => {
                const sConf = getStatusConfig(r.status);
                const SIcon = sConf.icon;
                const cat = wasteCategories.find(c => c.id === r.category) || wasteCategories[0];
                const CIcon = cat.icon;
                
                return (
                  <div key={r.id} className="bg-slate-50/50 border border-slate-100 hover:border-slate-200 rounded-2xl p-5 hover:shadow-md transition-all relative overflow-hidden">
                    
                    <div className={`absolute top-0 left-0 w-1.5 h-full bg-gradient-to-b ${cat.color}`} />
                    
                    <div className="flex gap-3 items-center mb-3 pl-3">
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center bg-gradient-to-tr ${cat.color} text-white shadow-sm shrink-0`}>
                        <CIcon className="w-5 h-5" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="font-bold text-slate-800 text-sm leading-tight">{r.description}</p>
                        <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">{cat.label}</span>
                      </div>
                    </div>
                    
                    <div className="pl-3 mb-3 flex items-center gap-2 flex-wrap">
                      <div className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 ${sConf.color}`}>
                        <SIcon className={`w-3.5 h-3.5 ${r.status === 'in-progress' ? 'animate-spin' : ''}`} />
                        <span className="text-[11px] font-black uppercase tracking-wider">{r.status}</span>
                      </div>
                      {r.pickupDate && (
                        <div className="px-3 py-1.5 rounded-lg bg-white text-slate-600 flex items-center gap-1.5 text-[11px] font-bold tracking-wider border border-slate-100">
                          <Calendar className="w-3.5 h-3.5" /> {new Date(r.pickupDate).toLocaleDateString()}
                        </div>
                      )}
                    </div>

                    <div className="bg-white p-2.5 rounded-xl flex items-center text-sm ml-3 border border-slate-100">
                      <div className="flex items-center gap-2 text-slate-600 font-medium min-w-0">
                        <MapPin className="w-3.5 h-3.5 text-slate-400 shrink-0" /> <span className="truncate text-xs font-bold">{r.location}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
      
      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background-color: #cbd5e1; border-radius: 20px; }
      `}</style>
    </div>
  );
}
