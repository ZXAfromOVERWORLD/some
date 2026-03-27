import { useState, useRef, useCallback, useEffect } from 'react';
import { MapPin, Send, CheckCircle2, Clock, ShieldAlert, Camera, X, ChevronDown, ArrowUpDown, Loader2, Search, Target, AlertTriangle } from 'lucide-react';
import { ComplaintMap } from '../ComplaintMap';
import { createComplaint, getMyComplaints, Complaint as ApiComplaint } from '../../services/complaintService';
import { uploadImage } from '../../services/api';
import { classifyWasteFromImage, WasteClassificationResponse } from '../../services/aiService';

interface UIComplaint {
  id: string;
  location: string;
  latitude: number;
  longitude: number;
  description: string;
  category: string;
  priority: 'low' | 'medium' | 'high' | string;
  photo?: string;
  status: 'pending' | 'investigating' | 'resolved' | string;
  date: string;
}

interface SearchResult {
  display_name: string;
  lat: string;
  lon: string;
}

export default function UserComplaintsPage() {
  const [markerPosition, setMarkerPosition] = useState<[number, number] | null>(null);
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState('');
  const [priority, setPriority] = useState<'low' | 'medium' | 'high'>('medium');
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);
  const [photoFile, setPhotoFile] = useState<File | null>(null);
  const [isAiClassifying, setIsAiClassifying] = useState(false);
  const [aiWaste, setAiWaste] = useState<WasteClassificationResponse | null>(null);
  const [sortBy, setSortBy] = useState<'date' | 'priority'>('date');
  const [filterStatus, setFilterStatus] = useState<'all' | 'pending' | 'investigating' | 'resolved' | string>('all');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Location search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedLocationName, setSelectedLocationName] = useState('');
  const searchTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  
  const [complaints, setComplaints] = useState<UIComplaint[]>([]);

  useEffect(() => {
    fetchComplaints();
  }, []);

  const fetchComplaints = async () => {
    setIsLoading(true);
    try {
      const data = await getMyComplaints();
      const mapped = data.map(mapApiToUI);
      setComplaints(mapped);
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to load complaints');
    } finally {
      setIsLoading(false);
    }
  };

  const mapApiToUI = (comp: ApiComplaint): UIComplaint => {
    let uiStatus = comp.status.toLowerCase();
    if (uiStatus === 'in progress') uiStatus = 'investigating';

    // Try to extract category from description if formatted as "[Category] Actual desc"
    let uiCategory = 'General';
    let uiDesc = comp.description;
    const match = comp.description.match(/^\[(.*?)\] (.*)/);
    if (match) {
      uiCategory = match[1];
      uiDesc = match[2];
    }

    return {
      id: comp._id,
      location: comp.location || 'Unknown location',
      latitude: 0,
      longitude: 0,
      description: uiDesc,
      category: uiCategory,
      priority: comp.priority?.toLowerCase() || 'medium',
      photo: comp.imageUrl,
      status: uiStatus,
      date: comp.createdAt,
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

  const handlePhotoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setPhotoFile(file);
      const reader = new FileReader();
      reader.onloadend = () => setPhotoPreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const removePhoto = () => {
    setPhotoPreview(null);
    setPhotoFile(null);
    setAiWaste(null);
    setIsAiClassifying(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  useEffect(() => {
    if (!photoFile) {
      setAiWaste(null);
      setIsAiClassifying(false);
      return;
    }

    let cancelled = false;
    setIsAiClassifying(true);
    setAiWaste(null);

    classifyWasteFromImage(photoFile)
      .then((ai) => {
        if (cancelled) return;
        setAiWaste(ai);

        const nextPriority =
          ai.classification === 'Non-biodegradable'
            ? 'high'
            : ai.classification === 'Biodegradable'
              ? 'low'
              : 'medium';

        setPriority(nextPriority);
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
  }, [photoFile]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!markerPosition || !description || !category) return;

    setIsSubmitting(true);
    
    try {
      let uploadedImageUrl = '';
      if (photoFile) {
        uploadedImageUrl = await uploadImage(photoFile);
      }

      const locName = selectedLocationName || `Lat: ${markerPosition[0].toFixed(4)}, Lng: ${markerPosition[1].toFixed(4)}`;
      const formattedDesc = `[${category}] ${description}`;
      // Capitalize first letter of priority for API
      const apiPriority = priority.charAt(0).toUpperCase() + priority.slice(1);

      const newComp = await createComplaint({
        description: formattedDesc,
        location: locName,
        priority: apiPriority,
        lat: markerPosition[0],
        lng: markerPosition[1],
        ...(uploadedImageUrl ? { imageUrl: uploadedImageUrl } : {}),
        ...(aiWaste
          ? {
              ai: {
                classification: aiWaste.classification,
                confidence: aiWaste.confidence ?? null,
                provider: 'waste_classifier',
                raw: aiWaste,
              },
            }
          : {}),
      });

      const updated = [mapApiToUI(newComp), ...complaints];
      setComplaints(updated);
      
      setDescription('');
      setCategory('');
      setPriority('medium');
      setMarkerPosition(null);
      setPhotoPreview(null);
      setPhotoFile(null);
      setAiWaste(null);
      setIsAiClassifying(false);
      setSearchQuery('');
      setSelectedLocationName('');
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err: any) {
      alert(err.response?.data?.message || 'Failed to submit complaint');
    } finally {
      setIsSubmitting(false);
    }
  };

  const categories = ['Overflowing Bin', 'Illegal Dumping', 'Missed Collection', 'Broken Bin', 'Odor Issue', 'Hazardous Waste', 'Other'];

  const getPriorityColors = (p: string) => {
    if (p === 'high') return 'bg-rose-100 text-rose-700 border-rose-200';
    if (p === 'medium') return 'bg-amber-100 text-amber-700 border-amber-200';
    return 'bg-emerald-100 text-emerald-700 border-emerald-200';
  };

  const getStatusConfig = (s: string) => {
    if (s === 'resolved') return { color: 'bg-emerald-100 text-emerald-700', icon: CheckCircle2 };
    if (s === 'investigating') return { color: 'bg-blue-100 text-blue-700', icon: Loader2 };
    return { color: 'bg-amber-100 text-amber-700', icon: Clock };
  };

  const filtered = complaints
    .filter(c => filterStatus === 'all' || c.status === filterStatus)
    .sort((a, b) => {
      if (sortBy === 'date') return new Date(b.date).getTime() - new Date(a.date).getTime();
      const p: { [key: string]: number } = { high: 3, medium: 2, low: 1 };
      return (p[b.priority] || 0) - (p[a.priority] || 0);
    });

  const isFormValid = markerPosition && description && category;

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-[60vh]">
        <Loader2 className="w-10 h-10 animate-spin text-rose-500" />
      </div>
    );
  }

  return (
    <div className="w-full max-w-[1400px] mx-auto animate-fade-in pb-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl md:text-4xl font-black text-slate-800 tracking-tight mb-2">Report an Issue</h1>
        <p className="text-slate-500 font-medium text-lg">Help us keep the city clean by reporting waste management issues.</p>
      </div>

      {/* Summary Tracking Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm flex items-center gap-4">
          <div className="w-12 h-12 bg-slate-50 text-slate-500 rounded-xl flex items-center justify-center font-bold text-lg border border-slate-100">
            {complaints.length}
          </div>
          <div>
            <p className="font-bold text-slate-800 text-sm">Total Reports</p>
            <p className="text-xs text-slate-500 font-medium">All time</p>
          </div>
        </div>
        
        <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm flex items-center gap-4">
          <div className="w-12 h-12 bg-amber-50 text-amber-600 rounded-xl flex items-center justify-center shadow-sm shadow-amber-500/10">
            <Clock className="w-6 h-6" />
          </div>
          <div>
            <p className="font-bold text-slate-800 text-sm">Pending</p>
            <p className="font-black text-xl text-amber-600">{complaints.filter(c => c.status === 'pending').length}</p>
          </div>
        </div>

        <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm flex items-center gap-4">
          <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-xl flex items-center justify-center shadow-sm shadow-blue-500/10">
            <Loader2 className="w-6 h-6" />
          </div>
          <div>
            <p className="font-bold text-slate-800 text-sm">In Progress</p>
            <p className="font-black text-xl text-blue-600">{complaints.filter(c => c.status === 'investigating').length}</p>
          </div>
        </div>

        <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm flex items-center gap-4">
          <div className="w-12 h-12 bg-emerald-50 text-emerald-600 rounded-xl flex items-center justify-center shadow-sm shadow-emerald-500/10">
            <CheckCircle2 className="w-6 h-6" />
          </div>
          <div>
            <p className="font-bold text-slate-800 text-sm">Resolved</p>
            <p className="font-black text-xl text-emerald-600">{complaints.filter(c => c.status === 'resolved').length}</p>
          </div>
        </div>
      </div>

      {error && (
        <div className="mb-8 bg-red-50 text-red-600 p-4 rounded-xl border border-red-100 font-medium flex items-center gap-2">
          <AlertTriangle className="w-5 h-5" />
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
        {/* Form Column */}
        <div className="xl:col-span-7 space-y-6">
          <div className="bg-white rounded-[2rem] p-6 sm:p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-slate-100">
            <div className="flex items-center gap-4 mb-8">
              <div className="w-14 h-14 bg-gradient-to-tr from-rose-500 to-orange-400 rounded-2xl flex items-center justify-center shadow-lg shadow-rose-500/20">
                <ShieldAlert className="w-7 h-7 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-slate-800">New Complaint</h2>
                <p className="text-slate-500 font-medium">Please provide accurate details</p>
              </div>
            </div>
            
            <form onSubmit={handleSubmit} className="space-y-8">
              {/* Location Search + Map */}
              <div>
                <label className="flex items-center gap-2 text-sm font-bold text-slate-700 uppercase tracking-wider mb-3">
                  <MapPin className="w-4 h-4 text-rose-500" /> Location
                </label>
                
                {/* Search Bar */}
                <div className="mb-3 flex gap-2">
                  <div className="relative flex-1">
                    <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => handleSearch(e.target.value)}
                      placeholder="Search for a location (e.g. MG Road, Delhi)..."
                      className="w-full pl-12 pr-5 py-4 bg-slate-50 border-2 border-slate-100 rounded-[1.25rem] focus:bg-white focus:outline-none focus:border-rose-400 focus:ring-4 focus:ring-rose-400/10 text-slate-800 font-medium placeholder:text-slate-400 transition-all"
                    />
                    {isSearching && (
                      <Loader2 className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-rose-400 animate-spin" />
                    )}
                    
                    {/* Search Results Dropdown */}
                    {searchResults.length > 0 && (
                      <div className="absolute z-50 top-full left-0 right-0 mt-2 bg-white rounded-2xl shadow-2xl border border-slate-100 overflow-hidden">
                        {searchResults.map((result, i) => (
                          <button
                            key={i}
                            type="button"
                            onClick={() => selectSearchResult(result)}
                            className="w-full text-left px-5 py-3.5 hover:bg-slate-50 flex items-start gap-3 transition-colors border-b border-slate-50 last:border-0"
                          >
                            <MapPin className="w-4 h-4 text-rose-500 mt-0.5 shrink-0" />
                            <span className="text-sm font-medium text-slate-700 leading-snug">{result.display_name}</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={handleLocateMe}
                    className="bg-slate-50 border-2 border-slate-100 text-slate-600 hover:border-rose-300 hover:text-rose-500 rounded-[1.25rem] px-5 flex items-center justify-center transition-all group"
                    title="Auto detect location"
                  >
                    <Target className="w-6 h-6 group-hover:scale-110 transition-transform" />
                  </button>
                </div>

                <div className="bg-slate-50 rounded-[1.5rem] overflow-hidden h-[250px] border-2 border-slate-100 relative group">
                  <ComplaintMap position={markerPosition} setPosition={(pos) => {
                    setMarkerPosition(pos);
                    setSelectedLocationName('');
                    setSearchQuery('');
                    setSearchResults([]);
                  }} compact />
                  {!markerPosition && (
                    <div className="absolute inset-0 pointer-events-none flex items-center justify-center bg-white/20 backdrop-blur-[1px]">
                      <div className="bg-white/90 px-4 py-2 rounded-xl shadow-lg border border-slate-200 text-sm font-bold text-slate-700">
                        Search above or click on map
                      </div>
                    </div>
                  )}
                </div>
                {markerPosition && (
                  <div className="mt-3 flex items-center gap-2 text-sm font-bold text-emerald-600 bg-emerald-50 px-4 py-3 rounded-xl border border-emerald-100">
                    <CheckCircle2 className="w-5 h-5 shrink-0" /> 
                    <span className="truncate">{selectedLocationName || `Lat: ${markerPosition[0].toFixed(4)}, Lng: ${markerPosition[1].toFixed(4)}`}</span>
                  </div>
                )}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                {/* Category */}
                <div>
                  <label className="block text-sm font-bold text-slate-700 uppercase tracking-wider mb-3">Category</label>
                  <div className="relative">
                    <select
                      value={category}
                      onChange={(e) => setCategory(e.target.value)}
                      className="w-full px-5 py-4 bg-slate-50 border-2 border-slate-100 rounded-[1.25rem] focus:bg-white focus:outline-none focus:border-rose-400 focus:ring-4 focus:ring-rose-400/10 text-slate-800 font-semibold appearance-none pr-12 transition-all cursor-pointer"
                      required
                    >
                      <option value="" disabled>Select issue type...</option>
                      {categories.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                    <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 pointer-events-none" />
                  </div>
                </div>

                {/* Priority */}
                <div>
                  <label className="block text-sm font-bold text-slate-700 uppercase tracking-wider mb-3">Priority</label>
                  <div className="flex bg-slate-50 p-1.5 rounded-[1.25rem] border border-slate-100">
                    {(['low', 'medium', 'high'] as const).map(p => (
                      <button
                        key={p}
                        type="button"
                        onClick={() => setPriority(p)}
                        className={`flex-1 py-2.5 rounded-xl text-sm font-bold capitalize transition-all ${
                          priority === p ? 'bg-white shadow-sm text-slate-800' : 'text-slate-500 hover:text-slate-700'
                        }`}
                      >
                        {p}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Description */}
              <div>
                <label className="block text-sm font-bold text-slate-700 uppercase tracking-wider mb-3">Description</label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Provide specific details about the problem..."
                  rows={4}
                  className="w-full px-5 py-4 bg-slate-50 border-2 border-slate-100 rounded-[1.25rem] focus:bg-white focus:outline-none focus:border-rose-400 focus:ring-4 focus:ring-rose-400/10 text-slate-800 font-medium placeholder:text-slate-400 transition-all resize-none"
                  required
                />
              </div>

              {/* Photo */}
              <div>
                <label className="block text-sm font-bold text-slate-700 uppercase tracking-wider mb-3">Photo Proof</label>
                {photoPreview ? (
                  <div className="relative rounded-[1.25rem] overflow-hidden group border-2 border-slate-100 inline-block">
                    <img src={photoPreview} alt="Preview" className="h-48 w-auto object-cover" />
                    <button
                      type="button"
                      onClick={removePhoto}
                      className="absolute top-3 right-3 bg-white/90 backdrop-blur-md p-2 rounded-full shadow-lg text-rose-500 hover:bg-rose-50 transition-colors"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full sm:w-auto px-8 py-8 border-2 border-dashed border-slate-200 hover:border-rose-300 rounded-[1.25rem] hover:bg-rose-50/50 transition-all flex flex-col items-center justify-center gap-3 cursor-pointer group"
                  >
                    <div className="w-12 h-12 bg-rose-100 text-rose-500 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                      <Camera className="w-6 h-6" />
                    </div>
                    <span className="font-bold text-slate-600">Upload a Photo</span>
                  </button>
                )}
                <input ref={fileInputRef} type="file" accept="image/*" onChange={handlePhotoChange} className="hidden" />

                {isAiClassifying && (
                  <p className="mt-3 text-xs font-semibold text-rose-700">
                    Analyzing image...
                  </p>
                )}

                {aiWaste && !isAiClassifying && (
                  <p className="mt-3 text-xs font-semibold text-rose-700">
                    AI detected: {aiWaste.classification} ({Math.round(aiWaste.confidence * 100)}% confidence)
                  </p>
                )}
              </div>

              <div className="pt-4 border-t border-slate-100">
                <button
                  type="submit"
                  disabled={!isFormValid || isSubmitting}
                  className={`w-full py-5 rounded-[1.25rem] font-bold text-lg transition-all flex items-center justify-center gap-2 ${
                    isFormValid && !isSubmitting
                      ? 'bg-gradient-to-r from-rose-500 to-orange-500 text-white shadow-lg shadow-rose-500/25 hover:shadow-xl hover:scale-[1.01]'
                      : 'bg-slate-100 text-slate-400 cursor-not-allowed'
                  }`}
                >
                  {isSubmitting ? <Loader2 className="w-6 h-6 animate-spin" /> : <><Send className="w-6 h-6" /> Submit Report</>}
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* History Column - FIXED: no viewport height constraint, uses auto height */}
        <div className="xl:col-span-5">
          <div className="bg-white rounded-[2rem] p-6 sm:p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-slate-100 xl:sticky xl:top-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
                <Clock className="w-6 h-6 text-slate-400" /> Recent History
              </h2>
              <div className="flex items-center gap-2">
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value as any)}
                  className="px-4 py-2 bg-slate-50 border border-slate-100 rounded-xl text-sm font-semibold text-slate-700 outline-none focus:border-rose-300 appearance-none cursor-pointer"
                >
                  <option value="all">All</option>
                  <option value="pending">Pending</option>
                  <option value="investigating">Investigating</option>
                  <option value="resolved">Resolved</option>
                </select>
                <button onClick={() => setSortBy(sortBy === 'date' ? 'priority' : 'date')} className="p-2.5 bg-slate-50 border border-slate-100 rounded-xl text-slate-600 hover:bg-slate-100 transition-colors">
                  <ArrowUpDown className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="space-y-4 max-h-[600px] overflow-y-auto pr-1 custom-scrollbar">
              {filtered.length === 0 ? (
                <div className="py-16 flex flex-col items-center justify-center text-slate-400">
                  <ShieldAlert className="w-16 h-16 opacity-20 mb-4" />
                  <p className="font-semibold">No complaints found</p>
                </div>
              ) : (
                filtered.map(c => {
                  const sConf = getStatusConfig(c.status);
                  const Icon = sConf.icon;
                  return (
                    <div key={c.id} className="bg-slate-50/50 border border-slate-100 hover:border-slate-200 rounded-2xl p-5 hover:shadow-md transition-all">
                      <div className="flex justify-between items-start mb-3 gap-3">
                        <div className="flex-1 min-w-0">
                          <div className="flex flex-wrap gap-2 mb-2">
                            <span className="px-3 py-1 bg-white text-slate-600 rounded-lg text-xs font-bold border border-slate-100">{c.category}</span>
                            <span className={`px-3 py-1 rounded-lg text-xs font-bold border ${getPriorityColors(c.priority)} capitalize`}>{c.priority}</span>
                          </div>
                          <p className="font-bold text-slate-800 leading-snug mb-2">{c.description}</p>
                          {c.photo && (
                            <img src={c.photo} alt="Issue" className="w-24 h-24 object-cover rounded-lg border border-slate-200" />
                          )}
                        </div>
                        <div className={`px-3 py-1.5 rounded-full flex items-center gap-1.5 shrink-0 ${sConf.color}`}>
                          <Icon className={`w-3.5 h-3.5 ${c.status === 'investigating' ? 'animate-spin' : ''}`} />
                          <span className="text-[11px] font-bold capitalize">{c.status}</span>
                        </div>
                      </div>
                      
                      <div className="bg-white p-3 rounded-xl flex items-center justify-between text-sm mt-3 border border-slate-100">
                        <div className="flex items-center gap-2 text-slate-600 font-medium min-w-0">
                          <MapPin className="w-4 h-4 text-emerald-500 shrink-0" /> <span className="truncate text-xs">{c.location}</span>
                        </div>
                        <div className="text-slate-400 font-bold text-xs shrink-0 ml-2">{new Date(c.date).toLocaleDateString()}</div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
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
