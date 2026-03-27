import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Fix Leaflet default icon paths
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface Bin {
  id: number | string;
  lat: number;
  lng: number;
  fillLevel: number;
  status: "empty" | "medium" | "full";
  predictedOverflow: string;
  location: string;
}

interface Truck {
  id: number | string;
  lat: number;
  lng: number;
  name: string;
  status: string;
}

interface UserRequest {
  id: number | string;
  userName: string;
  phone: string;
  address: string;
  lat: number;
  lng: number;
  wasteType: string;
  requestedDate: string;
  status: "pending" | "assigned" | "completed";
  priority: "high" | "medium" | "low";
}

interface LeafletMapProps {
  bins: Bin[];
  trucks: Truck[];
  userRequests: UserRequest[];
  showRoutes: boolean;
  routes: { positions: [number, number][]; color: string; type: string }[];
  onBinClick?: (bin: Bin) => void;
}

const getBinColor = (status: string) => {
  switch (status) {
    case "full":
      return "var(--destructive)";
    case "medium":
      return "var(--warning-yellow)";
    case "empty":
      return "var(--success-green)";
    default:
      return "var(--muted-foreground)";
  }
};

const createBinIcon = (color: string) => {
  return L.divIcon({
    className: "custom-marker bg-transparent border-none",
    html: `<div style="background-color: ${color}; width: 32px; height: 32px; border-radius: 50%; border: 2px solid rgba(255,255,255,0.9); display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
        <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
      </svg>
    </div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    popupAnchor: [0, -16],
  });
};

const createTruckIcon = () => {
  return L.divIcon({
    className: "custom-marker bg-transparent border-none",
    html: `<div style="background-color: var(--primary); width: 36px; height: 36px; border-radius: 50%; border: 2px solid white; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3); animation: pulse 2s ease-in-out infinite;">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
        <path d="M1 3h15v13H1zM16 8h4l3 3v5h-7V8z"/>
        <circle cx="5.5" cy="18.5" r="2.5"/>
        <circle cx="18.5" cy="18.5" r="2.5"/>
      </svg>
    </div>`,
    iconSize: [36, 36],
    iconAnchor: [18, 18],
    popupAnchor: [0, -18],
  });
};

const createUserRequestIcon = (priority: string, status: string) => {
  const color = status === "completed" ? "var(--success-green)" : priority === "high" ? "var(--destructive)" : priority === "medium" ? "var(--warning-yellow)" : "var(--info-blue)";
  const opacity = status === "completed" ? "0.5" : "1.0";
  
  return L.divIcon({
    className: "custom-marker bg-transparent border-none",
    html: `<div style="background-color: ${color}; opacity: ${opacity}; width: 34px; height: 34px; border-radius: 50%; border: 2px solid white; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
        <circle cx="12" cy="7" r="4"/>
      </svg>
    </div>`,
    iconSize: [34, 34],
    iconAnchor: [17, 17],
    popupAnchor: [0, -17],
  });
};

export default function LeafletMap({ bins, trucks, userRequests, showRoutes, routes, onBinClick }: LeafletMapProps) {
  const mapRef = useRef<L.Map | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const markersRef = useRef<L.Marker[]>([]);
  const routeLinesRef = useRef<L.Polyline[]>([]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    // Initialize map
    const map = L.map(containerRef.current).setView([28.6139, 77.2090], 12); // Delhi, India
    mapRef.current = map;

    // Add tile layer
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!mapRef.current) return;

    // Clear existing markers
    markersRef.current.forEach((marker) => marker.remove());
    markersRef.current = [];

    // Add bin markers
    bins?.forEach((bin) => {
      const marker = L.marker([bin.lat, bin.lng], {
        icon: createBinIcon(getBinColor(bin.status)),
      }).addTo(mapRef.current!);

      const popupContent = `
        <div class="min-w-[200px] p-1">
          <h4 class="font-semibold text-foreground text-lg mb-2">Bin #${bin.id}</h4>
          <div class="space-y-3">
            <div class="flex justify-between text-sm">
              <span class="text-muted-foreground">Location:</span>
              <span class="text-foreground font-medium">${bin.location}</span>
            </div>
            <div class="flex justify-between text-sm">
              <span class="text-muted-foreground">Fill Level:</span>
              <span class="text-foreground font-medium">${bin.fillLevel}%</span>
            </div>
            <div class="w-full bg-secondary/20 rounded-full h-2 overflow-hidden border border-border">
              <div class="h-full rounded-full transition-all" style="width: ${bin.fillLevel}%; background-color: ${getBinColor(bin.status)}"></div>
            </div>
            <div class="flex justify-between text-sm">
              <span class="text-muted-foreground">Predicted Overflow:</span>
              <span class="text-foreground font-medium">${bin.predictedOverflow}</span>
            </div>
            <div class="pt-3 border-t border-border mt-2">
              <span class="inline-block px-3 py-1 rounded-full text-xs font-medium ${
                bin.status === "full"
                  ? "bg-destructive/10 text-destructive border border-destructive/20"
                  : bin.status === "medium"
                  ? "bg-amber-500/10 text-amber-600 border border-amber-500/20"
                  : "bg-emerald-500/10 text-emerald-600 border border-emerald-500/20"
              }">
                ${bin.status.toUpperCase()}
              </span>
            </div>
          </div>
        </div>
      `;

      marker.bindPopup(popupContent);
      if (onBinClick) {
        marker.on("click", () => onBinClick(bin));
      }
      markersRef.current.push(marker);
    });

    // Add truck markers
    trucks?.forEach((truck) => {
      const marker = L.marker([truck.lat, truck.lng], {
        icon: createTruckIcon(),
      }).addTo(mapRef.current!);

      marker.bindPopup(`<div class="text-foreground font-medium p-1">${truck.name}</div>`);
      markersRef.current.push(marker);
    });

    // Add user request markers
    userRequests?.forEach((request) => {
      const marker = L.marker([request.lat, request.lng], {
        icon: createUserRequestIcon(request.priority, request.status),
      }).addTo(mapRef.current!);

      const popupContent = `
        <div class="min-w-[200px] p-1">
          <h4 class="font-semibold text-foreground text-lg mb-2">Request #${request.id}</h4>
          <div class="space-y-2">
            <div class="flex justify-between text-sm">
              <span class="text-muted-foreground">User Name:</span>
              <span class="text-foreground font-medium">${request.userName}</span>
            </div>
            <div class="flex justify-between text-sm">
              <span class="text-muted-foreground">Phone:</span>
              <span class="text-foreground font-medium">${request.phone}</span>
            </div>
            <div class="flex justify-between text-sm">
              <span class="text-muted-foreground">Address:</span>
              <span class="text-foreground font-medium">${request.address}</span>
            </div>
            <div class="flex justify-between text-sm">
              <span class="text-muted-foreground">Waste Type:</span>
              <span class="text-foreground font-medium">${request.wasteType}</span>
            </div>
            <div class="flex justify-between text-sm">
              <span class="text-muted-foreground">Requested Date:</span>
              <span class="text-foreground font-medium">${request.requestedDate}</span>
            </div>
            <div class="flex justify-between text-sm">
              <span class="text-muted-foreground">Status:</span>
              <span class="text-foreground font-medium capitalize">${request.status}</span>
            </div>
            <div class="flex justify-between text-sm">
              <span class="text-muted-foreground">Priority:</span>
              <span class="text-foreground font-medium capitalize">${request.priority}</span>
            </div>
          </div>
        </div>
      `;

      marker.bindPopup(popupContent);
      markersRef.current.push(marker);
    });
  }, [bins, trucks, userRequests, onBinClick]);

  useEffect(() => {
    if (!mapRef.current) return;

    // Clear existing routes
    routeLinesRef.current.forEach((line) => line.remove());
    routeLinesRef.current = [];

    if (showRoutes && routes) {
      routes.forEach((route) => {
        const polyline = L.polyline(route.positions, {
          color: route.color || 'var(--primary)',
          weight: 4,
          dashArray: "8,6",
          opacity: 0.8,
        }).addTo(mapRef.current!);
        routeLinesRef.current.push(polyline);
      });
    }
  }, [showRoutes, routes]);

  return <div ref={containerRef} className="absolute inset-0 rounded-xl overflow-hidden" />;
}