import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet default marker icon
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

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

interface DriverMapProps {
  currentLocation: [number, number];
  collectionPoints: CollectionPoint[];
  optimizedRoute: [number, number][];
  routeGeometry?: [number, number][];
  onMarkCompleted: (id: string) => void;
}

// Custom marker icons for bins
const createIcon = (color: string, completed: boolean = false) =>
  L.divIcon({
    className: 'custom-marker bg-transparent border-none',
    html: `<div style="background-color: ${color}; width: 30px; height: 30px; border-radius: 50%; border: 3px solid white; box-shadow: 0 4px 12px rgba(0,0,0,0.15); ${completed ? 'opacity: 0.5;' : ''}"></div>`,
    iconSize: [30, 30],
    iconAnchor: [15, 15],
  });

// Truck icon for driver's current location
const createTruckIcon = () =>
  L.divIcon({
    className: 'custom-truck-marker bg-transparent border-none',
    html: `
      <div style="width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #10B981, #059669); border-radius: 50%; border: 3px solid white; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M16 17H15C15 17.55 14.55 18 14 18C13.45 18 13 17.55 13 17H11C11 17.55 10.55 18 10 18C9.45 18 9 17.55 9 17H8V13H3V8H8V6H15V10H18L20 13V17H19C19 17.55 18.55 18 18 18C17.45 18 17 17.55 17 17H16Z" fill="white"/>
          <circle cx="10" cy="17" r="1" fill="white"/>
          <circle cx="18" cy="17" r="1" fill="white"/>
        </svg>
      </div>
    `,
    iconSize: [40, 40],
    iconAnchor: [20, 20],
  });

// User request marker (pin/dot shape)
const createUserRequestIcon = (completed: boolean = false) =>
  L.divIcon({
    className: 'custom-user-marker bg-transparent border-none',
    html: `
      <div style="display: flex; flex-direction: column; align-items: center; ${completed ? 'opacity: 0.5;' : ''}">
        <svg width="32" height="42" viewBox="0 0 24 34" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 4px 6px rgba(0,0,0,0.2));">
          <path d="M12 0C7.03 0 3 4.03 3 9C3 15.25 12 26 12 26C12 26 21 15.25 21 9C21 4.03 16.97 0 12 0Z" fill="#3B82F6"/>
          <circle cx="12" cy="9" r="4" fill="white"/>
        </svg>
      </div>
    `,
    iconSize: [32, 42],
    iconAnchor: [16, 42],
  });

export function DriverMap({ currentLocation, collectionPoints, optimizedRoute, routeGeometry, onMarkCompleted }: DriverMapProps) {
  const mapRef = useRef<L.Map | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const markersRef = useRef<Map<string, L.Marker>>(new Map());
  const routeLineRef = useRef<L.Polyline | null>(null);
  const currentLocationMarkerRef = useRef<L.Marker | null>(null);
  const [isMapReady, setIsMapReady] = useState(false);
  const initializingRef = useRef(false);

  // Initialize map once
  useEffect(() => {
    if (!containerRef.current || mapRef.current || initializingRef.current) return;

    initializingRef.current = true;

    const timeout = setTimeout(() => {
      if (!containerRef.current) {
        initializingRef.current = false;
        return;
      }

      try {
        const map = L.map(containerRef.current, {
          zoomControl: true,
          scrollWheelZoom: true,
        }).setView(currentLocation, 13);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '&copy; OpenStreetMap contributors',
          maxZoom: 19,
        }).addTo(map);

        setTimeout(() => map.invalidateSize(), 100);

        mapRef.current = map;
        setIsMapReady(true);
      } catch (error) {
        console.error('Error initializing map:', error);
      } finally {
        initializingRef.current = false;
      }
    }, 100);

    return () => {
      clearTimeout(timeout);
      markersRef.current.forEach((marker) => marker.remove());
      markersRef.current.clear();
      
      if (routeLineRef.current) routeLineRef.current.remove();
      if (currentLocationMarkerRef.current) currentLocationMarkerRef.current.remove();
      if (mapRef.current) {
        mapRef.current.off();
        mapRef.current.remove();
        mapRef.current = null;
      }
      
      setIsMapReady(false);
      initializingRef.current = false;
    };
  }, []);

  // Update map content when data changes
  useEffect(() => {
    if (!isMapReady || !mapRef.current) return;

    try {
      markersRef.current.forEach((marker) => marker.remove());
      markersRef.current.clear();

      if (currentLocationMarkerRef.current) {
        currentLocationMarkerRef.current.remove();
      }
      currentLocationMarkerRef.current = L.marker(currentLocation, {
        icon: createTruckIcon(),
      }).addTo(mapRef.current);

      const truckPopupDiv = document.createElement('div');
      truckPopupDiv.className = 'text-center p-1';
      truckPopupDiv.innerHTML = `
        <p class="font-semibold text-foreground">Your Current Location</p>
        <p class="text-xs text-muted-foreground mt-1">Truck Position</p>
      `;
      currentLocationMarkerRef.current.bindPopup(truckPopupDiv);

      collectionPoints.forEach((point) => {
        const color =
          point.status === 'completed'
            ? '#9CA3AF'
            : point.priority === 'high'
              ? '#EF4444'
              : point.priority === 'medium'
                ? '#F59E0B'
                : '#10B981';

        const markerIcon = (point.type === 'user-request' || point.type === 'complaint')
          ? createUserRequestIcon(point.status === 'completed')
          : createIcon(color, point.status === 'completed');

        const marker = L.marker([point.latitude, point.longitude], {
          icon: markerIcon,
        }).addTo(mapRef.current!);

        const popupDiv = document.createElement('div');
        popupDiv.className = 'p-0 m-0';
        
        let typeBadge = point.type === 'user-request'
          ? `<span style="background: #EFF6FF; color: #2563EB; border: 1px solid #BFDBFE; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 700; text-transform: uppercase;">User Request</span>`
          : point.type === 'complaint'
            ? `<span style="background: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 700; text-transform: uppercase;">Complaint</span>`
            : `<span style="background: #F3F4F6; color: #4B5563; border: 1px solid #E5E7EB; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 700; text-transform: uppercase;">Bin</span>`;

        let fillLevelHTML = '';
        if (point.fillLevel !== undefined) {
          fillLevelHTML = `
            <div style="margin-bottom: 12px;">
              <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
                <span style="color: #6B7280; font-weight: 600;">Fill Level</span>
                <span style="color: #111827; font-weight: 800;">${point.fillLevel}%</span>
              </div>
              <div style="width: 100%; background: #E5E7EB; border-radius: 999px; height: 8px; overflow: hidden;">
                <div style="height: 100%; border-radius: 999px; transition: all 0.3s; width: ${point.fillLevel}%; background-color: ${color}"></div>
              </div>
            </div>
          `;
        }

        let buttonHTML = '';
        if (point.status === 'pending') {
          buttonHTML = `
            <button 
              id="mark-btn-${point.id}"
              style="width: 100%; background: linear-gradient(to right, #10B981, #059669); color: white; border: none; padding: 10px; border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px; box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2); transition: all 0.2s;"
              onmouseover="this.style.transform='scale(1.02)'"
              onmouseout="this.style.transform='scale(1)'"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
              Mark Collected
            </button>
          `;
        }

        popupDiv.innerHTML = `
          <div style="min-width: 220px; font-family: 'Inter', system-ui, sans-serif;">
            <div style="margin-bottom: 8px;">${typeBadge}</div>
            <h3 style="margin: 0 0 4px 0; font-size: 15px; font-weight: 800; color: #111827;">${point.location}</h3>
            <p style="margin: 0 0 12px 0; font-size: 13px; color: #4B5563; line-height: 1.4;">${point.details}</p>
            ${fillLevelHTML}
            ${buttonHTML}
          </div>
        `;

        marker.bindPopup(popupDiv);

        if (point.status === 'pending') {
          marker.on('popupopen', () => {
            const btn = document.getElementById(`mark-btn-${point.id}`);
            if (btn) {
              btn.onclick = () => {
                onMarkCompleted(point.id);
                marker.closePopup();
              };
            }
          });
        }

        markersRef.current.set(point.id, marker);
      });

      if (routeLineRef.current) {
        routeLineRef.current.remove();
      }
      const line = (routeGeometry && routeGeometry.length > 1) ? routeGeometry : optimizedRoute;

      if (line.length > 1) {
        routeLineRef.current = L.polyline(line, {
          color: '#10B981',
          weight: 5,
          opacity: 0.8,
          dashArray: '10, 10',
          lineJoin: 'round',
          lineCap: 'round',
        }).addTo(mapRef.current);

        routeLineRef.current.bindPopup(`
          <div class="text-center p-1">
            <p class="font-semibold text-sm text-foreground">Optimized Collection Route</p>
            <p class="text-xs text-muted-foreground mt-1">${optimizedRoute.length - 1} stops</p>
          </div>
        `);
      }
    } catch (error) {
      console.error('Error updating map:', error);
    }
  }, [collectionPoints, optimizedRoute, onMarkCompleted, currentLocation, isMapReady]);

  return <div ref={containerRef} className="h-full w-full rounded-xl overflow-hidden shadow-sm border border-border" style={{ minHeight: '400px', position: 'relative', zIndex: 0 }} />;
}
