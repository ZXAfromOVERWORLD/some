import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Search, MapPin, Loader2 } from 'lucide-react';

// Fix Leaflet default marker icon
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface LocationResult {
  place_id: number;
  lat: string;
  lon: string;
  display_name: string;
}

interface LocationPickerMapProps {
  onLocationSelect: (lat: number, lng: number) => void;
  initialLocation?: [number, number];
  selectedLocation?: [number, number] | null;
}

export default function LocationPickerMap({ onLocationSelect, initialLocation = [28.6139, 77.2090], selectedLocation }: LocationPickerMapProps) {
  const mapRef = useRef<L.Map | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const markerRef = useRef<L.Marker | null>(null);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      zoomControl: false // We reposition it later or just leave it out of the top-left to avoid blocking the search bar
    }).setView(initialLocation, 12);

    L.control.zoom({ position: 'bottomright' }).addTo(map);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19,
    }).addTo(map);

    // Initial marker
    markerRef.current = L.marker(initialLocation, { draggable: true }).addTo(map);

    markerRef.current.on('dragend', (e) => {
      const position = e.target.getLatLng();
      onLocationSelect(position.lat, position.lng);
    });

    map.on('click', (e) => {
      if (markerRef.current) {
        markerRef.current.setLatLng(e.latlng);
        onLocationSelect(e.latlng.lat, e.latlng.lng);
      }
    });

    mapRef.current = map;

    setTimeout(() => {
      map.invalidateSize();
    }, 200);

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []); // Run only once to mount

  useEffect(() => {
    if (selectedLocation && mapRef.current && markerRef.current) {
      const newLatLng = new L.LatLng(selectedLocation[0], selectedLocation[1]);
      mapRef.current.flyTo(newLatLng, 15);
      markerRef.current.setLatLng(newLatLng);
    }
  }, [selectedLocation]);

  return (
    <div className="relative w-full h-[350px] rounded-lg shadow-inner z-0 overflow-hidden" style={{ zIndex: 1 }}>
      {/* Map Container */}
      <div ref={containerRef} className="w-full h-full" />
    </div>
  );
}
