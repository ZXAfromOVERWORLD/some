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

interface ComplaintMapProps {
  position: [number, number] | null;
  setPosition: (pos: [number, number]) => void;
  compact?: boolean;
}

export function ComplaintMap({ position, setPosition, compact }: ComplaintMapProps) {
  const mapRef = useRef<L.Map | null>(null);
  const markerRef = useRef<L.Marker | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isMapReady, setIsMapReady] = useState(false);
  const initializingRef = useRef(false);

  // Initialize map once
  useEffect(() => {
    if (!containerRef.current || mapRef.current || initializingRef.current) return;

    initializingRef.current = true;

    // Small delay to ensure container is rendered
    const timeout = setTimeout(() => {
      if (!containerRef.current) {
        initializingRef.current = false;
        return;
      }

      try {
        const map = L.map(containerRef.current, {
          zoomControl: true,
          scrollWheelZoom: true,
        }).setView([28.4595, 77.0266], 13);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '&copy; <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> contributors',
          maxZoom: 19,
        }).addTo(map);

        // Add click handler
        map.on('click', (e: L.LeafletMouseEvent) => {
          setPosition([e.latlng.lat, e.latlng.lng]);
        });

        // Invalidate size after a short delay to ensure proper rendering
        setTimeout(() => {
          map.invalidateSize();
        }, 100);

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
      if (markerRef.current) {
        try {
          markerRef.current.remove();
        } catch (e) {
          // Ignore cleanup errors
        }
        markerRef.current = null;
      }
      if (mapRef.current) {
        try {
          mapRef.current.off();
          mapRef.current.remove();
        } catch (e) {
          // Ignore cleanup errors
        }
        mapRef.current = null;
      }
      setIsMapReady(false);
      initializingRef.current = false;
    };
  }, []);

  // Update marker when position changes
  useEffect(() => {
    if (!isMapReady || !mapRef.current) return;

    try {
      if (position) {
        if (markerRef.current) {
          markerRef.current.setLatLng(position);
        } else {
          markerRef.current = L.marker(position).addTo(mapRef.current);
        }
        mapRef.current.panTo(position);
      } else {
        if (markerRef.current) {
          markerRef.current.remove();
          markerRef.current = null;
        }
      }
    } catch (error) {
      console.error('Error updating marker:', error);
    }
  }, [position, isMapReady]);

  return (
    <div 
      ref={containerRef} 
      className={`h-full w-full ${compact ? 'min-h-0' : 'min-h-[400px]'} rounded-xl overflow-hidden shadow-sm border border-border relative z-0`} 
    />
  );
}