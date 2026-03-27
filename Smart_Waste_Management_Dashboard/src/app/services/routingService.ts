export type LatLng = [number, number];

const OSRM_URL = import.meta.env.VITE_OSRM_URL || 'https://router.project-osrm.org';

// OSRM expects lon,lat pairs in the URL.
function toOsrmCoord([lat, lng]: LatLng): string {
  return `${lng},${lat}`;
}

export async function getRoadRoute(points: LatLng[]): Promise<{ geometry: LatLng[]; distanceMeters: number; durationSeconds: number }> {
  if (!points || points.length < 2) {
    return { geometry: points || [], distanceMeters: 0, durationSeconds: 0 };
  }

  const coords = points.map(toOsrmCoord).join(';');
  const url = `${OSRM_URL}/route/v1/driving/${coords}?overview=full&geometries=geojson&steps=false`;

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Routing failed (${res.status})`);
  const data = await res.json();
  const route = data?.routes?.[0];
  const coordsOut: [number, number][] = route?.geometry?.coordinates || [];

  // GeoJSON coords are [lon,lat] → convert back to [lat,lng]
  const geometry: LatLng[] = coordsOut.map(([lon, lat]) => [lat, lon]);
  return {
    geometry,
    distanceMeters: Number(route?.distance || 0),
    durationSeconds: Number(route?.duration || 0),
  };
}

