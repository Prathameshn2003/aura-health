import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { MapPin, Star, ExternalLink, Loader2, AlertCircle, RefreshCw, Navigation } from "lucide-react";

interface NearbyPlace {
  id: number;
  name: string;
  type: string;
  lat: number;
  lon: number;
  address?: string;
  distance?: number;
}

interface NearbyDoctorsProps {
  specialty?: string;
}

const OVERPASS_URL = "https://overpass-api.de/api/interpreter";

const calculateDistance = (lat1: number, lon1: number, lat2: number, lon2: number): number => {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
};

export const NearbyDoctors = ({ specialty = "gynecologist" }: NearbyDoctorsProps) => {
  const [places, setPlaces] = useState<NearbyPlace[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [userLocation, setUserLocation] = useState<{ lat: number; lng: number } | null>(null);
  const [locationDenied, setLocationDenied] = useState(false);
  const mapRef = useRef<HTMLDivElement>(null);
  const leafletMapRef = useRef<any>(null);

  // Get user location
  useEffect(() => {
    if (!navigator.geolocation) {
      setLocationDenied(true);
      setLoading(false);
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => setUserLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
      () => {
        setLocationDenied(true);
        setLoading(false);
      },
      { timeout: 10000 }
    );
  }, []);

  // Fetch nearby places from Overpass API
  const fetchNearbyPlaces = async (lat: number, lng: number) => {
    setLoading(true);
    setError(null);
    try {
      const radius = 5000; // 5km
      const query = `
        [out:json][timeout:25];
        (
          node["amenity"="hospital"](around:${radius},${lat},${lng});
          node["amenity"="clinic"](around:${radius},${lat},${lng});
          node["amenity"="doctors"](around:${radius},${lat},${lng});
          node["healthcare"="doctor"](around:${radius},${lat},${lng});
          node["healthcare"="centre"](around:${radius},${lat},${lng});
        );
        out body;
      `;
      const res = await fetch(OVERPASS_URL, {
        method: "POST",
        body: `data=${encodeURIComponent(query)}`,
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      });
      if (!res.ok) throw new Error("Overpass API request failed");
      const data = await res.json();

      const results: NearbyPlace[] = (data.elements || [])
        .filter((el: any) => el.tags?.name)
        .map((el: any) => ({
          id: el.id,
          name: el.tags.name,
          type: el.tags.amenity || el.tags.healthcare || "clinic",
          lat: el.lat,
          lon: el.lon,
          address: [el.tags["addr:street"], el.tags["addr:city"]].filter(Boolean).join(", ") || undefined,
          distance: calculateDistance(lat, lng, el.lat, el.lon),
        }))
        .sort((a: NearbyPlace, b: NearbyPlace) => (a.distance || 0) - (b.distance || 0))
        .slice(0, 20);

      setPlaces(results);
    } catch {
      setError("Unable to fetch nearby doctors right now.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (userLocation) fetchNearbyPlaces(userLocation.lat, userLocation.lng);
  }, [userLocation]);

  // Initialize Leaflet map
  useEffect(() => {
    if (!userLocation || !mapRef.current) return;

    const initMap = async () => {
      const L = (await import("leaflet")).default;
      await import("leaflet/dist/leaflet.css");

      // Fix default icon issue
      delete (L.Icon.Default.prototype as any)._getIconUrl;
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
        iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
        shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
      });

      if (leafletMapRef.current) leafletMapRef.current.remove();

      const map = L.map(mapRef.current!, { zoomControl: true }).setView([userLocation.lat, userLocation.lng], 14);
      leafletMapRef.current = map;

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '© OpenStreetMap contributors',
      }).addTo(map);

      // User marker
      const userIcon = L.divIcon({
        html: '<div style="background:#e11d48;width:16px;height:16px;border-radius:50%;border:3px solid white;box-shadow:0 0 6px rgba(0,0,0,0.3)"></div>',
        iconSize: [16, 16],
        className: "",
      });
      L.marker([userLocation.lat, userLocation.lng], { icon: userIcon })
        .addTo(map)
        .bindPopup("<strong>📍 Your Location</strong>");

      // Place markers
      places.forEach((place) => {
        const marker = L.marker([place.lat, place.lon]).addTo(map);
        marker.bindPopup(`
          <div style="min-width:180px">
            <strong>${place.name}</strong><br/>
            <span style="text-transform:capitalize;color:#666">${place.type}</span><br/>
            ${place.address ? `<span style="font-size:12px">${place.address}</span><br/>` : ""}
            ${place.distance ? `<span style="font-size:12px">${place.distance.toFixed(1)} km away</span><br/>` : ""}
            <a href="https://www.google.com/maps/search/?api=1&query=${place.lat},${place.lon}" target="_blank" style="color:#e11d48;font-size:13px;text-decoration:underline">Open in Google Maps</a>
          </div>
        `);
      });

      setTimeout(() => map.invalidateSize(), 100);
    };

    initMap();

    return () => {
      if (leafletMapRef.current) {
        leafletMapRef.current.remove();
        leafletMapRef.current = null;
      }
    };
  }, [userLocation, places]);

  const openGoogleMaps = () => {
    window.open("https://www.google.com/maps/search/hospital+near+me", "_blank");
  };

  const openPlaceInMaps = (place: NearbyPlace) => {
    window.open(`https://www.google.com/maps/search/?api=1&query=${place.lat},${place.lon}`, "_blank");
  };

  if (locationDenied) {
    return (
      <div className="glass-card rounded-2xl p-8 text-center space-y-4">
        <AlertCircle className="w-10 h-10 text-accent mx-auto" />
        <h3 className="font-heading text-lg font-semibold text-foreground">Location Access Required</h3>
        <p className="text-muted-foreground text-sm">
          Please enable location access in your browser to find nearby doctors.
        </p>
        <Button onClick={openGoogleMaps} className="gap-2">
          <ExternalLink className="w-4 h-4" />
          Open Nearby Doctors in Google Maps
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Map */}
      <div className="glass-card rounded-2xl overflow-hidden">
        <div ref={mapRef} className="w-full h-[350px] md:h-[450px] bg-muted" />
      </div>

      {error && (
        <div className="flex items-center justify-between gap-3 p-4 rounded-xl bg-accent/10 border border-accent/20">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-accent flex-shrink-0" />
            <p className="text-sm text-accent">{error}</p>
          </div>
          <div className="flex gap-2">
            <Button size="sm" variant="outline" onClick={() => userLocation && fetchNearbyPlaces(userLocation.lat, userLocation.lng)}>
              <RefreshCw className="w-3.5 h-3.5 mr-1" /> Retry
            </Button>
            <Button size="sm" variant="outline" onClick={openGoogleMaps}>
              <ExternalLink className="w-3.5 h-3.5 mr-1" /> Google Maps
            </Button>
          </div>
        </div>
      )}

      {/* Place cards */}
      <div className="glass-card rounded-2xl p-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-lg bg-teal/20 flex items-center justify-center">
            <MapPin className="w-5 h-5 text-teal" />
          </div>
          <div>
            <h3 className="font-heading text-lg font-semibold text-foreground">
              Nearby Hospitals & Clinics
            </h3>
            <p className="text-sm text-muted-foreground">
              {loading ? "Searching..." : `Found ${places.length} places within 5 km`}
            </p>
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
          </div>
        ) : places.length === 0 && !error ? (
          <div className="text-center py-8">
            <p className="text-muted-foreground mb-4">No nearby places found.</p>
            <Button variant="outline" onClick={openGoogleMaps}>
              <ExternalLink className="w-4 h-4 mr-2" /> Search on Google Maps
            </Button>
          </div>
        ) : (
          <div className="space-y-3">
            {places.map((place) => (
              <div
                key={place.id}
                className="p-4 rounded-xl bg-muted/50 hover:bg-muted transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-semibold text-foreground">{place.name}</h4>
                    <p className="text-sm text-primary capitalize">{place.type}</p>
                    {place.address && (
                      <p className="text-sm text-muted-foreground mt-1 flex items-center gap-1">
                        <MapPin className="w-3 h-3" />
                        {place.address}
                      </p>
                    )}
                    {place.distance !== undefined && (
                      <span className="text-sm text-muted-foreground mt-1 inline-block">
                        {place.distance.toFixed(1)} km away
                      </span>
                    )}
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => openPlaceInMaps(place)}
                    className="shrink-0 gap-1.5"
                  >
                    <Navigation className="w-3.5 h-3.5" />
                    Google Maps
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="mt-6 pt-4 border-t border-border">
          <Button variant="ghost" className="w-full" onClick={openGoogleMaps}>
            View More on Google Maps
            <ExternalLink className="w-4 h-4 ml-2" />
          </Button>
        </div>
      </div>
    </div>
  );
};
