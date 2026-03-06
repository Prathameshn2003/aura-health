import { useState, useEffect, useMemo } from "react";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { supabase } from "@/integrations/supabase/client";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue
} from "@/components/ui/select";
import {
  Loader2, Search, MapPin, Phone, Navigation, Stethoscope,
  Star, ShieldCheck, Sparkles, ExternalLink, MapPinOff, ChevronDown
} from "lucide-react";
import { toast } from "@/hooks/use-toast";

interface Doctor {
  id: string;
  name: string;
  specialization: string;
  hospital: string | null;
  location: string | null;
  city: string | null;
  latitude: number | null;
  longitude: number | null;
  phone: string | null;
  email: string | null;
  description: string | null;
  image_url: string | null;
  experience: number | null;
  verified: boolean | null;
  recommended: boolean | null;
  distance?: number;
}

const SPECIALIZATIONS = [
  "All",
  "Gynecologist",
  "Obstetrician",
  "PCOS Specialist",
  "Fertility Specialist",
  "Endocrinologist",
  "General Physician",
  "Dermatologist",
  "Nutritionist",
];

const calculateDistance = (lat1: number, lon1: number, lat2: number, lon2: number): number => {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) *
    Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
};

const FindSpecialist = () => {
  const [allDoctors, setAllDoctors] = useState<Doctor[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [citySearch, setCitySearch] = useState("");
  const [specialization, setSpecialization] = useState("All");
  const [userLocation, setUserLocation] = useState<{ lat: number; lng: number } | null>(null);
  const [locationDenied, setLocationDenied] = useState(false);
  const [visibleCount, setVisibleCount] = useState(10);
  const [searchRadius, setSearchRadius] = useState<number | null>(null);

  // Get user location
  useEffect(() => {
    if (!navigator.geolocation) {
      setLocationDenied(true);
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => setUserLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
      () => setLocationDenied(true),
      { timeout: 10000 }
    );
  }, []);

  // Fetch doctors
  useEffect(() => {
    const fetch = async () => {
      const { data, error } = await supabase
        .from("doctors")
        .select("*")
        .eq("is_active", true)
        .order("name");
      if (!error && data) setAllDoctors(data as Doctor[]);
      setLoading(false);
    };
    fetch();
  }, []);

  // Process doctors with distance, filtering, and 3-level fallback
  const { displayDoctors, resultMessage } = useMemo(() => {
    let docs = allDoctors.map((d) => ({
      ...d,
      distance: userLocation && d.latitude && d.longitude
        ? calculateDistance(userLocation.lat, userLocation.lng, d.latitude, d.longitude)
        : undefined,
    }));

    // Apply text search
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      docs = docs.filter(
        (d) =>
          d.name.toLowerCase().includes(q) ||
          d.specialization.toLowerCase().includes(q) ||
          d.hospital?.toLowerCase().includes(q) ||
          d.location?.toLowerCase().includes(q)
      );
    }

    // Apply city search
    if (citySearch) {
      const c = citySearch.toLowerCase();
      docs = docs.filter(
        (d) =>
          d.city?.toLowerCase().includes(c) ||
          d.location?.toLowerCase().includes(c)
      );
    }

    // Apply specialization filter
    if (specialization !== "All") {
      docs = docs.filter((d) => d.specialization === specialization);
    }

    // 3-level fallback for distance-based search
    let message = "";
    if (userLocation && !citySearch) {
      const within50 = docs.filter((d) => d.distance !== undefined && d.distance <= 50);
      if (within50.length > 0) {
        docs = within50;
        message = `${within50.length} doctors found within 50 km`;
      } else {
        const within200 = docs.filter((d) => d.distance !== undefined && d.distance <= 200);
        if (within200.length > 0) {
          docs = within200;
          message = "No doctors found within 50 km. Showing doctors in nearby cities.";
        } else {
          // Show recommended doctors as fallback
          const recommended = docs.filter((d) => d.recommended);
          if (recommended.length > 0) {
            docs = recommended;
            message = "No nearby doctors found. Showing recommended specialists.";
          } else {
            message = docs.length > 0 ? "Showing all available doctors" : "";
          }
        }
      }
    }

    // Sort: distance first (if available), then verified, then experience
    docs.sort((a, b) => {
      if (a.distance !== undefined && b.distance !== undefined) return a.distance - b.distance;
      if (a.distance !== undefined) return -1;
      if (b.distance !== undefined) return 1;
      if (a.verified && !b.verified) return -1;
      if (!a.verified && b.verified) return 1;
      return (b.experience || 0) - (a.experience || 0);
    });

    return { displayDoctors: docs, resultMessage: message };
  }, [allDoctors, userLocation, searchQuery, citySearch, specialization]);

  const visibleDoctors = displayDoctors.slice(0, visibleCount);

  const openGoogleMaps = () => {
    window.open("https://www.google.com/maps/search/gynecologist+near+me", "_blank");
  };

  const openDirections = (doc: Doctor) => {
    if (doc.latitude && doc.longitude) {
      window.open(`https://www.google.com/maps/dir/?api=1&destination=${doc.latitude},${doc.longitude}`, "_blank");
    } else if (doc.location) {
      window.open(`https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(doc.location)}`, "_blank");
    }
  };

  const openOnMap = (doc: Doctor) => {
    if (doc.latitude && doc.longitude) {
      window.open(`https://www.google.com/maps/search/?api=1&query=${doc.latitude},${doc.longitude}`, "_blank");
    } else if (doc.location) {
      window.open(`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(doc.name + " " + doc.location)}`, "_blank");
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="pt-24 pb-16">
        <div className="container mx-auto px-4 max-w-6xl">
          {/* Hero */}
          <div className="glass-card rounded-2xl p-6 md:p-8 mb-8 bg-gradient-to-br from-teal/10 via-primary/5 to-accent/10">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-12 h-12 rounded-xl bg-teal/20 flex items-center justify-center">
                <Stethoscope className="w-6 h-6 text-teal" />
              </div>
              <div>
                <h1 className="font-heading text-2xl md:text-3xl font-bold text-foreground">
                  Find a Specialist
                </h1>
                <p className="text-muted-foreground text-sm">
                  Discover trusted women's health specialists near you
                </p>
              </div>
            </div>

            {locationDenied && (
              <div className="mt-4 p-4 rounded-xl bg-accent/10 border border-accent/20 flex items-start gap-3">
                <MapPinOff className="w-5 h-5 text-accent mt-0.5 shrink-0" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-foreground">Location permission denied</p>
                  <p className="text-xs text-muted-foreground mt-1">Search doctors by city instead, or open Google Maps.</p>
                </div>
                <Button size="sm" variant="outline" onClick={openGoogleMaps} className="shrink-0 gap-1.5">
                  <ExternalLink className="w-3.5 h-3.5" /> Google Maps
                </Button>
              </div>
            )}
          </div>

          {/* Filters */}
          <div className="flex flex-col sm:flex-row gap-3 mb-6">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search doctor or hospital..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            {locationDenied && (
              <div className="relative flex-1 sm:max-w-[200px]">
                <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search by city..."
                  value={citySearch}
                  onChange={(e) => setCitySearch(e.target.value)}
                  className="pl-10"
                />
              </div>
            )}
            <Select value={specialization} onValueChange={setSpecialization}>
              <SelectTrigger className="sm:w-[200px]">
                <SelectValue placeholder="Specialization" />
              </SelectTrigger>
              <SelectContent>
                {SPECIALIZATIONS.map((s) => (
                  <SelectItem key={s} value={s}>{s}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Result message */}
          {resultMessage && !loading && (
            <p className="text-sm text-muted-foreground mb-4 flex items-center gap-2">
              <MapPin className="w-4 h-4 text-teal" />
              {resultMessage}
            </p>
          )}

          {/* Doctor cards */}
          {loading ? (
            <div className="flex flex-col items-center justify-center py-16 gap-3">
              <Loader2 className="w-8 h-8 animate-spin text-primary" />
              <p className="text-muted-foreground text-sm">Finding doctors near you...</p>
            </div>
          ) : displayDoctors.length === 0 ? (
            <div className="glass-card rounded-2xl p-10 text-center space-y-4">
              <Stethoscope className="w-12 h-12 text-muted-foreground mx-auto" />
              <p className="text-muted-foreground">No doctors found matching your criteria.</p>
              <Button onClick={openGoogleMaps} className="gap-2">
                <ExternalLink className="w-4 h-4" /> Find on Google Maps
              </Button>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {visibleDoctors.map((doc) => (
                  <DoctorCard
                    key={doc.id}
                    doctor={doc}
                    onDirections={() => openDirections(doc)}
                    onViewMap={() => openOnMap(doc)}
                  />
                ))}
              </div>

              {visibleCount < displayDoctors.length && (
                <div className="flex justify-center mt-8">
                  <Button
                    variant="outline"
                    onClick={() => setVisibleCount((c) => c + 10)}
                    className="gap-2"
                  >
                    <ChevronDown className="w-4 h-4" />
                    Load More ({displayDoctors.length - visibleCount} remaining)
                  </Button>
                </div>
              )}
            </>
          )}
        </div>
      </main>
      <Footer />
    </div>
  );
};

// Doctor Card Component
const DoctorCard = ({
  doctor,
  onDirections,
  onViewMap,
}: {
  doctor: Doctor;
  onDirections: () => void;
  onViewMap: () => void;
}) => (
  <div className="glass-card rounded-xl p-5 hover:shadow-glow transition-all group">
    <div className="flex items-start gap-4">
      {/* Photo */}
      <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center shrink-0 overflow-hidden">
        {doctor.image_url ? (
          <img src={doctor.image_url} alt={doctor.name} className="w-16 h-16 rounded-full object-cover" />
        ) : (
          <Stethoscope className="w-7 h-7 text-primary" />
        )}
      </div>

      <div className="flex-1 min-w-0">
        {/* Name & badges */}
        <div className="flex items-start justify-between gap-2">
          <div>
            <h3 className="font-heading font-semibold text-foreground truncate">{doctor.name}</h3>
            <p className="text-sm text-accent">
              {doctor.specialization}
              {doctor.experience ? ` · ${doctor.experience} yrs exp` : ""}
            </p>
          </div>
          <div className="flex gap-1.5 shrink-0">
            {doctor.verified && (
              <Badge variant="secondary" className="gap-1 text-xs bg-teal/15 text-teal border-teal/30">
                <ShieldCheck className="w-3 h-3" /> Verified
              </Badge>
            )}
          </div>
        </div>

        {/* Hospital */}
        {doctor.hospital && (
          <p className="text-sm text-muted-foreground mt-1">{doctor.hospital}</p>
        )}

        {/* Location & distance */}
        <div className="flex flex-wrap items-center gap-3 mt-2 text-xs text-muted-foreground">
          {doctor.location && (
            <span className="flex items-center gap-1">
              <MapPin className="w-3 h-3" /> {doctor.location}
            </span>
          )}
          {doctor.distance !== undefined && (
            <span className="font-medium text-teal">{doctor.distance.toFixed(1)} km away</span>
          )}
        </div>

        {/* AI Recommended badge */}
        {doctor.recommended && (
          <div className="mt-2">
            <Badge className="gap-1 text-xs bg-accent/15 text-accent border-accent/30">
              <Sparkles className="w-3 h-3" /> AI Recommended
            </Badge>
          </div>
        )}

        {/* Action buttons */}
        <div className="flex flex-wrap gap-2 mt-3">
          {doctor.phone && (
            <Button size="sm" variant="outline" asChild className="gap-1.5 text-xs">
              <a href={`tel:${doctor.phone}`}>
                <Phone className="w-3.5 h-3.5" /> Call
              </a>
            </Button>
          )}
          <Button size="sm" variant="outline" onClick={onDirections} className="gap-1.5 text-xs">
            <Navigation className="w-3.5 h-3.5" /> Directions
          </Button>
          <Button size="sm" variant="outline" onClick={onViewMap} className="gap-1.5 text-xs">
            <MapPin className="w-3.5 h-3.5" /> View on Map
          </Button>
        </div>
      </div>
    </div>
  </div>
);

export default FindSpecialist;
