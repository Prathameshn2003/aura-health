import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { MapPin, Star, ExternalLink, Loader2, AlertCircle } from "lucide-react";

interface Doctor {
  name: string;
  specialty: string;
  address: string;
  rating?: number;
  distance?: string;
  placeId: string;
}

interface NearbyDoctorsProps {
  specialty?: string;
}

export const NearbyDoctors = ({ specialty = "gynecologist" }: NearbyDoctorsProps) => {
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [userLocation, setUserLocation] = useState<{ lat: number; lng: number } | null>(null);

  useEffect(() => {
    // Get user's location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          });
        },
        (err) => {
          console.error("Location error:", err);
          // Default to a major city if location denied
          setUserLocation({ lat: 28.6139, lng: 77.209 }); // Delhi
          setError("Location access denied. Showing results for Delhi.");
        }
      );
    } else {
      setUserLocation({ lat: 28.6139, lng: 77.209 });
      setError("Geolocation not supported. Showing results for Delhi.");
    }
  }, []);

  useEffect(() => {
    if (!userLocation) return;

    const searchDoctors = async () => {
      setLoading(true);
      try {
        // Simulated doctors data based on specialty
        // In production, this would call Google Places API via edge function
        const mockDoctors: Doctor[] = [
          {
            name: "Dr. Priya Sharma",
            specialty: "Gynecologist & Obstetrician",
            address: "Apollo Hospital, Sarita Vihar",
            rating: 4.8,
            distance: "2.5 km",
            placeId: "place1",
          },
          {
            name: "Dr. Anjali Gupta",
            specialty: "Endocrinologist",
            address: "Max Healthcare, Saket",
            rating: 4.6,
            distance: "4.2 km",
            placeId: "place2",
          },
          {
            name: "Dr. Meera Kapoor",
            specialty: "Women's Health Specialist",
            address: "Fortis Hospital, Vasant Kunj",
            rating: 4.9,
            distance: "5.8 km",
            placeId: "place3",
          },
          {
            name: "Dr. Sunita Reddy",
            specialty: "Gynecologist",
            address: "AIIMS, New Delhi",
            rating: 4.7,
            distance: "7.1 km",
            placeId: "place4",
          },
        ];

        // Simulate API delay
        await new Promise((resolve) => setTimeout(resolve, 800));
        setDoctors(mockDoctors);
        setLoading(false);
      } catch (err) {
        setError("Failed to load nearby doctors");
        setLoading(false);
      }
    };

    searchDoctors();
  }, [userLocation, specialty]);

  const openInMaps = (doctor: Doctor) => {
    const query = encodeURIComponent(`${doctor.name} ${doctor.address}`);
    window.open(`https://www.google.com/maps/search/?api=1&query=${query}`, "_blank");
  };

  if (loading) {
    return (
      <div className="glass-card rounded-2xl p-8 text-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-4" />
        <p className="text-muted-foreground">Finding nearby specialists...</p>
      </div>
    );
  }

  return (
    <div className="glass-card rounded-2xl p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-lg bg-teal/20 flex items-center justify-center">
          <MapPin className="w-5 h-5 text-teal" />
        </div>
        <div>
          <h3 className="font-heading text-lg font-semibold text-foreground">
            Nearby Women's Health Specialists
          </h3>
          <p className="text-sm text-muted-foreground">
            Consult with qualified professionals
          </p>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-sm text-accent mb-4 p-3 bg-accent/10 rounded-lg">
          <AlertCircle className="w-4 h-4" />
          {error}
        </div>
      )}

      <div className="space-y-4">
        {doctors.map((doctor) => (
          <div
            key={doctor.placeId}
            className="p-4 rounded-xl bg-muted/50 hover:bg-muted transition-colors"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h4 className="font-semibold text-foreground">{doctor.name}</h4>
                <p className="text-sm text-primary">{doctor.specialty}</p>
                <p className="text-sm text-muted-foreground mt-1 flex items-center gap-1">
                  <MapPin className="w-3 h-3" />
                  {doctor.address}
                </p>
                <div className="flex items-center gap-4 mt-2">
                  {doctor.rating && (
                    <span className="flex items-center gap-1 text-sm text-accent">
                      <Star className="w-3 h-3 fill-current" />
                      {doctor.rating}
                    </span>
                  )}
                  {doctor.distance && (
                    <span className="text-sm text-muted-foreground">
                      {doctor.distance}
                    </span>
                  )}
                </div>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => openInMaps(doctor)}
                className="shrink-0"
              >
                <ExternalLink className="w-4 h-4 mr-1" />
                Open Map
              </Button>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 pt-4 border-t border-border">
        <Button
          variant="ghost"
          className="w-full"
          onClick={() =>
            window.open(
              `https://www.google.com/maps/search/${specialty}+near+me`,
              "_blank"
            )
          }
        >
          View More on Google Maps
          <ExternalLink className="w-4 h-4 ml-2" />
        </Button>
      </div>
    </div>
  );
};
