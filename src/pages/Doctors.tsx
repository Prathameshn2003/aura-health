import { useState, useEffect } from "react";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { supabase } from "@/integrations/supabase/client";
import { Input } from "@/components/ui/input";
import { Loader2, Search, MapPin, Phone, Mail, Stethoscope } from "lucide-react";

interface Doctor {
  id: string;
  name: string;
  specialization: string;
  hospital: string | null;
  location: string | null;
  phone: string | null;
  email: string | null;
  description: string | null;
  image_url: string | null;
}

const Doctors = () => {
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [filteredDoctors, setFilteredDoctors] = useState<Doctor[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    const fetchDoctors = async () => {
      const { data, error } = await supabase
        .from('doctors')
        .select('*')
        .eq('is_active', true)
        .order('name');

      if (!error && data) {
        setDoctors(data);
        setFilteredDoctors(data);
      }
      setLoading(false);
    };

    fetchDoctors();
  }, []);

  useEffect(() => {
    if (searchQuery) {
      setFilteredDoctors(
        doctors.filter(d =>
          d.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          d.specialization.toLowerCase().includes(searchQuery.toLowerCase()) ||
          d.location?.toLowerCase().includes(searchQuery.toLowerCase())
        )
      );
    } else {
      setFilteredDoctors(doctors);
    }
  }, [searchQuery, doctors]);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="pt-24 pb-16">
        <div className="container mx-auto px-4">
          <div className="mb-10">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 rounded-lg bg-teal/20 flex items-center justify-center">
                <Stethoscope className="w-5 h-5 text-teal" />
              </div>
              <h1 className="font-heading text-3xl md:text-4xl font-bold text-foreground">
                Find Doctors
              </h1>
            </div>
            <p className="text-muted-foreground">
              Connect with qualified gynecologists and women's health specialists
            </p>
          </div>

          <div className="relative max-w-md mb-8">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search by name, specialization, or location..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-accent" />
            </div>
          ) : filteredDoctors.length === 0 ? (
            <div className="glass-card rounded-xl p-8 text-center">
              <p className="text-muted-foreground">
                {doctors.length === 0 ? "No doctors available at the moment." : "No doctors match your search."}
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredDoctors.map((doctor) => (
                <div key={doctor.id} className="glass-card rounded-xl p-6 hover:shadow-glow transition-all">
                  <div className="flex items-start gap-4 mb-4">
                    <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
                      {doctor.image_url ? (
                        <img 
                          src={doctor.image_url} 
                          alt={doctor.name}
                          className="w-16 h-16 rounded-full object-cover"
                        />
                      ) : (
                        <Stethoscope className="w-8 h-8 text-primary" />
                      )}
                    </div>
                    <div>
                      <h3 className="font-heading font-semibold text-foreground">{doctor.name}</h3>
                      <p className="text-sm text-accent">{doctor.specialization}</p>
                      {doctor.hospital && (
                        <p className="text-sm text-muted-foreground">{doctor.hospital}</p>
                      )}
                    </div>
                  </div>

                  {doctor.description && (
                    <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                      {doctor.description}
                    </p>
                  )}

                  <div className="space-y-2 text-sm">
                    {doctor.location && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <MapPin className="w-4 h-4" />
                        <span>{doctor.location}</span>
                      </div>
                    )}
                    {doctor.phone && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <Phone className="w-4 h-4" />
                        <a href={`tel:${doctor.phone}`} className="hover:text-foreground">{doctor.phone}</a>
                      </div>
                    )}
                    {doctor.email && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <Mail className="w-4 h-4" />
                        <a href={`mailto:${doctor.email}`} className="hover:text-foreground">{doctor.email}</a>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Doctors;
