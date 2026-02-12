import { useState, useEffect } from "react";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { supabase } from "@/integrations/supabase/client";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Loader2, Search, BookOpen, ExternalLink, ChevronRight } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface HealthResource {
  id: string;
  title: string;
  category: string;
  description: string | null;
  external_link: string | null;
}

const HealthResources = () => {
  const [resources, setResources] = useState<HealthResource[]>([]);
  const [filteredResources, setFilteredResources] = useState<HealthResource[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("all");

  useEffect(() => {
    const fetchResources = async () => {
      const { data, error } = await supabase
        .from('health_resources')
        .select('*')
        .eq('status', 'Published')
        .eq('is_active', true)
        .order('created_at', { ascending: false });

      if (!error && data) {
        setResources(data);
        setFilteredResources(data);
      }
      setLoading(false);
    };

    fetchResources();
  }, []);

  useEffect(() => {
    let filtered = resources;
    
    if (searchQuery) {
      filtered = filtered.filter(r =>
        r.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        r.description?.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }
    
    if (categoryFilter !== "all") {
      filtered = filtered.filter(r => r.category === categoryFilter);
    }
    
    setFilteredResources(filtered);
  }, [searchQuery, categoryFilter, resources]);

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Menstrual': return 'bg-primary/15 text-primary';
      case 'PCOS': return 'bg-accent/15 text-accent';
      case 'Menopause': return 'bg-teal/15 text-teal';
      default: return 'bg-secondary text-secondary-foreground';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Menstrual': return '🩸';
      case 'PCOS': return '💜';
      case 'Menopause': return '🌸';
      default: return '📚';
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="pt-20 sm:pt-24 pb-16">
        <div className="container mx-auto px-4">
          {/* Header Section */}
          <div className="mb-8 sm:mb-10 animate-fade-up">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-11 h-11 sm:w-12 sm:h-12 rounded-xl bg-secondary flex items-center justify-center">
                <BookOpen className="w-5 h-5 sm:w-6 sm:h-6 text-foreground" />
              </div>
              <div>
                <h1 className="font-heading text-2xl sm:text-3xl md:text-4xl font-bold text-foreground">
                  Health Resources
                </h1>
                <p className="text-sm sm:text-base text-muted-foreground hidden sm:block">
                  Educational content for women's health
                </p>
              </div>
            </div>
            <p className="text-sm text-muted-foreground sm:hidden">
              Learn about women's health topics
            </p>
          </div>

          {/* Filters */}
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 mb-6 sm:mb-8 animate-fade-up delay-100">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search resources..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 h-11"
              />
            </div>
            <Select value={categoryFilter} onValueChange={setCategoryFilter}>
              <SelectTrigger className="w-full sm:w-48 h-11">
                <SelectValue placeholder="Filter by category" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Categories</SelectItem>
                <SelectItem value="Menstrual">Menstrual</SelectItem>
                <SelectItem value="PCOS">PCOS</SelectItem>
                <SelectItem value="Menopause">Menopause</SelectItem>
                <SelectItem value="General Wellness">General Wellness</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Content */}
          {loading ? (
            <div className="flex flex-col items-center justify-center py-16">
              <Loader2 className="w-8 h-8 animate-spin text-primary mb-3" />
              <p className="text-sm text-muted-foreground">Loading resources...</p>
            </div>
          ) : filteredResources.length === 0 ? (
            <div className="glass-card rounded-xl sm:rounded-2xl p-8 sm:p-12 text-center animate-fade-up">
              <BookOpen className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
              <p className="text-muted-foreground">
                {resources.length === 0 ? "No health resources available at the moment." : "No resources match your filters."}
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
              {filteredResources.map((resource, index) => (
                <div 
                  key={resource.id} 
                  className="glass-card card-hover rounded-xl sm:rounded-2xl p-5 sm:p-6 flex flex-col animate-fade-up"
                  style={{ animationDelay: `${index * 50}ms` }}
                >
                  <div className="flex items-start justify-between gap-3 mb-3">
                    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-2xs sm:text-xs rounded-full ${getCategoryColor(resource.category)}`}>
                      <span>{getCategoryIcon(resource.category)}</span>
                      {resource.category}
                    </span>
                  </div>
                  
                  <h3 className="font-heading text-base sm:text-lg font-semibold text-foreground mb-2 line-clamp-2">
                    {resource.title}
                  </h3>
                  
                  {resource.description && (
                    <p className="text-sm text-muted-foreground mb-4 flex-1 line-clamp-3">
                      {resource.description}
                    </p>
                  )}

                  {resource.external_link && (
                    <Button variant="outline" size="sm" className="gap-2 w-full mt-auto h-10" asChild>
                      <a href={resource.external_link} target="_blank" rel="noopener noreferrer">
                        <ExternalLink className="w-4 h-4" />
                        Learn More
                        <ChevronRight className="w-4 h-4 ml-auto" />
                      </a>
                    </Button>
                  )}
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

export default HealthResources;