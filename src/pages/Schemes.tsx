import { useState, useEffect } from "react";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { supabase } from "@/integrations/supabase/client";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Loader2, Search, FileText, ExternalLink, CheckCircle, ChevronDown, ChevronUp } from "lucide-react";

interface Scheme {
  id: string;
  name: string;
  description: string | null;
  eligibility: string | null;
  benefits: string | null;
  how_to_apply: string | null;
  website: string | null;
  category: string | null;
}

const Schemes = () => {
  const [schemes, setSchemes] = useState<Scheme[]>([]);
  const [filteredSchemes, setFilteredSchemes] = useState<Scheme[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    const fetchSchemes = async () => {
      const { data, error } = await supabase
        .from('schemes')
        .select('*')
        .eq('is_active', true)
        .order('name');

      if (!error && data) {
        setSchemes(data);
        setFilteredSchemes(data);
      }
      setLoading(false);
    };

    fetchSchemes();
  }, []);

  useEffect(() => {
    if (searchQuery) {
      setFilteredSchemes(
        schemes.filter(s =>
          s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          s.category?.toLowerCase().includes(searchQuery.toLowerCase()) ||
          s.description?.toLowerCase().includes(searchQuery.toLowerCase())
        )
      );
    } else {
      setFilteredSchemes(schemes);
    }
  }, [searchQuery, schemes]);

  const getCategoryColor = (category: string | null) => {
    switch (category?.toLowerCase()) {
      case 'health': return 'bg-teal/15 text-teal';
      case 'financial': return 'bg-accent/15 text-accent';
      case 'education': return 'bg-primary/15 text-primary';
      default: return 'bg-secondary text-secondary-foreground';
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
              <div className="w-11 h-11 sm:w-12 sm:h-12 rounded-xl bg-accent/15 flex items-center justify-center">
                <FileText className="w-5 h-5 sm:w-6 sm:h-6 text-accent" />
              </div>
              <div>
                <h1 className="font-heading text-2xl sm:text-3xl md:text-4xl font-bold text-foreground">
                  Govt. Schemes
                </h1>
                <p className="text-sm sm:text-base text-muted-foreground hidden sm:block">
                  Health schemes and benefits for women
                </p>
              </div>
            </div>
            <p className="text-sm text-muted-foreground sm:hidden">
              Explore government health schemes
            </p>
          </div>

          {/* Search */}
          <div className="relative max-w-md mb-6 sm:mb-8 animate-fade-up delay-100">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search schemes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 h-11"
            />
          </div>

          {/* Content */}
          {loading ? (
            <div className="flex flex-col items-center justify-center py-16">
              <Loader2 className="w-8 h-8 animate-spin text-accent mb-3" />
              <p className="text-sm text-muted-foreground">Loading schemes...</p>
            </div>
          ) : filteredSchemes.length === 0 ? (
            <div className="glass-card rounded-xl sm:rounded-2xl p-8 sm:p-12 text-center animate-fade-up">
              <FileText className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
              <p className="text-muted-foreground">
                {schemes.length === 0 ? "No schemes available at the moment." : "No schemes match your search."}
              </p>
            </div>
          ) : (
            <div className="space-y-3 sm:space-y-4">
              {filteredSchemes.map((scheme, index) => (
                <div 
                  key={scheme.id} 
                  className="glass-card rounded-xl sm:rounded-2xl overflow-hidden animate-fade-up"
                  style={{ animationDelay: `${index * 50}ms` }}
                >
                  <button
                    className="w-full p-4 sm:p-6 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset"
                    onClick={() => setExpandedId(expandedId === scheme.id ? null : scheme.id)}
                  >
                    <div className="flex items-start justify-between gap-3 sm:gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex flex-wrap items-center gap-2 mb-2">
                          {scheme.category && (
                            <span className={`px-2.5 py-1 text-2xs sm:text-xs rounded-full ${getCategoryColor(scheme.category)}`}>
                              {scheme.category}
                            </span>
                          )}
                        </div>
                        <h3 className="font-heading text-base sm:text-lg font-semibold text-foreground mb-1.5">
                          {scheme.name}
                        </h3>
                        {scheme.description && (
                          <p className="text-sm text-muted-foreground line-clamp-2">
                            {scheme.description}
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <div className="hidden sm:flex w-10 h-10 rounded-lg bg-accent/10 items-center justify-center">
                          <FileText className="w-5 h-5 text-accent" />
                        </div>
                        {expandedId === scheme.id ? (
                          <ChevronUp className="w-5 h-5 text-muted-foreground" />
                        ) : (
                          <ChevronDown className="w-5 h-5 text-muted-foreground" />
                        )}
                      </div>
                    </div>
                  </button>

                  {expandedId === scheme.id && (
                    <div className="px-4 sm:px-6 pb-4 sm:pb-6 pt-2 border-t border-border space-y-4 animate-fade-in">
                      {scheme.eligibility && (
                        <div className="bg-muted/50 rounded-lg p-4">
                          <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2 text-sm sm:text-base">
                            <CheckCircle className="w-4 h-4 text-teal flex-shrink-0" />
                            Eligibility
                          </h4>
                          <p className="text-sm text-muted-foreground whitespace-pre-line pl-6">
                            {scheme.eligibility}
                          </p>
                        </div>
                      )}

                      {scheme.benefits && (
                        <div className="bg-muted/50 rounded-lg p-4">
                          <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2 text-sm sm:text-base">
                            <CheckCircle className="w-4 h-4 text-teal flex-shrink-0" />
                            Benefits
                          </h4>
                          <p className="text-sm text-muted-foreground whitespace-pre-line pl-6">
                            {scheme.benefits}
                          </p>
                        </div>
                      )}

                      {scheme.how_to_apply && (
                        <div className="bg-muted/50 rounded-lg p-4">
                          <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2 text-sm sm:text-base">
                            <CheckCircle className="w-4 h-4 text-teal flex-shrink-0" />
                            How to Apply
                          </h4>
                          <p className="text-sm text-muted-foreground whitespace-pre-line pl-6">
                            {scheme.how_to_apply}
                          </p>
                        </div>
                      )}

                      {scheme.website && (
                        <Button variant="default" size="sm" className="gap-2 mt-2" asChild>
                          <a href={scheme.website} target="_blank" rel="noopener noreferrer">
                            <ExternalLink className="w-4 h-4" />
                            Official Website
                          </a>
                        </Button>
                      )}
                    </div>
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

export default Schemes;