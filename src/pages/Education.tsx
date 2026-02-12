import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Link } from "react-router-dom";
import { BookOpen, ChevronRight, Droplets, Activity, Thermometer, Heart } from "lucide-react";

const educationTopics = [
  {
    title: "Menstrual Health",
    description: "Understanding your menstrual cycle, tracking periods, and managing symptoms effectively.",
    icon: Droplets,
    color: "text-primary",
    bgColor: "bg-primary/20",
    path: "/modules/menstrual",
    topics: ["Cycle phases", "Period tracking", "PMS management", "Irregular periods"],
  },
  {
    title: "PCOS Guide",
    description: "Comprehensive information about Polycystic Ovary Syndrome, symptoms, and management.",
    icon: Activity,
    color: "text-accent",
    bgColor: "bg-accent/20",
    path: "/modules/pcos",
    topics: ["Symptoms", "Diagnosis", "Lifestyle changes", "Treatment options"],
  },
  {
    title: "Menopause Information",
    description: "Navigate perimenopause and menopause with knowledge about stages, symptoms, and wellness.",
    icon: Thermometer,
    color: "text-teal",
    bgColor: "bg-teal/20",
    path: "/modules/menopause",
    topics: ["Perimenopause", "Menopause stages", "Symptom relief", "Bone health"],
  },
  {
    title: "General Wellness",
    description: "Holistic health tips for nutrition, exercise, mental health, and overall wellbeing.",
    icon: Heart,
    color: "text-destructive",
    bgColor: "bg-destructive/20",
    path: "/health-resources",
    topics: ["Nutrition", "Exercise", "Mental health", "Sleep hygiene"],
  },
];

const Education = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="pt-24 pb-16">
        <div className="container mx-auto px-4">
          <div className="mb-10">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 rounded-lg bg-secondary flex items-center justify-center">
                <BookOpen className="w-5 h-5 text-secondary-foreground" />
              </div>
              <h1 className="font-heading text-3xl md:text-4xl font-bold text-foreground">
                Health Education
              </h1>
            </div>
            <p className="text-muted-foreground max-w-2xl">
              Explore comprehensive educational resources about women's health topics. 
              Knowledge is the first step towards better health.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
            {educationTopics.map((topic) => (
              <Link
                key={topic.title}
                to={topic.path}
                className="glass-card rounded-xl p-6 hover:shadow-glow transition-all group"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className={`w-12 h-12 rounded-xl ${topic.bgColor} flex items-center justify-center`}>
                    <topic.icon className={`w-6 h-6 ${topic.color}`} />
                  </div>
                  <ChevronRight className="w-5 h-5 text-muted-foreground group-hover:text-accent group-hover:translate-x-1 transition-all" />
                </div>
                
                <h2 className="font-heading text-xl font-semibold text-foreground mb-2">
                  {topic.title}
                </h2>
                <p className="text-sm text-muted-foreground mb-4">
                  {topic.description}
                </p>
                
                <div className="flex flex-wrap gap-2">
                  {topic.topics.map((t) => (
                    <span 
                      key={t}
                      className="px-2 py-1 text-xs rounded-full bg-muted text-muted-foreground"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </Link>
            ))}
          </div>

          {/* Quick Resources */}
          <div className="glass-card rounded-xl p-6 md:p-8">
            <h2 className="font-heading text-xl font-semibold text-foreground mb-4">
              Quick Resources
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <Link 
                to="/hygiene" 
                className="p-4 rounded-lg bg-muted hover:bg-muted/80 transition-colors text-center"
              >
                <span className="text-sm font-medium text-foreground">Hygiene Tips</span>
              </Link>
              <Link 
                to="/health-resources" 
                className="p-4 rounded-lg bg-muted hover:bg-muted/80 transition-colors text-center"
              >
                <span className="text-sm font-medium text-foreground">Health Resources</span>
              </Link>
              <Link 
                to="/doctors" 
                className="p-4 rounded-lg bg-muted hover:bg-muted/80 transition-colors text-center"
              >
                <span className="text-sm font-medium text-foreground">Find Doctors</span>
              </Link>
              <Link 
                to="/schemes" 
                className="p-4 rounded-lg bg-muted hover:bg-muted/80 transition-colors text-center"
              >
                <span className="text-sm font-medium text-foreground">Govt. Schemes</span>
              </Link>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Education;
