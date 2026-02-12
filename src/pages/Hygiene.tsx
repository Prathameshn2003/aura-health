import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Droplets, CheckCircle, AlertTriangle, Sparkles, Heart } from "lucide-react";

const hygieneTopics = [
  {
    title: "Menstrual Hygiene",
    icon: Droplets,
    color: "text-primary",
    bgColor: "bg-primary/20",
    tips: [
      "Change sanitary products every 4-6 hours",
      "Wash hands before and after changing products",
      "Use clean, breathable cotton underwear",
      "Dispose of sanitary products properly",
      "Track your cycle for better preparation",
    ],
  },
  {
    title: "Daily Intimate Care",
    icon: Sparkles,
    color: "text-accent",
    bgColor: "bg-accent/20",
    tips: [
      "Clean the external area with plain water",
      "Avoid harsh soaps or douching",
      "Wipe from front to back",
      "Wear breathable cotton underwear",
      "Change underwear daily",
    ],
  },
  {
    title: "Warning Signs",
    icon: AlertTriangle,
    color: "text-destructive",
    bgColor: "bg-destructive/20",
    tips: [
      "Unusual discharge with strong odor",
      "Persistent itching or burning",
      "Pain during urination",
      "Irregular bleeding",
      "Consult a doctor if symptoms persist",
    ],
  },
  {
    title: "Healthy Habits",
    icon: Heart,
    color: "text-teal",
    bgColor: "bg-teal/20",
    tips: [
      "Stay hydrated throughout the day",
      "Maintain a balanced diet",
      "Exercise regularly",
      "Get adequate sleep",
      "Manage stress through relaxation",
    ],
  },
];

const productGuide = [
  {
    name: "Sanitary Pads",
    pros: ["Easy to use", "Widely available", "Various sizes"],
    cons: ["Can feel bulky", "May cause rashes", "Not eco-friendly"],
  },
  {
    name: "Tampons",
    pros: ["Discreet", "Good for swimming", "Freedom of movement"],
    cons: ["TSS risk if left too long", "Learning curve", "Not for everyone"],
  },
  {
    name: "Menstrual Cups",
    pros: ["Eco-friendly", "Cost-effective", "Long-lasting"],
    cons: ["Learning curve", "Requires sterilization", "Initial investment"],
  },
  {
    name: "Period Underwear",
    pros: ["Reusable", "Comfortable", "Backup protection"],
    cons: ["Higher upfront cost", "Requires washing", "May feel different"],
  },
];

const Hygiene = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="pt-24 pb-16">
        <div className="container mx-auto px-4">
          <div className="mb-10">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 rounded-lg bg-teal/20 flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-teal" />
              </div>
              <h1 className="font-heading text-3xl md:text-4xl font-bold text-foreground">
                Hygiene & Wellness
              </h1>
            </div>
            <p className="text-muted-foreground">
              Essential tips and guidance for women's health and hygiene
            </p>
          </div>

          {/* Hygiene Topics */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
            {hygieneTopics.map((topic) => (
              <div key={topic.title} className="glass-card rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className={`w-10 h-10 rounded-lg ${topic.bgColor} flex items-center justify-center`}>
                    <topic.icon className={`w-5 h-5 ${topic.color}`} />
                  </div>
                  <h2 className="font-heading text-xl font-semibold text-foreground">
                    {topic.title}
                  </h2>
                </div>
                <ul className="space-y-3">
                  {topic.tips.map((tip, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 text-teal mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-muted-foreground">{tip}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          {/* Product Guide */}
          <div className="mb-12">
            <h2 className="font-heading text-2xl font-semibold text-foreground mb-6">
              Menstrual Product Guide
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {productGuide.map((product) => (
                <div key={product.name} className="glass-card rounded-xl p-5">
                  <h3 className="font-semibold text-foreground mb-4">{product.name}</h3>
                  
                  <div className="mb-3">
                    <p className="text-xs font-medium text-teal mb-2">Pros</p>
                    <ul className="space-y-1">
                      {product.pros.map((pro, i) => (
                        <li key={i} className="text-xs text-muted-foreground flex items-center gap-1">
                          <span className="w-1 h-1 rounded-full bg-teal" />
                          {pro}
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <div>
                    <p className="text-xs font-medium text-destructive mb-2">Cons</p>
                    <ul className="space-y-1">
                      {product.cons.map((con, i) => (
                        <li key={i} className="text-xs text-muted-foreground flex items-center gap-1">
                          <span className="w-1 h-1 rounded-full bg-destructive" />
                          {con}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Disclaimer */}
          <div className="glass-card rounded-xl p-6 bg-muted/50">
            <p className="text-sm text-muted-foreground text-center">
              <strong>Disclaimer:</strong> This information is for educational purposes only and should not replace professional medical advice. 
              Please consult a healthcare provider for personalized guidance.
            </p>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Hygiene;
