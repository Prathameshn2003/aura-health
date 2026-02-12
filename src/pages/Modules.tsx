import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Link } from "react-router-dom";
import { 
  Calendar, 
  Activity, 
  Thermometer, 
  ArrowRight,
  CheckCircle2,
  Brain
} from "lucide-react";

const modules = [
  {
    id: "menstrual",
    icon: Calendar,
    title: "Menstrual Health Tracking",
    description: "Comprehensive cycle tracking with AI-powered predictions and personalized insights for your reproductive health.",
    features: [
      "Cycle prediction & tracking",
      "Symptom logging",
      "Pattern analysis",
      "Health recommendations",
    ],
    color: "primary",
    path: "/modules/menstrual",
  },
  {
    id: "pcos",
    icon: Activity,
    title: "PCOS Risk Prediction",
    description: "Advanced machine learning model to assess your PCOS risk with explainable results and actionable health guidance.",
    features: [
      "AI-powered risk assessment",
      "Explainable predictions",
      "Personalized diet plans",
      "Exercise recommendations",
    ],
    color: "accent",
    path: "/modules/pcos",
  },
  {
    id: "menopause",
    icon: Thermometer,
    title: "Menopause Stage Prediction",
    description: "Understand your menopause journey with intelligent stage predictions and comprehensive support resources.",
    features: [
      "Stage identification",
      "Symptom management",
      "Lifestyle guidance",
      "Community support",
    ],
    color: "teal",
    path: "/modules/menopause",
  },
];

const colorClasses = {
  primary: {
    bg: "bg-primary/20",
    text: "text-primary",
    border: "border-primary/30",
    hover: "hover:border-primary",
  },
  accent: {
    bg: "bg-accent/20",
    text: "text-accent",
    border: "border-accent/30",
    hover: "hover:border-accent",
  },
  teal: {
    bg: "bg-teal/20",
    text: "text-teal",
    border: "border-teal/30",
    hover: "hover:border-teal",
  },
};

const Modules = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="pt-24 pb-16">
        <div className="container mx-auto px-4">
          {/* Header */}
          <div className="text-center max-w-2xl mx-auto mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 mb-6">
              <Brain className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium text-foreground">AI-Powered Health Modules</span>
            </div>
            <h1 className="font-heading text-3xl md:text-4xl lg:text-5xl font-bold text-foreground mb-4">
              Comprehensive Health Insights
            </h1>
            <p className="text-lg text-muted-foreground">
              Each module provides education, AI-powered predictions, explainable results, 
              and personalized recommendations tailored to your unique health profile.
            </p>
          </div>

          {/* Modules Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {modules.map((module, index) => {
              const colors = colorClasses[module.color as keyof typeof colorClasses];
              return (
                <Link
                  key={module.id}
                  to={module.path}
                  className={`glass-card rounded-2xl p-6 md:p-8 border-2 ${colors.border} ${colors.hover} transition-all duration-300 hover:shadow-glow hover:-translate-y-2 group animate-fade-up`}
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className={`w-16 h-16 rounded-2xl ${colors.bg} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                    <module.icon className={`w-8 h-8 ${colors.text}`} />
                  </div>
                  
                  <h3 className="font-heading text-xl font-bold text-foreground mb-3">
                    {module.title}
                  </h3>
                  
                  <p className="text-muted-foreground text-sm leading-relaxed mb-6">
                    {module.description}
                  </p>

                  <ul className="space-y-3 mb-6">
                    {module.features.map((feature) => (
                      <li key={feature} className="flex items-center gap-2 text-sm">
                        <CheckCircle2 className={`w-4 h-4 ${colors.text} flex-shrink-0`} />
                        <span className="text-foreground">{feature}</span>
                      </li>
                    ))}
                  </ul>

                  <div className={`flex items-center gap-2 ${colors.text} font-medium text-sm group-hover:gap-3 transition-all`}>
                    Start Assessment
                    <ArrowRight className="w-4 h-4" />
                  </div>
                </Link>
              );
            })}
          </div>

          {/* How It Works */}
          <div className="mt-20 max-w-4xl mx-auto">
            <h2 className="font-heading text-2xl md:text-3xl font-bold text-foreground text-center mb-12">
              How Each Module Works
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              {[
                { step: "1", title: "Learn", desc: "Understand the condition" },
                { step: "2", title: "Assess", desc: "Complete questionnaire" },
                { step: "3", title: "Predict", desc: "Get AI-powered results" },
                { step: "4", title: "Act", desc: "Follow recommendations" },
              ].map((item, i) => (
                <div key={item.step} className="text-center">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center mx-auto mb-4 text-foreground font-bold">
                    {item.step}
                  </div>
                  <h4 className="font-heading font-semibold text-foreground mb-1">{item.title}</h4>
                  <p className="text-sm text-muted-foreground">{item.desc}</p>
                  {i < 3 && (
                    <ArrowRight className="w-5 h-5 text-muted-foreground mx-auto mt-4 hidden md:block rotate-0" />
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Modules;
