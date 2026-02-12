import { Link } from "react-router-dom";
import { 
  Calendar, 
  Activity, 
  Thermometer, 
  Salad, 
  MessageCircle, 
  Stethoscope,
  ArrowRight
} from "lucide-react";

const features = [
  {
    icon: Calendar,
    title: "Menstrual Tracking",
    description: "Track your cycle with precision. Get predictions, insights, and personalized health recommendations.",
    color: "primary",
    path: "/modules/menstrual",
  },
  {
    icon: Activity,
    title: "PCOS Risk Prediction",
    description: "AI-powered PCOS risk assessment with explainable results and actionable health guidance.",
    color: "accent",
    path: "/modules/pcos",
  },
  {
    icon: Thermometer,
    title: "Menopause Insights",
    description: "Understand your menopause stage with intelligent predictions and personalized support.",
    color: "teal",
    path: "/modules/menopause",
  },
  {
    icon: Salad,
    title: "Diet & Fitness Plans",
    description: "Personalized nutrition and exercise recommendations tailored to your health profile.",
    color: "secondary",
    path: "/recommendations",
  },
  {
    icon: MessageCircle,
    title: "AI Health Assistant",
    description: "24/7 intelligent chatbot for instant health queries and empathetic support.",
    color: "primary",
    path: "/chatbot",
  },
  {
    icon: Stethoscope,
    title: "Connect with Doctors",
    description: "Find verified healthcare providers, NGOs, and government health schemes near you.",
    color: "teal",
    path: "/doctors",
  },
];

const colorClasses = {
  primary: "bg-primary/20 text-primary group-hover:bg-primary group-hover:text-primary-foreground",
  accent: "bg-accent/20 text-accent group-hover:bg-accent group-hover:text-accent-foreground",
  teal: "bg-teal/20 text-teal group-hover:bg-teal group-hover:text-teal-foreground",
  secondary: "bg-secondary text-secondary-foreground group-hover:bg-secondary/80",
};

export const FeaturesSection = () => {
  return (
    <section className="py-20 md:py-32 bg-background">
      <div className="container mx-auto px-4">
        {/* Section Header */}
        <div className="text-center max-w-2xl mx-auto mb-16">
          <h2 className="font-heading text-3xl md:text-4xl font-bold text-foreground mb-4">
            Comprehensive Health Modules
          </h2>
          <p className="text-muted-foreground text-lg">
            Everything you need to understand, track, and improve your health – 
            powered by AI and backed by medical expertise.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <Link
              key={feature.title}
              to={feature.path}
              className="group glass-card rounded-2xl p-6 md:p-8 hover:shadow-glow transition-all duration-300 hover:-translate-y-1 animate-fade-up"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className={`w-14 h-14 rounded-xl flex items-center justify-center mb-5 transition-colors duration-300 ${colorClasses[feature.color as keyof typeof colorClasses]}`}>
                <feature.icon className="w-7 h-7" />
              </div>
              <h3 className="font-heading text-xl font-semibold text-foreground mb-3 group-hover:text-accent transition-colors">
                {feature.title}
              </h3>
              <p className="text-muted-foreground text-sm leading-relaxed mb-4">
                {feature.description}
              </p>
              <div className="flex items-center text-sm font-medium text-accent opacity-0 group-hover:opacity-100 transition-opacity">
                Learn More
                <ArrowRight className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" />
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
};
