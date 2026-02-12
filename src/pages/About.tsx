import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Heart, Shield, Brain, Users } from "lucide-react";

const values = [
  {
    icon: Heart,
    title: "Women-First",
    description: "Every feature is designed with women's unique health needs in mind.",
  },
  {
    icon: Brain,
    title: "AI-Powered",
    description: "Advanced machine learning for accurate, personalized health insights.",
  },
  {
    icon: Shield,
    title: "Privacy-Focused",
    description: "Your health data is encrypted and never shared without consent.",
  },
  {
    icon: Users,
    title: "Expert-Backed",
    description: "Developed in consultation with healthcare professionals.",
  },
];

const About = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="pt-24 pb-16">
        {/* Hero */}
        <section className="py-16 md:py-24 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-secondary/10 to-transparent" />
          <div className="container mx-auto px-4 relative z-10">
            <div className="max-w-3xl mx-auto text-center">
              <h1 className="font-heading text-4xl md:text-5xl font-bold text-foreground mb-6">
                Empowering Women Through{" "}
                <span className="gradient-text">Intelligent Health</span>
              </h1>
              <p className="text-lg text-muted-foreground leading-relaxed">
                NaariCare was founded with a simple mission: to make quality women's 
                healthcare accessible, understandable, and personalized for every woman, 
                everywhere.
              </p>
            </div>
          </div>
        </section>

        {/* Mission */}
        <section className="py-16 md:py-20">
          <div className="container mx-auto px-4">
            <div className="grid md:grid-cols-2 gap-12 items-center max-w-5xl mx-auto">
              <div>
                <h2 className="font-heading text-3xl font-bold text-foreground mb-4">
                  Our Mission
                </h2>
                <p className="text-muted-foreground mb-4">
                  We believe every woman deserves access to intelligent health insights 
                  that help them understand their bodies better. NaariCare combines 
                  cutting-edge AI technology with medical expertise to provide 
                  personalized predictions, recommendations, and support.
                </p>
                <p className="text-muted-foreground">
                  From menstrual health tracking to PCOS risk prediction and menopause 
                  support, we're building a comprehensive platform that grows with you 
                  through every stage of life.
                </p>
              </div>
              <div className="glass-card rounded-2xl p-8">
                <div className="grid grid-cols-2 gap-4 text-center">
                  <div className="p-4">
                    <div className="text-3xl font-bold gradient-text mb-1">50K+</div>
                    <div className="text-sm text-muted-foreground">Active Users</div>
                  </div>
                  <div className="p-4">
                    <div className="text-3xl font-bold gradient-text mb-1">95%</div>
                    <div className="text-sm text-muted-foreground">Accuracy Rate</div>
                  </div>
                  <div className="p-4">
                    <div className="text-3xl font-bold gradient-text mb-1">500+</div>
                    <div className="text-sm text-muted-foreground">Partner Doctors</div>
                  </div>
                  <div className="p-4">
                    <div className="text-3xl font-bold gradient-text mb-1">24/7</div>
                    <div className="text-sm text-muted-foreground">AI Support</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Values */}
        <section className="py-16 md:py-20 bg-muted/30">
          <div className="container mx-auto px-4">
            <h2 className="font-heading text-3xl font-bold text-foreground text-center mb-12">
              Our Values
            </h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-5xl mx-auto">
              {values.map((value, i) => (
                <div key={value.title} className="glass-card rounded-2xl p-6 text-center animate-fade-up" style={{ animationDelay: `${i * 0.1}s` }}>
                  <div className="w-14 h-14 rounded-xl bg-primary/20 flex items-center justify-center mx-auto mb-4">
                    <value.icon className="w-7 h-7 text-primary" />
                  </div>
                  <h3 className="font-heading font-semibold text-foreground mb-2">
                    {value.title}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {value.description}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  );
};

export default About;
