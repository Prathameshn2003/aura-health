const stats = [
  { value: "50K+", label: "Active Users" },
  { value: "95%", label: "Prediction Accuracy" },
  { value: "24/7", label: "AI Support" },
  { value: "500+", label: "Partner Doctors" },
];

export const StatsSection = () => {
  return (
    <section className="py-16 bg-gradient-to-r from-primary/10 via-secondary/10 to-primary/10">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <div
              key={stat.label}
              className="text-center animate-fade-up"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="font-heading text-3xl md:text-4xl font-bold gradient-text mb-2">
                {stat.value}
              </div>
              <div className="text-sm text-muted-foreground font-medium">
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
