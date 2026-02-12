import { AlertTriangle } from "lucide-react";

export const HealthDisclaimer = () => {
  return (
    <div className="p-4 rounded-xl bg-accent/10 border border-accent/20">
      <div className="flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-accent flex-shrink-0 mt-0.5" />
        <div>
          <h4 className="font-semibold text-foreground text-sm mb-1">
            Health Disclaimer
          </h4>
          <p className="text-sm text-muted-foreground">
            This assessment is for educational purposes only and is not a medical diagnosis. 
            The results should not replace professional medical advice, diagnosis, or treatment. 
            Always consult with a qualified healthcare provider for any health concerns.
          </p>
        </div>
      </div>
    </div>
  );
};
