import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Calendar,
  Activity,
  TrendingUp,
  ChevronRight,
  Sparkles,
  CheckCircle2,
  Circle,
  Droplets,
} from "lucide-react";

interface QuickStartCardProps {
  cycleCount: number;
  hasLoggedToday: boolean;
  onLogPeriod: () => void;
  onLogSymptoms: () => void;
  onViewInsights: () => void;
}

export function QuickStartCard({
  cycleCount,
  hasLoggedToday,
  onLogPeriod,
  onLogSymptoms,
  onViewInsights,
}: QuickStartCardProps) {
  const steps = [
    {
      title: "Log Your Period",
      description: "Tap when your period starts",
      completed: cycleCount > 0,
      action: onLogPeriod,
      icon: Droplets,
      color: "text-primary",
    },
    {
      title: "Track Today's Symptoms",
      description: "How are you feeling?",
      completed: hasLoggedToday,
      action: onLogSymptoms,
      icon: Activity,
      color: "text-teal",
    },
    {
      title: "View Your Insights",
      description: cycleCount >= 2 ? "Check your patterns" : "Need 2+ cycles",
      completed: cycleCount >= 2,
      action: onViewInsights,
      icon: TrendingUp,
      color: "text-purple-500",
      disabled: cycleCount < 2,
    },
  ];

  const completedCount = steps.filter(s => s.completed).length;
  const progress = (completedCount / steps.length) * 100;

  return (
    <Card className="glass-card border-0 overflow-hidden">
      {/* Progress Bar */}
      <div className="h-1.5 bg-muted">
        <div
          className="h-full bg-gradient-to-r from-primary to-teal transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>

      <CardContent className="p-4 sm:p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-primary" />
            <h3 className="font-heading font-semibold text-foreground">
              Quick Start
            </h3>
          </div>
          <Badge variant="secondary" className="text-xs">
            {completedCount}/{steps.length} done
          </Badge>
        </div>

        <div className="space-y-3">
          {steps.map((step, index) => (
            <button
              key={index}
              onClick={step.action}
              disabled={step.disabled}
              className={`w-full flex items-center gap-4 p-3 rounded-xl transition-all ${
                step.disabled
                  ? "opacity-50 cursor-not-allowed bg-muted/30"
                  : "hover:bg-muted/50 cursor-pointer"
              } ${step.completed ? "bg-muted/30" : ""}`}
            >
              {/* Status Icon */}
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
                  step.completed
                    ? "bg-teal/20"
                    : "bg-muted"
                }`}
              >
                {step.completed ? (
                  <CheckCircle2 className="w-5 h-5 text-teal" />
                ) : (
                  <step.icon className={`w-5 h-5 ${step.color}`} />
                )}
              </div>

              {/* Content */}
              <div className="flex-1 text-left">
                <p className={`font-medium text-sm ${
                  step.completed ? "text-muted-foreground line-through" : "text-foreground"
                }`}>
                  {step.title}
                </p>
                <p className="text-xs text-muted-foreground">
                  {step.description}
                </p>
              </div>

              {/* Arrow */}
              {!step.completed && !step.disabled && (
                <ChevronRight className="w-5 h-5 text-muted-foreground" />
              )}
            </button>
          ))}
        </div>

        {completedCount === steps.length && (
          <div className="mt-4 p-3 rounded-xl bg-teal/10 border border-teal/20 text-center">
            <p className="text-sm text-teal font-medium">
              🎉 Great job! Keep tracking to improve predictions
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
