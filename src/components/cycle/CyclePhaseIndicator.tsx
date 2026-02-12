import { useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Droplets,
  Flower2,
  Sun,
  Moon,
  Info,
  Calendar,
} from "lucide-react";
import { differenceInDays, parseISO, addDays, format } from "date-fns";

interface CyclePhaseIndicatorProps {
  lastPeriodStart: string | null;
  averageCycleLength: number;
  averagePeriodLength: number;
  prediction: {
    predicted_start_date: string;
    days_until: number;
  } | null;
}

const phases = [
  {
    name: "Menstrual",
    icon: Droplets,
    color: "text-primary",
    bgColor: "bg-primary/10",
    borderColor: "border-primary/30",
    description: "Your period is here. Take it easy and rest when needed.",
    tips: ["Stay hydrated", "Use heat for cramps", "Rest when needed"],
  },
  {
    name: "Follicular",
    icon: Flower2,
    color: "text-teal",
    bgColor: "bg-teal/10",
    borderColor: "border-teal/30",
    description: "Energy is rising! Great time for new projects.",
    tips: ["High energy time", "Great for workouts", "Try new things"],
  },
  {
    name: "Ovulation",
    icon: Sun,
    color: "text-accent",
    bgColor: "bg-accent/10",
    borderColor: "border-accent/30",
    description: "Peak fertility window. You might feel most social.",
    tips: ["Peak energy", "Fertile window", "Feeling confident"],
  },
  {
    name: "Luteal",
    icon: Moon,
    color: "text-purple-500",
    bgColor: "bg-purple-500/10",
    borderColor: "border-purple-500/30",
    description: "PMS time. Be gentle with yourself.",
    tips: ["Energy may dip", "Cravings are normal", "Self-care matters"],
  },
];

export function CyclePhaseIndicator({
  lastPeriodStart,
  averageCycleLength,
  averagePeriodLength,
  prediction,
}: CyclePhaseIndicatorProps) {
  const { currentPhase, dayOfCycle, cycleProgress, nextPhaseDate } = useMemo(() => {
    if (!lastPeriodStart) {
      return { currentPhase: null, dayOfCycle: 0, cycleProgress: 0, nextPhaseDate: null };
    }

    const lastStart = parseISO(lastPeriodStart);
    const today = new Date();
    const dayOfCycle = differenceInDays(today, lastStart) + 1;
    
    // Determine current phase
    let phaseIndex = 0;
    let nextPhaseDate: Date | null = null;

    if (dayOfCycle <= averagePeriodLength) {
      // Menstrual phase
      phaseIndex = 0;
      nextPhaseDate = addDays(lastStart, averagePeriodLength);
    } else if (dayOfCycle <= Math.floor(averageCycleLength * 0.4)) {
      // Follicular phase (after period, before ovulation)
      phaseIndex = 1;
      nextPhaseDate = addDays(lastStart, Math.floor(averageCycleLength * 0.4));
    } else if (dayOfCycle <= Math.floor(averageCycleLength * 0.55)) {
      // Ovulation phase (around day 14 of 28-day cycle)
      phaseIndex = 2;
      nextPhaseDate = addDays(lastStart, Math.floor(averageCycleLength * 0.55));
    } else {
      // Luteal phase (after ovulation until next period)
      phaseIndex = 3;
      nextPhaseDate = addDays(lastStart, averageCycleLength);
    }

    const cycleProgress = Math.min(100, (dayOfCycle / averageCycleLength) * 100);

    return {
      currentPhase: phases[phaseIndex],
      dayOfCycle: dayOfCycle > averageCycleLength ? dayOfCycle : dayOfCycle,
      cycleProgress,
      nextPhaseDate,
    };
  }, [lastPeriodStart, averageCycleLength, averagePeriodLength]);

  if (!currentPhase || !lastPeriodStart) {
    return (
      <Card className="glass-card border-0">
        <CardContent className="p-4 sm:p-6 text-center">
          <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mx-auto mb-3">
            <Calendar className="w-6 h-6 text-muted-foreground" />
          </div>
          <p className="text-foreground font-medium mb-1">No cycle data yet</p>
          <p className="text-sm text-muted-foreground">
            Log your first period to see your cycle phase
          </p>
        </CardContent>
      </Card>
    );
  }

  const PhaseIcon = currentPhase.icon;

  return (
    <Card className={`glass-card border-0 overflow-hidden ${currentPhase.borderColor} border-l-4`}>
      <CardContent className="p-4 sm:p-6">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`w-12 h-12 rounded-xl ${currentPhase.bgColor} flex items-center justify-center`}>
              <PhaseIcon className={`w-6 h-6 ${currentPhase.color}`} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-heading font-semibold text-foreground">
                  {currentPhase.name} Phase
                </h3>
                <Badge variant="secondary" className="text-xs">
                  Day {dayOfCycle}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                of ~{averageCycleLength} day cycle
              </p>
            </div>
          </div>
        </div>

        {/* Cycle Progress */}
        <div className="mb-4">
          <div className="flex justify-between text-xs text-muted-foreground mb-1.5">
            <span>Cycle Progress</span>
            <span>{Math.round(cycleProgress)}%</span>
          </div>
          <Progress value={cycleProgress} className="h-2" />
          <div className="flex justify-between mt-1">
            {phases.map((phase, index) => (
              <div
                key={phase.name}
                className={`w-6 h-6 rounded-full flex items-center justify-center -mt-4 ${
                  phase.name === currentPhase.name
                    ? phase.bgColor
                    : "bg-muted"
                }`}
                style={{ marginLeft: index === 0 ? "0" : "auto" }}
              >
                <phase.icon className={`w-3 h-3 ${
                  phase.name === currentPhase.name ? phase.color : "text-muted-foreground"
                }`} />
              </div>
            ))}
          </div>
        </div>

        {/* Description */}
        <p className="text-sm text-foreground/80 mb-4">
          {currentPhase.description}
        </p>

        {/* Tips */}
        <div className="flex flex-wrap gap-2">
          {currentPhase.tips.map((tip, index) => (
            <Badge key={index} variant="outline" className={`text-xs ${currentPhase.bgColor}`}>
              {tip}
            </Badge>
          ))}
        </div>

        {/* Next Period Info */}
        {prediction && prediction.days_until > 0 && (
          <div className="mt-4 pt-4 border-t border-border">
            <div className="flex items-center gap-2 text-sm">
              <Info className="w-4 h-4 text-muted-foreground" />
              <span className="text-muted-foreground">
                Next period in <span className="font-medium text-foreground">~{prediction.days_until} days</span>
                {" "}({format(parseISO(prediction.predicted_start_date), "MMM d")})
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
