import { memo } from "react";
import { motion } from "framer-motion";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface ScoreItem {
  name: string;
  score: number;
  maxScore: number;
  description?: string;
  color?: string;
}

interface ScoreBreakdownProps {
  title: string;
  items: ScoreItem[];
  showTrend?: boolean;
}

export const ScoreBreakdown = memo(({ title, items, showTrend = true }: ScoreBreakdownProps) => {
  const getPercentage = (score: number, max: number) => Math.round((score / max) * 100);
  
  const getColor = (percentage: number) => {
    if (percentage <= 30) return { bg: "bg-teal", text: "text-teal" };
    if (percentage <= 60) return { bg: "bg-accent", text: "text-accent" };
    return { bg: "bg-destructive", text: "text-destructive" };
  };

  const getTrend = (percentage: number) => {
    if (percentage <= 30) return { icon: TrendingDown, label: "Low" };
    if (percentage <= 60) return { icon: Minus, label: "Moderate" };
    return { icon: TrendingUp, label: "High" };
  };

  return (
    <div className="glass-card rounded-2xl p-4 sm:p-6">
      <h3 className="font-heading text-base sm:text-lg font-semibold text-foreground mb-4 sm:mb-6">
        {title}
      </h3>
      
      <div className="space-y-4 sm:space-y-5">
        {items.map((item, index) => {
          const percentage = getPercentage(item.score, item.maxScore);
          const colorStyles = item.color ? { bg: item.color, text: item.color.replace("bg-", "text-") } : getColor(percentage);
          const trend = getTrend(percentage);
          const TrendIcon = trend.icon;

          return (
            <motion.div
              key={item.name}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="space-y-2"
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 sm:gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-foreground text-sm sm:text-base">{item.name}</span>
                  {showTrend && (
                    <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-xs ${colorStyles.bg}/20 ${colorStyles.text}`}>
                      <TrendIcon className="w-3 h-3" />
                      <span>{trend.label}</span>
                    </div>
                  )}
                </div>
                <span className={`font-bold text-sm ${colorStyles.text}`}>
                  {item.score}/{item.maxScore} ({percentage}%)
                </span>
              </div>
              
              <div className="h-3 bg-muted rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${percentage}%` }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  className={`h-full ${colorStyles.bg} rounded-full relative`}
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                </motion.div>
              </div>
              
              {item.description && (
                <p className="text-xs text-muted-foreground">{item.description}</p>
              )}
            </motion.div>
          );
        })}
      </div>
      
      {/* Legend */}
      <div className="flex flex-wrap items-center justify-center gap-3 sm:gap-4 mt-4 sm:mt-6 pt-4 border-t border-border">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-teal" />
          <span className="text-xs text-muted-foreground">Low (0-30%)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-accent" />
          <span className="text-xs text-muted-foreground">Moderate (31-60%)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-destructive" />
          <span className="text-xs text-muted-foreground">High (61-100%)</span>
        </div>
      </div>
    </div>
  );
});

ScoreBreakdown.displayName = "ScoreBreakdown";
