import { memo, ReactNode } from "react";
import { motion } from "framer-motion";
import { Info } from "lucide-react";

interface ResultSummaryCardProps {
  title: string;
  status: "positive" | "negative" | "neutral";
  icon: ReactNode;
  description: string;
  details?: string[];
}

export const ResultSummaryCard = memo(({ 
  title, 
  status, 
  icon, 
  description, 
  details 
}: ResultSummaryCardProps) => {
  const statusStyles = {
    positive: {
      bg: "bg-teal/10",
      border: "border-teal/30",
      iconBg: "bg-teal/20",
      text: "text-teal",
    },
    negative: {
      bg: "bg-destructive/10",
      border: "border-destructive/30",
      iconBg: "bg-destructive/20",
      text: "text-destructive",
    },
    neutral: {
      bg: "bg-accent/10",
      border: "border-accent/30",
      iconBg: "bg-accent/20",
      text: "text-accent",
    },
  };

  const styles = statusStyles[status];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-2xl p-4 sm:p-6 border-2 ${styles.bg} ${styles.border}`}
    >
      <div className="flex flex-col sm:flex-row items-start gap-4">
        <div className={`w-14 h-14 sm:w-16 sm:h-16 rounded-xl ${styles.iconBg} flex items-center justify-center flex-shrink-0`}>
          <div className={styles.text}>{icon}</div>
        </div>
        <div className="flex-1">
          <h3 className={`font-heading text-lg sm:text-xl font-bold ${styles.text} mb-1`}>
            {title}
          </h3>
          <p className="text-foreground/80 text-sm sm:text-base mb-3">{description}</p>
          
          {details && details.length > 0 && (
            <div className="space-y-2">
              {details.map((detail, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="flex items-start gap-2"
                >
                  <Info className="w-4 h-4 text-muted-foreground flex-shrink-0 mt-0.5" />
                  <span className="text-sm text-muted-foreground">{detail}</span>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
});

ResultSummaryCard.displayName = "ResultSummaryCard";
