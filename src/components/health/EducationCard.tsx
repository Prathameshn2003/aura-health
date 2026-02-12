import { memo } from "react";
import { motion } from "framer-motion";
import { LucideIcon } from "lucide-react";

interface EducationCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
  items?: string[];
  color?: "primary" | "accent" | "teal";
  delay?: number;
}

const colorMap = {
  primary: {
    iconBg: "bg-primary/20",
    iconColor: "text-primary",
    border: "border-primary/20",
    hoverBorder: "hover:border-primary/40",
  },
  accent: {
    iconBg: "bg-accent/20",
    iconColor: "text-accent",
    border: "border-accent/20",
    hoverBorder: "hover:border-accent/40",
  },
  teal: {
    iconBg: "bg-teal/20",
    iconColor: "text-teal",
    border: "border-teal/20",
    hoverBorder: "hover:border-teal/40",
  },
};

export const EducationCard = memo(({ icon: Icon, title, description, items, color = "primary", delay = 0 }: EducationCardProps) => {
  const colors = colorMap[color];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
      className={`glass-card rounded-2xl p-6 border ${colors.border} ${colors.hoverBorder} transition-all hover:shadow-lg`}
    >
      <div className="flex items-start gap-4">
        <div className={`w-12 h-12 rounded-xl ${colors.iconBg} flex items-center justify-center flex-shrink-0`}>
          <Icon className={`w-6 h-6 ${colors.iconColor}`} />
        </div>
        <div className="flex-1">
          <h3 className="font-heading text-lg font-semibold text-foreground mb-2">{title}</h3>
          {description && <p className="text-muted-foreground text-sm mb-3">{description}</p>}
          {items && items.length > 0 && (
            <ul className="space-y-2">
              {items.map((item, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: delay + (index * 0.05) }}
                  className="flex items-start gap-2 text-sm"
                >
                  <span className={`w-1.5 h-1.5 rounded-full ${colors.iconBg.replace("/20", "")} mt-2 flex-shrink-0`} />
                  <span className="text-foreground/80">{item}</span>
                </motion.li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </motion.div>
  );
});

EducationCard.displayName = "EducationCard";
