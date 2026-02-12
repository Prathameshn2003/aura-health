import { memo } from "react";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Moon,
  Brain,
  AlertTriangle,
  Stethoscope,
  Info,
  ChevronRight,
  Heart,
  Sparkles,
  Calendar,
  Droplets,
  Target,
} from "lucide-react";
import { Link } from "react-router-dom";
import { 
  RadialBarChart, 
  RadialBar, 
  ResponsiveContainer,
  PolarAngleAxis
} from "recharts";

interface CycleInsightsProps {
  insights: {
    averageCycleLength: number;
    averagePeriodLength: number;
    cycleVariability: number;
    isRegular: boolean;
    pcosRiskFlag: boolean;
    pcosRiskScore: number;
    stressCorrelation: string | null;
    sleepCorrelation: string | null;
    commonSymptoms: string[];
    needsDoctorConsultation: boolean;
    consultationReasons: string[];
  };
  cycleCount: number;
}

export const CycleInsights = memo(function CycleInsights({ insights, cycleCount }: CycleInsightsProps) {
  const getHealthScore = () => {
    let score = 100;
    if (!insights.isRegular) score -= 20;
    if (insights.pcosRiskFlag) score -= 30;
    if (insights.cycleVariability > 5) score -= 10;
    if (insights.needsDoctorConsultation) score -= 15;
    return Math.max(0, score);
  };

  const healthScore = getHealthScore();
  
  const healthScoreData = [{
    name: "Health",
    value: healthScore,
    fill: healthScore >= 80 ? "hsl(var(--teal))" : healthScore >= 50 ? "hsl(var(--accent))" : "hsl(var(--destructive))"
  }];

  return (
    <div className="space-y-6">
      {/* Cycle Health Score - Visual Gauge */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Card className="glass-card border-0 overflow-hidden">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2">
              <Heart className="w-5 h-5 text-primary" />
              Cycle Health Score
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col md:flex-row items-center gap-6">
              {/* Radial Chart */}
              <div className="w-40 h-40 relative">
                <ResponsiveContainer width="100%" height="100%">
                  <RadialBarChart
                    cx="50%"
                    cy="50%"
                    innerRadius="70%"
                    outerRadius="100%"
                    barSize={12}
                    data={healthScoreData}
                    startAngle={180}
                    endAngle={-180}
                  >
                    <PolarAngleAxis
                      type="number"
                      domain={[0, 100]}
                      angleAxisId={0}
                      tick={false}
                    />
                    <RadialBar
                      background={{ fill: "hsl(var(--muted))" }}
                      dataKey="value"
                      cornerRadius={10}
                      angleAxisId={0}
                    />
                  </RadialBarChart>
                </ResponsiveContainer>
                <div className="absolute inset-0 flex items-center justify-center flex-col">
                  <span className={`text-4xl font-bold ${
                    healthScore >= 80 ? "text-teal" : healthScore >= 50 ? "text-accent" : "text-destructive"
                  }`}>
                    {healthScore}
                  </span>
                  <span className="text-xs text-muted-foreground">out of 100</span>
                </div>
              </div>
              
              {/* Score Interpretation */}
              <div className="flex-1 text-center md:text-left">
                <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium mb-2 ${
                  healthScore >= 80 
                    ? "bg-teal/20 text-teal" 
                    : healthScore >= 50 
                    ? "bg-accent/20 text-accent" 
                    : "bg-destructive/20 text-destructive"
                }`}>
                  {healthScore >= 80 ? "🌸 Healthy" : healthScore >= 50 ? "⚡ Moderate" : "⚠️ Needs Attention"}
                </div>
                <p className="text-sm text-muted-foreground">
                  {healthScore >= 80 
                    ? "Your cycle appears healthy! Keep tracking to maintain awareness."
                    : healthScore >= 50 
                    ? "Some patterns may need attention. Continue monitoring and consider lifestyle adjustments."
                    : "We've noticed some concerns. Please review the recommendations below."
                  }
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Key Stats - Visual Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { icon: Calendar, label: "Cycle Length", value: `${insights.averageCycleLength} days`, color: "text-primary" },
          { icon: Droplets, label: "Period Length", value: `${insights.averagePeriodLength} days`, color: "text-accent" },
          { icon: Target, label: "Regularity", value: insights.isRegular ? "Regular" : "Variable", color: insights.isRegular ? "text-teal" : "text-accent" },
          { icon: Activity, label: "Cycles Tracked", value: cycleCount.toString(), color: "text-primary" },
        ].map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card className="glass-card border-0 p-4">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-lg bg-muted flex items-center justify-center ${stat.color}`}>
                  <stat.icon className="w-5 h-5" />
                </div>
                <div>
                  <div className={`text-xl font-bold ${stat.color}`}>{stat.value}</div>
                  <div className="text-xs text-muted-foreground">{stat.label}</div>
                </div>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Cycle Regularity Visual */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Card className="glass-card border-0">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              {insights.isRegular ? (
                <TrendingUp className="w-5 h-5 text-teal" />
              ) : (
                <TrendingDown className="w-5 h-5 text-accent" />
              )}
              Cycle Regularity Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Cycle Variability</span>
                <span className={`text-sm font-medium ${
                  insights.cycleVariability <= 3 ? "text-teal" : 
                  insights.cycleVariability <= 6 ? "text-accent" : "text-destructive"
                }`}>
                  ±{insights.cycleVariability} days
                </span>
              </div>
              <Progress 
                value={Math.min((insights.cycleVariability / 10) * 100, 100)} 
                className="h-3"
              />
            </div>
            <div className="flex items-center justify-between mt-4 p-3 rounded-lg bg-muted/50">
              <span className="text-sm text-muted-foreground">Variation Range</span>
              <Badge variant={insights.isRegular ? "default" : "secondary"} className="text-sm">
                ±{insights.cycleVariability} days
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground mt-3">
              {insights.isRegular 
                ? "Your cycles are consistent. This makes predictions more accurate! 🎯"
                : "Your cycles vary. Predictions are estimates — listen to your body's signals."}
            </p>
          </CardContent>
        </Card>
      </motion.div>

      {/* PCOS Risk Alert */}
      {insights.pcosRiskFlag && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <Alert className="border-accent/50 bg-accent/10">
            <AlertTriangle className="h-5 w-5 text-accent" />
            <AlertTitle className="text-accent font-semibold">Pattern Alert</AlertTitle>
            <AlertDescription className="text-foreground/80">
              <p className="mb-2">
                Your cycle shows repeated irregular patterns that may be associated with hormonal imbalance. 
                <strong className="text-foreground"> This is not a diagnosis.</strong>
              </p>
              <p className="text-sm mb-3">
                Common conditions like PCOS share similar symptoms. Consider consulting a healthcare provider 
                for proper evaluation.
              </p>
              <Link to="/modules/pcos">
                <Button variant="outline" size="sm" className="gap-2">
                  Learn about PCOS <ChevronRight className="w-4 h-4" />
                </Button>
              </Link>
            </AlertDescription>
          </Alert>
        </motion.div>
      )}

      {/* Doctor Consultation */}
      {insights.needsDoctorConsultation && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <Alert className="border-primary/50 bg-primary/10">
            <Stethoscope className="h-5 w-5 text-primary" />
            <AlertTitle className="text-primary font-semibold">Consider Medical Consultation</AlertTitle>
            <AlertDescription>
              <p className="text-foreground/80 mb-3">
                Based on your tracking data, we gently recommend speaking with a healthcare provider:
              </p>
              <ul className="space-y-2 mb-3">
                {insights.consultationReasons.map((reason, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-foreground/70">
                    <Info className="w-4 h-4 flex-shrink-0 mt-0.5 text-primary" />
                    {reason}
                  </li>
                ))}
              </ul>
              <Link to="/doctors">
                <Button variant="outline" size="sm" className="gap-2">
                  Find Nearby Doctors <ChevronRight className="w-4 h-4" />
                </Button>
              </Link>
            </AlertDescription>
          </Alert>
        </motion.div>
      )}

      {/* Correlations & Patterns */}
      {(insights.stressCorrelation || insights.sleepCorrelation) && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="glass-card border-0">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Sparkles className="w-5 h-5 text-purple-500" />
                Patterns Detected
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {insights.stressCorrelation && (
                <div className="flex items-start gap-3 p-4 rounded-xl bg-gradient-to-r from-red-500/10 to-transparent border border-red-500/20">
                  <Brain className="w-6 h-6 text-red-500 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-foreground">Stress Impact</p>
                    <p className="text-sm text-muted-foreground">{insights.stressCorrelation}</p>
                  </div>
                </div>
              )}
              {insights.sleepCorrelation && (
                <div className="flex items-start gap-3 p-4 rounded-xl bg-gradient-to-r from-indigo-500/10 to-transparent border border-indigo-500/20">
                  <Moon className="w-6 h-6 text-indigo-500 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-foreground">Sleep Impact</p>
                    <p className="text-sm text-muted-foreground">{insights.sleepCorrelation}</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Common Symptoms - Visual Tags */}
      {insights.commonSymptoms.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="glass-card border-0">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Activity className="w-5 h-5 text-teal" />
                Your Common Symptoms
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {insights.commonSymptoms.map((symptom, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.05 }}
                  >
                    <Badge variant="secondary" className="capitalize px-3 py-1 text-sm">
                      {symptom}
                    </Badge>
                  </motion.div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground mt-4">
                These symptoms appear regularly before or during your period.
              </p>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Disclaimer */}
      <div className="p-4 rounded-xl bg-muted/50 border border-border">
        <p className="text-xs text-muted-foreground text-center">
          <strong>Important:</strong> This information is for awareness only and is not medical advice. 
          Always consult a qualified healthcare provider for diagnosis and treatment.
        </p>
      </div>
    </div>
  );
});
