import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Calendar,
  Activity,
  TrendingUp,
  Bell,
  ChevronRight,
  ChevronLeft,
  Sparkles,
  Heart,
  CheckCircle2,
  Play,
  Droplets,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface OnboardingGuideProps {
  onComplete: () => void;
  onSkip: () => void;
}

const steps = [
  {
    icon: Droplets,
    title: "Welcome to Your Cycle Tracker",
    subtitle: "Let's get you started in 4 simple steps",
    description: "Track your menstrual cycle, understand your body patterns, and get personalized predictions. It takes just 2 minutes to set up!",
    image: "🌸",
    color: "text-primary",
    bgColor: "bg-primary/10",
  },
  {
    icon: Calendar,
    title: "Step 1: Log Your Period",
    subtitle: "Tell us when your last period started",
    description: "Simply tap any date on the calendar to mark when your period began. Don't worry about exact dates - estimates work too!",
    tips: [
      "Tap the date when your period started",
      "You can log past periods too",
      "Each entry helps improve predictions",
    ],
    image: "📅",
    color: "text-teal",
    bgColor: "bg-teal/10",
  },
  {
    icon: Activity,
    title: "Step 2: Track Your Symptoms",
    subtitle: "Log how you're feeling each day",
    description: "Record symptoms like cramps, mood changes, and energy levels. This helps identify patterns unique to your body.",
    tips: [
      "Flow intensity: Light to Heavy",
      "Mood: Happy, Neutral, Sad, etc.",
      "Physical symptoms: Cramps, Headache, Fatigue",
    ],
    image: "📝",
    color: "text-accent",
    bgColor: "bg-accent/10",
  },
  {
    icon: TrendingUp,
    title: "Step 3: View Your Insights",
    subtitle: "Understand your unique patterns",
    description: "After logging 2+ cycles, you'll see personalized insights including average cycle length, symptom patterns, and health scores.",
    tips: [
      "Cycle Health Score",
      "Average cycle & period length",
      "Stress & sleep correlations",
    ],
    image: "📊",
    color: "text-purple-500",
    bgColor: "bg-purple-500/10",
  },
  {
    icon: Bell,
    title: "Step 4: Get Smart Reminders",
    subtitle: "Never be caught off guard",
    description: "Set up notifications to remind you before your period starts. Customize when and how you want to be notified.",
    tips: [
      "Reminders 3, 2, 1 days before",
      "Choose your preferred time",
      "Works even with irregular cycles",
    ],
    image: "🔔",
    color: "text-pink-500",
    bgColor: "bg-pink-500/10",
  },
];

export function OnboardingGuide({ onComplete, onSkip }: OnboardingGuideProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const step = steps[currentStep];
  const isLastStep = currentStep === steps.length - 1;
  const isFirstStep = currentStep === 0;

  const nextStep = () => {
    if (isLastStep) {
      onComplete();
    } else {
      setCurrentStep(prev => prev + 1);
    }
  };

  const prevStep = () => {
    if (!isFirstStep) {
      setCurrentStep(prev => prev - 1);
    }
  };

  return (
    <div className="max-w-lg mx-auto">
      {/* Progress Indicator */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex gap-1.5">
          {steps.map((_, index) => (
            <div
              key={index}
              className={`h-1.5 rounded-full transition-all duration-300 ${
                index === currentStep
                  ? "w-8 bg-primary"
                  : index < currentStep
                  ? "w-4 bg-primary/50"
                  : "w-4 bg-muted"
              }`}
            />
          ))}
        </div>
        {!isFirstStep && (
          <Button variant="ghost" size="sm" onClick={onSkip}>
            Skip
          </Button>
        )}
      </div>

      {/* Step Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
        >
          <Card className="glass-card border-0 overflow-hidden">
            {/* Header */}
            <div className={`${step.bgColor} p-8 text-center`}>
              <div className="text-6xl mb-4">{step.image}</div>
              <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full ${step.bgColor} border border-current/20 mb-3`}>
                <step.icon className={`w-4 h-4 ${step.color}`} />
                <span className={`text-xs font-medium ${step.color}`}>
                  {isFirstStep ? "Getting Started" : `Step ${currentStep} of ${steps.length - 1}`}
                </span>
              </div>
              <h2 className="font-heading text-2xl font-bold text-foreground mb-2">
                {step.title}
              </h2>
              <p className="text-muted-foreground text-sm">{step.subtitle}</p>
            </div>

            {/* Body */}
            <CardContent className="p-6">
              <p className="text-foreground/80 text-center mb-6">
                {step.description}
              </p>

              {/* Tips */}
              {step.tips && (
                <div className="space-y-3 mb-6">
                  {step.tips.map((tip, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-3 p-3 rounded-xl bg-muted/50"
                    >
                      <div className={`w-8 h-8 rounded-full ${step.bgColor} flex items-center justify-center flex-shrink-0`}>
                        <CheckCircle2 className={`w-4 h-4 ${step.color}`} />
                      </div>
                      <span className="text-sm text-foreground">{tip}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Navigation */}
              <div className="flex gap-3">
                {!isFirstStep && (
                  <Button
                    variant="outline"
                    onClick={prevStep}
                    className="flex-1"
                  >
                    <ChevronLeft className="w-4 h-4 mr-1" />
                    Back
                  </Button>
                )}
                <Button
                  onClick={nextStep}
                  className={`flex-1 gap-2 ${isFirstStep ? "w-full" : ""}`}
                >
                  {isFirstStep ? (
                    <>
                      <Play className="w-4 h-4" />
                      Let's Get Started
                    </>
                  ) : isLastStep ? (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Start Tracking
                    </>
                  ) : (
                    <>
                      Next
                      <ChevronRight className="w-4 h-4" />
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </AnimatePresence>

      {/* Help Text */}
      <p className="text-xs text-muted-foreground text-center mt-4">
        <Heart className="w-3 h-3 inline mr-1" />
        Your data is private and secure
      </p>
    </div>
  );
}
