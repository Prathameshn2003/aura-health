import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Droplet,
  Frown,
  Smile,
  Meh,
  Moon,
  Activity,
  Brain,
  ThermometerSun,
  Sparkles,
  Heart,
  CircleAlert,
  Loader2,
  Check,
} from "lucide-react";
import { format } from "date-fns";

interface SymptomLoggerProps {
  date: Date;
  onSave: (symptoms: SymptomData) => Promise<void>;
  saving: boolean;
  initialData?: Partial<SymptomData>;
}

export interface SymptomData {
  flow_intensity: string | null;
  cramps: string | null;
  mood: string | null;
  fatigue: string | null;
  bloating: string | null;
  headache: string | null;
  breast_tenderness: string | null;
  acne: string | null;
  stress_level: number | null;
  sleep_hours: number | null;
  notes: string | null;
}

const flowOptions = [
  { value: "light", label: "Light", icon: "💧" },
  { value: "moderate", label: "Moderate", icon: "💧💧" },
  { value: "heavy", label: "Heavy", icon: "💧💧💧" },
  { value: "very_heavy", label: "Very Heavy", icon: "🌊" },
];

const severityOptions = [
  { value: "none", label: "None", color: "bg-muted" },
  { value: "mild", label: "Mild", color: "bg-teal/30" },
  { value: "moderate", label: "Moderate", color: "bg-accent/30" },
  { value: "severe", label: "Severe", color: "bg-destructive/30" },
];

const moodOptions = [
  { value: "happy", label: "Happy", icon: Smile, color: "text-teal" },
  { value: "neutral", label: "Neutral", icon: Meh, color: "text-muted-foreground" },
  { value: "sad", label: "Sad", icon: Frown, color: "text-blue-500" },
  { value: "irritable", label: "Irritable", icon: CircleAlert, color: "text-accent" },
  { value: "anxious", label: "Anxious", icon: Brain, color: "text-purple-500" },
];

export function SymptomLogger({ date, onSave, saving, initialData }: SymptomLoggerProps) {
  const [symptoms, setSymptoms] = useState<SymptomData>({
    flow_intensity: initialData?.flow_intensity || null,
    cramps: initialData?.cramps || "none",
    mood: initialData?.mood || "neutral",
    fatigue: initialData?.fatigue || "none",
    bloating: initialData?.bloating || "none",
    headache: initialData?.headache || "none",
    breast_tenderness: initialData?.breast_tenderness || "none",
    acne: initialData?.acne || "none",
    stress_level: initialData?.stress_level || 3,
    sleep_hours: initialData?.sleep_hours || 7,
    notes: initialData?.notes || null,
  });
  const [saved, setSaved] = useState(false);

  const handleSave = async () => {
    await onSave(symptoms);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const updateSymptom = <K extends keyof SymptomData>(key: K, value: SymptomData[K]) => {
    setSymptoms(prev => ({ ...prev, [key]: value }));
    setSaved(false);
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="font-heading text-lg font-semibold text-foreground">
          Log Symptoms for {format(date, "MMMM d")}
        </h3>
        <p className="text-sm text-muted-foreground mt-1">
          Track how you're feeling today
        </p>
      </div>

      {/* Flow Intensity */}
      <Card className="glass-card border-0">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Droplet className="w-5 h-5 text-primary" />
            Flow Intensity
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {flowOptions.map(option => (
              <button
                key={option.value}
                onClick={() => updateSymptom("flow_intensity", option.value)}
                className={`p-3 rounded-xl border-2 transition-all ${
                  symptoms.flow_intensity === option.value
                    ? "border-primary bg-primary/10"
                    : "border-border hover:border-primary/50"
                }`}
              >
                <div className="text-2xl mb-1">{option.icon}</div>
                <div className="text-xs font-medium">{option.label}</div>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Mood */}
      <Card className="glass-card border-0">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Heart className="w-5 h-5 text-pink-500" />
            Mood
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {moodOptions.map(option => {
              const Icon = option.icon;
              return (
                <button
                  key={option.value}
                  onClick={() => updateSymptom("mood", option.value)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-full border-2 transition-all ${
                    symptoms.mood === option.value
                      ? "border-primary bg-primary/10"
                      : "border-border hover:border-primary/50"
                  }`}
                >
                  <Icon className={`w-4 h-4 ${option.color}`} />
                  <span className="text-sm font-medium">{option.label}</span>
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Symptoms Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Cramps */}
        <Card className="glass-card border-0">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <ThermometerSun className="w-4 h-4 text-orange-500" />
              Cramps
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="flex gap-1">
              {severityOptions.map(option => (
                <button
                  key={option.value}
                  onClick={() => updateSymptom("cramps", option.value)}
                  className={`flex-1 px-2 py-1.5 text-xs rounded-lg transition-all ${
                    symptoms.cramps === option.value
                      ? `${option.color} ring-2 ring-primary`
                      : "bg-muted/50 hover:bg-muted"
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Fatigue */}
        <Card className="glass-card border-0">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Activity className="w-4 h-4 text-yellow-500" />
              Fatigue
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="flex gap-1">
              {severityOptions.map(option => (
                <button
                  key={option.value}
                  onClick={() => updateSymptom("fatigue", option.value)}
                  className={`flex-1 px-2 py-1.5 text-xs rounded-lg transition-all ${
                    symptoms.fatigue === option.value
                      ? `${option.color} ring-2 ring-primary`
                      : "bg-muted/50 hover:bg-muted"
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Bloating */}
        <Card className="glass-card border-0">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Sparkles className="w-4 h-4 text-blue-500" />
              Bloating
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="flex gap-1">
              {severityOptions.map(option => (
                <button
                  key={option.value}
                  onClick={() => updateSymptom("bloating", option.value)}
                  className={`flex-1 px-2 py-1.5 text-xs rounded-lg transition-all ${
                    symptoms.bloating === option.value
                      ? `${option.color} ring-2 ring-primary`
                      : "bg-muted/50 hover:bg-muted"
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Headache */}
        <Card className="glass-card border-0">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Brain className="w-4 h-4 text-purple-500" />
              Headache
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="flex gap-1">
              {severityOptions.map(option => (
                <button
                  key={option.value}
                  onClick={() => updateSymptom("headache", option.value)}
                  className={`flex-1 px-2 py-1.5 text-xs rounded-lg transition-all ${
                    symptoms.headache === option.value
                      ? `${option.color} ring-2 ring-primary`
                      : "bg-muted/50 hover:bg-muted"
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Stress & Sleep */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Card className="glass-card border-0">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Brain className="w-4 h-4 text-red-500" />
              Stress Level: {symptoms.stress_level}/5
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <Slider
              value={[symptoms.stress_level || 3]}
              onValueChange={([value]) => updateSymptom("stress_level", value)}
              min={1}
              max={5}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>Low</span>
              <span>High</span>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-0">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Moon className="w-4 h-4 text-indigo-500" />
              Sleep: {symptoms.sleep_hours} hours
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <Slider
              value={[symptoms.sleep_hours || 7]}
              onValueChange={([value]) => updateSymptom("sleep_hours", value)}
              min={0}
              max={12}
              step={0.5}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>0h</span>
              <span>12h</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Notes */}
      <Card className="glass-card border-0">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Notes (Optional)</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <Textarea
            placeholder="Any additional notes about how you're feeling..."
            value={symptoms.notes || ""}
            onChange={(e) => updateSymptom("notes", e.target.value || null)}
            className="min-h-[80px] resize-none"
          />
        </CardContent>
      </Card>

      {/* Save Button */}
      <Button
        onClick={handleSave}
        disabled={saving}
        className="w-full"
        size="lg"
      >
        {saving ? (
          <>
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            Saving...
          </>
        ) : saved ? (
          <>
            <Check className="w-4 h-4 mr-2" />
            Saved!
          </>
        ) : (
          "Save Symptoms"
        )}
      </Button>
    </div>
  );
}
