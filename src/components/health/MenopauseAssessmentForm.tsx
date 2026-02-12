import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { 
  ArrowRight, 
  ArrowLeft, 
  User, 
  Thermometer, 
  Heart,
  Sparkles
} from "lucide-react";
import { MenopauseInputData } from "@/lib/ml-predictions";

interface MenopauseAssessmentFormProps {
  onSubmit: (data: MenopauseInputData) => void;
}

type FormStep = 'personal' | 'hormones' | 'symptoms';

export const MenopauseAssessmentForm = ({ onSubmit }: MenopauseAssessmentFormProps) => {
  const [step, setStep] = useState<FormStep>('personal');
  const [formData, setFormData] = useState<MenopauseInputData>({
    age: 45,
    estrogenLevel: 50,
    fshLevel: 25,
    yearsSinceLastPeriod: 0,
    irregularPeriods: false,
    missedPeriods: false,
    hotFlashes: false,
    nightSweats: false,
    sleepProblems: false,
    vaginalDryness: false,
    jointPain: false,
  });

  const steps: FormStep[] = ['personal', 'hormones', 'symptoms'];
  const currentStepIndex = steps.indexOf(step);

  const updateField = <K extends keyof MenopauseInputData>(field: K, value: MenopauseInputData[K]) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const nextStep = () => {
    const nextIndex = currentStepIndex + 1;
    if (nextIndex < steps.length) {
      setStep(steps[nextIndex]);
    }
  };

  const prevStep = () => {
    const prevIndex = currentStepIndex - 1;
    if (prevIndex >= 0) {
      setStep(steps[prevIndex]);
    }
  };

  const handleSubmit = () => {
    onSubmit(formData);
  };

  const stepIcons = {
    personal: User,
    hormones: Thermometer,
    symptoms: Heart,
  };

  const stepLabels = {
    personal: 'Personal Info',
    hormones: 'Hormone Levels',
    symptoms: 'Symptoms',
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Progress Steps */}
      <div className="flex items-center justify-center gap-2 mb-8">
        {steps.map((s, i) => {
          const Icon = stepIcons[s];
          return (
            <div key={s} className="flex items-center">
              <button
                onClick={() => i <= currentStepIndex && setStep(s)}
                className={`w-14 h-14 rounded-full flex flex-col items-center justify-center text-xs font-medium transition-all duration-300 ${
                  step === s
                    ? "bg-teal text-teal-foreground scale-110 shadow-lg"
                    : i < currentStepIndex
                    ? "bg-primary text-primary-foreground cursor-pointer hover:scale-105"
                    : "bg-muted text-muted-foreground"
                }`}
              >
                <Icon className="w-5 h-5 mb-0.5" />
                <span className="text-[10px]">{i + 1}/{steps.length}</span>
              </button>
              {i < steps.length - 1 && (
                <div className={`w-12 md:w-20 h-0.5 transition-colors duration-300 ${
                  i < currentStepIndex ? "bg-primary" : "bg-border"
                }`} />
              )}
            </div>
          );
        })}
      </div>

      <div className="text-center mb-6">
        <h2 className="font-heading text-xl font-semibold text-foreground">
          {stepLabels[step]}
        </h2>
      </div>

      <div className="glass-card rounded-2xl p-6 md:p-8 animate-fade-up">
        {/* Personal Info Step */}
        {step === 'personal' && (
          <div className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="age">Age (years)</Label>
              <Slider
                value={[formData.age]}
                onValueChange={([v]) => updateField('age', v)}
                min={30}
                max={60}
                step={1}
                className="mt-3"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>30</span>
                <span className="font-semibold text-lg text-teal">{formData.age} years</span>
                <span>60</span>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Years Since Last Period</Label>
              <Slider
                value={[formData.yearsSinceLastPeriod]}
                onValueChange={([v]) => updateField('yearsSinceLastPeriod', v)}
                min={0}
                max={5}
                step={0.1}
                className="mt-3"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>Still having periods</span>
                <span className={`font-semibold text-lg ${formData.yearsSinceLastPeriod >= 1 ? 'text-primary' : 'text-teal'}`}>
                  {formData.yearsSinceLastPeriod.toFixed(1)} years
                </span>
                <span>5+ years</span>
              </div>
              {formData.yearsSinceLastPeriod >= 1 && (
                <p className="text-sm text-primary mt-2 p-2 rounded-lg bg-primary/10">
                  12+ months without period typically indicates menopause
                </p>
              )}
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div
                className={`p-4 rounded-xl border-2 transition-all cursor-pointer ${
                  formData.irregularPeriods
                    ? "border-accent bg-accent/10"
                    : "border-border hover:border-accent/50"
                }`}
                onClick={() => updateField('irregularPeriods', !formData.irregularPeriods)}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-sm">Irregular Periods</span>
                  <Switch
                    checked={formData.irregularPeriods}
                    onCheckedChange={(v) => updateField('irregularPeriods', v)}
                  />
                </div>
              </div>
              <div
                className={`p-4 rounded-xl border-2 transition-all cursor-pointer ${
                  formData.missedPeriods
                    ? "border-accent bg-accent/10"
                    : "border-border hover:border-accent/50"
                }`}
                onClick={() => updateField('missedPeriods', !formData.missedPeriods)}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-sm">Missed Periods</span>
                  <Switch
                    checked={formData.missedPeriods}
                    onCheckedChange={(v) => updateField('missedPeriods', v)}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Hormones Step */}
        {step === 'hormones' && (
          <div className="space-y-6">
            <div className="p-4 rounded-xl bg-muted/50 mb-4">
              <p className="text-sm text-muted-foreground">
                <strong>Note:</strong> If you have recent blood test results, enter them below.
                Otherwise, use estimated values based on your symptoms.
              </p>
            </div>

            <div className="space-y-2">
              <Label>Estrogen Level</Label>
              <Slider
                value={[formData.estrogenLevel]}
                onValueChange={([v]) => updateField('estrogenLevel', v)}
                min={10}
                max={100}
                step={1}
                className="mt-3"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>10 (Low)</span>
                <span className={`font-semibold text-lg ${formData.estrogenLevel <= 30 ? 'text-destructive' : formData.estrogenLevel >= 60 ? 'text-teal' : 'text-accent'}`}>
                  {formData.estrogenLevel}
                </span>
                <span>100 (High)</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Lower estrogen levels are common during/after menopause
              </p>
            </div>

            <div className="space-y-2">
              <Label>FSH Level (Follicle Stimulating Hormone)</Label>
              <Slider
                value={[formData.fshLevel]}
                onValueChange={([v]) => updateField('fshLevel', v)}
                min={5}
                max={80}
                step={1}
                className="mt-3"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>5 (Low)</span>
                <span className={`font-semibold text-lg ${formData.fshLevel >= 40 ? 'text-primary' : formData.fshLevel >= 25 ? 'text-accent' : 'text-teal'}`}>
                  {formData.fshLevel}
                </span>
                <span>80 (High)</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Higher FSH levels (above 25) often indicate perimenopause or menopause
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4 p-4 rounded-xl bg-muted/30">
              <div className="text-center">
                <div className="text-2xl font-bold text-foreground">{formData.estrogenLevel}</div>
                <div className="text-sm text-muted-foreground">Estrogen</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-foreground">{formData.fshLevel}</div>
                <div className="text-sm text-muted-foreground">FSH</div>
              </div>
            </div>
          </div>
        )}

        {/* Symptoms Step */}
        {step === 'symptoms' && (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground mb-4">
              Select all symptoms you're experiencing:
            </p>
            
            {[
              { key: 'hotFlashes', label: 'Hot Flashes', emoji: '🔥', desc: 'Sudden feeling of warmth in upper body' },
              { key: 'nightSweats', label: 'Night Sweats', emoji: '🌙', desc: 'Excessive sweating during sleep' },
              { key: 'sleepProblems', label: 'Sleep Problems', emoji: '😴', desc: 'Difficulty falling or staying asleep' },
              { key: 'vaginalDryness', label: 'Vaginal Dryness', emoji: '💧', desc: 'Dryness or discomfort' },
              { key: 'jointPain', label: 'Joint Pain', emoji: '🦴', desc: 'Aching or stiffness in joints' },
            ].map(({ key, label, emoji, desc }) => (
              <div
                key={key}
                className={`p-4 rounded-xl border-2 transition-all cursor-pointer ${
                  formData[key as keyof MenopauseInputData]
                    ? "border-teal bg-teal/10"
                    : "border-border hover:border-teal/50"
                }`}
                onClick={() => updateField(key as keyof MenopauseInputData, !formData[key as keyof MenopauseInputData] as any)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{emoji}</span>
                    <div>
                      <div className="font-medium text-foreground">{label}</div>
                      <div className="text-sm text-muted-foreground">{desc}</div>
                    </div>
                  </div>
                  <Switch
                    checked={formData[key as keyof MenopauseInputData] as boolean}
                    onCheckedChange={(v) => updateField(key as keyof MenopauseInputData, v as any)}
                  />
                </div>
              </div>
            ))}

            {/* Symptom Summary */}
            <div className="mt-6 p-4 rounded-xl bg-muted/50">
              <div className="text-sm text-muted-foreground">
                Symptoms selected: <span className="font-semibold text-foreground">
                  {[formData.hotFlashes, formData.nightSweats, formData.sleepProblems, 
                    formData.vaginalDryness, formData.jointPain].filter(Boolean).length} of 5
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Navigation Buttons */}
        <div className="flex justify-between mt-8 pt-6 border-t border-border">
          <Button
            variant="ghost"
            onClick={prevStep}
            disabled={currentStepIndex === 0}
            className="gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Previous
          </Button>

          {currentStepIndex < steps.length - 1 ? (
            <Button onClick={nextStep} className="gap-2 bg-teal hover:bg-teal/90">
              Next
              <ArrowRight className="w-4 h-4" />
            </Button>
          ) : (
            <Button onClick={handleSubmit} className="gap-2 bg-teal hover:bg-teal/90">
              <Sparkles className="w-4 h-4" />
              Get Assessment
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};
