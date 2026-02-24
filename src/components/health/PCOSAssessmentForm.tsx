import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { 
  ArrowRight, 
  ArrowLeft, 
  User, 
  Activity, 
  Heart, 
  Microscope,
  Sparkles,
  TestTube,
  AlertTriangle,
  ShieldCheck
} from "lucide-react";
import { PCOSInputData } from "@/lib/ml-predictions";
import { motion } from "framer-motion";

interface PCOSAssessmentFormProps {
  onSubmit: (data: PCOSInputData) => void;
}

type FormStep = 'consent' | 'personal' | 'symptoms' | 'lifestyle' | 'clinical' | 'bloodwork';

export const PCOSAssessmentForm = ({ onSubmit }: PCOSAssessmentFormProps) => {
  const [step, setStep] = useState<FormStep>('consent');
  const [consentChecked, setConsentChecked] = useState({ reports: false, accuracy: false });
  const [formData, setFormData] = useState<PCOSInputData>({
    age: 25,
    height: 160,
    weight: 55,
    bmi: 21.5,
    cycleRegular: true,
    cycleLength: 28,
    weightGain: false,
    hairGrowth: false,
    skinDarkening: false,
    hairLoss: false,
    pimples: false,
    fastFood: false,
    regularExercise: true,
    follicleLeft: 5,
    follicleRight: 5,
    endometrium: 8,
    lh: 5.0,
    fsh: 6.0,
    testosterone: 30,
    insulin: 10,
  });

  const steps: FormStep[] = ['consent', 'personal', 'symptoms', 'lifestyle', 'clinical', 'bloodwork'];
  const currentStepIndex = steps.indexOf(step);

  const updateField = <K extends keyof PCOSInputData>(field: K, value: PCOSInputData[K]) => {
    setFormData(prev => {
      const updated = { ...prev, [field]: value };
      // Auto-calculate BMI when height or weight changes
      if (field === 'height' || field === 'weight') {
        const h = field === 'height' ? (value as number) : prev.height;
        const w = field === 'weight' ? (value as number) : prev.weight;
        if (h > 0) {
          updated.bmi = Math.round((w / ((h / 100) ** 2)) * 10) / 10;
        }
      }
      return updated;
    });
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

  const stepIcons: Record<FormStep, any> = {
    consent: ShieldCheck,
    personal: User,
    symptoms: Activity,
    lifestyle: Heart,
    clinical: Microscope,
    bloodwork: TestTube,
  };

  const stepLabels: Record<FormStep, string> = {
    consent: 'Confirmation',
    personal: 'Personal Info',
    symptoms: 'Symptoms',
    lifestyle: 'Lifestyle',
    clinical: 'Ultrasound Data',
    bloodwork: 'Blood Test Report',
  };

  const canProceedFromConsent = consentChecked.reports && consentChecked.accuracy;

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Progress Steps */}
      <div className="flex items-center justify-center gap-1 mb-8 overflow-x-auto">
        {steps.map((s, i) => {
          const Icon = stepIcons[s];
          return (
            <div key={s} className="flex items-center">
              <button
                onClick={() => i <= currentStepIndex && setStep(s)}
                className={`w-10 h-10 sm:w-12 sm:h-12 rounded-full flex items-center justify-center text-xs font-medium transition-all duration-300 ${
                  step === s
                    ? "bg-accent text-accent-foreground scale-110 shadow-lg"
                    : i < currentStepIndex
                    ? "bg-teal text-teal-foreground cursor-pointer hover:scale-105"
                    : "bg-muted text-muted-foreground"
                }`}
              >
                <Icon className="w-4 h-4 sm:w-5 sm:h-5" />
              </button>
              {i < steps.length - 1 && (
                <div className={`w-4 sm:w-8 h-0.5 transition-colors duration-300 ${
                  i < currentStepIndex ? "bg-teal" : "bg-border"
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
        <p className="text-xs text-muted-foreground mt-1">
          Step {currentStepIndex + 1} of {steps.length}
        </p>
      </div>

      <div className="glass-card rounded-2xl p-6 md:p-8 animate-fade-up">
        {/* Consent Step */}
        {step === 'consent' && (
          <div className="space-y-6">
            <div className="p-4 rounded-xl bg-accent/10 border border-accent/20">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-accent mt-0.5 flex-shrink-0" />
                <div>
                  <h3 className="font-semibold text-foreground mb-1">Assessment Mode Confirmation</h3>
                  <p className="text-sm text-muted-foreground">
                    This clinical assessment requires accurate values from your medical reports. 
                    Please confirm the following before proceeding:
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div
                className={`p-4 rounded-xl border-2 transition-all cursor-pointer ${
                  consentChecked.reports ? "border-teal bg-teal/10" : "border-border hover:border-teal/50"
                }`}
                onClick={() => setConsentChecked(prev => ({ ...prev, reports: !prev.reports }))}
              >
                <div className="flex items-start gap-3">
                  <Checkbox
                    checked={consentChecked.reports}
                    onCheckedChange={(v) => setConsentChecked(prev => ({ ...prev, reports: !!v }))}
                    className="mt-0.5"
                  />
                  <div>
                    <p className="font-medium text-foreground text-sm">
                      I confirm that I am entering values from official medical reports
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Ultrasound (pelvic sonography) and blood hormone profile reports are required
                    </p>
                  </div>
                </div>
              </div>

              <div
                className={`p-4 rounded-xl border-2 transition-all cursor-pointer ${
                  consentChecked.accuracy ? "border-teal bg-teal/10" : "border-border hover:border-teal/50"
                }`}
                onClick={() => setConsentChecked(prev => ({ ...prev, accuracy: !prev.accuracy }))}
              >
                <div className="flex items-start gap-3">
                  <Checkbox
                    checked={consentChecked.accuracy}
                    onCheckedChange={(v) => setConsentChecked(prev => ({ ...prev, accuracy: !!v }))}
                    className="mt-0.5"
                  />
                  <div>
                    <p className="font-medium text-foreground text-sm">
                      I understand that incorrect information may affect prediction accuracy
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Estimated or guessed values may lead to inaccurate results
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-muted/50">
              <p className="text-xs text-muted-foreground">
                <strong>📋 Required Reports:</strong> Pelvic ultrasound (follicle count, endometrium thickness) 
                and hormone profile blood test (LH, FSH, Testosterone, Insulin).
                If you don't have these, please consult a gynecologist first.
              </p>
            </div>
          </div>
        )}

        {/* Personal Info Step */}
        {step === 'personal' && (
          <div className="space-y-6">
            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="age">Age (years)</Label>
                <Input
                  id="age"
                  type="number"
                  min={15}
                  max={55}
                  value={formData.age}
                  onChange={(e) => updateField('age', Number(e.target.value))}
                  className="text-lg"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="height">Height (cm)</Label>
                <Input
                  id="height"
                  type="number"
                  min={100}
                  max={200}
                  value={formData.height}
                  onChange={(e) => updateField('height', Number(e.target.value))}
                  className="text-lg"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="weight">Weight (kg)</Label>
                <Input
                  id="weight"
                  type="number"
                  min={30}
                  max={150}
                  step={0.5}
                  value={formData.weight}
                  onChange={(e) => updateField('weight', Number(e.target.value))}
                  className="text-lg"
                />
              </div>
            </div>

            <div className="p-4 rounded-xl bg-muted/50">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Calculated BMI:</span>
                <span className={`text-lg font-bold ${
                  formData.bmi < 18.5 ? 'text-accent' : 
                  formData.bmi < 25 ? 'text-teal' : 
                  formData.bmi < 30 ? 'text-accent' : 'text-destructive'
                }`}>
                  {formData.bmi} kg/m²
                </span>
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {formData.bmi < 18.5 ? 'Underweight' : 
                 formData.bmi < 25 ? 'Normal weight' : 
                 formData.bmi < 30 ? 'Overweight' : 'Obese'}
              </p>
            </div>

            <div className="p-4 rounded-xl bg-accent/10 border border-accent/20">
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-base">Menstrual Cycle</Label>
                  <p className="text-sm text-muted-foreground mt-1">Is your cycle regular?</p>
                </div>
                <div className="flex items-center gap-3">
                  <span className={!formData.cycleRegular ? "font-semibold text-destructive" : "text-muted-foreground"}>
                    Irregular
                  </span>
                  <Switch
                    checked={formData.cycleRegular}
                    onCheckedChange={(v) => updateField('cycleRegular', v)}
                  />
                  <span className={formData.cycleRegular ? "font-semibold text-teal" : "text-muted-foreground"}>
                    Regular
                  </span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Average Cycle Length (days)</Label>
              <Slider
                value={[formData.cycleLength]}
                onValueChange={([v]) => updateField('cycleLength', v)}
                min={15}
                max={90}
                step={1}
                className="mt-3"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>15 days</span>
                <span className="font-semibold text-accent">{formData.cycleLength} days</span>
                <span>90 days</span>
              </div>
            </div>
          </div>
        )}

        {/* Symptoms Step */}
        {step === 'symptoms' && (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground mb-4">
              Select all symptoms you're currently experiencing:
            </p>
            
            {[
              { key: 'weightGain', label: 'Weight Gain', desc: 'Unexplained weight gain or difficulty losing weight' },
              { key: 'hairGrowth', label: 'Excess Hair Growth (Hirsutism)', desc: 'Unwanted hair on face, chest, or back' },
              { key: 'skinDarkening', label: 'Skin Darkening (Acanthosis Nigricans)', desc: 'Dark patches on neck, underarms, or groin' },
              { key: 'hairLoss', label: 'Hair Thinning/Loss', desc: 'Thinning hair on scalp (androgenic alopecia)' },
              { key: 'pimples', label: 'Acne / Pimples', desc: 'Persistent or hormonal acne' },
            ].map(({ key, label, desc }) => (
              <div
                key={key}
                className={`p-4 rounded-xl border-2 transition-all cursor-pointer ${
                  formData[key as keyof PCOSInputData]
                    ? "border-accent bg-accent/10"
                    : "border-border hover:border-accent/50"
                }`}
                onClick={() => updateField(key as keyof PCOSInputData, !formData[key as keyof PCOSInputData] as any)}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-foreground">{label}</div>
                    <div className="text-sm text-muted-foreground">{desc}</div>
                  </div>
                  <Switch
                    checked={formData[key as keyof PCOSInputData] as boolean}
                    onCheckedChange={(v) => updateField(key as keyof PCOSInputData, v as any)}
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Lifestyle Step */}
        {step === 'lifestyle' && (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground mb-4">Tell us about your lifestyle habits:</p>
            
            <div
              className={`p-4 rounded-xl border-2 transition-all cursor-pointer ${
                formData.fastFood ? "border-destructive bg-destructive/10" : "border-border hover:border-accent/50"
              }`}
              onClick={() => updateField('fastFood', !formData.fastFood)}
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium text-foreground">Fast Food Consumption</div>
                  <div className="text-sm text-muted-foreground">Do you frequently consume fast food/junk food?</div>
                </div>
                <div className="flex items-center gap-2">
                  <span className={!formData.fastFood ? "text-teal font-medium" : "text-muted-foreground"}>No</span>
                  <Switch checked={formData.fastFood} onCheckedChange={(v) => updateField('fastFood', v)} />
                  <span className={formData.fastFood ? "text-destructive font-medium" : "text-muted-foreground"}>Yes</span>
                </div>
              </div>
            </div>

            <div
              className={`p-4 rounded-xl border-2 transition-all cursor-pointer ${
                formData.regularExercise ? "border-teal bg-teal/10" : "border-border hover:border-accent/50"
              }`}
              onClick={() => updateField('regularExercise', !formData.regularExercise)}
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium text-foreground">Regular Exercise</div>
                  <div className="text-sm text-muted-foreground">Do you exercise regularly (at least 3x/week)?</div>
                </div>
                <div className="flex items-center gap-2">
                  <span className={!formData.regularExercise ? "text-destructive font-medium" : "text-muted-foreground"}>No</span>
                  <Switch checked={formData.regularExercise} onCheckedChange={(v) => updateField('regularExercise', v)} />
                  <span className={formData.regularExercise ? "text-teal font-medium" : "text-muted-foreground"}>Yes</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Clinical (Ultrasound) Step */}
        {step === 'clinical' && (
          <div className="space-y-6">
            <div className="p-4 rounded-xl bg-destructive/10 border border-destructive/20">
              <p className="text-sm text-destructive font-medium">
                🩺 Enter values from your pelvic ultrasound (sonography) report. These are mandatory for accurate prediction.
              </p>
            </div>

            <div className="space-y-2">
              <Label>🥚 Follicle Count - Left Ovary</Label>
              <Slider
                value={[formData.follicleLeft]}
                onValueChange={([v]) => updateField('follicleLeft', v)}
                min={0}
                max={30}
                step={1}
                className="mt-3"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>0</span>
                <span className={`font-semibold ${formData.follicleLeft >= 12 ? 'text-destructive' : 'text-accent'}`}>
                  {formData.follicleLeft} follicles
                </span>
                <span>30</span>
              </div>
            </div>

            <div className="space-y-2">
              <Label>🥚 Follicle Count - Right Ovary</Label>
              <Slider
                value={[formData.follicleRight]}
                onValueChange={([v]) => updateField('follicleRight', v)}
                min={0}
                max={30}
                step={1}
                className="mt-3"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>0</span>
                <span className={`font-semibold ${formData.follicleRight >= 12 ? 'text-destructive' : 'text-accent'}`}>
                  {formData.follicleRight} follicles
                </span>
                <span>30</span>
              </div>
            </div>

            <div className="space-y-2">
              <Label>📄 Endometrium Thickness (mm)</Label>
              <Slider
                value={[formData.endometrium]}
                onValueChange={([v]) => updateField('endometrium', v)}
                min={1}
                max={20}
                step={0.5}
                className="mt-3"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>1 mm</span>
                <span className="font-semibold text-accent">{formData.endometrium} mm</span>
                <span>20 mm</span>
              </div>
            </div>

            {(formData.follicleLeft + formData.follicleRight >= 20) && (
              <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
                ⚠️ High follicle count detected ({formData.follicleLeft + formData.follicleRight} total). This may indicate polycystic ovaries.
              </div>
            )}
          </div>
        )}

        {/* Blood Test Step */}
        {step === 'bloodwork' && (
          <div className="space-y-6">
            <div className="p-4 rounded-xl bg-destructive/10 border border-destructive/20">
              <p className="text-sm text-destructive font-medium">
                🧪 Enter values from your hormone profile blood test. These are mandatory for accurate prediction.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="lh">LH (mIU/mL)</Label>
                <Input
                  id="lh"
                  type="number"
                  min={0.5}
                  max={80}
                  step={0.1}
                  value={formData.lh}
                  onChange={(e) => updateField('lh', Number(e.target.value))}
                  placeholder="e.g., 10.5"
                />
                <p className="text-xs text-muted-foreground">Luteinizing Hormone • Normal: 2-15 mIU/mL</p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="fsh">FSH (mIU/mL)</Label>
                <Input
                  id="fsh"
                  type="number"
                  min={0.5}
                  max={80}
                  step={0.1}
                  value={formData.fsh}
                  onChange={(e) => updateField('fsh', Number(e.target.value))}
                  placeholder="e.g., 6.0"
                />
                <p className="text-xs text-muted-foreground">Follicle Stimulating Hormone • Normal: 3-10 mIU/mL</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="testosterone">Testosterone (ng/dL)</Label>
                <Input
                  id="testosterone"
                  type="number"
                  min={0}
                  max={200}
                  step={1}
                  value={formData.testosterone}
                  onChange={(e) => updateField('testosterone', Number(e.target.value))}
                  placeholder="e.g., 30"
                />
                <p className="text-xs text-muted-foreground">Total Testosterone • Normal: 15-70 ng/dL</p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="insulin">Insulin (µIU/mL)</Label>
                <Input
                  id="insulin"
                  type="number"
                  min={0}
                  max={100}
                  step={0.5}
                  value={formData.insulin}
                  onChange={(e) => updateField('insulin', Number(e.target.value))}
                  placeholder="e.g., 10"
                />
                <p className="text-xs text-muted-foreground">Fasting Insulin • Normal: 2-25 µIU/mL</p>
              </div>
            </div>

            {/* LH/FSH Ratio Alert */}
            {formData.fsh > 0 && formData.lh / formData.fsh > 2 && (
              <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
                ⚠️ LH/FSH ratio is {(formData.lh / formData.fsh).toFixed(1)} (elevated). A ratio above 2:1 is often associated with PCOS.
              </div>
            )}

            {formData.testosterone > 70 && (
              <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
                ⚠️ Testosterone level is elevated ({formData.testosterone} ng/dL). This is a common indicator of PCOS.
              </div>
            )}

            {formData.insulin > 25 && (
              <div className="p-3 rounded-lg bg-accent/10 border border-accent/20 text-sm text-accent">
                ⚠️ Insulin level is elevated ({formData.insulin} µIU/mL). This may indicate insulin resistance.
              </div>
            )}

            {/* Summary Grid */}
            <div className="grid grid-cols-4 gap-3 p-4 rounded-xl bg-muted/50">
              <div className="text-center">
                <div className="text-lg font-bold text-foreground">{formData.lh}</div>
                <div className="text-xs text-muted-foreground">LH</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-foreground">{formData.fsh}</div>
                <div className="text-xs text-muted-foreground">FSH</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-foreground">{formData.testosterone}</div>
                <div className="text-xs text-muted-foreground">Testosterone</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-foreground">{formData.insulin}</div>
                <div className="text-xs text-muted-foreground">Insulin</div>
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

          {step === 'consent' ? (
            <Button 
              onClick={nextStep} 
              className="gap-2" 
              disabled={!canProceedFromConsent}
            >
              Proceed to Assessment
              <ArrowRight className="w-4 h-4" />
            </Button>
          ) : currentStepIndex < steps.length - 1 ? (
            <Button onClick={nextStep} className="gap-2">
              Next
              <ArrowRight className="w-4 h-4" />
            </Button>
          ) : (
            <Button onClick={handleSubmit} className="gap-2 bg-accent hover:bg-accent/90">
              <Sparkles className="w-4 h-4" />
              Get Clinical Assessment
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};
