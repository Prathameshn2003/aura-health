import { useState, useEffect, useCallback, useMemo } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "./useAuth";
import { toast } from "./use-toast";
import { differenceInDays, addDays, format, subDays, parseISO, isValid } from "date-fns";

interface CycleLog {
  id: string;
  user_id: string;
  start_date: string;
  end_date: string | null;
  cycle_length: number | null;
  period_length: number | null;
  symptoms: Record<string, unknown> | null;
  notes: string | null;
  flow_intensity: string | null;
  cramps: string | null;
  mood: string | null;
  acne: string | null;
  fatigue: string | null;
  bloating: string | null;
  stress_level: number | null;
  sleep_hours: number | null;
  headache: string | null;
  breast_tenderness: string | null;
  created_at: string;
}

interface CycleSettings {
  id: string;
  user_id: string;
  notification_enabled: boolean;
  reminder_days: number[];
  notification_time: string;
  hide_notification_text: boolean;
  allow_advanced_analysis: boolean;
  average_cycle_length: number | null;
  average_period_length: number | null;
  cycle_variability: number | null;
  pcos_risk_flag: boolean;
  pcos_risk_score: number | null;
}

interface CyclePrediction {
  predicted_start_date: string;
  predicted_end_date: string | null;
  confidence_level: "high" | "medium" | "low";
  days_until: number;
  based_on_cycles: number;
}

interface CycleInsights {
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
}

const DEFAULT_CYCLE_LENGTH = 28;
const DEFAULT_PERIOD_LENGTH = 5;

export function useCycleTracking() {
  const { user } = useAuth();
  const [cycleLogs, setCycleLogs] = useState<CycleLog[]>([]);
  const [settings, setSettings] = useState<CycleSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // Fetch cycle logs and settings
  const fetchData = useCallback(async () => {
    if (!user) {
      setLoading(false);
      return;
    }

    try {
      // Fetch cycle logs and settings in parallel for faster loading
      const [logsResult, settingsResult] = await Promise.all([
        supabase
          .from("cycle_logs")
          .select("*")
          .eq("user_id", user.id)
          .order("start_date", { ascending: false })
          .limit(12),
        supabase
          .from("user_cycle_settings")
          .select("*")
          .eq("user_id", user.id)
          .maybeSingle(),
      ]);

      if (logsResult.error) throw logsResult.error;
      setCycleLogs((logsResult.data as CycleLog[]) || []);

      const settingsData = settingsResult.data;
      const settingsError = settingsResult.error;
      if (settingsError && settingsError.code !== "PGRST116") throw settingsError;

      if (!settingsData) {
        // Create default settings
        const { data: newSettings, error: createError } = await supabase
          .from("user_cycle_settings")
          .insert({ user_id: user.id })
          .select()
          .single();
        
        if (createError) throw createError;
        setSettings(newSettings as CycleSettings);
      } else {
        setSettings(settingsData as CycleSettings);
      }
    } catch (error) {
      console.error("Error fetching cycle data:", error);
      toast({ title: "Error", description: "Failed to load cycle data", variant: "destructive" });
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Calculate cycle insights
  const insights = useMemo((): CycleInsights => {
    const completedCycles = cycleLogs.filter(log => log.cycle_length && log.cycle_length > 0);
    
    if (completedCycles.length < 2) {
      return {
        averageCycleLength: DEFAULT_CYCLE_LENGTH,
        averagePeriodLength: DEFAULT_PERIOD_LENGTH,
        cycleVariability: 0,
        isRegular: true,
        pcosRiskFlag: false,
        pcosRiskScore: 0,
        stressCorrelation: null,
        sleepCorrelation: null,
        commonSymptoms: [],
        needsDoctorConsultation: false,
        consultationReasons: [],
      };
    }

    // Calculate weighted average (recent cycles have more weight)
    const weights = completedCycles.map((_, i) => Math.pow(0.8, i));
    const totalWeight = weights.reduce((a, b) => a + b, 0);
    
    const weightedCycleLength = completedCycles.reduce((sum, log, i) => 
      sum + (log.cycle_length || DEFAULT_CYCLE_LENGTH) * weights[i], 0
    ) / totalWeight;

    const periodLengths = cycleLogs.filter(log => log.period_length).map(log => log.period_length!);
    const averagePeriodLength = periodLengths.length > 0 
      ? periodLengths.reduce((a, b) => a + b, 0) / periodLengths.length 
      : DEFAULT_PERIOD_LENGTH;

    // Calculate variability (standard deviation)
    const cycleLengths = completedCycles.map(log => log.cycle_length || DEFAULT_CYCLE_LENGTH);
    const mean = cycleLengths.reduce((a, b) => a + b, 0) / cycleLengths.length;
    const variance = cycleLengths.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / cycleLengths.length;
    const cycleVariability = Math.sqrt(variance);

    // Determine regularity
    const isRegular = cycleVariability < 4;

    // PCOS Risk Assessment (non-diagnostic)
    let pcosRiskScore = 0;
    const consultationReasons: string[] = [];

    // Long cycles (>35 days)
    const longCycles = completedCycles.filter(log => (log.cycle_length || 0) > 35).length;
    if (longCycles >= 3) {
      pcosRiskScore += 30;
      consultationReasons.push("Cycles longer than 35 days for 3+ months");
    }

    // High variability
    if (cycleVariability > 7) {
      pcosRiskScore += 20;
      consultationReasons.push("High cycle variability detected");
    }

    // Symptoms check
    const symptomCounts = { acne: 0, hair_growth: 0, hair_loss: 0, fatigue: 0 };
    cycleLogs.forEach(log => {
      if (log.acne && log.acne !== "none") symptomCounts.acne++;
      if (log.fatigue && log.fatigue !== "none") symptomCounts.fatigue++;
    });

    if (symptomCounts.acne >= 3) pcosRiskScore += 15;
    if (symptomCounts.fatigue >= 3) pcosRiskScore += 10;

    // Very short or very long cycles
    const extremeCycles = completedCycles.filter(log => 
      (log.cycle_length || 0) < 21 || (log.cycle_length || 0) > 40
    ).length;
    if (extremeCycles >= 2) {
      consultationReasons.push("Cycle length outside normal range (21-40 days)");
    }

    // Missed periods check
    const sortedLogs = [...cycleLogs].sort((a, b) => 
      new Date(b.start_date).getTime() - new Date(a.start_date).getTime()
    );
    if (sortedLogs.length >= 2) {
      const lastCycle = sortedLogs[0];
      const previousCycle = sortedLogs[1];
      const daysBetween = differenceInDays(
        parseISO(lastCycle.start_date),
        parseISO(previousCycle.start_date)
      );
      if (daysBetween > 60) {
        consultationReasons.push("Missed period detected (gap > 60 days)");
      }
    }

    // Stress correlation analysis
    let stressCorrelation: string | null = null;
    const cyclesWithStress = cycleLogs.filter(log => log.stress_level !== null && log.cycle_length);
    if (cyclesWithStress.length >= 3) {
      const highStressCycles = cyclesWithStress.filter(log => (log.stress_level || 0) >= 4);
      const highStressDelays = highStressCycles.filter(log => 
        (log.cycle_length || DEFAULT_CYCLE_LENGTH) > weightedCycleLength + 3
      );
      if (highStressDelays.length >= 2) {
        stressCorrelation = `High stress appears to delay your period by ~${Math.round((highStressDelays.reduce((sum, log) => sum + ((log.cycle_length || DEFAULT_CYCLE_LENGTH) - weightedCycleLength), 0) / highStressDelays.length))} days`;
      }
    }

    // Sleep correlation analysis
    let sleepCorrelation: string | null = null;
    const cyclesWithSleep = cycleLogs.filter(log => log.sleep_hours !== null && log.cycle_length);
    if (cyclesWithSleep.length >= 3) {
      const lowSleepCycles = cyclesWithSleep.filter(log => (log.sleep_hours || 8) < 6);
      const lowSleepDelays = lowSleepCycles.filter(log => 
        (log.cycle_length || DEFAULT_CYCLE_LENGTH) > weightedCycleLength + 3
      );
      if (lowSleepDelays.length >= 2) {
        sleepCorrelation = "Poor sleep patterns correlate with longer cycles";
      }
    }

    // Common symptoms
    const symptomFrequency: Record<string, number> = {};
    cycleLogs.forEach(log => {
      if (log.cramps && log.cramps !== "none") symptomFrequency["cramps"] = (symptomFrequency["cramps"] || 0) + 1;
      if (log.bloating && log.bloating !== "none") symptomFrequency["bloating"] = (symptomFrequency["bloating"] || 0) + 1;
      if (log.mood && log.mood !== "neutral") symptomFrequency["mood changes"] = (symptomFrequency["mood changes"] || 0) + 1;
      if (log.headache && log.headache !== "none") symptomFrequency["headaches"] = (symptomFrequency["headaches"] || 0) + 1;
      if (log.fatigue && log.fatigue !== "none") symptomFrequency["fatigue"] = (symptomFrequency["fatigue"] || 0) + 1;
    });

    const commonSymptoms = Object.entries(symptomFrequency)
      .filter(([, count]) => count >= 2)
      .sort((a, b) => b[1] - a[1])
      .map(([symptom]) => symptom);

    const needsDoctorConsultation = pcosRiskScore >= 40 || consultationReasons.length >= 2;

    return {
      averageCycleLength: Math.round(weightedCycleLength),
      averagePeriodLength: Math.round(averagePeriodLength),
      cycleVariability: Math.round(cycleVariability * 10) / 10,
      isRegular,
      pcosRiskFlag: pcosRiskScore >= 40,
      pcosRiskScore: Math.min(pcosRiskScore, 100),
      stressCorrelation,
      sleepCorrelation,
      commonSymptoms,
      needsDoctorConsultation,
      consultationReasons,
    };
  }, [cycleLogs]);

  // Calculate prediction
  const prediction = useMemo((): CyclePrediction | null => {
    if (cycleLogs.length === 0) return null;

    const latestCycle = cycleLogs[0];
    const lastStartDate = parseISO(latestCycle.start_date);
    
    if (!isValid(lastStartDate)) return null;

    const cycleLength = insights.averageCycleLength;
    const periodLength = insights.averagePeriodLength;

    const predictedStart = addDays(lastStartDate, cycleLength);
    const predictedEnd = addDays(predictedStart, periodLength - 1);
    const daysUntil = differenceInDays(predictedStart, new Date());

    // Determine confidence level
    let confidenceLevel: "high" | "medium" | "low" = "low";
    const completedCycles = cycleLogs.filter(log => log.cycle_length).length;
    
    if (completedCycles >= 6 && insights.isRegular) {
      confidenceLevel = "high";
    } else if (completedCycles >= 3) {
      confidenceLevel = insights.isRegular ? "high" : "medium";
    }

    return {
      predicted_start_date: format(predictedStart, "yyyy-MM-dd"),
      predicted_end_date: format(predictedEnd, "yyyy-MM-dd"),
      confidence_level: confidenceLevel,
      days_until: daysUntil,
      based_on_cycles: completedCycles,
    };
  }, [cycleLogs, insights]);

  // Get notification schedule
  const getNotificationSchedule = useMemo(() => {
    if (!prediction || !settings?.notification_enabled) return [];

    const schedule: { date: string; message: string; daysBefor: number }[] = [];
    const reminderDays = settings.reminder_days || [3, 2, 1];
    const isIrregular = !insights.isRegular;

    // Use 5-3-1 for irregular, 3-2-1 for regular
    const effectiveDays = isIrregular ? [5, 3, 1] : reminderDays;

    effectiveDays.forEach(days => {
      const notifyDate = subDays(parseISO(prediction.predicted_start_date), days);
      if (notifyDate >= new Date()) {
        const message = days === 1 
          ? "Your period may start tomorrow. This is an estimate 🌸"
          : days === 2
          ? "Your period may start in 2 days. Take care 💗"
          : days === 3
          ? "Your period may start in 3 days. Stay prepared 🩷"
          : `Your period may start in about ${days} days (estimate) 🌸`;
        
        schedule.push({
          date: format(notifyDate, "yyyy-MM-dd"),
          message: isIrregular ? message.replace("may", "might") : message,
          daysBefor: days,
        });
      }
    });

    return schedule;
  }, [prediction, settings, insights]);

  // Log a new period
  const logPeriod = async (startDate: Date, symptoms?: Partial<CycleLog>) => {
    if (!user) return;

    setSaving(true);
    try {
      // Calculate cycle length if there's a previous cycle
      let cycleLength: number | null = null;
      if (cycleLogs.length > 0) {
        const lastCycle = cycleLogs[0];
        cycleLength = differenceInDays(startDate, parseISO(lastCycle.start_date));
      }

      // Build insert data with only valid fields
      const insertData: Record<string, unknown> = {
        user_id: user.id,
        start_date: format(startDate, "yyyy-MM-dd"),
        cycle_length: cycleLength,
      };

      // Add symptom fields if provided (filter out unknown fields)
      const validFields = [
        'flow_intensity', 'cramps', 'mood', 'acne', 'fatigue', 'bloating',
        'headache', 'breast_tenderness', 'stress_level', 'sleep_hours', 'notes'
      ];
      if (symptoms) {
        validFields.forEach(field => {
          if (field in symptoms && symptoms[field as keyof typeof symptoms] !== undefined) {
            insertData[field] = symptoms[field as keyof typeof symptoms];
          }
        });
      }

      const { data, error } = await supabase
        .from("cycle_logs")
        .insert(insertData as never)
        .select()
        .single();

      if (error) throw error;

      setCycleLogs(prev => [data as CycleLog, ...prev]);
      toast({ title: "Period logged", description: "Your cycle has been recorded 🌸" });

      // Update settings with new averages
      await updateCycleSettings();

      return data;
    } catch (error) {
      console.error("Error logging period:", error);
      toast({ title: "Error", description: "Failed to log period", variant: "destructive" });
    } finally {
      setSaving(false);
    }
  };

  // End current period
  const endPeriod = async (endDate: Date) => {
    if (!user || cycleLogs.length === 0) return;

    setSaving(true);
    try {
      const currentCycle = cycleLogs[0];
      const periodLength = differenceInDays(endDate, parseISO(currentCycle.start_date)) + 1;

      const { error } = await supabase
        .from("cycle_logs")
        .update({
          end_date: format(endDate, "yyyy-MM-dd"),
          period_length: periodLength,
        } as never)
        .eq("id", currentCycle.id);

      if (error) throw error;

      setCycleLogs(prev => 
        prev.map(log => 
          log.id === currentCycle.id 
            ? { ...log, end_date: format(endDate, "yyyy-MM-dd"), period_length: periodLength }
            : log
        )
      );

      toast({ title: "Period ended", description: "Your cycle data has been updated" });
    } catch (error) {
      console.error("Error ending period:", error);
      toast({ title: "Error", description: "Failed to update period", variant: "destructive" });
    } finally {
      setSaving(false);
    }
  };

  // Log daily symptoms
  const logSymptoms = async (date: Date, symptoms: Record<string, unknown>) => {
    if (!user) return;

    setSaving(true);
    try {
      // Find or create log for this date
      const dateStr = format(date, "yyyy-MM-dd");
      const existingLog = cycleLogs.find(log => log.start_date === dateStr);

      // Build update/insert data with only valid fields
      const validFields = [
        'flow_intensity', 'cramps', 'mood', 'acne', 'fatigue', 'bloating',
        'headache', 'breast_tenderness', 'stress_level', 'sleep_hours', 'notes'
      ];
      const cleanSymptoms: Record<string, unknown> = {};
      validFields.forEach(field => {
        if (field in symptoms && symptoms[field] !== undefined) {
          cleanSymptoms[field] = symptoms[field];
        }
      });

      if (existingLog) {
        const { error } = await supabase
          .from("cycle_logs")
          .update(cleanSymptoms as never)
          .eq("id", existingLog.id);

        if (error) throw error;

        setCycleLogs(prev => 
          prev.map(log => log.id === existingLog.id ? { ...log, ...cleanSymptoms } as CycleLog : log)
        );
      } else {
        const insertData = {
          user_id: user.id,
          start_date: dateStr,
          ...cleanSymptoms,
        };

        const { data, error } = await supabase
          .from("cycle_logs")
          .insert(insertData as never)
          .select()
          .single();

        if (error) throw error;
        setCycleLogs(prev => [data as CycleLog, ...prev]);
      }

      toast({ title: "Symptoms logged", description: "Your health data has been saved" });
    } catch (error) {
      console.error("Error logging symptoms:", error);
      toast({ title: "Error", description: "Failed to save symptoms", variant: "destructive" });
    } finally {
      setSaving(false);
    }
  };

  // Update cycle settings
  const updateCycleSettings = async () => {
    if (!user) return;

    try {
      const { error } = await supabase
        .from("user_cycle_settings")
        .update({
          average_cycle_length: insights.averageCycleLength,
          average_period_length: insights.averagePeriodLength,
          cycle_variability: insights.cycleVariability,
          pcos_risk_flag: insights.pcosRiskFlag,
          pcos_risk_score: insights.pcosRiskScore,
          last_calculated_at: new Date().toISOString(),
        })
        .eq("user_id", user.id);

      if (error) throw error;
    } catch (error) {
      console.error("Error updating settings:", error);
    }
  };

  // Update notification preferences
  const updateNotificationSettings = async (updates: Partial<CycleSettings>) => {
    if (!user || !settings) return;

    setSaving(true);
    try {
      const { error } = await supabase
        .from("user_cycle_settings")
        .update(updates)
        .eq("user_id", user.id);

      if (error) throw error;

      setSettings(prev => prev ? { ...prev, ...updates } : null);
      toast({ title: "Settings updated", description: "Your preferences have been saved" });
    } catch (error) {
      console.error("Error updating settings:", error);
      toast({ title: "Error", description: "Failed to save settings", variant: "destructive" });
    } finally {
      setSaving(false);
    }
  };

  // Delete all cycle data
  const deleteAllData = async () => {
    if (!user) return;

    setSaving(true);
    try {
      await supabase.from("cycle_logs").delete().eq("user_id", user.id);
      await supabase.from("cycle_predictions").delete().eq("user_id", user.id);
      await supabase.from("user_cycle_settings").delete().eq("user_id", user.id);

      setCycleLogs([]);
      setSettings(null);

      toast({ title: "Data deleted", description: "All your cycle data has been removed" });
    } catch (error) {
      console.error("Error deleting data:", error);
      toast({ title: "Error", description: "Failed to delete data", variant: "destructive" });
    } finally {
      setSaving(false);
    }
  };

  return {
    cycleLogs,
    settings,
    loading,
    saving,
    prediction,
    insights,
    notificationSchedule: getNotificationSchedule,
    logPeriod,
    endPeriod,
    logSymptoms,
    updateNotificationSettings,
    deleteAllData,
    refetch: fetchData,
  };
}
