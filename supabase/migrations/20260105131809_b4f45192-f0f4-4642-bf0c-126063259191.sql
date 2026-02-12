-- Add new columns to cycle_logs for enhanced tracking
ALTER TABLE public.cycle_logs 
ADD COLUMN IF NOT EXISTS flow_intensity TEXT CHECK (flow_intensity IN ('light', 'moderate', 'heavy', 'very_heavy')),
ADD COLUMN IF NOT EXISTS cramps TEXT CHECK (cramps IN ('none', 'mild', 'moderate', 'severe')),
ADD COLUMN IF NOT EXISTS mood TEXT CHECK (mood IN ('happy', 'neutral', 'sad', 'irritable', 'anxious')),
ADD COLUMN IF NOT EXISTS acne TEXT CHECK (acne IN ('none', 'mild', 'moderate', 'severe')),
ADD COLUMN IF NOT EXISTS hair_growth TEXT CHECK (hair_growth IN ('none', 'mild', 'noticeable', 'excessive')),
ADD COLUMN IF NOT EXISTS hair_loss TEXT CHECK (hair_loss IN ('none', 'mild', 'moderate', 'severe')),
ADD COLUMN IF NOT EXISTS fatigue TEXT CHECK (fatigue IN ('none', 'mild', 'moderate', 'severe')),
ADD COLUMN IF NOT EXISTS bloating TEXT CHECK (bloating IN ('none', 'mild', 'moderate', 'severe')),
ADD COLUMN IF NOT EXISTS stress_level INTEGER CHECK (stress_level >= 1 AND stress_level <= 5),
ADD COLUMN IF NOT EXISTS sleep_hours NUMERIC(3,1) CHECK (sleep_hours >= 0 AND sleep_hours <= 24),
ADD COLUMN IF NOT EXISTS physical_activity TEXT CHECK (physical_activity IN ('none', 'light', 'moderate', 'intense')),
ADD COLUMN IF NOT EXISTS is_period_start BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS is_period_end BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS predicted_start DATE,
ADD COLUMN IF NOT EXISTS actual_vs_predicted INTEGER,
ADD COLUMN IF NOT EXISTS headache TEXT CHECK (headache IN ('none', 'mild', 'moderate', 'severe')),
ADD COLUMN IF NOT EXISTS breast_tenderness TEXT CHECK (breast_tenderness IN ('none', 'mild', 'moderate', 'severe'));

-- Create user_cycle_settings table for preferences
CREATE TABLE IF NOT EXISTS public.user_cycle_settings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE UNIQUE,
  timezone TEXT DEFAULT 'UTC',
  notification_enabled BOOLEAN DEFAULT true,
  reminder_days INTEGER[] DEFAULT ARRAY[3, 2, 1],
  notification_time TEXT DEFAULT 'morning',
  hide_notification_text BOOLEAN DEFAULT true,
  allow_advanced_analysis BOOLEAN DEFAULT true,
  average_cycle_length INTEGER,
  average_period_length INTEGER,
  cycle_variability NUMERIC(4,2),
  last_calculated_at TIMESTAMP WITH TIME ZONE,
  pcos_risk_flag BOOLEAN DEFAULT false,
  pcos_risk_score NUMERIC(5,2),
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS on user_cycle_settings
ALTER TABLE public.user_cycle_settings ENABLE ROW LEVEL SECURITY;

-- RLS policies for user_cycle_settings
CREATE POLICY "Users can view their own cycle settings"
ON public.user_cycle_settings
FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own cycle settings"
ON public.user_cycle_settings
FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own cycle settings"
ON public.user_cycle_settings
FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own cycle settings"
ON public.user_cycle_settings
FOR DELETE
USING (auth.uid() = user_id);

-- Create cycle_predictions table for storing predictions
CREATE TABLE IF NOT EXISTS public.cycle_predictions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  predicted_start_date DATE NOT NULL,
  predicted_end_date DATE,
  confidence_level TEXT CHECK (confidence_level IN ('high', 'medium', 'low')),
  prediction_method TEXT,
  based_on_cycles INTEGER,
  cycle_length_used INTEGER,
  is_active BOOLEAN DEFAULT true,
  was_accurate BOOLEAN,
  actual_start_date DATE,
  deviation_days INTEGER,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS on cycle_predictions
ALTER TABLE public.cycle_predictions ENABLE ROW LEVEL SECURITY;

-- RLS policies for cycle_predictions
CREATE POLICY "Users can view their own predictions"
ON public.cycle_predictions
FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own predictions"
ON public.cycle_predictions
FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own predictions"
ON public.cycle_predictions
FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own predictions"
ON public.cycle_predictions
FOR DELETE
USING (auth.uid() = user_id);

-- Create function to auto-update updated_at
CREATE OR REPLACE FUNCTION public.update_cycle_settings_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = public;

-- Create trigger for user_cycle_settings
DROP TRIGGER IF EXISTS update_user_cycle_settings_updated_at ON public.user_cycle_settings;
CREATE TRIGGER update_user_cycle_settings_updated_at
BEFORE UPDATE ON public.user_cycle_settings
FOR EACH ROW
EXECUTE FUNCTION public.update_cycle_settings_updated_at();