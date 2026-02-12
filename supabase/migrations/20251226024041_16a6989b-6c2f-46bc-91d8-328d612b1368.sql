-- Create enum for error severity
CREATE TYPE public.error_severity AS ENUM ('info', 'warning', 'error', 'critical');

-- Create enum for error status
CREATE TYPE public.error_status AS ENUM ('detected', 'analyzed', 'fix_applied', 'verified', 'closed');

-- Create enum for service status
CREATE TYPE public.service_status AS ENUM ('healthy', 'warning', 'down');

-- System Errors table for centralized error management
CREATE TABLE public.system_errors (
    id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
    error_message TEXT NOT NULL,
    severity error_severity NOT NULL DEFAULT 'error',
    affected_module TEXT,
    affected_page TEXT,
    status error_status NOT NULL DEFAULT 'detected',
    root_cause TEXT,
    fix_applied TEXT,
    users_affected INTEGER DEFAULT 0,
    retry_attempts INTEGER DEFAULT 0,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_time_minutes INTEGER,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Feature Flags table
CREATE TABLE public.feature_flags (
    id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
    feature_name TEXT NOT NULL UNIQUE,
    feature_key TEXT NOT NULL UNIQUE,
    description TEXT,
    is_enabled BOOLEAN NOT NULL DEFAULT true,
    last_toggled_by UUID,
    last_toggled_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- System Health Checks table
CREATE TABLE public.system_health_checks (
    id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
    service_name TEXT NOT NULL,
    service_key TEXT NOT NULL UNIQUE,
    status service_status NOT NULL DEFAULT 'healthy',
    response_time_ms INTEGER,
    last_successful_check TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Admin Action Logs table
CREATE TABLE public.admin_logs (
    id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
    admin_id UUID NOT NULL,
    admin_email TEXT NOT NULL,
    action_type TEXT NOT NULL,
    action_description TEXT NOT NULL,
    affected_resource TEXT,
    affected_resource_id UUID,
    metadata JSONB,
    ip_address TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- System Alerts table
CREATE TABLE public.system_alerts (
    id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
    alert_type TEXT NOT NULL,
    message TEXT NOT NULL,
    severity error_severity NOT NULL DEFAULT 'warning',
    is_acknowledged BOOLEAN NOT NULL DEFAULT false,
    acknowledged_by UUID,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS on all tables
ALTER TABLE public.system_errors ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.feature_flags ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.system_health_checks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.admin_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.system_alerts ENABLE ROW LEVEL SECURITY;

-- RLS Policies - Only admins can access these tables
CREATE POLICY "Admins can view system errors" ON public.system_errors
FOR SELECT USING (has_role(auth.uid(), 'admin'));

CREATE POLICY "Admins can insert system errors" ON public.system_errors
FOR INSERT WITH CHECK (has_role(auth.uid(), 'admin'));

CREATE POLICY "Admins can update system errors" ON public.system_errors
FOR UPDATE USING (has_role(auth.uid(), 'admin'));

CREATE POLICY "Admins can delete system errors" ON public.system_errors
FOR DELETE USING (has_role(auth.uid(), 'admin'));

CREATE POLICY "Admins can view feature flags" ON public.feature_flags
FOR SELECT USING (has_role(auth.uid(), 'admin'));

CREATE POLICY "Admins can insert feature flags" ON public.feature_flags
FOR INSERT WITH CHECK (has_role(auth.uid(), 'admin'));

CREATE POLICY "Admins can update feature flags" ON public.feature_flags
FOR UPDATE USING (has_role(auth.uid(), 'admin'));

CREATE POLICY "Admins can view system health" ON public.system_health_checks
FOR SELECT USING (has_role(auth.uid(), 'admin'));

CREATE POLICY "Admins can manage system health" ON public.system_health_checks
FOR ALL USING (has_role(auth.uid(), 'admin'));

CREATE POLICY "Admins can view admin logs" ON public.admin_logs
FOR SELECT USING (has_role(auth.uid(), 'admin'));

CREATE POLICY "Admins can insert admin logs" ON public.admin_logs
FOR INSERT WITH CHECK (has_role(auth.uid(), 'admin'));

CREATE POLICY "Admins can view system alerts" ON public.system_alerts
FOR SELECT USING (has_role(auth.uid(), 'admin'));

CREATE POLICY "Admins can manage system alerts" ON public.system_alerts
FOR ALL USING (has_role(auth.uid(), 'admin'));

-- Insert default feature flags
INSERT INTO public.feature_flags (feature_name, feature_key, description, is_enabled) VALUES
('Doctor Map', 'doctor_map', 'Find nearby doctors and gynecologists', true),
('AI Chatbot', 'ai_chatbot', 'AI-powered health guidance chatbot', true),
('PCOS Prediction', 'pcos_prediction', 'PCOS risk assessment module', true),
('Menstrual Tracking', 'menstrual_tracking', 'Period and cycle tracking', true),
('Menopause Assessment', 'menopause_assessment', 'Menopause stage assessment', true),
('NGO Directory', 'ngo_directory', 'NGO and support organizations listing', true),
('Government Schemes', 'govt_schemes', 'Government health schemes information', true);

-- Insert default system health checks
INSERT INTO public.system_health_checks (service_name, service_key, status, response_time_ms, last_successful_check) VALUES
('Authentication Service', 'auth_service', 'healthy', 45, now()),
('Database', 'database', 'healthy', 12, now()),
('Google Maps API', 'maps_api', 'healthy', 120, now()),
('AI Service', 'ai_service', 'healthy', 200, now());

-- Triggers for updated_at
CREATE TRIGGER update_system_errors_updated_at
BEFORE UPDATE ON public.system_errors
FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_feature_flags_updated_at
BEFORE UPDATE ON public.feature_flags
FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_system_health_updated_at
BEFORE UPDATE ON public.system_health_checks
FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();