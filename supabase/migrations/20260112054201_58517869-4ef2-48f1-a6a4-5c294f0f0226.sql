-- Drop the trigger first (correct trigger name)
DROP TRIGGER IF EXISTS trigger_create_alert ON public.system_errors;

-- Drop the trigger function with CASCADE
DROP FUNCTION IF EXISTS public.create_alert_on_error() CASCADE;

-- Now drop the unused tables
DROP TABLE IF EXISTS public.admin_logs CASCADE;
DROP TABLE IF EXISTS public.system_errors CASCADE;
DROP TABLE IF EXISTS public.system_alerts CASCADE;
DROP TABLE IF EXISTS public.system_health_checks CASCADE;
DROP TABLE IF EXISTS public.feature_flags CASCADE;