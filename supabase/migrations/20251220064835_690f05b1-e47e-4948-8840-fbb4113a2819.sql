-- Create health_resources table
CREATE TABLE public.health_resources (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  title TEXT NOT NULL,
  category TEXT NOT NULL CHECK (category IN ('Menstrual', 'PCOS', 'Menopause', 'General Wellness')),
  description TEXT,
  external_link TEXT,
  status TEXT NOT NULL DEFAULT 'Draft' CHECK (status IN ('Published', 'Draft')),
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.health_resources ENABLE ROW LEVEL SECURITY;

-- Policy: Anyone can view active/published resources
CREATE POLICY "Anyone can view active published resources"
ON public.health_resources
FOR SELECT
USING (is_active = true AND status = 'Published');

-- Policy: Admins can view all resources
CREATE POLICY "Admins can view all resources"
ON public.health_resources
FOR SELECT
USING (public.has_role(auth.uid(), 'admin'));

-- Policy: Admins can insert resources
CREATE POLICY "Admins can insert resources"
ON public.health_resources
FOR INSERT
WITH CHECK (public.has_role(auth.uid(), 'admin'));

-- Policy: Admins can update resources
CREATE POLICY "Admins can update resources"
ON public.health_resources
FOR UPDATE
USING (public.has_role(auth.uid(), 'admin'));

-- Policy: Admins can delete resources
CREATE POLICY "Admins can delete resources"
ON public.health_resources
FOR DELETE
USING (public.has_role(auth.uid(), 'admin'));

-- Add updated_at trigger
CREATE TRIGGER update_health_resources_updated_at
BEFORE UPDATE ON public.health_resources
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();