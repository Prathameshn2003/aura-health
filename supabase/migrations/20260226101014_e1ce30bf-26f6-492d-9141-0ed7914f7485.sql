
-- Create medical-documents storage bucket (private)
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES ('medical-documents', 'medical-documents', false, 10485760, ARRAY['application/pdf', 'image/jpeg', 'image/png']);

-- Storage RLS policies
CREATE POLICY "Users can upload their own medical documents"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'medical-documents' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can view their own medical documents"
ON storage.objects FOR SELECT
USING (bucket_id = 'medical-documents' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can delete their own medical documents"
ON storage.objects FOR DELETE
USING (bucket_id = 'medical-documents' AND auth.uid()::text = (storage.foldername(name))[1]);

-- Create documents metadata table
CREATE TABLE public.documents (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL,
  file_name TEXT NOT NULL,
  file_path TEXT NOT NULL,
  file_type TEXT NOT NULL,
  file_size INTEGER NOT NULL,
  uploaded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

-- RLS policies
CREATE POLICY "Users can insert their own documents"
ON public.documents FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view their own documents"
ON public.documents FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own documents"
ON public.documents FOR DELETE
USING (auth.uid() = user_id);
