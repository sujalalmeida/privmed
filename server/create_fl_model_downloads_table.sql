-- SQL Script to create/update fl_model_downloads table
-- This table tracks which labs have downloaded which global model versions
-- Run this in Supabase SQL Editor

-- Drop and recreate table to ensure correct schema
DROP TABLE IF EXISTS public.fl_model_downloads;

CREATE TABLE public.fl_model_downloads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    lab_label TEXT NOT NULL,
    global_model_version INTEGER NOT NULL,
    downloaded_at TIMESTAMPTZ DEFAULT NOW(),
    storage_path TEXT,
    
    -- Track performance improvement after download
    accuracy_before_download FLOAT,
    accuracy_after_download FLOAT,
    improvement FLOAT,
    improvement_percentage FLOAT,
    
    -- Node accuracy: the lab's current model accuracy on shared test set
    -- This is the single accuracy number for this lab after downloading global model
    node_accuracy FLOAT,
    
    -- Prevent duplicate downloads from being tracked
    UNIQUE(lab_label, global_model_version)
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_fl_model_downloads_lab ON public.fl_model_downloads(lab_label);
CREATE INDEX IF NOT EXISTS idx_fl_model_downloads_version ON public.fl_model_downloads(global_model_version);

-- Enable RLS
ALTER TABLE public.fl_model_downloads ENABLE ROW LEVEL SECURITY;

-- Create policy for service role access
CREATE POLICY "Service role has full access to fl_model_downloads" ON public.fl_model_downloads
    FOR ALL USING (true) WITH CHECK (true);

-- Add comment for documentation
COMMENT ON TABLE public.fl_model_downloads IS 'Tracks which labs downloaded which global model versions, their node accuracy, and performance improvements';
COMMENT ON COLUMN public.fl_model_downloads.node_accuracy IS 'Lab node accuracy after downloading global model, evaluated on shared test set';
