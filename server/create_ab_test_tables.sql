-- ============================================================================
-- Push Global Model to Labs & A/B Test Dashboard Tables
-- Run this in your Supabase SQL Editor
-- ============================================================================

-- ============================================================================
-- FEATURE 1: Push Global Model to Labs
-- ============================================================================

-- Table to track model download history (required by existing code)
CREATE TABLE IF NOT EXISTS public.fl_model_downloads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    lab_label TEXT NOT NULL,
    global_model_version INT NOT NULL,
    downloaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    storage_path TEXT,
    accuracy_before_download FLOAT,
    accuracy_after_download FLOAT,
    improvement FLOAT
);

-- Index for fl_model_downloads
CREATE INDEX IF NOT EXISTS idx_fl_model_downloads_lab ON public.fl_model_downloads(lab_label);
CREATE INDEX IF NOT EXISTS idx_fl_model_downloads_version ON public.fl_model_downloads(global_model_version);

-- RLS for fl_model_downloads
ALTER TABLE public.fl_model_downloads ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Service role has full access to fl_model_downloads" ON public.fl_model_downloads
    FOR ALL USING (true) WITH CHECK (true);

-- Table to track model broadcast events initiated by admin
CREATE TABLE IF NOT EXISTS public.fl_model_broadcasts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    global_model_version INT NOT NULL,
    initiated_by TEXT,  -- admin user who pushed
    status TEXT DEFAULT 'pending',  -- pending, in_progress, completed
    labs_notified INT DEFAULT 0,
    labs_downloaded INT DEFAULT 0
);

-- Table to track per-lab sync status for each broadcast
CREATE TABLE IF NOT EXISTS public.fl_lab_sync_status (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    broadcast_id UUID REFERENCES public.fl_model_broadcasts(id) ON DELETE CASCADE,
    lab_label TEXT NOT NULL,
    notified_at TIMESTAMPTZ,
    downloaded_at TIMESTAMPTZ,
    auto_sync_enabled BOOLEAN DEFAULT false,
    status TEXT DEFAULT 'pending'  -- pending, notified, downloaded, failed
);

-- Table to store lab auto-sync preferences
CREATE TABLE IF NOT EXISTS public.fl_lab_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lab_label TEXT UNIQUE NOT NULL,
    auto_sync_enabled BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================================
-- FEATURE 2: A/B Test Dashboard
-- ============================================================================

-- Table to store A/B test configurations and results
CREATE TABLE IF NOT EXISTS public.fl_ab_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    test_name TEXT,
    model_a_type TEXT,  -- 'local' or 'global'
    model_a_version TEXT,  -- lab label or global version
    model_b_type TEXT,
    model_b_version TEXT,
    test_dataset_path TEXT,
    num_samples INT,
    model_a_accuracy FLOAT,
    model_b_accuracy FLOAT,
    accuracy_delta FLOAT,
    model_a_predictions JSONB,  -- [{patient_id, predicted, actual, confidence}, ...]
    model_b_predictions JSONB,
    confusion_matrix_a JSONB,
    confusion_matrix_b JSONB,
    per_class_metrics_a JSONB,  -- {class: {precision, recall, f1}}
    per_class_metrics_b JSONB,
    statistical_significance JSONB,  -- {p_value, is_significant, test_used}
    status TEXT DEFAULT 'pending'  -- pending, running, completed, failed
);

-- Table for held-out test dataset management
CREATE TABLE IF NOT EXISTS public.fl_test_patient_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    original_record_id UUID,  -- Reference to original patient_records
    lab_label TEXT,
    patient_data JSONB NOT NULL,  -- Full patient data for testing
    actual_diagnosis TEXT,  -- Ground truth label
    is_active BOOLEAN DEFAULT true  -- Can be deactivated without deletion
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_fl_model_broadcasts_status ON public.fl_model_broadcasts(status);
CREATE INDEX IF NOT EXISTS idx_fl_model_broadcasts_created_at ON public.fl_model_broadcasts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_lab_sync_status_broadcast_id ON public.fl_lab_sync_status(broadcast_id);
CREATE INDEX IF NOT EXISTS idx_fl_lab_sync_status_lab_label ON public.fl_lab_sync_status(lab_label);
CREATE INDEX IF NOT EXISTS idx_fl_ab_tests_status ON public.fl_ab_tests(status);
CREATE INDEX IF NOT EXISTS idx_fl_ab_tests_created_at ON public.fl_ab_tests(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_test_patient_records_lab ON public.fl_test_patient_records(lab_label);

-- RLS Policies (optional - adjust based on your auth setup)
ALTER TABLE public.fl_model_broadcasts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fl_lab_sync_status ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fl_lab_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fl_ab_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fl_test_patient_records ENABLE ROW LEVEL SECURITY;

-- Allow service role full access
CREATE POLICY "Service role has full access to fl_model_broadcasts" ON public.fl_model_broadcasts
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service role has full access to fl_lab_sync_status" ON public.fl_lab_sync_status
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service role has full access to fl_lab_preferences" ON public.fl_lab_preferences
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service role has full access to fl_ab_tests" ON public.fl_ab_tests
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service role has full access to fl_test_patient_records" ON public.fl_test_patient_records
    FOR ALL USING (true) WITH CHECK (true);

-- ============================================================================
-- Helpful views for analytics
-- ============================================================================

-- View to get broadcast summary with lab counts
CREATE OR REPLACE VIEW public.v_broadcast_summary AS
SELECT 
    b.id,
    b.created_at,
    b.global_model_version,
    b.initiated_by,
    b.status,
    COUNT(s.id) as total_labs,
    COUNT(CASE WHEN s.status = 'notified' OR s.status = 'downloaded' THEN 1 END) as labs_notified,
    COUNT(CASE WHEN s.status = 'downloaded' THEN 1 END) as labs_downloaded
FROM public.fl_model_broadcasts b
LEFT JOIN public.fl_lab_sync_status s ON b.id = s.broadcast_id
GROUP BY b.id, b.created_at, b.global_model_version, b.initiated_by, b.status;

-- View to get A/B test summary
CREATE OR REPLACE VIEW public.v_ab_test_summary AS
SELECT 
    id,
    created_at,
    test_name,
    model_a_type,
    model_a_version,
    model_b_type,
    model_b_version,
    num_samples,
    model_a_accuracy,
    model_b_accuracy,
    accuracy_delta,
    CASE 
        WHEN model_b_accuracy > model_a_accuracy THEN 'Model B'
        WHEN model_a_accuracy > model_b_accuracy THEN 'Model A'
        ELSE 'Tie'
    END as winner,
    status
FROM public.fl_ab_tests;
