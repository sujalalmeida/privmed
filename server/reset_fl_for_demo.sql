-- ============================================================================
-- RESET FL STATE FOR DEMO VIDEO
-- Run this in Supabase Dashboard → SQL Editor
-- This will clear all federated learning data to start fresh for a demo
-- ============================================================================

-- 1. Delete all FL client updates (local model submissions from labs)
DELETE FROM public.fl_client_updates;

-- 2. Delete all FL global models (aggregated models)
DELETE FROM public.fl_global_models;

-- 3. Delete all FL round metrics
DELETE FROM public.fl_round_metrics;

-- 4. Delete all FL runs
DELETE FROM public.fl_runs;

-- 5. Delete all FL round history (only if table exists)
DO $$ 
BEGIN
  IF EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'fl_round_history') THEN
    DELETE FROM public.fl_round_history;
  END IF;
END $$;

-- 6. Delete all FL model downloads tracking (only if table exists)
DO $$ 
BEGIN
  IF EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'fl_model_downloads') THEN
    DELETE FROM public.fl_model_downloads;
  END IF;
END $$;

-- 7. Delete FL test patient records used by A/B or evaluation dashboards (only if table exists)
DO $$
BEGIN
  IF EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'fl_test_patient_records') THEN
    DELETE FROM public.fl_test_patient_records;
  END IF;
END $$;

-- 8. (Optional) Reset patient records if you want fresh prediction history too
-- Uncomment the next line if you want to clear patient data too
-- DELETE FROM public.patient_records;

-- 9. Clear privacy-preserving global report demo data if those tables exist
DO $$
BEGIN
  IF EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'patient_reports_global') THEN
    DELETE FROM public.patient_reports_global;
  END IF;
  IF EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'patient_reports_global_audit') THEN
    DELETE FROM public.patient_reports_global_audit;
  END IF;
  IF EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'doctor_feedback_global_reports') THEN
    DELETE FROM public.doctor_feedback_global_reports;
  END IF;
END $$;

-- 10. Verify cleanup - show counts
SELECT 'fl_client_updates' as table_name, COUNT(*) as row_count FROM public.fl_client_updates
UNION ALL
SELECT 'fl_global_models', COUNT(*) FROM public.fl_global_models
UNION ALL
SELECT 'fl_round_metrics', COUNT(*) FROM public.fl_round_metrics
UNION ALL
SELECT 'fl_runs', COUNT(*) FROM public.fl_runs
UNION ALL
SELECT 'fl_model_downloads', COUNT(*) FROM public.fl_model_downloads
UNION ALL
SELECT 'patient_reports_global',
  CASE
    WHEN EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'patient_reports_global')
      THEN (SELECT COUNT(*) FROM public.patient_reports_global)
    ELSE 0
  END
UNION ALL
SELECT 'patient_reports_global_audit',
  CASE
    WHEN EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'patient_reports_global_audit')
      THEN (SELECT COUNT(*) FROM public.patient_reports_global_audit)
    ELSE 0
  END
UNION ALL
SELECT 'doctor_feedback_global_reports',
  CASE
    WHEN EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'doctor_feedback_global_reports')
      THEN (SELECT COUNT(*) FROM public.doctor_feedback_global_reports)
    ELSE 0
  END;

-- 11. Confirm success
SELECT 'FL and encrypted report data reset complete! Ready for demo.' as status;
