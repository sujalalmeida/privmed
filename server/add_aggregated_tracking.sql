-- Add column to track which round an update was aggregated in
-- Run this in Supabase Dashboard → SQL Editor

-- Add aggregated_in_round column (NULL means not yet aggregated)
ALTER TABLE public.fl_client_updates 
ADD COLUMN IF NOT EXISTS aggregated_in_round INT DEFAULT NULL;

-- Add index for faster queries on unaggregated updates
CREATE INDEX IF NOT EXISTS idx_fl_client_updates_unaggregated 
ON public.fl_client_updates(aggregated_in_round) 
WHERE aggregated_in_round IS NULL;

-- Verify the column was added
SELECT column_name, data_type, is_nullable
FROM information_schema.columns 
WHERE table_name = 'fl_client_updates' AND table_schema = 'public'
ORDER BY ordinal_position;
