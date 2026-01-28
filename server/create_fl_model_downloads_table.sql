-- SQL Script to create fl_model_downloads table
-- This table tracks which labs have downloaded which global model versions

CREATE TABLE IF NOT EXISTS fl_model_downloads (
    id BIGSERIAL PRIMARY KEY,
    lab_label TEXT NOT NULL,
    global_model_version INTEGER NOT NULL,
    downloaded_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Optional: Track performance improvement after download
    accuracy_before_download NUMERIC,
    accuracy_after_download NUMERIC,
    improvement_percentage NUMERIC,
    
    -- Prevent duplicate downloads from being tracked
    UNIQUE(lab_label, global_model_version)
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_fl_model_downloads_lab ON fl_model_downloads(lab_label);
CREATE INDEX IF NOT EXISTS idx_fl_model_downloads_version ON fl_model_downloads(global_model_version);

-- Add comment for documentation
COMMENT ON TABLE fl_model_downloads IS 'Tracks which labs downloaded which global model versions and their performance improvements';

-- Grant permissions (adjust based on your setup)
-- ALTER TABLE fl_model_downloads ENABLE ROW LEVEL SECURITY;
