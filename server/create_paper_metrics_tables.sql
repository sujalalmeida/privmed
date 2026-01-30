-- PrivMed FL Paper Metrics Tables
-- Additional tables for collecting experiment-level and detailed per-round metrics
-- Run this in Supabase Dashboard → SQL Editor after create_required_tables.sql

-- ============================================================================
-- Experiment Log: One row per experiment run
-- ============================================================================
create table if not exists public.fl_experiment_log (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  
  experiment_id text not null unique,
  experiment_name text,
  description text,
  
  -- Model accuracies
  centralized_accuracy double precision,
  federated_accuracy double precision,
  federated_he_accuracy double precision,  -- nullable for future HE implementation
  
  -- Loss metrics
  final_global_loss double precision,
  final_validation_loss double precision,
  
  -- Per-class metrics (stored as JSONB for flexibility)
  -- Format: {"healthy": {"precision": 0.95, "recall": 0.93, "specificity": 0.97, "f1": 0.94, "support": 100}, ...}
  per_class_metrics jsonb,
  
  -- AUC-ROC metrics
  auc_roc_macro double precision,
  auc_roc_per_class jsonb,  -- Format: {"healthy": 0.95, "diabetes": 0.92, ...}
  
  -- Metadata
  total_rounds int,
  num_clients int,
  model_type text,
  random_seed int default 42,
  notes text
);

-- Index for fast lookup by experiment_id
create index if not exists idx_fl_experiment_log_experiment_id 
  on public.fl_experiment_log(experiment_id);

-- ============================================================================
-- Round Detailed Metrics: One row per (round, lab) + one row per round for global
-- ============================================================================
create table if not exists public.fl_round_detailed_metrics (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  
  experiment_id text,  -- links to fl_experiment_log
  round int not null,
  
  -- Lab identification (null for global aggregated metrics)
  lab_label text,
  is_global boolean not null default false,
  
  -- Accuracy metrics
  global_accuracy double precision,
  local_accuracy double precision,
  
  -- Loss metrics
  global_train_loss double precision,
  global_val_loss double precision,
  local_train_loss double precision,
  local_val_loss double precision,
  
  -- Training details
  num_examples int,
  grad_norm double precision,
  weight_update_magnitude double precision,
  
  -- Per-class metrics for this round/lab (optional, mainly for global)
  per_class_metrics jsonb,
  
  -- Metadata
  training_time_seconds double precision,
  aggregation_method text default 'FedAvg'
);

-- Indexes for efficient querying
create index if not exists idx_fl_round_detailed_metrics_experiment 
  on public.fl_round_detailed_metrics(experiment_id);
create index if not exists idx_fl_round_detailed_metrics_round 
  on public.fl_round_detailed_metrics(round);
create index if not exists idx_fl_round_detailed_metrics_lab 
  on public.fl_round_detailed_metrics(lab_label);
create index if not exists idx_fl_round_detailed_metrics_exp_round 
  on public.fl_round_detailed_metrics(experiment_id, round);

-- ============================================================================
-- Centralized Baseline Metrics: Store baseline model performance
-- ============================================================================
create table if not exists public.fl_centralized_baselines (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  
  experiment_id text,
  model_name text not null,  -- e.g., "Gradient Boosting", "Logistic Regression"
  
  -- Performance metrics
  accuracy double precision not null,
  loss double precision,
  
  -- Per-class metrics
  per_class_metrics jsonb not null,
  
  -- AUC-ROC
  auc_roc_macro double precision,
  auc_roc_per_class jsonb,
  
  -- Training details
  training_samples int,
  validation_samples int,
  test_samples int,
  training_time_seconds double precision,
  
  -- Model configuration
  model_config jsonb,  -- hyperparameters, etc.
  
  -- Storage
  model_path text
);

-- Index for lookup
create index if not exists idx_fl_centralized_baselines_experiment 
  on public.fl_centralized_baselines(experiment_id);

-- ============================================================================
-- Per-Class Performance Tracking: Detailed class-level metrics over time
-- ============================================================================
create table if not exists public.fl_per_class_performance (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  
  experiment_id text,
  round int,
  model_type text not null,  -- 'centralized', 'federated', 'federated_he'
  lab_label text,  -- null for global metrics
  
  class_name text not null,  -- 'healthy', 'diabetes', 'hypertension', 'heart_disease'
  
  -- Metrics
  precision double precision,
  recall double precision,  -- sensitivity
  specificity double precision,
  f1_score double precision,
  support int,  -- number of samples in this class
  
  -- ROC metrics
  auc_roc double precision,
  
  -- Confusion matrix elements (one-vs-rest)
  true_positives int,
  false_positives int,
  true_negatives int,
  false_negatives int
);

-- Indexes
create index if not exists idx_fl_per_class_performance_experiment 
  on public.fl_per_class_performance(experiment_id);
create index if not exists idx_fl_per_class_performance_round 
  on public.fl_per_class_performance(round);
create index if not exists idx_fl_per_class_performance_class 
  on public.fl_per_class_performance(class_name);

-- ============================================================================
-- Lab Data Imbalance Tracking: Samples per lab per round
-- ============================================================================
create table if not exists public.fl_lab_data_distribution (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  
  experiment_id text,
  round int,
  lab_label text not null,
  
  -- Sample counts
  total_samples int not null,
  samples_per_class jsonb,  -- {"healthy": 100, "diabetes": 50, ...}
  
  -- Data quality metrics
  missing_values_pct double precision,
  duplicate_records int default 0
);

-- Indexes
create index if not exists idx_fl_lab_data_distribution_experiment 
  on public.fl_lab_data_distribution(experiment_id);
create index if not exists idx_fl_lab_data_distribution_lab 
  on public.fl_lab_data_distribution(lab_label);

-- ============================================================================
-- Model Predictions Storage: For ROC curve generation
-- ============================================================================
create table if not exists public.fl_model_predictions (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  
  experiment_id text,
  round int,
  model_type text not null,  -- 'centralized', 'federated', 'federated_he'
  
  -- Prediction data (stored as JSONB for efficient storage)
  -- Format: [{"true_label": 0, "pred_label": 0, "proba": [0.9, 0.05, 0.03, 0.02]}, ...]
  predictions jsonb not null,
  
  -- Summary stats
  total_predictions int,
  dataset_type text default 'test'  -- 'train', 'val', 'test'
);

-- Indexes
create index if not exists idx_fl_model_predictions_experiment 
  on public.fl_model_predictions(experiment_id);

-- ============================================================================
-- Row Level Security
-- ============================================================================
alter table public.fl_experiment_log enable row level security;
alter table public.fl_round_detailed_metrics enable row level security;
alter table public.fl_centralized_baselines enable row level security;
alter table public.fl_per_class_performance enable row level security;
alter table public.fl_lab_data_distribution enable row level security;
alter table public.fl_model_predictions enable row level security;

-- Allow public read access for analysis (adjust as needed)
drop policy if exists "public read fl_experiment_log" on public.fl_experiment_log;
create policy "public read fl_experiment_log"
on public.fl_experiment_log
for select
using (true);

drop policy if exists "public read fl_round_detailed_metrics" on public.fl_round_detailed_metrics;
create policy "public read fl_round_detailed_metrics"
on public.fl_round_detailed_metrics
for select
using (true);

drop policy if exists "public read fl_centralized_baselines" on public.fl_centralized_baselines;
create policy "public read fl_centralized_baselines"
on public.fl_centralized_baselines
for select
using (true);

drop policy if exists "public read fl_per_class_performance" on public.fl_per_class_performance;
create policy "public read fl_per_class_performance"
on public.fl_per_class_performance
for select
using (true);

drop policy if exists "public read fl_lab_data_distribution" on public.fl_lab_data_distribution;
create policy "public read fl_lab_data_distribution"
on public.fl_lab_data_distribution
for select
using (true);

drop policy if exists "public read fl_model_predictions" on public.fl_model_predictions;
create policy "public read fl_model_predictions"
on public.fl_model_predictions
for select
using (true);

-- ============================================================================
-- Comments for documentation
-- ============================================================================
comment on table public.fl_experiment_log is 
  'Stores experiment-level summary metrics for federated learning paper analysis';

comment on table public.fl_round_detailed_metrics is 
  'Stores per-round and per-lab detailed metrics including loss and accuracy';

comment on table public.fl_centralized_baselines is 
  'Stores centralized baseline model performance for comparison';

comment on table public.fl_per_class_performance is 
  'Tracks per-class precision, recall, specificity, F1, and AUC-ROC over rounds';

comment on table public.fl_lab_data_distribution is 
  'Tracks data imbalance across labs for analysis and visualization';

comment on table public.fl_model_predictions is 
  'Stores raw predictions and probabilities for ROC curve generation';
