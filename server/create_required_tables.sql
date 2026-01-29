-- MedSafe required tables for backend + FL metadata
-- Run this in Supabase Dashboard → SQL Editor.

create extension if not exists pgcrypto;

-- ============================================================================
-- Drop old patient_records table if it exists (schema update)
-- ============================================================================
drop table if exists public.patient_records cascade;

-- ============================================================================
-- Create new patient_records with clinical schema
-- ============================================================================
create table public.patient_records (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  lab_label text not null,
  
  -- Demographics
  patient_id text,
  age int not null,
  sex text not null default 'M',
  height_cm float,
  weight_kg float,
  bmi float,
  
  -- Vital Signs
  systolic_bp int,
  diastolic_bp int,
  heart_rate int,
  
  -- Blood Chemistry - Glucose Metabolism
  fasting_glucose int,
  hba1c float,
  insulin float,
  
  -- Blood Chemistry - Lipid Panel
  total_cholesterol int,
  ldl_cholesterol int,
  hdl_cholesterol int,
  triglycerides int,
  
  -- Cardiac Assessment
  chest_pain_type int,     -- 1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic
  resting_ecg int,         -- 0=normal, 1=ST-T abnormality, 2=LVH
  max_heart_rate int,
  exercise_angina int,     -- 0=no, 1=yes
  st_depression float,
  st_slope int,            -- 1=upsloping, 2=flat, 3=downsloping
  
  -- Medical History (binary flags)
  smoking_status int,      -- 0=never, 1=former, 2=current
  family_history_cvd int,
  family_history_diabetes int,
  prior_hypertension int,
  prior_diabetes int,
  prior_heart_disease int,
  
  -- Medications (binary flags)
  on_bp_medication int,
  on_diabetes_medication int,
  on_cholesterol_medication int,
  
  -- Prediction Results
  diagnosis int,           -- 0=healthy, 1=diabetes, 2=hypertension, 3=heart_disease
  diagnosis_label text,
  confidence float,
  probabilities jsonb,
  
  -- Legacy columns for backward compatibility
  gender text,
  blood_type text,
  discomfort_level int,
  symptom_duration int,
  prior_conditions text,
  bp_sys int,
  bp_dia int,
  cholesterol int,
  glucose int,
  family_history text,
  medication_use text,
  disease_label int,
  disease_type text,
  smoker_status text
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_patient_records_lab ON public.patient_records(lab_label);
CREATE INDEX IF NOT EXISTS idx_patient_records_diagnosis ON public.patient_records(diagnosis);
CREATE INDEX IF NOT EXISTS idx_patient_records_created ON public.patient_records(created_at DESC);

-- Stores client-side (lab) model update metadata.
create table if not exists public.fl_client_updates (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),

  run_id uuid null,
  round int not null default 1,
  client_user_id uuid null,
  client_label text not null,

  local_accuracy double precision,
  grad_norm double precision,
  num_examples int,
  storage_path text
);

-- Stores global models produced by aggregation.
create table if not exists public.fl_global_models (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),

  version int not null unique,
  model_type text,
  storage_path text
);

-- Tracks federated learning runs.
create table if not exists public.fl_runs (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),

  status text not null default 'running',
  num_rounds int not null default 1,
  model_type text
);

-- Ensure fl_round_metrics has the columns the backend expects (table may already exist).
create table if not exists public.fl_round_metrics (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),

  run_id uuid null,
  round int not null,
  global_accuracy double precision null,
  aggregated_grad_norm double precision null
);

alter table public.fl_round_metrics
  add column if not exists created_at timestamptz not null default now();
alter table public.fl_round_metrics
  add column if not exists run_id uuid null;
alter table public.fl_round_metrics
  add column if not exists round int;
alter table public.fl_round_metrics
  add column if not exists global_accuracy double precision null;
alter table public.fl_round_metrics
  add column if not exists aggregated_grad_norm double precision null;

-- Security: lock down patient_records and server-only FL tables.
alter table public.patient_records enable row level security;
alter table public.fl_global_models enable row level security;
alter table public.fl_runs enable row level security;
alter table public.fl_round_metrics enable row level security;

-- Allow the frontend to read fl_client_updates (for showing update history).
alter table public.fl_client_updates enable row level security;
drop policy if exists "public read fl_client_updates" on public.fl_client_updates;
create policy "public read fl_client_updates"
on public.fl_client_updates
for select
using (true);

