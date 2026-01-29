-- MedSafe required tables for backend + FL metadata
-- Run this in Supabase Dashboard → SQL Editor.

create extension if not exists pgcrypto;

-- Stores per-lab patient records used for local training.
create table if not exists public.patient_records (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),

  lab_label text not null,

  age int,
  gender text,
  blood_type text,
  discomfort_level int,
  symptom_duration int,
  prior_conditions text,

  bmi double precision,
  smoker_status text,
  heart_rate int,
  bp_sys int,
  bp_dia int,
  cholesterol int,
  glucose int,
  family_history text,
  medication_use text,

  disease_label int,
  disease_type text
);

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

