-- HE-encrypted patient reports for the global dashboard.
-- Run this in Supabase SQL Editor.

create extension if not exists pgcrypto;

create table if not exists public.patient_reports_global (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),

  lab_id text not null,
  lab_label text not null,
  patient_id_hash text not null,

  prediction text not null,
  confidence double precision,
  clinical_reasoning text,

  plaintext_metadata jsonb not null default '{}'::jsonb,
  encrypted_numeric_fields jsonb not null,
  encrypted boolean not null default true,
  encryption_scheme text not null default 'HE-CKKS',
  lab_key_id text
);

create index if not exists idx_patient_reports_global_created_at
  on public.patient_reports_global(created_at desc);
create index if not exists idx_patient_reports_global_lab_label
  on public.patient_reports_global(lab_label);
create index if not exists idx_patient_reports_global_prediction
  on public.patient_reports_global(prediction);

create table if not exists public.patient_reports_global_audit (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  event_type text not null,
  actor_role text not null,
  actor_label text,
  details jsonb not null default '{}'::jsonb
);

create index if not exists idx_patient_reports_global_audit_created_at
  on public.patient_reports_global_audit(created_at desc);

create table if not exists public.doctor_feedback_global_reports (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  report_id uuid not null references public.patient_reports_global(id) on delete cascade,
  reviewer_id text not null,
  reviewer_name text,
  reviewer_role text default 'central_admin',

  agree boolean not null,
  correct_diagnosis int,
  correct_diagnosis_label text,
  remarks text,

  constraint unique_feedback_per_global_report_reviewer unique (report_id, reviewer_id)
);

create index if not exists idx_doctor_feedback_global_reports_report
  on public.doctor_feedback_global_reports(report_id);
create index if not exists idx_doctor_feedback_global_reports_created_at
  on public.doctor_feedback_global_reports(created_at desc);

create or replace function update_doctor_feedback_global_reports_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists doctor_feedback_global_reports_updated_at on public.doctor_feedback_global_reports;
create trigger doctor_feedback_global_reports_updated_at
  before update on public.doctor_feedback_global_reports
  for each row
  execute function update_doctor_feedback_global_reports_updated_at();

alter table public.patient_reports_global enable row level security;
alter table public.patient_reports_global_audit enable row level security;
alter table public.doctor_feedback_global_reports enable row level security;
