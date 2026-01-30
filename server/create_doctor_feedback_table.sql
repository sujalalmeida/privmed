-- Doctor Feedback Table for PrivMed
-- Stores feedback from admin/doctors on AI predictions (Agree/Disagree)
-- Run this in Supabase Dashboard → SQL Editor

-- ============================================================================
-- Doctor Feedback Table
-- ============================================================================
-- Stores feedback from reviewers (admin/doctors) on patient record AI predictions.
-- One feedback per (record_id, reviewer_id) - latest wins via upsert.

create table if not exists public.doctor_feedback (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  
  -- The patient record being reviewed
  record_id uuid not null references public.patient_records(id) on delete cascade,
  
  -- Who gave the feedback
  reviewer_id text not null,       -- user id or email of reviewer
  reviewer_name text,              -- display name of reviewer
  reviewer_role text default 'central_admin',  -- 'central_admin' or 'lab'
  
  -- Feedback on AI prediction
  agree boolean not null,          -- true = agree with AI, false = disagree
  correct_diagnosis int,           -- if disagree, optional correct label (0-3)
  correct_diagnosis_label text,    -- 'healthy', 'diabetes', 'hypertension', 'heart_disease'
  remarks text,                    -- optional comment/notes
  
  -- Constraint: one feedback per record per reviewer
  constraint unique_feedback_per_record_reviewer unique (record_id, reviewer_id)
);

-- Indexes for efficient queries
create index if not exists idx_doctor_feedback_record on public.doctor_feedback(record_id);
create index if not exists idx_doctor_feedback_reviewer on public.doctor_feedback(reviewer_id);
create index if not exists idx_doctor_feedback_agree on public.doctor_feedback(agree);
create index if not exists idx_doctor_feedback_created on public.doctor_feedback(created_at desc);

-- Trigger to update updated_at on modification
create or replace function update_doctor_feedback_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists doctor_feedback_updated_at on public.doctor_feedback;
create trigger doctor_feedback_updated_at
  before update on public.doctor_feedback
  for each row
  execute function update_doctor_feedback_updated_at();

-- ============================================================================
-- Row Level Security
-- ============================================================================
alter table public.doctor_feedback enable row level security;

-- Allow backend (service role) full access
drop policy if exists "service_role_full_access" on public.doctor_feedback;
create policy "service_role_full_access"
on public.doctor_feedback
for all
using (true)
with check (true);

-- Allow authenticated users to read feedback
drop policy if exists "authenticated_read_feedback" on public.doctor_feedback;
create policy "authenticated_read_feedback"
on public.doctor_feedback
for select
using (auth.role() = 'authenticated');

-- Comment for documentation
comment on table public.doctor_feedback is 'Stores doctor/admin feedback on AI predictions (agree/disagree) for closed-loop learning';
comment on column public.doctor_feedback.agree is 'true = doctor agrees with AI prediction, false = disagrees';
comment on column public.doctor_feedback.correct_diagnosis is '0=healthy, 1=diabetes, 2=hypertension, 3=heart_disease - only set when disagree';
