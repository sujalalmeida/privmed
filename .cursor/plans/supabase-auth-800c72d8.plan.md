---
name: Lab Enhancements and Multiclass Federated Learning
overview: ""
todos:
  - id: b4fdfeb6-1d70-45ee-832a-bbe67dc0eb5b
    content: Add full_name, contact_phone, assigned_lab_id, and lab_name columns to users table via migration
    status: pending
  - id: a72a3d50-cc00-42de-9964-da67a05391c6
    content: Create src/lib/supabase.ts with initialized Supabase client using env variables
    status: pending
  - id: c27fba48-5b20-4105-a19b-e695cb4e1e22
    content: Add login/signup toggle and signup form fields to Login.tsx
    status: pending
  - id: a092b749-64ca-49f9-8268-76283d120dcc
    content: Replace mock auth with Supabase auth functions (login, signup, logout) in AuthContext.tsx
    status: pending
  - id: 23cf4538-5233-4001-b112-9d93c968dbf8
    content: Ensure .env.local has correct Supabase URL and anon key
    status: pending
isProject: false
---

# Lab Enhancements and Multiclass Federated Learning

## Backend

- Update data model to multiclass disease prediction:
- patient_records: add `disease_type text` (e.g., 'healthy','diabetes','hypertension','heart_disease').
- Switch model to multiclass Logistic Regression or DecisionTreeClassifier.
- Train per-lab on latest lab-specific data (query `patient_records` by `lab_label`).
- Extend `/lab/add_patient_data`:
- Accept extra fields: bmi, smoker_status, heart_rate, bp_sys, bp_dia, cholesterol, glucose, family_history, medication_use.
- Insert row to `patient_records` with predicted `disease_label` and `disease_type`.
- Retrain per-lab model on that lab’s records; compute weight delta and local accuracy.
- Upload weights to Supabase Storage at `models/local/<lab_label>/<timestamp>.pkl`.
- Insert into `fl_client_updates` with `client_label`, `local_accuracy`, `grad_norm`, `num_examples`, and `storage_path`.
- Add optional `/lab/send_model_update` to trigger a retrain/send without a new patient.
- Keep `/admin/aggregate_models` to read all latest local models from Storage (or fallback to server/models), average weights, compute global accuracy, upload to `models/global/v{version}.pkl`, insert `fl_round_metrics` and `fl_global_models`.

## Supabase

- Tables/migrations:
- `alter table public.patient_records add column if not exists disease_type text`.
- `alter table public.fl_client_updates add column if not exists storage_path text`.
- Storage:
- Create bucket `models` (Dashboard → Storage) and make it public or serve via signed URLs.

## Sample CSV

- Update `server/data/patient_data.csv` with new columns: bmi, smoker_status, heart_rate, bp_sys, bp_dia, cholesterol, glucose, family_history, medication_use; disease_label maps to disease_type.

## Frontend

- Lab: `PatientDataCollection.tsx`
- Add inputs for new fields.
- On submit: call `/lab/add_patient_data`; show a card with:
  - Risk bar (0–100%) and predicted disease type chip.
  - "Model update sent" message with timestamp.
- Add a Model Update History panel (lab-only) pulling from `fl_client_updates` filtered by `client_label`; list round/time, local_accuracy, grad_norm, link to model file if available.
- Admin: `ModelAggregation.tsx`
- Add a table: lab label, last update time, last local_accuracy.
- After Aggregate Models, show new global version and accuracy.

## Dev/Run

- Ensure `server/.env` has Supabase vars.
- Create Storage bucket `models`.
- Start backend (Flask) and frontend.
- Lab A & Lab B: submit records; see predictions and history.
- Admin: aggregate; see accuracy and version update.