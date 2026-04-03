# AGENTS.md — Federated Learning Medical Diagnosis System

## Project Overview
This is a Federated Learning (FL) web application for medical diagnosis.
Multiple labs each have a local ML model. Labs train locally and send
model WEIGHTS (not data) to a global server. The global server aggregates
using FedAvg and sends improved weights back to labs.

## Architecture
- Each lab: local dataset, local ML model, patient prediction form
- Global server: receives weights, runs FedAvg aggregation, redistributes
- Doctor approval system: doctors approve/reject predictions;
  approval rate >= 0.8 gives that lab a 1.5x weight multiplier in FedAvg
- Terminology rule: weight UPDATES go TO the global server;
  weights come BACK to labs. Never swap these in UI labels.

## What Is Already Built
- Lab login and dashboard
- Patient data input form with Predict button
- Local model per lab making predictions with confidence score
- Global model page showing aggregated accuracy
- Doctor approval/rejection flow

## What Needs to Be Implemented
1. FedAvg fixes (see prompt 1)
2. Homomorphic Encryption (HE) architecture understanding (see prompt 2)
3. HE implementation with TenSEAL (see prompt 3)

## Key Technical Decisions
- CKKS scheme via TenSEAL library
- CKKS params: poly_modulus_degree=8192, coeff_mod_bit_sizes=[60,40,40,60], scale=2**40
- Secret key NEVER leaves the lab. Server only gets: ciphertext + public context + sample count
- Server aggregates on ciphertexts only — never decrypts
- HE applies to: model weights + numerical patient report fields
- Text fields: standard encryption + TLS only (no HE on text)
- Accuracy: always report test accuracy from 80/20 split, never training accuracy

## Known Limitations to Preserve in Comments
- Non-IID data across labs not handled (FedProx is future work)
- HE doesn't protect against gradient inversion alone; differential privacy is complementary
- Threat model: honest-but-curious server
- HE does not protect against: malicious weight poisoning, membership inference, post-decryption exposure

## How to Run
npm install

cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_role_key

cd server
source .venv/bin/activate
python app.py


npm run dev

## How to Test
[fill in — e.g. `pytest tests/`, or how to manually test the predict flow]

## File Structure
[briefly describe where your main files are — e.g. /lab_server/, /global_server/, /models/, /templates/]