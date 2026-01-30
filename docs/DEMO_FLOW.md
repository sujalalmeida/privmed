# MedSafe Federated Learning Demo Flow

This guide provides step-by-step instructions for demonstrating MedSafe's federated learning capabilities. Follow this flow to show how global model accuracy improves as more labs contribute data and participate in aggregation rounds.

## Prerequisites

1. **Start the Backend Server**
   ```bash
   cd server
   python app.py
   ```
   The server will run on `http://127.0.0.1:5001`

2. **Start the Frontend**
   ```bash
   npm run dev
   ```
   The frontend will run on `http://localhost:5174`

## Demo Accounts

| Role | Email | Password | Lab |
|------|-------|----------|-----|
| Lab A | `lab@demo.com` | any | Lab A |
| Lab B | `lab2@demo.com` | any | Lab B |
| Admin | `admin@demo.com` | any | Central Admin |

---

## Round 1: Initial Data Collection and Model Training

### Step 1: Lab A - Enter Patient Data

1. **Login** as Lab A: `lab@demo.com` (any password)
2. Navigate to **Clinical Data Entry** tab
3. Enter **5-10 patient records** with varying diagnoses:
   - Mix of healthy, diabetes, hypertension, and heart disease cases
   - For each patient:
     - Fill in demographics (age, sex, height, weight)
     - Enter vital signs (blood pressure, heart rate)
     - Add blood chemistry values (glucose, HbA1c, cholesterol)
     - Fill cardiac markers if relevant
     - Submit each patient record
   - **Important**: Each submission saves the record to the database with `lab_label = lab_A`

### Step 2: Lab A - Send Model Update

1. Navigate to **Model Training** tab
2. Click **Send Model Update**
3. Wait for confirmation: "Model update sent successfully"
4. Note the local accuracy displayed (e.g., 85%)

### Step 3: Lab B - Enter Patient Data

1. **Logout** and **Login** as Lab B: `lab2@demo.com`
2. Navigate to **Clinical Data Entry** tab
3. Enter **5-10 patient records** (similar diversity as Lab A)
4. Submit each record
   - Records are saved with `lab_label = lab_B`

### Step 4: Lab B - Send Model Update

1. Navigate to **Model Training** tab
2. Click **Send Model Update**
3. Wait for confirmation
4. Note the local accuracy

### Step 5: Admin - Aggregate Models (Round 1)

1. **Logout** and **Login** as Admin: `admin@demo.com`
2. Navigate to **Model Aggregation** tab
3. Verify both labs appear in "Participating Labs" section:
   - Lab A: ✓ Ready (X samples)
   - Lab B: ✓ Ready (X samples)
4. Click **Aggregate Models Now**
5. **Record the results**:
   - Global Model Version: v1
   - Global Accuracy: XX.X%
   - Number of contributing labs: 2
   - Total samples: combined from both labs

---

## Round 2: Adding More Data and Re-aggregating

### Step 6: Lab A - Add More Patients

1. Login as Lab A: `lab@demo.com`
2. Navigate to **Clinical Data Entry**
3. Enter **5 more patient records**
4. Submit each record
5. Navigate to **Model Training**
6. Click **Send Model Update**
7. Note the improved local accuracy

### Step 7: Lab B - Add More Patients

1. Login as Lab B: `lab2@demo.com`
2. Enter **5 more patient records** in Clinical Data Entry
3. Submit each record
4. Navigate to Model Training → **Send Model Update**

### Step 8: Admin - Aggregate Models (Round 2)

1. Login as Admin: `admin@demo.com`
2. Navigate to **Model Aggregation**
3. Click **Aggregate Models Now**
4. **Compare with Round 1**:
   - Global Model Version: v2
   - Global Accuracy: Should be **higher** than v1
   - Total samples: Should be larger

---

## Round 3 and Beyond

Repeat the cycle:
1. Labs add more patient data
2. Labs send model updates
3. Admin aggregates models

**Expected behavior**: Global accuracy should **increase or stabilize** as more data is added.

---

## Verifying Accuracy Improvement

### Check FL Performance Tab

1. As Admin, navigate to **FL Performance** tab
2. View the accuracy chart showing improvement over rounds
3. The table shows per-round metrics:
   - Round number
   - Global accuracy
   - Number of participating labs
   - Total samples

### Check A/B Testing Tab

1. Navigate to **A/B Testing** tab
2. Configure a test:
   - Model A: Local lab model (e.g., Lab A)
   - Model B: Latest global model (e.g., v2)
3. Run the test to compare local vs. global model accuracy
4. The global model should outperform individual lab models

---

## Troubleshooting

### Lab has 0 records / "No patient data found"

**Cause**: Lab label mismatch between Clinical Data Entry and Model Training

**Solution**: 
- Ensure you're logged in as the correct lab user (lab@demo.com for Lab A, lab2@demo.com for Lab B)
- The lab_label is now derived from `user.labName` in both places
- Check server logs for `Searching for patient records with lab_label: 'lab_A'`

### Accuracy is not improving

**Possible causes**:
1. **Not enough data diversity**: Add patients with different diagnoses
2. **Same data repeated**: Enter new, varied patient data each round
3. **Only one lab participating**: Both labs need to send updates before aggregation

**Check**:
- In Admin → Model Aggregation, verify both labs show as "Ready"
- Check the "Total Samples" count increases each round

### Model version not incrementing

**Cause**: Aggregation may have failed silently

**Solution**:
- Check server console for errors
- Ensure both labs have sent model updates
- Verify Supabase connection is working

---

## Architecture Notes

### Data Flow

```
Lab A: Clinical Data Entry → patient_records (lab_label='lab_A')
                           ↓
       Model Training → fl_client_updates → Supabase Storage

Lab B: Clinical Data Entry → patient_records (lab_label='lab_B')
                           ↓
       Model Training → fl_client_updates → Supabase Storage

Admin: Aggregate → Downloads lab models → FedAvg → fl_global_models
                                                 → fl_round_metrics
```

### Key Database Tables

| Table | Purpose |
|-------|---------|
| `patient_records` | Stores patient data per lab |
| `fl_client_updates` | Tracks lab model updates |
| `fl_global_models` | Stores aggregated global models |
| `fl_round_metrics` | Records accuracy per aggregation round |

### Random Seeds

All training and aggregation operations use `random_state=42` for reproducibility. This ensures:
- Same data produces same model
- Demo results are consistent and reproducible

---

## Quick Demo Script (5 minutes)

1. **Pre-populate**: Before the demo, enter 5 patients each for Lab A and Lab B
2. **Show**: Login as Admin → Model Aggregation → Show both labs ready
3. **Aggregate**: Click Aggregate → Show v1 accuracy (e.g., 75%)
4. **Add data**: Login as Lab A → Add 3 patients → Send Update
5. **Add data**: Login as Lab B → Add 3 patients → Send Update
6. **Re-aggregate**: Login as Admin → Aggregate → Show v2 accuracy (e.g., 80%)
7. **Explain**: "Accuracy improved because more data was federated from multiple labs without sharing raw patient data"
