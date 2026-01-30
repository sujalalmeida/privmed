# PrivMed FL Paper Metrics Collection & Export

Complete guide for collecting and exporting federated learning metrics for research paper analysis.

## Table of Contents

1. [Overview](#overview)
2. [Database Schema](#database-schema)
3. [Setup Instructions](#setup-instructions)
4. [Data Collection](#data-collection)
5. [Export Metrics](#export-metrics)
6. [Generating Figures](#generating-figures)
7. [Metrics Reference](#metrics-reference)

---

## Overview

This system collects comprehensive metrics for federated learning experiments including:

- **Experiment-level**: Centralized vs Federated vs Federated+HE comparison
- **Per-round**: Global accuracy, loss, convergence metrics
- **Per-lab per-round**: Local accuracy, loss, data imbalance, weight updates
- **Per-class**: Precision, recall, specificity, F1-score, AUC-ROC for each disease class
- **Raw predictions**: For custom ROC curve generation

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Supabase Database                     │
│  ┌────────────────────────────────────────────────────┐ │
│  │ fl_experiment_log                                  │ │
│  │ fl_round_detailed_metrics                          │ │
│  │ fl_per_class_performance                           │ │
│  │ fl_centralized_baselines                           │ │
│  │ fl_lab_data_distribution                           │ │
│  │ fl_model_predictions                               │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                         ▲
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
    │ FL      │    │ Baseline│    │ Export  │
    │ Pipeline│    │ Training│    │ Script  │
    └─────────┘    └─────────┘    └─────────┘
```

---

## Database Schema

### 1. Setup Tables

Run the SQL scripts in order:

```bash
# 1. Core FL tables (if not already created)
# Run in Supabase Dashboard → SQL Editor
cat server/create_required_tables.sql

# 2. Paper metrics tables
cat server/create_paper_metrics_tables.sql
```

### Key Tables

#### `fl_experiment_log`
One row per experiment with summary metrics:
- `centralized_accuracy`: Baseline accuracy
- `federated_accuracy`: Final FL model accuracy
- `federated_he_accuracy`: FL+HE accuracy (nullable)
- `per_class_metrics`: JSONB with per-class precision/recall/F1
- `auc_roc_macro`: Macro-averaged AUC-ROC

#### `fl_round_detailed_metrics`
One row per (round, lab) or per (round, global):
- `is_global`: Boolean flag (true for aggregated metrics)
- `global_accuracy`, `local_accuracy`
- `global_train_loss`, `global_val_loss`
- `local_train_loss`, `local_val_loss`
- `num_examples`, `grad_norm`

#### `fl_per_class_performance`
One row per (experiment, round, class):
- `class_name`: 'healthy', 'diabetes', 'hypertension', 'heart_disease'
- `precision`, `recall`, `specificity`, `f1_score`
- `support`: Number of samples
- `auc_roc`: Per-class AUC-ROC

#### `fl_centralized_baselines`
One row per centralized baseline model:
- `model_name`: 'Gradient Boosting', 'Logistic Regression', etc.
- `accuracy`, `loss`
- `per_class_metrics`: JSONB
- `auc_roc_macro`

#### `fl_lab_data_distribution`
One row per (experiment, lab):
- `total_samples`
- `samples_per_class`: JSONB with counts per disease class

#### `fl_model_predictions`
Raw predictions for ROC curves:
- `predictions`: JSONB array of `{true_label, pred_label, proba: [...]}`
- `total_predictions`

---

## Setup Instructions

### 1. Install Dependencies

```bash
cd server

# Core dependencies (should already be installed)
pip install scikit-learn pandas numpy

# For metrics and export
pip install supabase python-dotenv

# Optional: for visualization
pip install matplotlib seaborn
```

### 2. Environment Variables

Ensure `.env` file has Supabase credentials:

```bash
# server/.env or repo root .env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key  # Preferred for server operations
# or
SUPABASE_ANON_KEY=your-anon-key
```

### 3. Initialize Database

Run both SQL scripts in Supabase Dashboard → SQL Editor:

```sql
-- 1. Run create_required_tables.sql (if not done)
-- 2. Run create_paper_metrics_tables.sql
```

Verify tables exist:

```sql
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name LIKE 'fl_%'
ORDER BY table_name;
```

---

## Data Collection

### Workflow

```
1. Train Centralized Baseline
   ↓
2. Run Federated Learning Rounds (with logging enabled)
   ↓
3. Finalize Experiment
   ↓
4. Export Metrics
```

### Step 1: Train Centralized Baseline

```bash
cd server/scripts

# Train and log baseline
python train_centralized_baseline.py \
  --experiment-id baseline_20260130 \
  --log-to-db

# Output:
# - models/baseline_global_model.pkl
# - models/centralized_baseline_metrics.json
# - Database: fl_centralized_baselines, fl_per_class_performance
```

**What gets logged:**
- Overall accuracy and loss
- Per-class precision, recall, specificity, F1, AUC-ROC
- Training time and sample counts
- Feature importances

### Step 2: Run Federated Learning with Logging

#### Option A: Integrate FLMetricsLogger into app.py

Add to `server/app.py`:

```python
from fl_logging import FLMetricsLogger

# In your FL aggregation endpoint:
# Initialize logger (once per experiment)
experiment_id = "fl_exp_20260130_001"
logger = FLMetricsLogger(sb(), experiment_id)

# When a lab sends an update:
logger.log_lab_update(
    round_num=current_round,
    lab_label=lab_label,
    local_accuracy=local_accuracy,
    num_examples=len(y_train),
    grad_norm=grad_norm,
    model=model,  # Optional: for loss computation
    X_train=X_train,  # Optional
    y_train=y_train,  # Optional
    training_time=training_time
)

# When aggregating:
logger.log_global_aggregation(
    round_num=current_round,
    global_model=global_model,
    X_test=X_test,
    y_test=y_test,
    X_train=X_train_combined,  # Optional
    y_train=y_train_combined,  # Optional
    aggregated_grad_norm=aggregated_grad_norm,
    participating_labs=['lab_A', 'lab_B'],
    store_predictions=True  # Store for ROC curves
)

# At experiment end:
logger.finalize_experiment(
    centralized_accuracy=0.87,  # From baseline
    federated_accuracy=0.89,    # From final round
    total_rounds=10,
    num_clients=2,
    notes="First experiment with new dataset"
)
```

#### Option B: Use Standalone Helper Functions

For quick integration without full logger:

```python
from fl_logging import log_client_update_metrics, log_global_round_metrics

# Log client update
log_client_update_metrics(
    sb(), experiment_id, round_num,
    lab_label, local_accuracy, num_examples, grad_norm
)

# Log global round
log_global_round_metrics(
    sb(), experiment_id, round_num,
    global_accuracy, global_loss, aggregated_grad_norm
)
```

### Step 3: Log Lab Data Distribution

Track data imbalance:

```python
logger.log_lab_data_distribution(
    round_num=1,
    lab_label='lab_A',
    y_train=y_train_lab_A
)
```

### Step 4: Finalize Experiment

After all rounds complete:

```python
logger.finalize_experiment(
    centralized_accuracy=0.87,
    federated_accuracy=0.89,
    federated_he_accuracy=None,  # Not implemented yet
    total_rounds=10,
    num_clients=2,
    notes="Baseline FL experiment"
)
```

---

## Export Metrics

### Export All Metrics

```bash
cd server/scripts

# Export all experiments
python export_paper_metrics.py --all --output-dir ../../paper_data

# Export specific experiment(s)
python export_paper_metrics.py \
  --experiment-id fl_exp_20260130_001 baseline_20260130 \
  --output-dir ../../paper_data

# Export summary only (no predictions)
python export_paper_metrics.py \
  --summary-only \
  --output-dir ../../paper_data
```

### Output Files

```
paper_data/
├── experiment_summary.csv              # Centralized vs FL comparison
├── experiment_summary_detailed.json    # With nested metrics
├── rounds_summary.csv                  # Per-round global metrics
├── per_lab_per_round.csv              # Per-lab metrics
├── per_class_metrics.csv              # Precision, recall, F1, etc.
├── lab_data_distribution.csv          # Data imbalance
├── centralized_baselines.csv          # Baseline performance
├── centralized_baselines_detailed.json
├── model_predictions.json             # For ROC curves
├── predictions_summary.csv
└── README.md                          # Usage guide
```

---

## Generating Figures

### Figure 1: Accuracy vs Federated Rounds

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
rounds = pd.read_csv('paper_data/rounds_summary.csv')

# Plot
plt.figure(figsize=(10, 6))
for exp_id in rounds['experiment_id'].unique():
    data = rounds[rounds['experiment_id'] == exp_id]
    plt.plot(data['round'], data['global_accuracy'], 
             marker='o', label=exp_id)

plt.xlabel('Federated Round', fontsize=12)
plt.ylabel('Global Accuracy', fontsize=12)
plt.title('Federated Learning Convergence', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_fl_convergence.png', dpi=300)
plt.show()
```

### Figure 2: Loss vs Rounds

```python
# Global loss over rounds
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Training loss
for exp_id in rounds['experiment_id'].unique():
    data = rounds[rounds['experiment_id'] == exp_id]
    ax1.plot(data['round'], data['global_train_loss'], 
             marker='o', label=exp_id)
ax1.set_xlabel('Round')
ax1.set_ylabel('Training Loss')
ax1.set_title('Global Training Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Validation loss
for exp_id in rounds['experiment_id'].unique():
    data = rounds[rounds['experiment_id'] == exp_id]
    ax2.plot(data['round'], data['global_val_loss'], 
             marker='s', label=exp_id)
ax2.set_xlabel('Round')
ax2.set_ylabel('Validation Loss')
ax2.set_title('Global Validation Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig2_loss_convergence.png', dpi=300)
plt.show()
```

### Figure 3: Centralized vs FL vs FL+HE Bar Chart

```python
import numpy as np

exp_summary = pd.read_csv('paper_data/experiment_summary.csv')

# Get latest experiment
exp = exp_summary.iloc[-1]

models = ['Centralized', 'Federated', 'FL + HE']
accuracies = [
    exp['centralized_accuracy'],
    exp['federated_accuracy'],
    exp['federated_he_accuracy'] if pd.notna(exp['federated_he_accuracy']) else 0
]

# Remove FL+HE if not available
if accuracies[2] == 0:
    models = models[:2]
    accuracies = accuracies[:2]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14)
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('fig3_model_comparison.png', dpi=300)
plt.show()
```

### Figure 4: Per-Class Performance (Heatmap)

```python
import seaborn as sns

per_class = pd.read_csv('paper_data/per_class_metrics.csv')

# Filter for final round and federated model
final_round = per_class['round'].max()
final_metrics = per_class[
    (per_class['round'] == final_round) & 
    (per_class['model_type'] == 'federated') &
    (per_class['lab_label'].isna())
]

# Pivot for heatmap
metrics_to_plot = ['precision', 'recall', 'specificity', 'f1_score']
heatmap_data = final_metrics.pivot_table(
    index='class_name',
    values=metrics_to_plot
)

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0, vmax=1, cbar_kws={'label': 'Score'})
plt.title('Per-Class Performance Metrics (Final FL Model)', fontsize=14)
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Disease Class', fontsize=12)
plt.tight_layout()
plt.savefig('fig4_per_class_heatmap.png', dpi=300)
plt.show()
```

### Figure 5: Data Imbalance (Samples per Lab)

```python
lab_dist = pd.read_csv('paper_data/lab_data_distribution.csv')

# Group by lab
lab_summary = lab_dist.groupby('lab_label')[
    ['samples_healthy', 'samples_diabetes', 
     'samples_hypertension', 'samples_heart_disease']
].mean()

# Stacked bar chart
lab_summary.plot(kind='bar', stacked=True, figsize=(10, 6),
                 color=['#2ecc71', '#3498db', '#e67e22', '#e74c3c'])
plt.xlabel('Lab', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.title('Data Distribution Across Labs', fontsize=14)
plt.legend(title='Disease Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('fig5_data_imbalance.png', dpi=300)
plt.show()
```

### Figure 6: Per-Lab Accuracy Comparison

```python
per_lab = pd.read_csv('paper_data/per_lab_per_round.csv')

# Get latest round
final_round = per_lab['round'].max()
final_lab_metrics = per_lab[per_lab['round'] == final_round]

plt.figure(figsize=(8, 6))
bars = plt.bar(final_lab_metrics['lab_label'], 
               final_lab_metrics['local_accuracy'],
               color=['#3498db', '#2ecc71'])
plt.ylabel('Local Accuracy', fontsize=12)
plt.xlabel('Lab', fontsize=12)
plt.title(f'Per-Lab Accuracy (Round {final_round})', fontsize=14)
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('fig6_per_lab_accuracy.png', dpi=300)
plt.show()
```

### Figure 7: ROC Curves

```python
import json
from sklearn.metrics import roc_curve, auc

# Load predictions
with open('paper_data/model_predictions.json', 'r') as f:
    pred_data = json.load(f)

# Get final round predictions for federated model
fed_preds = [p for p in pred_data 
             if p['model_type'] == 'federated'][-1]

predictions = fed_preds['predictions']

# Extract data
y_true = [p['true_label'] for p in predictions]
y_proba = np.array([p['proba'] for p in predictions])

# Plot ROC for each class
plt.figure(figsize=(10, 8))

class_names = ['Healthy', 'Diabetes', 'Hypertension', 'Heart Disease']
colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']

for i, (class_name, color) in enumerate(zip(class_names, colors)):
    # One-vs-rest
    y_true_binary = (np.array(y_true) == i).astype(int)
    y_score = y_proba[:, i]
    
    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{class_name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Federated Model', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig7_roc_curves.png', dpi=300)
plt.show()
```

---

## Metrics Reference

### Accuracy
- **Definition**: (TP + TN) / (TP + TN + FP + FN)
- **Used for**: Overall model performance
- **Tables**: `fl_round_detailed_metrics`, `fl_experiment_log`

### Loss (Cross-Entropy)
- **Definition**: -Σ y_true * log(y_pred)
- **Used for**: Convergence analysis
- **Tables**: `fl_round_detailed_metrics`

### Precision (Per-Class)
- **Definition**: TP / (TP + FP)
- **Interpretation**: Of predicted positives, how many are correct?
- **Tables**: `fl_per_class_performance`

### Recall / Sensitivity (Per-Class)
- **Definition**: TP / (TP + FN)
- **Interpretation**: Of actual positives, how many did we catch?
- **Tables**: `fl_per_class_performance`

### Specificity (Per-Class)
- **Definition**: TN / (TN + FP)
- **Interpretation**: Of actual negatives, how many did we correctly identify?
- **Tables**: `fl_per_class_performance`

### F1-Score (Per-Class)
- **Definition**: 2 * (Precision * Recall) / (Precision + Recall)
- **Used for**: Balanced metric when classes are imbalanced
- **Tables**: `fl_per_class_performance`

### AUC-ROC
- **Definition**: Area Under Receiver Operating Characteristic curve
- **Range**: 0.5 (random) to 1.0 (perfect)
- **Used for**: Model discrimination ability
- **Tables**: `fl_per_class_performance`, `fl_experiment_log`

### Gradient Norm / Weight Update Magnitude
- **Definition**: ‖Δw‖ = L2 norm of parameter updates
- **Used for**: Analyzing convergence speed and update magnitudes
- **Tables**: `fl_round_detailed_metrics`

---

## Troubleshooting

### Issue: No data in export

**Check:**
1. Did you run the SQL scripts to create tables?
2. Is logging enabled in your FL pipeline?
3. Are experiment IDs correct?

```python
# Verify data exists
from supabase import create_client
sb = create_client(url, key)

# Check experiment log
result = sb.table('fl_experiment_log').select('*').execute()
print(f"Experiments: {len(result.data)}")

# Check round metrics
result = sb.table('fl_round_detailed_metrics').select('*').execute()
print(f"Round metrics: {len(result.data)}")
```

### Issue: Missing per-class metrics

**Ensure:**
- `fl_metrics.py` is imported in your training code
- `evaluate_model_comprehensive()` is called after model training
- Results are logged to database

### Issue: Predictions file too large

**Solution:**
- Use `--summary-only` flag to skip predictions
- Or sample predictions before storing:

```python
# In fl_logging.py, log_global_aggregation():
if len(predictions_data) > 10000:
    import random
    predictions_data = random.sample(predictions_data, 10000)
```

---

## Next Steps

1. **Implement HE (Homomorphic Encryption)** - Currently `federated_he_accuracy` is nullable
2. **Add more visualization scripts** - Create automated plotting pipeline
3. **Extend metrics** - Add fairness metrics, calibration, etc.
4. **Automate experiment tracking** - MLflow or Weights & Biases integration

---

## References

- **Scikit-learn Metrics**: https://scikit-learn.org/stable/modules/model_evaluation.html
- **ROC Curves**: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
- **Federated Learning**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"

---

**Last Updated**: January 30, 2026  
**Version**: 1.0.0  
**Authors**: PrivMed Team
