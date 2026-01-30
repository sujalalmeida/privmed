# MedSafe FL Paper Metrics - Quick Start Guide

Get up and running with comprehensive FL metrics collection in 5 minutes.

## Prerequisites

- MedSafe FL system running (server/app.py)
- Supabase database configured
- Python 3.8+ with scikit-learn, pandas, numpy installed

## 5-Minute Setup

### Step 1: Create Database Tables (2 minutes)

```bash
# 1. Open Supabase Dashboard → SQL Editor
# 2. Copy and run server/create_paper_metrics_tables.sql
# 3. Verify tables created:

SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' AND table_name LIKE 'fl_%'
ORDER BY table_name;

# You should see:
# - fl_centralized_baselines
# - fl_experiment_log
# - fl_lab_data_distribution
# - fl_model_predictions
# - fl_per_class_performance
# - fl_round_detailed_metrics
```

### Step 2: Train Centralized Baseline (1 minute)

```bash
cd server/scripts

python train_centralized_baseline.py \
  --experiment-id baseline_$(date +%Y%m%d) \
  --log-to-db

# Output:
# ✅ Model saved to models/baseline_global_model.pkl
# ✅ Logged to fl_centralized_baselines
# ✅ Logged per-class metrics for 4 classes
```

### Step 3: Add Logging to FL Pipeline (2 minutes)

Open `server/app.py` and add at the top:

```python
from fl_logging import FLMetricsLogger

# Add global variable
_metrics_logger = None
```

In your aggregation function (e.g., around line 1650), add:

```python
# After loading test data:
X_test, y_test = load_test_dataset()

# Initialize logger (do once)
global _metrics_logger
if _metrics_logger is None:
    _metrics_logger = FLMetricsLogger(sb(), "fl_exp_001")

# Log aggregation
evaluation = _metrics_logger.log_global_aggregation(
    round_num=version,
    global_model=global_model,
    X_test=X_test,
    y_test=y_test,
    store_predictions=True
)

# Use evaluation['accuracy'] instead of manual calculation
global_accuracy = evaluation['accuracy']
```

### Step 4: Run FL Rounds

```bash
# Start your FL server and run rounds as usual
# Metrics are automatically logged to database

# Check logs for confirmation:
# ✓ Logged lab update: lab_A (round 1)
# ✓ Logged global metrics (accuracy: 0.8567)
# ✓ Logged per-class metrics for 4 classes
```

### Step 5: Export Metrics for Paper

```bash
cd server/scripts

python export_paper_metrics.py \
  --all \
  --output-dir ../../paper_data

# Output:
# 📊 Exporting experiment summary...
#   ✓ Saved to paper_data/experiment_summary.csv
# 📊 Exporting per-round metrics...
#   ✓ Saved to paper_data/rounds_summary.csv
# ... (more files)
# ✅ Export complete! All files saved to: paper_data
```

## What You Get

### Exported Files

```
paper_data/
├── experiment_summary.csv          # Centralized vs FL comparison
├── rounds_summary.csv              # Accuracy & loss per round
├── per_lab_per_round.csv          # Lab-level metrics
├── per_class_metrics.csv          # Precision, recall, F1, etc.
├── lab_data_distribution.csv      # Data imbalance
└── README.md                      # Usage guide
```

### Example: Plot Convergence

```python
import pandas as pd
import matplotlib.pyplot as plt

rounds = pd.read_csv('paper_data/rounds_summary.csv')

plt.figure(figsize=(10, 6))
plt.plot(rounds['round'], rounds['global_accuracy'], marker='o')
plt.xlabel('Federated Round')
plt.ylabel('Global Accuracy')
plt.title('FL Convergence')
plt.grid(True)
plt.savefig('fl_convergence.png', dpi=300)
```

## Troubleshooting

### "No data in export"

Check if metrics are being logged:

```python
from supabase import create_client
import os

sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY'))

# Check experiment log
result = sb.table('fl_experiment_log').select('*').execute()
print(f"Experiments: {len(result.data)}")

# Check round metrics
result = sb.table('fl_round_detailed_metrics').select('*').execute()
print(f"Rounds: {len(result.data)}")
```

### "Import error: fl_logging not found"

Make sure you're running from the correct directory:

```bash
cd server
python -c "import fl_logging; print('OK')"

# Or add to PYTHONPATH:
export PYTHONPATH="${PYTHONPATH}:/path/to/medsafe/server"
```

### "Tables don't exist"

Re-run the SQL script:

```sql
-- In Supabase SQL Editor
\i server/create_paper_metrics_tables.sql
```

## Next Steps

1. **Full Integration**: See `server/INTEGRATION_EXAMPLE.py` for complete examples
2. **Detailed Guide**: Read `docs/FL_PAPER_METRICS_GUIDE.md` for comprehensive documentation
3. **Customize Logging**: Modify `fl_logging.py` to add custom metrics
4. **Generate Figures**: Use examples in guide to create paper-ready plots

## Minimal Integration (Alternative)

If you want even simpler integration without the full logger:

```python
# In app.py aggregation endpoint
from fl_logging import log_global_round_metrics

# After computing global_accuracy:
log_global_round_metrics(
    sb(), "exp_001", round_num, 
    global_accuracy, global_loss=None
)
```

Then export as usual.

## Files Reference

| File | Purpose |
|------|---------|
| `server/fl_metrics.py` | Core metrics computation (precision, recall, AUC, etc.) |
| `server/fl_logging.py` | FL pipeline logging (wrapper for database inserts) |
| `server/scripts/train_centralized_baseline.py` | Train and log centralized baseline |
| `server/scripts/export_paper_metrics.py` | Export all metrics to CSV/JSON |
| `server/create_paper_metrics_tables.sql` | Database schema |
| `server/INTEGRATION_EXAMPLE.py` | Integration examples |
| `docs/FL_PAPER_METRICS_GUIDE.md` | Complete documentation |

## Support

For issues or questions:
1. Check `docs/FL_PAPER_METRICS_GUIDE.md` for detailed docs
2. Review `INTEGRATION_EXAMPLE.py` for code examples
3. Verify database tables exist and have RLS policies

---

**Time Investment**: ~5 minutes setup, automatic thereafter  
**Benefit**: Publication-ready metrics and figures with minimal effort
