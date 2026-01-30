# MedSafe FL Metrics - Quick Reference Card

**One-page reference for common tasks**

---

## 🚀 Quick Start (5 minutes)

```bash
# 1. Database (run SQL in Supabase Dashboard)
cat server/create_paper_metrics_tables.sql

# 2. Train baseline
cd server/scripts
python train_centralized_baseline.py --experiment-id baseline_001 --log-to-db

# 3. Add to app.py (in aggregation function)
from fl_logging import FLMetricsLogger
logger = FLMetricsLogger(sb(), "exp_001")
evaluation = logger.log_global_aggregation(round_num, global_model, X_test, y_test)

# 4. Export
python export_paper_metrics.py --all --output-dir ../../paper_data

# 5. Plot
import pandas as pd; import matplotlib.pyplot as plt
rounds = pd.read_csv('paper_data/rounds_summary.csv')
plt.plot(rounds['round'], rounds['global_accuracy']); plt.savefig('convergence.png')
```

---

## 📁 File Locations

| File | Location |
|------|----------|
| **Core Modules** | |
| Metrics computation | `server/fl_metrics.py` |
| FL logging | `server/fl_logging.py` |
| **Scripts** | |
| Baseline training | `server/scripts/train_centralized_baseline.py` |
| Export | `server/scripts/export_paper_metrics.py` |
| **Database** | |
| Schema | `server/create_paper_metrics_tables.sql` |
| **Docs** | |
| Complete guide | `docs/FL_PAPER_METRICS_GUIDE.md` |
| Quick start | `docs/QUICKSTART_FL_METRICS.md` |
| Integration | `server/INTEGRATION_EXAMPLE.py` |

---

## 📊 Database Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `fl_experiment_log` | Experiment summary | `centralized_accuracy`, `federated_accuracy` |
| `fl_round_detailed_metrics` | Round & lab metrics | `round`, `lab_label`, `global_accuracy`, `local_accuracy` |
| `fl_per_class_performance` | Per-class metrics | `class_name`, `precision`, `recall`, `f1_score`, `auc_roc` |
| `fl_centralized_baselines` | Baseline models | `model_name`, `accuracy`, `per_class_metrics` |
| `fl_lab_data_distribution` | Data imbalance | `lab_label`, `total_samples`, `samples_per_class` |
| `fl_model_predictions` | ROC curves | `predictions` (JSONB) |

---

## 🔧 Integration Snippets

### Minimal (3 lines)
```python
from fl_logging import FLMetricsLogger
logger = FLMetricsLogger(sb(), "exp_001")
logger.log_global_aggregation(round_num, global_model, X_test, y_test)
```

### Full (in aggregation endpoint)
```python
from fl_logging import FLMetricsLogger

# Initialize (once)
logger = FLMetricsLogger(sb(), "exp_001")

# After aggregation
evaluation = logger.log_global_aggregation(
    round_num=current_round,
    global_model=global_model,
    X_test=X_test,
    y_test=y_test,
    store_predictions=True
)

# Use results
global_accuracy = evaluation['accuracy']
```

---

## 📈 Exported Files

| File | Use For |
|------|---------|
| `experiment_summary.csv` | Centralized vs FL comparison, AUC-ROC |
| `rounds_summary.csv` | Accuracy/loss vs rounds |
| `per_lab_per_round.csv` | Per-lab analysis |
| `per_class_metrics.csv` | Precision, recall, F1 per class |
| `lab_data_distribution.csv` | Data imbalance plots |
| `model_predictions.json` | ROC curves |

---

## 📊 Figure Generation

### Convergence
```python
import pandas as pd; import matplotlib.pyplot as plt
rounds = pd.read_csv('paper_data/rounds_summary.csv')
plt.plot(rounds['round'], rounds['global_accuracy'])
plt.xlabel('Round'); plt.ylabel('Accuracy')
plt.savefig('convergence.png', dpi=300)
```

### Model Comparison
```python
exp = pd.read_csv('paper_data/experiment_summary.csv').iloc[-1]
models = ['Centralized', 'Federated']
accs = [exp['centralized_accuracy'], exp['federated_accuracy']]
plt.bar(models, accs); plt.ylabel('Accuracy')
plt.savefig('comparison.png', dpi=300)
```

### ROC Curves
```python
import json
from sklearn.metrics import roc_curve, auc
with open('paper_data/model_predictions.json') as f:
    data = json.load(f)[-1]
predictions = data['predictions']
# ... (see full guide for complete code)
```

---

## 🔍 Metrics Collected

| Category | Metrics |
|----------|---------|
| **Overall** | Accuracy, Loss (train/val) |
| **Per-class** | Precision, Recall, Specificity, F1, AUC-ROC |
| **Per-round** | Global accuracy, Local accuracy, Grad norm |
| **Distribution** | Samples per lab, Per-class counts |

---

## 🛠️ Common Commands

### Check data exists
```python
from supabase import create_client; import os
sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY'))
print(sb.table('fl_experiment_log').select('*').execute().data)
```

### Test connection
```python
from supabase import create_client; import os
sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY'))
result = sb.table('fl_experiment_log').select('*').limit(1).execute()
print('✓ Connected' if result else '✗ Failed')
```

### Export specific experiment
```bash
python export_paper_metrics.py --experiment-id exp_001 --output-dir ./paper_data
```

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Import error | `cd server && export PYTHONPATH=$(pwd)` |
| No data | Check experiment_id matches |
| Missing tables | Re-run SQL script in Supabase |
| No predictions | Set `store_predictions=True` |

---

## 📚 Documentation Quick Links

- **5-min setup**: `docs/QUICKSTART_FL_METRICS.md`
- **Complete guide**: `docs/FL_PAPER_METRICS_GUIDE.md`
- **Integration**: `server/INTEGRATION_EXAMPLE.py`
- **Architecture**: `ARCHITECTURE_DIAGRAM.md`
- **Checklist**: `IMPLEMENTATION_CHECKLIST.md`

---

## 🎯 Key Functions

### fl_metrics.py
```python
evaluate_model_comprehensive(model, X, y)  # Returns all metrics
compute_per_class_metrics(y_true, y_pred, y_proba)
compute_loss(model, X, y)
```

### fl_logging.py
```python
logger = FLMetricsLogger(sb, experiment_id)
logger.log_lab_update(round, lab, accuracy, samples, grad_norm)
logger.log_global_aggregation(round, model, X_test, y_test)
logger.finalize_experiment()
```

---

## ✅ Validation Checklist

- [ ] Database tables created (6 tables)
- [ ] Baseline trained and logged
- [ ] FL integration added (3 lines minimum)
- [ ] Ran FL rounds successfully
- [ ] Exported data (7 files created)
- [ ] Generated at least 3 figures
- [ ] Metrics make sense (accuracy 0-1, etc.)

---

## 📞 Support

1. Check docs: `docs/FL_PAPER_METRICS_GUIDE.md`
2. Review examples: `server/INTEGRATION_EXAMPLE.py`
3. Verify database: Run SQL queries to check data
4. Test modules: `python server/fl_metrics.py`

---

**Version**: 1.0.0 | **Date**: January 30, 2026
