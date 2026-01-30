# PrivMed FL Paper Metrics Collection System

**Status**: ✅ Complete - Ready for Use  
**Version**: 1.0.0  
**Date**: January 30, 2026

## Overview

Comprehensive data collection and export system for PrivMed federated learning research paper. Automatically collects experiment-level, per-round, per-lab, and per-class metrics with minimal integration effort.

## What's Included

### 📊 Data Collection

1. **Database Schema** (`create_paper_metrics_tables.sql`)
   - 6 new tables for comprehensive metrics storage
   - JSONB support for nested data (per-class metrics, predictions)
   - Row-level security configured

2. **Metrics Computation** (`fl_metrics.py`)
   - Per-class: precision, recall, specificity, F1, AUC-ROC
   - Loss computation (training & validation, cross-entropy)
   - Confusion matrices and one-vs-rest metrics
   - Multi-class support for 4 disease classes

3. **FL Pipeline Logging** (`fl_logging.py`)
   - `FLMetricsLogger` class for structured logging
   - Automatic per-round and per-lab metric capture
   - Integration helpers for existing code

4. **Centralized Baseline** (`scripts/train_centralized_baseline.py`)
   - Extended baseline training with full metrics
   - Automatic database logging
   - Feature importance tracking

5. **Export System** (`scripts/export_paper_metrics.py`)
   - CSV exports for plotting (pandas/matplotlib/R)
   - JSON exports for detailed analysis
   - 7 export tables ready for paper figures

### 📈 Metrics Collected

| Category | Metrics | Storage |
|----------|---------|---------|
| **Experiment-level** | Centralized accuracy, Federated accuracy, FL+HE accuracy | `fl_experiment_log` |
| **Per-round** | Global accuracy, training loss, validation loss, grad norm | `fl_round_detailed_metrics` |
| **Per-lab per-round** | Local accuracy, local loss, samples, weight updates | `fl_round_detailed_metrics` |
| **Per-class** | Precision, recall, specificity, F1, AUC-ROC | `fl_per_class_performance` |
| **Data distribution** | Samples per lab, per-class breakdown | `fl_lab_data_distribution` |
| **Predictions** | Raw predictions for ROC curves | `fl_model_predictions` |

### 📁 Files Structure

```
server/
├── fl_metrics.py                           # Core metrics computation
├── fl_logging.py                           # FL pipeline logging
├── create_paper_metrics_tables.sql         # Database schema
├── INTEGRATION_EXAMPLE.py                  # Integration guide
├── scripts/
│   ├── train_centralized_baseline.py       # Enhanced baseline trainer
│   └── export_paper_metrics.py             # Export to CSV/JSON
└── models/
    ├── baseline_global_model.pkl           # Centralized model
    └── centralized_baseline_metrics.json   # Baseline metrics

docs/
├── FL_PAPER_METRICS_GUIDE.md               # Complete documentation
└── QUICKSTART_FL_METRICS.md                # 5-minute setup guide

paper_data/                                  # Export output (created on export)
├── experiment_summary.csv
├── rounds_summary.csv
├── per_lab_per_round.csv
├── per_class_metrics.csv
├── lab_data_distribution.csv
├── centralized_baselines.csv
├── model_predictions.json
└── README.md
```

## Quick Start

### 1. Database Setup (2 minutes)

```bash
# Run in Supabase SQL Editor
cat server/create_paper_metrics_tables.sql
```

### 2. Train Baseline (1 minute)

```bash
cd server/scripts
python train_centralized_baseline.py --experiment-id baseline_20260130 --log-to-db
```

### 3. Integrate into FL Pipeline (2 minutes)

Add to `server/app.py`:

```python
from fl_logging import FLMetricsLogger

# Initialize once
logger = FLMetricsLogger(sb(), "fl_exp_001")

# In aggregation endpoint:
evaluation = logger.log_global_aggregation(
    round_num=current_round,
    global_model=global_model,
    X_test=X_test,
    y_test=y_test,
    store_predictions=True
)
```

See `server/INTEGRATION_EXAMPLE.py` for complete examples.

### 4. Export Metrics

```bash
cd server/scripts
python export_paper_metrics.py --all --output-dir ../../paper_data
```

### 5. Generate Figures

```python
import pandas as pd
import matplotlib.pyplot as plt

rounds = pd.read_csv('paper_data/rounds_summary.csv')
plt.plot(rounds['round'], rounds['global_accuracy'])
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.savefig('convergence.png', dpi=300)
```

See `docs/FL_PAPER_METRICS_GUIDE.md` for all figure examples.

## Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| **QUICKSTART_FL_METRICS.md** | 5-minute setup guide | Quick integration |
| **FL_PAPER_METRICS_GUIDE.md** | Complete reference (database, metrics, figures) | Deep understanding |
| **INTEGRATION_EXAMPLE.py** | Code examples for app.py | Implementation |
| **README.md** (this file) | System overview | Everyone |

## Key Features

### ✅ Automatic Metrics

- **Per-class metrics**: Computed automatically during evaluation
- **Specificity**: Multi-class one-vs-rest calculation
- **AUC-ROC**: Per-class and macro-averaged
- **Loss tracking**: Training and validation loss at every round

### ✅ Minimal Integration

- Add 3-5 lines to your aggregation endpoint
- No changes to core FL logic (FedAvg)
- Backward compatible with existing pipeline

### ✅ Publication-Ready Output

- CSV files ready for pandas/matplotlib/seaborn/R
- JSON files for detailed analysis
- README with usage examples included

### ✅ Extensible

- Easy to add custom metrics in `fl_metrics.py`
- Modular design: use full logger or individual functions
- Database schema supports nested data (JSONB)

## Figures You Can Generate

From the exported data, you can generate all figures for your paper:

1. **Accuracy vs Rounds** - FL convergence (rounds_summary.csv)
2. **Loss vs Rounds** - Training/validation loss (rounds_summary.csv)
3. **Centralized vs FL Bar Chart** - Model comparison (experiment_summary.csv)
4. **Per-Class Performance Heatmap** - Precision/recall/F1 (per_class_metrics.csv)
5. **ROC Curves** - Multi-class ROC (model_predictions.json + per_class_metrics.csv)
6. **Data Imbalance** - Samples per lab (lab_data_distribution.csv)
7. **Per-Lab Accuracy** - Lab comparison (per_lab_per_round.csv)
8. **Convergence Analysis** - Detailed loss/accuracy (rounds_summary.csv + per_lab_per_round.csv)

All examples provided in `docs/FL_PAPER_METRICS_GUIDE.md`.

## Architecture

```
┌─────────────────────────────────────────┐
│      PrivMed FL Pipeline (app.py)       │
│  ┌───────────────────────────────────┐  │
│  │  Lab Training                     │  │
│  │  ↓                                │  │
│  │  fl_logging.log_lab_update()     │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  Global Aggregation               │  │
│  │  ↓                                │  │
│  │  fl_logging.log_global_aggreg()  │  │
│  │  (uses fl_metrics internally)    │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Supabase Database               │
│  • fl_experiment_log                    │
│  • fl_round_detailed_metrics            │
│  • fl_per_class_performance             │
│  • fl_centralized_baselines             │
│  • fl_lab_data_distribution             │
│  • fl_model_predictions                 │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│   export_paper_metrics.py               │
│  ↓                                       │
│  CSV + JSON exports                     │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│   Python/R Analysis & Visualization     │
│  • matplotlib/seaborn                   │
│  • pandas                               │
│  • sklearn.metrics (ROC)                │
└─────────────────────────────────────────┘
```

## Future Work

- [ ] **Homomorphic Encryption**: Add `federated_he_accuracy` collection
- [ ] **Differential Privacy**: Track privacy budget (ε, δ)
- [ ] **Fairness Metrics**: Per-demographic performance analysis
- [ ] **Communication Cost**: Track bytes transferred per round
- [ ] **Automated Plotting**: Generate all figures with one command
- [ ] **MLflow Integration**: Experiment tracking UI
- [ ] **LaTeX Tables**: Auto-generate paper tables

## Dependencies

### Core (Required)
- `scikit-learn` - Metrics computation
- `pandas` - Data handling
- `numpy` - Numerical operations
- `supabase` - Database client

### Optional (for visualization)
- `matplotlib` - Plotting
- `seaborn` - Advanced visualization

### Install
```bash
pip install scikit-learn pandas numpy supabase python-dotenv matplotlib seaborn
```

## Validation

### Test Metrics Module

```bash
cd server
python fl_metrics.py
# Output: ✅ All tests passed!
```

### Test Export

```bash
cd server/scripts
python export_paper_metrics.py --summary-only --output-dir /tmp/test_export
ls /tmp/test_export
# Should see: experiment_summary.csv, rounds_summary.csv, etc.
```

## Troubleshooting

### No data in exports
1. Verify tables exist: `SELECT * FROM fl_experiment_log LIMIT 1;`
2. Check logging is enabled in FL pipeline
3. Verify experiment_id matches

### Import errors
```bash
cd server
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -c "import fl_metrics, fl_logging; print('OK')"
```

### Missing per-class metrics
- Ensure `evaluate_model_comprehensive()` is called
- Check that test set has all 4 classes
- Verify `log_global_aggregation()` is called with `store_predictions=True`

## Performance Notes

- **Prediction Storage**: Can be large (10MB+ per round). Sample if needed:
  ```python
  # In fl_logging.py
  if len(predictions) > 10000:
      predictions = random.sample(predictions, 10000)
  ```

- **Database Growth**: ~1MB per experiment. Archive old experiments:
  ```sql
  DELETE FROM fl_model_predictions WHERE created_at < NOW() - INTERVAL '90 days';
  ```

- **Export Time**: ~1-5 seconds for typical experiments

## Support & Contribution

For issues or questions:
1. Check documentation in `docs/`
2. Review `INTEGRATION_EXAMPLE.py`
3. Verify database schema with SQL script

Contributions welcome:
- Additional metrics in `fl_metrics.py`
- New export formats in `export_paper_metrics.py`
- Visualization templates in guide

## License

Same as PrivMed project.

## Acknowledgments

Built for PrivMed federated learning research project.

---

**Version History**

- **1.0.0** (2026-01-30): Initial release
  - Complete metrics collection system
  - 6 database tables
  - Export to CSV/JSON
  - Comprehensive documentation

---

**Quick Links**

- 📖 [Complete Guide](./docs/FL_PAPER_METRICS_GUIDE.md)
- 🚀 [Quick Start](./docs/QUICKSTART_FL_METRICS.md)
- 💻 [Integration Examples](./server/INTEGRATION_EXAMPLE.py)
- 🗃️ [Database Schema](./server/create_paper_metrics_tables.sql)

---

**Ready to use!** Follow the Quick Start guide to begin collecting metrics in 5 minutes.
