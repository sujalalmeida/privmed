# MedSafe FL Paper Metrics - Implementation Checklist

Use this checklist to ensure complete setup and integration of the FL metrics system.

## Phase 1: Initial Setup ☐

### Database Setup
- [ ] Open Supabase Dashboard → SQL Editor
- [ ] Copy contents of `server/create_paper_metrics_tables.sql`
- [ ] Execute SQL script
- [ ] Verify tables created:
  ```sql
  SELECT table_name FROM information_schema.tables 
  WHERE table_schema = 'public' AND table_name LIKE 'fl_%'
  ORDER BY table_name;
  ```
- [ ] Confirm 6 new tables exist:
  - [ ] `fl_experiment_log`
  - [ ] `fl_round_detailed_metrics`
  - [ ] `fl_per_class_performance`
  - [ ] `fl_centralized_baselines`
  - [ ] `fl_lab_data_distribution`
  - [ ] `fl_model_predictions`

### Environment Setup
- [ ] Verify `.env` file exists with Supabase credentials
- [ ] Test credentials:
  ```python
  from supabase import create_client
  import os
  sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY'))
  result = sb.table('fl_experiment_log').select('*').limit(1).execute()
  print("✓ Connection successful" if result else "✗ Connection failed")
  ```

### Dependencies
- [ ] Verify Python dependencies installed:
  ```bash
  python -c "import pandas, numpy, sklearn, supabase; print('✓ All dependencies available')"
  ```
- [ ] If missing, install:
  ```bash
  pip install pandas numpy scikit-learn supabase python-dotenv
  ```

### Files Verification
- [ ] Verify new files exist:
  - [ ] `server/fl_metrics.py`
  - [ ] `server/fl_logging.py`
  - [ ] `server/create_paper_metrics_tables.sql`
  - [ ] `server/INTEGRATION_EXAMPLE.py`
  - [ ] `server/scripts/train_centralized_baseline.py`
  - [ ] `server/scripts/export_paper_metrics.py`
  - [ ] `docs/FL_PAPER_METRICS_GUIDE.md`
  - [ ] `docs/QUICKSTART_FL_METRICS.md`
  - [ ] `FL_METRICS_README.md`
  - [ ] `IMPLEMENTATION_SUMMARY.md`
  - [ ] `setup_fl_metrics.sh`

### Test Core Modules
- [ ] Test `fl_metrics.py`:
  ```bash
  cd server
  python fl_metrics.py
  # Should output: ✅ All tests passed!
  ```

---

## Phase 2: Centralized Baseline ☐

### Data Preparation
- [ ] Verify training data exists:
  ```bash
  ls -lh server/data/combined_train.csv
  ls -lh server/data/combined_validation.csv
  ls -lh server/data/combined_test.csv
  ```
- [ ] If missing, run data preparation:
  ```bash
  cd server/scripts
  python prepare_dataset.py
  ```

### Train Baseline
- [ ] Choose experiment ID (e.g., `baseline_20260130`)
- [ ] Train centralized baseline:
  ```bash
  cd server/scripts
  python train_centralized_baseline.py \
    --experiment-id baseline_20260130 \
    --log-to-db
  ```
- [ ] Verify outputs created:
  - [ ] `server/models/baseline_global_model.pkl`
  - [ ] `server/models/centralized_baseline_metrics.json`
- [ ] Verify database records:
  ```python
  # Check fl_centralized_baselines
  result = sb.table('fl_centralized_baselines').select('*').execute()
  print(f"Baseline records: {len(result.data)}")
  
  # Check fl_per_class_performance
  result = sb.table('fl_per_class_performance') \
    .select('*') \
    .eq('model_type', 'centralized') \
    .execute()
  print(f"Per-class records: {len(result.data)}")
  ```
- [ ] Note baseline accuracy: ________________

---

## Phase 3: FL Pipeline Integration ☐

### Review Integration Examples
- [ ] Read `server/INTEGRATION_EXAMPLE.py`
- [ ] Identify your aggregation endpoint in `server/app.py`
- [ ] Identify lab training endpoint (if separate)

### Add Imports
- [ ] Add to top of `server/app.py`:
  ```python
  from fl_logging import FLMetricsLogger
  from fl_metrics import evaluate_model_comprehensive
  ```

### Initialize Logger
- [ ] Choose integration approach:
  - [ ] **Option A**: Global logger (simpler, single experiment)
  - [ ] **Option B**: Request-scoped logger (multiple experiments)
- [ ] Add logger initialization code
- [ ] Choose experiment ID: ________________

### Integration Points

#### Lab Training Endpoint
- [ ] Locate lab training function in `app.py`
- [ ] Add after model training completes:
  ```python
  logger.log_lab_update(
      round_num=current_round,
      lab_label=lab_label,
      local_accuracy=local_accuracy,
      num_examples=len(X_train),
      grad_norm=grad_norm
  )
  ```
- [ ] Test: Train a local model and verify database update

#### Aggregation Endpoint
- [ ] Locate aggregation function in `app.py`
- [ ] Add after FedAvg aggregation:
  ```python
  evaluation = logger.log_global_aggregation(
      round_num=current_round,
      global_model=global_model,
      X_test=X_test,
      y_test=y_test,
      store_predictions=True
  )
  global_accuracy = evaluation['accuracy']
  ```
- [ ] Test: Run aggregation and verify database update

#### Data Distribution (Optional)
- [ ] Add data distribution logging (run once per lab):
  ```python
  logger.log_lab_data_distribution(
      round_num=1,
      lab_label=lab_label,
      y_train=y_train
  )
  ```

### Test Integration
- [ ] Start FL server:
  ```bash
  cd server
  python app.py
  ```
- [ ] Run 1-2 FL rounds
- [ ] Check console for logging messages:
  - [ ] "✓ Logged lab update: lab_A"
  - [ ] "✓ Logged global metrics"
  - [ ] "✓ Logged per-class metrics"
- [ ] Verify database records:
  ```python
  # Check round metrics
  result = sb.table('fl_round_detailed_metrics').select('*').execute()
  print(f"Round records: {len(result.data)}")
  ```

---

## Phase 4: Run Complete FL Experiment ☐

### Experiment Planning
- [ ] Experiment ID: ________________
- [ ] Number of rounds: ________________
- [ ] Number of labs: ________________
- [ ] Notes/description: ________________

### Execute Experiment
- [ ] Start FL server
- [ ] Run all planned rounds
- [ ] Monitor logging messages
- [ ] Verify data being collected in database

### Finalize Experiment
- [ ] After all rounds complete, finalize:
  ```python
  logger.finalize_experiment(
      centralized_accuracy=0.87,  # From baseline
      federated_accuracy=0.89,    # From final round
      total_rounds=10,
      num_clients=2,
      notes="Description of experiment"
  )
  ```
- [ ] Verify experiment log created:
  ```python
  result = sb.table('fl_experiment_log') \
    .select('*') \
    .eq('experiment_id', 'your_exp_id') \
    .execute()
  print(result.data)
  ```

---

## Phase 5: Export Metrics ☐

### Prepare Export Directory
- [ ] Create output directory:
  ```bash
  mkdir -p paper_data
  ```

### Run Export
- [ ] Export all metrics:
  ```bash
  cd server/scripts
  python export_paper_metrics.py \
    --all \
    --output-dir ../../paper_data
  ```
- [ ] Or export specific experiment:
  ```bash
  python export_paper_metrics.py \
    --experiment-id your_exp_id \
    --output-dir ../../paper_data
  ```

### Verify Exports
- [ ] Check files created:
  ```bash
  ls -lh paper_data/
  ```
- [ ] Verify expected files:
  - [ ] `experiment_summary.csv`
  - [ ] `rounds_summary.csv`
  - [ ] `per_lab_per_round.csv`
  - [ ] `per_class_metrics.csv`
  - [ ] `lab_data_distribution.csv`
  - [ ] `centralized_baselines.csv`
  - [ ] `model_predictions.json`
  - [ ] `README.md`

### Quick Data Check
- [ ] Load and inspect exports:
  ```python
  import pandas as pd
  
  # Experiment summary
  exp = pd.read_csv('paper_data/experiment_summary.csv')
  print(f"Experiments: {len(exp)}")
  print(exp[['experiment_id', 'centralized_accuracy', 'federated_accuracy']])
  
  # Rounds
  rounds = pd.read_csv('paper_data/rounds_summary.csv')
  print(f"Rounds: {len(rounds)}")
  print(rounds[['round', 'global_accuracy']].head())
  
  # Per-class
  per_class = pd.read_csv('paper_data/per_class_metrics.csv')
  print(f"Per-class records: {len(per_class)}")
  ```

---

## Phase 6: Generate Figures ☐

### Setup Plotting Environment
- [ ] Install visualization libraries (if not already):
  ```bash
  pip install matplotlib seaborn
  ```

### Figure 1: Accuracy vs Rounds
- [ ] Copy code from `docs/FL_PAPER_METRICS_GUIDE.md`
- [ ] Generate figure
- [ ] Verify output: `fig1_fl_convergence.png`
- [ ] Review figure quality and labels

### Figure 2: Loss vs Rounds
- [ ] Copy code from guide
- [ ] Generate figure
- [ ] Verify output: `fig2_loss_convergence.png`

### Figure 3: Model Comparison Bar Chart
- [ ] Copy code from guide
- [ ] Generate figure
- [ ] Verify output: `fig3_model_comparison.png`

### Figure 4: Per-Class Performance
- [ ] Copy code from guide
- [ ] Generate figure
- [ ] Verify output: `fig4_per_class_heatmap.png`

### Figure 5: ROC Curves
- [ ] Copy code from guide
- [ ] Generate figure
- [ ] Verify output: `fig7_roc_curves.png`

### Figure 6: Data Imbalance
- [ ] Copy code from guide
- [ ] Generate figure
- [ ] Verify output: `fig5_data_imbalance.png`

### Figure 7: Per-Lab Accuracy
- [ ] Copy code from guide
- [ ] Generate figure
- [ ] Verify output: `fig6_per_lab_accuracy.png`

### Additional Figures
- [ ] Create any custom figures needed for paper
- [ ] Ensure all figures are high-resolution (300 DPI)
- [ ] Verify axes labels, titles, and legends

---

## Phase 7: Documentation Review ☐

### Quick Start Guide
- [ ] Read `docs/QUICKSTART_FL_METRICS.md`
- [ ] Verify all steps match your setup

### Complete Guide
- [ ] Read `docs/FL_PAPER_METRICS_GUIDE.md`
- [ ] Bookmark for reference during paper writing

### Integration Examples
- [ ] Review `server/INTEGRATION_EXAMPLE.py`
- [ ] Compare with your actual integration

### Architecture
- [ ] Review `ARCHITECTURE_DIAGRAM.md`
- [ ] Understand data flow

---

## Phase 8: Validation ☐

### Data Completeness
- [ ] Verify all expected metrics collected:
  - [ ] Centralized baseline accuracy: ☐
  - [ ] Per-round global accuracy: ☐
  - [ ] Per-round global loss: ☐
  - [ ] Per-lab local accuracy: ☐
  - [ ] Per-class precision/recall/F1: ☐
  - [ ] AUC-ROC (macro and per-class): ☐
  - [ ] Data distribution per lab: ☐

### Metric Validation
- [ ] Spot-check metrics make sense:
  - [ ] Accuracies between 0 and 1: ☐
  - [ ] Loss values reasonable: ☐
  - [ ] Per-class metrics sum correctly: ☐
  - [ ] Round numbers sequential: ☐

### Reproducibility
- [ ] Document experiment parameters:
  - [ ] Random seed: ________________
  - [ ] Model type: ________________
  - [ ] Learning rate: ________________
  - [ ] Number of rounds: ________________
- [ ] Save configuration for reproducibility

---

## Phase 9: Paper Writing Support ☐

### Tables for Paper
- [ ] Experiment summary table (from `experiment_summary.csv`)
- [ ] Per-class performance table (from `per_class_metrics.csv`)
- [ ] Data distribution table (from `lab_data_distribution.csv`)

### Figures for Paper
- [ ] All required figures generated (see Phase 6)
- [ ] Figures saved at 300+ DPI
- [ ] Figure captions prepared

### Statistical Analysis
- [ ] Calculate standard deviations if multiple runs
- [ ] Perform significance tests if comparing models
- [ ] Document any assumptions or limitations

---

## Phase 10: Future Enhancements ☐

### Optional Extensions
- [ ] Add Homomorphic Encryption metrics
- [ ] Implement differential privacy tracking
- [ ] Add fairness metrics
- [ ] Track communication costs
- [ ] Automate figure generation
- [ ] Integrate with MLflow

### Maintenance
- [ ] Document any custom modifications
- [ ] Archive old experiment data periodically
- [ ] Update documentation as needed

---

## Troubleshooting Checklist ☐

If something doesn't work:

### Database Issues
- [ ] Verify Supabase credentials in `.env`
- [ ] Check tables exist in database
- [ ] Verify RLS policies allow inserts
- [ ] Test connection with simple query

### Import Errors
- [ ] Check Python path includes `server/`
- [ ] Verify all dependencies installed
- [ ] Check for typos in import statements

### No Data in Exports
- [ ] Verify experiment_id matches
- [ ] Check logging is actually running
- [ ] Query database directly to confirm data exists

### Missing Metrics
- [ ] Ensure `log_global_aggregation()` is called
- [ ] Verify test dataset is available
- [ ] Check for errors in console output

---

## Completion ☐

- [ ] All phases completed
- [ ] Data exported successfully
- [ ] Figures generated
- [ ] Documentation reviewed
- [ ] Ready for paper writing

**Completion Date**: ________________

**Notes**:
_________________________________________________
_________________________________________________
_________________________________________________

---

## Quick Reference

### Key Commands
```bash
# Train baseline
python server/scripts/train_centralized_baseline.py --log-to-db

# Export metrics
python server/scripts/export_paper_metrics.py --all --output-dir paper_data

# Test metrics module
python server/fl_metrics.py

# Run setup script
./setup_fl_metrics.sh
```

### Key Files
- Database: `server/create_paper_metrics_tables.sql`
- Metrics: `server/fl_metrics.py`
- Logging: `server/fl_logging.py`
- Export: `server/scripts/export_paper_metrics.py`
- Guide: `docs/FL_PAPER_METRICS_GUIDE.md`

### Support
- Documentation: `docs/`
- Examples: `server/INTEGRATION_EXAMPLE.py`
- Architecture: `ARCHITECTURE_DIAGRAM.md`

---

**Remember**: This system is designed to be flexible. You can implement parts incrementally and customize as needed!
