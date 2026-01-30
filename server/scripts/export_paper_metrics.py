"""
Export PrivMed FL Paper Metrics

Exports all collected federated learning metrics to CSV and JSON formats
for paper analysis and visualization.

Exports:
1. Experiment summary (centralized vs federated comparison)
2. Per-round metrics (accuracy, loss, convergence)
3. Per-lab per-round metrics (data imbalance, local performance)
4. Per-class metrics (precision, recall, specificity, F1, AUC-ROC)
5. Lab data distribution (samples per lab)
6. Model predictions (for ROC curve generation)

Usage:
    python export_paper_metrics.py --experiment-id EXP_ID --output-dir ./paper_data
    python export_paper_metrics.py --all --output-dir ./paper_data
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Supabase
try:
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv()
    SUPABASE_AVAILABLE = True
except ImportError:
    print("Error: Supabase client not available. Install with: pip install supabase python-dotenv")
    sys.exit(1)


def get_supabase_client():
    """Get Supabase client."""
    url = os.environ.get("SUPABASE_URL") or os.environ.get("VITE_SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("VITE_SUPABASE_ANON_KEY")
    
    if not url or not key:
        print("Error: Supabase credentials not found in environment")
        print("Set SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_ANON_KEY)")
        sys.exit(1)
    
    return create_client(url, key)


class PaperMetricsExporter:
    """Export federated learning metrics for paper analysis."""
    
    def __init__(self, supabase_client, output_dir: str = "./paper_data"):
        """
        Initialize exporter.
        
        Args:
            supabase_client: Supabase client instance
            output_dir: Directory to save exported files
        """
        self.sb = supabase_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}")
    
    def export_experiment_summary(self, experiment_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Export experiment-level summary.
        
        Columns: experiment_id, centralized_accuracy, fed_accuracy, fed_he_accuracy,
                 auc_roc_macro, total_rounds, num_clients, created_at
        
        Args:
            experiment_ids: List of experiment IDs to export (None for all)
        
        Returns:
            DataFrame with experiment summaries
        """
        print("\n📊 Exporting experiment summary...")
        
        query = self.sb.table('fl_experiment_log').select('*')
        if experiment_ids:
            query = query.in_('experiment_id', experiment_ids)
        
        result = query.execute()
        
        if not result.data:
            print("  ⚠️  No experiment data found")
            return pd.DataFrame()
        
        # Flatten JSONB fields for easier analysis
        records = []
        for row in result.data:
            record = {
                'experiment_id': row['experiment_id'],
                'experiment_name': row.get('experiment_name'),
                'centralized_accuracy': row.get('centralized_accuracy'),
                'federated_accuracy': row.get('federated_accuracy'),
                'federated_he_accuracy': row.get('federated_he_accuracy'),
                'final_global_loss': row.get('final_global_loss'),
                'final_validation_loss': row.get('final_validation_loss'),
                'auc_roc_macro': row.get('auc_roc_macro'),
                'total_rounds': row.get('total_rounds'),
                'num_clients': row.get('num_clients'),
                'model_type': row.get('model_type'),
                'random_seed': row.get('random_seed'),
                'created_at': row.get('created_at'),
                'notes': row.get('notes')
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Save to CSV
        output_path = self.output_dir / "experiment_summary.csv"
        df.to_csv(output_path, index=False)
        print(f"  ✓ Saved to {output_path} ({len(df)} experiments)")
        
        # Also save detailed JSON with per-class metrics
        json_path = self.output_dir / "experiment_summary_detailed.json"
        with open(json_path, 'w') as f:
            json.dump(result.data, f, indent=2, default=str)
        print(f"  ✓ Saved detailed JSON to {json_path}")
        
        return df
    
    def export_rounds_summary(self, experiment_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Export per-round global metrics.
        
        Columns: experiment_id, round, global_accuracy, global_train_loss,
                 global_val_loss, aggregated_grad_norm
        
        Args:
            experiment_ids: List of experiment IDs to export (None for all)
        
        Returns:
            DataFrame with per-round metrics
        """
        print("\n📊 Exporting per-round metrics...")
        
        query = self.sb.table('fl_round_detailed_metrics') \
            .select('*') \
            .eq('is_global', True) \
            .order('experiment_id') \
            .order('round')
        
        if experiment_ids:
            query = query.in_('experiment_id', experiment_ids)
        
        result = query.execute()
        
        if not result.data:
            print("  ⚠️  No round metrics found")
            return pd.DataFrame()
        
        df = pd.DataFrame(result.data)
        
        # Select relevant columns
        columns = ['experiment_id', 'round', 'global_accuracy', 'global_train_loss',
                   'global_val_loss', 'grad_norm', 'aggregation_method', 'created_at']
        df = df[[col for col in columns if col in df.columns]]
        
        # Rename for clarity
        df = df.rename(columns={'grad_norm': 'aggregated_grad_norm'})
        
        # Save to CSV
        output_path = self.output_dir / "rounds_summary.csv"
        df.to_csv(output_path, index=False)
        print(f"  ✓ Saved to {output_path} ({len(df)} rounds)")
        
        return df
    
    def export_per_lab_per_round(self, experiment_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Export per-lab per-round metrics.
        
        Columns: experiment_id, round, lab_label, local_accuracy, local_train_loss,
                 local_val_loss, num_examples, weight_update_magnitude
        
        Args:
            experiment_ids: List of experiment IDs to export (None for all)
        
        Returns:
            DataFrame with per-lab per-round metrics
        """
        print("\n📊 Exporting per-lab per-round metrics...")
        
        query = self.sb.table('fl_round_detailed_metrics') \
            .select('*') \
            .eq('is_global', False) \
            .order('experiment_id') \
            .order('round') \
            .order('lab_label')
        
        if experiment_ids:
            query = query.in_('experiment_id', experiment_ids)
        
        result = query.execute()
        
        if not result.data:
            print("  ⚠️  No per-lab metrics found")
            return pd.DataFrame()
        
        df = pd.DataFrame(result.data)
        
        # Select relevant columns
        columns = ['experiment_id', 'round', 'lab_label', 'local_accuracy',
                   'local_train_loss', 'local_val_loss', 'num_examples',
                   'grad_norm', 'weight_update_magnitude', 'training_time_seconds', 'created_at']
        df = df[[col for col in columns if col in df.columns]]
        
        # Save to CSV
        output_path = self.output_dir / "per_lab_per_round.csv"
        df.to_csv(output_path, index=False)
        print(f"  ✓ Saved to {output_path} ({len(df)} lab-round pairs)")
        
        return df
    
    def export_per_class_metrics(self, experiment_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Export per-class performance metrics.
        
        Columns: experiment_id, round, model_type, class_name, precision, recall,
                 specificity, f1_score, support, auc_roc
        
        Args:
            experiment_ids: List of experiment IDs to export (None for all)
        
        Returns:
            DataFrame with per-class metrics
        """
        print("\n📊 Exporting per-class metrics...")
        
        query = self.sb.table('fl_per_class_performance') \
            .select('*') \
            .order('experiment_id') \
            .order('round') \
            .order('class_name')
        
        if experiment_ids:
            query = query.in_('experiment_id', experiment_ids)
        
        result = query.execute()
        
        if not result.data:
            print("  ⚠️  No per-class metrics found")
            return pd.DataFrame()
        
        df = pd.DataFrame(result.data)
        
        # Select relevant columns
        columns = ['experiment_id', 'round', 'model_type', 'lab_label', 'class_name',
                   'precision', 'recall', 'specificity', 'f1_score', 'support', 'auc_roc']
        df = df[[col for col in columns if col in df.columns]]
        
        # Save to CSV
        output_path = self.output_dir / "per_class_metrics.csv"
        df.to_csv(output_path, index=False)
        print(f"  ✓ Saved to {output_path} ({len(df)} class-level records)")
        
        return df
    
    def export_lab_data_distribution(self, experiment_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Export lab data distribution (samples per lab).
        
        Columns: experiment_id, round, lab_label, total_samples, samples_per_class (JSON)
        
        Args:
            experiment_ids: List of experiment IDs to export (None for all)
        
        Returns:
            DataFrame with lab data distribution
        """
        print("\n📊 Exporting lab data distribution...")
        
        query = self.sb.table('fl_lab_data_distribution') \
            .select('*') \
            .order('experiment_id') \
            .order('lab_label')
        
        if experiment_ids:
            query = query.in_('experiment_id', experiment_ids)
        
        result = query.execute()
        
        if not result.data:
            print("  ⚠️  No lab data distribution found")
            return pd.DataFrame()
        
        df = pd.DataFrame(result.data)
        
        # Flatten samples_per_class for easier analysis
        if 'samples_per_class' in df.columns:
            # Extract individual class counts
            for class_name in ['healthy', 'diabetes', 'hypertension', 'heart_disease']:
                df[f'samples_{class_name}'] = df['samples_per_class'].apply(
                    lambda x: x.get(class_name, 0) if isinstance(x, dict) else 0
                )
        
        # Save to CSV
        output_path = self.output_dir / "lab_data_distribution.csv"
        df.to_csv(output_path, index=False)
        print(f"  ✓ Saved to {output_path} ({len(df)} lab records)")
        
        return df
    
    def export_centralized_baselines(self, experiment_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Export centralized baseline model metrics.
        
        Args:
            experiment_ids: List of experiment IDs to export (None for all)
        
        Returns:
            DataFrame with centralized baseline metrics
        """
        print("\n📊 Exporting centralized baselines...")
        
        query = self.sb.table('fl_centralized_baselines').select('*')
        
        if experiment_ids:
            query = query.in_('experiment_id', experiment_ids)
        
        result = query.execute()
        
        if not result.data:
            print("  ⚠️  No centralized baseline data found")
            return pd.DataFrame()
        
        # Save full JSON for per-class metrics
        json_path = self.output_dir / "centralized_baselines_detailed.json"
        with open(json_path, 'w') as f:
            json.dump(result.data, f, indent=2, default=str)
        print(f"  ✓ Saved detailed JSON to {json_path}")
        
        # Create summary CSV
        records = []
        for row in result.data:
            record = {
                'experiment_id': row.get('experiment_id'),
                'model_name': row.get('model_name'),
                'accuracy': row.get('accuracy'),
                'loss': row.get('loss'),
                'auc_roc_macro': row.get('auc_roc_macro'),
                'training_samples': row.get('training_samples'),
                'test_samples': row.get('test_samples'),
                'training_time_seconds': row.get('training_time_seconds'),
                'created_at': row.get('created_at')
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        output_path = self.output_dir / "centralized_baselines.csv"
        df.to_csv(output_path, index=False)
        print(f"  ✓ Saved summary to {output_path} ({len(df)} baselines)")
        
        return df
    
    def export_model_predictions(self, experiment_ids: Optional[List[str]] = None,
                                 limit: int = 10000) -> dict:
        """
        Export model predictions for ROC curve generation.
        
        Warning: This can be large. Use limit to control size.
        
        Args:
            experiment_ids: List of experiment IDs to export (None for all)
            limit: Maximum predictions per query
        
        Returns:
            Dict mapping experiment_id to predictions
        """
        print(f"\n📊 Exporting model predictions (limit: {limit} per experiment)...")
        
        query = self.sb.table('fl_model_predictions') \
            .select('*') \
            .order('experiment_id') \
            .order('round')
        
        if experiment_ids:
            query = query.in_('experiment_id', experiment_ids)
        
        result = query.limit(limit).execute()
        
        if not result.data:
            print("  ⚠️  No prediction data found")
            return {}
        
        # Save as JSON (predictions field contains arrays)
        json_path = self.output_dir / "model_predictions.json"
        with open(json_path, 'w') as f:
            json.dump(result.data, f, indent=2, default=str)
        print(f"  ✓ Saved to {json_path} ({len(result.data)} prediction sets)")
        
        # Create summary CSV
        records = []
        for row in result.data:
            record = {
                'experiment_id': row.get('experiment_id'),
                'round': row.get('round'),
                'model_type': row.get('model_type'),
                'total_predictions': row.get('total_predictions'),
                'dataset_type': row.get('dataset_type'),
                'created_at': row.get('created_at')
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        summary_path = self.output_dir / "predictions_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"  ✓ Saved summary to {summary_path}")
        
        return {row['experiment_id']: row for row in result.data}
    
    def export_all(self, experiment_ids: Optional[List[str]] = None):
        """
        Export all metrics.
        
        Args:
            experiment_ids: List of experiment IDs to export (None for all)
        """
        print("=" * 60)
        print("Exporting All PrivMed FL Paper Metrics")
        print("=" * 60)
        
        # Export all tables
        self.export_experiment_summary(experiment_ids)
        self.export_rounds_summary(experiment_ids)
        self.export_per_lab_per_round(experiment_ids)
        self.export_per_class_metrics(experiment_ids)
        self.export_lab_data_distribution(experiment_ids)
        self.export_centralized_baselines(experiment_ids)
        self.export_model_predictions(experiment_ids)
        
        # Create README
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"""# PrivMed FL Paper Metrics Export

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Files

### CSV Files (for plotting/analysis)

1. **experiment_summary.csv**
   - Experiment-level comparison: centralized vs federated vs federated+HE
   - Columns: experiment_id, centralized_accuracy, federated_accuracy, auc_roc_macro, etc.

2. **rounds_summary.csv**
   - Per-round global metrics: accuracy, loss, convergence
   - Columns: experiment_id, round, global_accuracy, global_train_loss, global_val_loss, aggregated_grad_norm

3. **per_lab_per_round.csv**
   - Per-lab per-round metrics: local accuracy, loss, samples
   - Columns: experiment_id, round, lab_label, local_accuracy, num_examples, etc.

4. **per_class_metrics.csv**
   - Per-class performance: precision, recall, specificity, F1, AUC-ROC
   - Columns: experiment_id, round, model_type, class_name, precision, recall, specificity, f1_score, auc_roc

5. **lab_data_distribution.csv**
   - Data imbalance across labs
   - Columns: experiment_id, lab_label, total_samples, samples_healthy, samples_diabetes, etc.

6. **centralized_baselines.csv**
   - Centralized baseline model performance
   - Columns: experiment_id, model_name, accuracy, loss, auc_roc_macro

### JSON Files (detailed data)

1. **experiment_summary_detailed.json**
   - Full experiment data including nested per-class metrics

2. **centralized_baselines_detailed.json**
   - Detailed centralized baseline metrics with per-class breakdown

3. **model_predictions.json**
   - Raw predictions for ROC curve generation (can be large)

## Usage with Python/Pandas

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
exp_summary = pd.read_csv('experiment_summary.csv')
rounds = pd.read_csv('rounds_summary.csv')
per_lab = pd.read_csv('per_lab_per_round.csv')
per_class = pd.read_csv('per_class_metrics.csv')

# Example: Plot accuracy vs rounds
plt.figure(figsize=(10, 6))
for exp_id in rounds['experiment_id'].unique():
    data = rounds[rounds['experiment_id'] == exp_id]
    plt.plot(data['round'], data['global_accuracy'], label=exp_id)
plt.xlabel('Federated Round')
plt.ylabel('Global Accuracy')
plt.legend()
plt.title('Federated Learning Convergence')
plt.savefig('fl_convergence.png')
```

## Figures to Generate

Based on these exports, you can generate:

1. **Accuracy vs Rounds** (rounds_summary.csv)
2. **Loss vs Rounds** (rounds_summary.csv)
3. **Centralized vs FL vs FL+HE Bar Chart** (experiment_summary.csv)
4. **Per-Lab Accuracy Comparison** (per_lab_per_round.csv)
5. **Per-Class Performance** (per_class_metrics.csv)
6. **Data Imbalance** (lab_data_distribution.csv)
7. **ROC Curves** (model_predictions.json + per_class_metrics.csv)
8. **Convergence Analysis** (rounds_summary.csv)
""")
        
        print(f"\n✅ Export complete! All files saved to: {self.output_dir}")
        print(f"   See {readme_path} for usage instructions")


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(
        description='Export PrivMed FL metrics for paper analysis'
    )
    parser.add_argument('--experiment-id', type=str, nargs='+',
                       help='Experiment ID(s) to export (default: all)')
    parser.add_argument('--output-dir', type=str, default='./paper_data',
                       help='Output directory (default: ./paper_data)')
    parser.add_argument('--all', action='store_true',
                       help='Export all tables')
    parser.add_argument('--summary-only', action='store_true',
                       help='Export only summary tables (no predictions)')
    
    args = parser.parse_args()
    
    # Get Supabase client
    sb = get_supabase_client()
    
    # Create exporter
    exporter = PaperMetricsExporter(sb, args.output_dir)
    
    # Export based on options
    if args.all or (not args.summary_only and not args.experiment_id):
        exporter.export_all(args.experiment_id)
    else:
        # Export specific tables
        if args.summary_only:
            exporter.export_experiment_summary(args.experiment_id)
            exporter.export_rounds_summary(args.experiment_id)
            exporter.export_per_class_metrics(args.experiment_id)
        else:
            exporter.export_all(args.experiment_id)


if __name__ == "__main__":
    main()
