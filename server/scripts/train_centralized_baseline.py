"""
Train Centralized Baseline and Log Metrics for PrivMed FL Paper

Extended version of train_baseline_model.py that:
1. Trains centralized model on all training data
2. Evaluates with comprehensive per-class metrics
3. Logs results to Supabase for paper analysis
4. Saves predictions for ROC curve generation

Usage:
    python train_centralized_baseline.py [--experiment-id EXP_ID] [--log-to-db]
"""

import os
import sys
import pickle
import json
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fl_metrics import (
    evaluate_model_comprehensive,
    format_per_class_metrics_for_db,
    print_evaluation_summary,
    extract_predictions_for_roc,
    compute_data_distribution
)

# Try to import Supabase (optional for standalone use)
try:
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv()
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("Warning: Supabase not available. Results will only be saved locally.")

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
MODELS_DIR = SCRIPT_DIR.parent / "models"

# Feature columns (must match lab_model.py)
FEATURE_COLUMNS = [
    'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'fasting_glucose', 'hba1c', 'total_cholesterol', 'ldl_cholesterol',
    'hdl_cholesterol', 'triglycerides', 'max_heart_rate', 'st_depression',
    'sex_encoded', 'chest_pain_type', 'resting_ecg', 'exercise_angina',
    'st_slope', 'smoking_status', 'family_history_cvd', 'family_history_diabetes',
    'prior_hypertension', 'prior_diabetes', 'prior_heart_disease',
    'on_bp_medication', 'on_diabetes_medication', 'on_cholesterol_medication'
]

CLASS_NAMES = ['healthy', 'diabetes', 'hypertension', 'heart_disease']


def get_supabase_client():
    """Get Supabase client if available."""
    if not SUPABASE_AVAILABLE:
        return None
    
    url = os.environ.get("SUPABASE_URL") or os.environ.get("VITE_SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("VITE_SUPABASE_ANON_KEY")
    
    if not url or not key:
        print("Warning: Supabase credentials not found in environment")
        return None
    
    return create_client(url, key)


def load_data():
    """Load train, validation, and test datasets."""
    train_path = DATA_DIR / "combined_train.csv"
    val_path = DATA_DIR / "combined_validation.csv"
    test_path = DATA_DIR / "combined_test.csv"
    
    if not train_path.exists():
        print(f"❌ Training data not found at {train_path}")
        print("   Run prepare_dataset.py first.")
        sys.exit(1)
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if val_path.exists() else None
    test_df = pd.read_csv(test_path) if test_path.exists() else None
    
    return train_df, val_df, test_df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Extract features and target from dataframe."""
    # Ensure all feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            if col == 'sex_encoded':
                df[col] = (df['sex'] == 'F').astype(int) if 'sex' in df.columns else 0
            else:
                df[col] = 0
    
    X = df[FEATURE_COLUMNS].fillna(0).values
    y = df['diagnosis'].values
    
    return X, y


def train_models(X_train, y_train, X_val, y_val):
    """Train multiple model types and compare performance."""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=500, 
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Validation set performance
        val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'val_accuracy': val_accuracy,
            'training_time': training_time
        }
        
        print(f"    • CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"    • Validation Accuracy: {val_accuracy:.4f}")
        print(f"    • Training Time: {training_time:.2f}s")
    
    return results


def select_best_model(results: dict) -> tuple:
    """Select the best performing model."""
    best_name = max(results.keys(), key=lambda x: results[x]['val_accuracy'])
    best_model = results[best_name]['model']
    best_training_time = results[best_name]['training_time']
    
    print(f"\n🏆 Best model: {best_name}")
    print(f"   Validation Accuracy: {results[best_name]['val_accuracy']:.4f}")
    
    return best_name, best_model, best_training_time


def save_model_and_artifacts(model, scaler, model_name: str, 
                             evaluation: dict, config: dict):
    """Save model, scaler, and comprehensive metadata."""
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Save model bundle
    bundle = {
        'model': model,
        'scaler': scaler,
        'features': FEATURE_COLUMNS,
        'class_names': CLASS_NAMES,
        'model_type': model_name,
        'training_date': datetime.now().isoformat(),
        'config': config
    }
    
    model_path = MODELS_DIR / "baseline_global_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save feature importances
    if hasattr(model, 'feature_importances_'):
        importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        
        importance_path = MODELS_DIR / "feature_importances.json"
        with open(importance_path, 'w') as f:
            json.dump(sorted_importances, f, indent=2)
        print(f"💾 Feature importances saved to: {importance_path}")
        
        # Print top 10
        print("\n🔍 Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(list(sorted_importances.items())[:10]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
    
    # Save comprehensive evaluation metrics
    metrics_path = MODELS_DIR / "centralized_baseline_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f"💾 Evaluation metrics saved to: {metrics_path}")
    
    return str(model_path)


def log_to_database(experiment_id: str, model_name: str, evaluation: dict,
                    training_samples: int, val_samples: int, test_samples: int,
                    training_time: float, model_path: str, config: dict):
    """Log centralized baseline results to Supabase."""
    sb = get_supabase_client()
    if sb is None:
        print("\n⚠️  Skipping database logging (Supabase not available)")
        return
    
    try:
        # Log to fl_centralized_baselines table
        baseline_record = {
            'experiment_id': experiment_id,
            'model_name': model_name,
            'accuracy': evaluation['accuracy'],
            'loss': evaluation['loss'],
            'per_class_metrics': format_per_class_metrics_for_db(evaluation['per_class_metrics']),
            'auc_roc_macro': evaluation.get('auc_roc_macro'),
            'auc_roc_per_class': evaluation.get('auc_roc_per_class'),
            'training_samples': training_samples,
            'validation_samples': val_samples,
            'test_samples': test_samples,
            'training_time_seconds': training_time,
            'model_config': config,
            'model_path': model_path
        }
        
        result = sb.table('fl_centralized_baselines').insert(baseline_record).execute()
        print(f"\n✅ Logged to fl_centralized_baselines (ID: {result.data[0]['id']})")
        
        # Log per-class performance to fl_per_class_performance table
        for class_name, metrics in evaluation['per_class_metrics'].items():
            class_record = {
                'experiment_id': experiment_id,
                'round': None,  # No round for centralized
                'model_type': 'centralized',
                'lab_label': None,  # No lab for centralized
                'class_name': class_name,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'specificity': metrics['specificity'],
                'f1_score': metrics['f1'],
                'support': metrics['support'],
                'auc_roc': metrics.get('auc_roc')
            }
            sb.table('fl_per_class_performance').insert(class_record).execute()
        
        print(f"✅ Logged per-class metrics for {len(evaluation['per_class_metrics'])} classes")
        
        # Log predictions for ROC curves (optional, can be large)
        predictions_data = extract_predictions_for_roc(
            np.array(evaluation['predictions']['y_true']),
            np.array(evaluation['predictions']['y_pred']),
            np.array(evaluation['predictions']['y_proba'])
        )
        
        predictions_record = {
            'experiment_id': experiment_id,
            'round': None,
            'model_type': 'centralized',
            'predictions': predictions_data,
            'total_predictions': len(predictions_data),
            'dataset_type': 'test'
        }
        
        sb.table('fl_model_predictions').insert(predictions_record).execute()
        print(f"✅ Logged {len(predictions_data)} predictions for ROC analysis")
        
    except Exception as e:
        print(f"\n❌ Error logging to database: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main training pipeline with comprehensive logging."""
    parser = argparse.ArgumentParser(description='Train centralized baseline model with logging')
    parser.add_argument('--experiment-id', type=str, 
                       default=f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help='Experiment ID for tracking')
    parser.add_argument('--log-to-db', action='store_true',
                       help='Log results to Supabase database')
    parser.add_argument('--model-type', type=str, default='auto',
                       choices=['auto', 'logistic', 'random_forest', 'gradient_boosting'],
                       help='Model type to train (auto selects best)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PrivMed Centralized Baseline Training with Metrics Logging")
    print("=" * 60)
    print(f"\nExperiment ID: {args.experiment_id}")
    print(f"Log to DB: {args.log_to_db}")
    
    # Load data
    print("\n📂 Loading datasets...")
    train_df, val_df, test_df = load_data()
    print(f"  • Train: {len(train_df)} samples")
    print(f"  • Validation: {len(val_df) if val_df is not None else 0} samples")
    print(f"  • Test: {len(test_df) if test_df is not None else 0} samples")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df) if val_df is not None else (X_train[:100], y_train[:100])
    X_test, y_test = prepare_features(test_df) if test_df is not None else (None, None)
    
    print(f"  • Feature count: {X_train.shape[1]}")
    print(f"  • Classes: {np.unique(y_train)}")
    print(f"  • Class distribution: {compute_data_distribution(y_train)}")
    
    # Scale features
    print("\n⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    
    # Train models
    print("\n🚀 Training models...")
    results = train_models(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Select best model
    best_name, best_model, training_time = select_best_model(results)
    
    # Comprehensive evaluation on test set
    if X_test_scaled is not None:
        print("\n📊 Comprehensive evaluation on test set...")
        evaluation = evaluate_model_comprehensive(
            best_model, X_test_scaled, y_test, 
            model_type='linear',  # All models use scaled features
            class_names=CLASS_NAMES
        )
        print_evaluation_summary(evaluation, title=f"{best_name} - Test Set Evaluation")
    else:
        print("\n⚠️  No test set available, using validation metrics")
        evaluation = {'accuracy': results[best_name]['val_accuracy']}
    
    # Model configuration
    config = {
        'model_type': best_name,
        'random_seed': 42,
        'features': FEATURE_COLUMNS,
        'scaler': 'StandardScaler'
    }
    
    # Save model and artifacts
    model_path = save_model_and_artifacts(
        best_model, scaler, best_name, evaluation, config
    )
    
    # Log to database if requested
    if args.log_to_db:
        print("\n📤 Logging to database...")
        log_to_database(
            experiment_id=args.experiment_id,
            model_name=best_name,
            evaluation=evaluation,
            training_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=len(X_test) if X_test is not None else 0,
            training_time=training_time,
            model_path=model_path,
            config=config
        )
    
    print("\n" + "=" * 60)
    print("✅ Training and logging complete!")
    print("=" * 60)
    
    # Summary
    if 'accuracy' in evaluation:
        if evaluation['accuracy'] < 0.75:
            print("\n⚠️  Warning: Model accuracy is below 75% threshold.")
            print("   Consider collecting more data or tuning hyperparameters.")
        else:
            print(f"\n✅ Model meets accuracy requirement: {evaluation['accuracy']:.2%}")
    
    print(f"\n📁 Results saved to: {MODELS_DIR}")
    print(f"   • Model: baseline_global_model.pkl")
    print(f"   • Metrics: centralized_baseline_metrics.json")
    if args.log_to_db:
        print(f"   • Database: fl_centralized_baselines, fl_per_class_performance")


if __name__ == "__main__":
    main()
