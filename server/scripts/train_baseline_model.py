"""
Train Baseline Model for PrivMed Federated Learning

This script trains a baseline global model on the combined real dataset.
The model will be shipped as the initial global model for federated learning.

Usage:
    python train_baseline_model.py
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
import warnings
warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
MODELS_DIR = SCRIPT_DIR.parent / "models"

# Feature columns for model (must match lab_model.py)
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


def load_data():
    """Load train, validation, and test datasets."""
    train_path = DATA_DIR / "combined_train.csv"
    val_path = DATA_DIR / "combined_validation.csv"
    test_path = DATA_DIR / "combined_test.csv"
    
    if not train_path.exists():
        print("❌ Training data not found! Run prepare_dataset.py first.")
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
            # Add missing columns with defaults
            if col == 'sex_encoded':
                df[col] = (df['sex'] == 'F').astype(int) if 'sex' in df.columns else 0
            else:
                df[col] = 0
    
    X = df[FEATURE_COLUMNS].fillna(0).values
    y = df['diagnosis'].values
    
    return X, y


def train_and_evaluate_models(X_train, y_train, X_val, y_val):
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
        model.fit(X_train, y_train)
        
        # Cross-validation on training data
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Validation set performance
        val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'val_accuracy': val_accuracy
        }
        
        print(f"    • CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"    • Validation Accuracy: {val_accuracy:.4f}")
    
    return results


def select_best_model(results: dict) -> tuple:
    """Select the best performing model."""
    best_name = max(results.keys(), key=lambda x: results[x]['val_accuracy'])
    best_model = results[best_name]['model']
    
    print(f"\n🏆 Best model: {best_name}")
    print(f"   Validation Accuracy: {results[best_name]['val_accuracy']:.4f}")
    
    return best_name, best_model


def evaluate_on_test(model, X_test, y_test):
    """Comprehensive evaluation on held-out test set."""
    print("\n" + "=" * 60)
    print("Test Set Evaluation (Held-out data)")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 Overall Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📈 Confusion Matrix:")
    print(f"{'':15} " + " ".join([f"{name:12}" for name in CLASS_NAMES]))
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:15} " + " ".join([f"{val:12}" for val in row]))
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
    
    metrics = {
        CLASS_NAMES[i]: {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
        for i in range(len(CLASS_NAMES))
    }
    
    return {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': metrics
    }


def save_model(model, scaler, model_name: str, test_metrics: dict):
    """Save model, scaler, and metadata."""
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Save model bundle (model + scaler + features)
    bundle = {
        'model': model,
        'scaler': scaler,
        'features': FEATURE_COLUMNS,
        'class_names': CLASS_NAMES,
        'model_type': model_name
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
        
        # Print top 10 features
        print("\n🔍 Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(list(sorted_importances.items())[:10]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
    
    # Save test metrics
    metrics_path = MODELS_DIR / "baseline_model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"💾 Test metrics saved to: {metrics_path}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("PrivMed Baseline Model Training")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading datasets...")
    train_df, val_df, test_df = load_data()
    print(f"  • Train: {len(train_df)} samples")
    print(f"  • Validation: {len(val_df) if val_df is not None else 0} samples")
    print(f"  • Test: {len(test_df) if test_df is not None else 0} samples")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df) if val_df is not None else (None, None)
    X_test, y_test = prepare_features(test_df) if test_df is not None else (None, None)
    
    print(f"  • Feature count: {X_train.shape[1]}")
    print(f"  • Classes: {np.unique(y_train)}")
    
    # Scale features
    print("\n⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    
    # Train and compare models
    print("\n🚀 Training models...")
    results = train_and_evaluate_models(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Select best model
    best_name, best_model = select_best_model(results)
    
    # Evaluate on test set
    if X_test_scaled is not None:
        test_metrics = evaluate_on_test(best_model, X_test_scaled, y_test)
    else:
        test_metrics = {'accuracy': results[best_name]['val_accuracy']}
    
    # Save model and artifacts
    save_model(best_model, scaler, best_name, test_metrics)
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    
    # Check accuracy threshold
    if test_metrics['accuracy'] < 0.75:
        print("\n⚠️  Warning: Model accuracy is below 75% threshold.")
        print("   Consider collecting more data or tuning hyperparameters.")
    else:
        print(f"\n✅ Model meets accuracy requirement: {test_metrics['accuracy']:.2%}")


if __name__ == "__main__":
    main()
