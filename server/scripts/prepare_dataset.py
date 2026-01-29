"""
Prepare Unified Dataset for MedSafe Federated Learning

This script:
1. Loads raw datasets (heart disease, diabetes, hypertension, healthy)
2. Maps them to a unified clinical schema
3. Handles missing values with imputation
4. Balances classes using undersampling/SMOTE
5. Splits into train/validation/test sets
6. Creates lab-specific data splits for federated learning simulation

Usage:
    python prepare_dataset.py
"""

import os
import sys
import json
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"

# Unified schema columns
UNIFIED_COLUMNS = {
    # Demographics
    'patient_id': str,
    'age': int,
    'sex': str,
    'height_cm': float,
    'weight_kg': float,
    'bmi': float,
    
    # Vital Signs
    'systolic_bp': int,
    'diastolic_bp': int,
    'heart_rate': int,
    
    # Blood Chemistry - Diabetes Markers
    'fasting_glucose': int,
    'hba1c': float,
    'insulin': float,
    
    # Blood Chemistry - Lipid Panel
    'total_cholesterol': int,
    'ldl_cholesterol': int,
    'hdl_cholesterol': int,
    'triglycerides': int,
    
    # Cardiac Markers
    'chest_pain_type': int,
    'resting_ecg': int,
    'max_heart_rate': int,
    'exercise_angina': int,
    'st_depression': float,
    'st_slope': int,
    
    # Medical History
    'smoking_status': int,
    'family_history_cvd': int,
    'family_history_diabetes': int,
    'prior_hypertension': int,
    'prior_diabetes': int,
    'prior_heart_disease': int,
    
    # Current Medications
    'on_bp_medication': int,
    'on_diabetes_medication': int,
    'on_cholesterol_medication': int,
    
    # Target
    'diagnosis': int,
    'diagnosis_source': str
}

# Feature columns for model (excludes patient_id, diagnosis_source)
MODEL_FEATURES = [
    'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'fasting_glucose', 'hba1c', 'total_cholesterol', 'ldl_cholesterol',
    'hdl_cholesterol', 'triglycerides', 'max_heart_rate', 'st_depression',
    'sex_encoded', 'chest_pain_type', 'resting_ecg', 'exercise_angina',
    'st_slope', 'smoking_status', 'family_history_cvd', 'family_history_diabetes',
    'prior_hypertension', 'prior_diabetes', 'prior_heart_disease',
    'on_bp_medication', 'on_diabetes_medication', 'on_cholesterol_medication'
]


def load_raw_datasets() -> Dict[str, pd.DataFrame]:
    """Load all raw datasets from disk."""
    datasets = {}
    
    # Heart Disease UCI
    heart_path = RAW_DIR / "heart_disease_uci.csv"
    if heart_path.exists():
        datasets['heart'] = pd.read_csv(heart_path)
        print(f"  ✓ Loaded heart disease: {len(datasets['heart'])} records")
    
    # Pima Diabetes
    diabetes_path = RAW_DIR / "pima_diabetes.csv"
    if diabetes_path.exists():
        datasets['diabetes'] = pd.read_csv(diabetes_path)
        print(f"  ✓ Loaded diabetes: {len(datasets['diabetes'])} records")
    
    # Hypertension (synthetic)
    hypertension_path = RAW_DIR / "hypertension_synthetic.csv"
    if hypertension_path.exists():
        datasets['hypertension'] = pd.read_csv(hypertension_path)
        print(f"  ✓ Loaded hypertension: {len(datasets['hypertension'])} records")
    
    # Healthy (synthetic)
    healthy_path = RAW_DIR / "healthy_synthetic.csv"
    if healthy_path.exists():
        datasets['healthy'] = pd.read_csv(healthy_path)
        print(f"  ✓ Loaded healthy: {len(datasets['healthy'])} records")
    
    return datasets


def create_empty_unified_row() -> Dict[str, Any]:
    """Create an empty row with default values."""
    return {
        'patient_id': str(uuid.uuid4()),
        'age': 0,
        'sex': 'M',
        'height_cm': np.nan,
        'weight_kg': np.nan,
        'bmi': np.nan,
        'systolic_bp': np.nan,
        'diastolic_bp': np.nan,
        'heart_rate': np.nan,
        'fasting_glucose': np.nan,
        'hba1c': np.nan,
        'insulin': np.nan,
        'total_cholesterol': np.nan,
        'ldl_cholesterol': np.nan,
        'hdl_cholesterol': np.nan,
        'triglycerides': np.nan,
        'chest_pain_type': 4,  # Asymptomatic default
        'resting_ecg': 0,
        'max_heart_rate': np.nan,
        'exercise_angina': 0,
        'st_depression': 0.0,
        'st_slope': 2,  # Flat default
        'smoking_status': 0,
        'family_history_cvd': 0,
        'family_history_diabetes': 0,
        'prior_hypertension': 0,
        'prior_diabetes': 0,
        'prior_heart_disease': 0,
        'on_bp_medication': 0,
        'on_diabetes_medication': 0,
        'on_cholesterol_medication': 0,
        'diagnosis': 0,
        'diagnosis_source': 'unknown'
    }


def map_heart_disease(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map UCI Heart Disease dataset to unified schema.
    Diagnosis: 0 = healthy, 3 = heart_disease
    """
    unified_rows = []
    
    for _, row in df.iterrows():
        unified = create_empty_unified_row()
        unified['patient_id'] = str(uuid.uuid4())
        unified['age'] = int(row['age']) if pd.notna(row['age']) else 50
        unified['sex'] = 'M' if row['sex'] == 1 else 'F'
        
        # Map chest pain type (1-4 in UCI, same in unified)
        unified['chest_pain_type'] = int(row['cp']) if pd.notna(row['cp']) else 4
        
        # Blood pressure
        unified['systolic_bp'] = int(row['trestbps']) if pd.notna(row['trestbps']) else 120
        
        # Cholesterol
        unified['total_cholesterol'] = int(row['chol']) if pd.notna(row['chol']) else 200
        
        # Fasting blood sugar > 120 mg/dl (fbs = 1 means high)
        if pd.notna(row['fbs']) and row['fbs'] == 1:
            unified['fasting_glucose'] = 140
        else:
            unified['fasting_glucose'] = 95
        
        # Resting ECG (0, 1, 2 in UCI, same in unified)
        unified['resting_ecg'] = int(row['restecg']) if pd.notna(row['restecg']) else 0
        
        # Max heart rate achieved
        unified['max_heart_rate'] = int(row['thalach']) if pd.notna(row['thalach']) else 150
        
        # Exercise induced angina (1 = yes, 0 = no)
        unified['exercise_angina'] = int(row['exang']) if pd.notna(row['exang']) else 0
        
        # ST depression
        unified['st_depression'] = float(row['oldpeak']) if pd.notna(row['oldpeak']) else 0.0
        
        # ST slope (1, 2, 3 in UCI, same in unified)
        unified['st_slope'] = int(row['slope']) if pd.notna(row['slope']) else 2
        
        # Estimate other values
        unified['bmi'] = np.random.normal(27, 4)
        unified['diastolic_bp'] = int(unified['systolic_bp'] * 0.65)
        unified['heart_rate'] = int(unified['max_heart_rate'] * 0.5) if unified['max_heart_rate'] else 70
        
        # Target: 0 = healthy, 1 = heart disease → map to 0 or 3
        if row['target'] == 1:
            unified['diagnosis'] = 3  # heart_disease
            unified['prior_heart_disease'] = 0  # No prior, this is the diagnosis
        else:
            unified['diagnosis'] = 0  # healthy
        
        unified['diagnosis_source'] = 'heart_uci'
        unified_rows.append(unified)
    
    return pd.DataFrame(unified_rows)


def map_diabetes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Pima Indians Diabetes dataset to unified schema.
    Diagnosis: 0 = healthy, 1 = diabetes
    """
    unified_rows = []
    
    for _, row in df.iterrows():
        unified = create_empty_unified_row()
        unified['patient_id'] = str(uuid.uuid4())
        unified['age'] = int(row['age']) if pd.notna(row['age']) else 35
        unified['sex'] = 'F'  # Pima dataset is all female
        
        # BMI
        unified['bmi'] = float(row['bmi']) if pd.notna(row['bmi']) and row['bmi'] > 0 else 28.0
        
        # Blood pressure
        bp = row['blood_pressure']
        unified['diastolic_bp'] = int(bp) if pd.notna(bp) and bp > 0 else 80
        unified['systolic_bp'] = int(unified['diastolic_bp'] * 1.5) + 40
        
        # Glucose
        glucose = row['glucose']
        unified['fasting_glucose'] = int(glucose) if pd.notna(glucose) and glucose > 0 else 100
        
        # Insulin
        insulin = row['insulin']
        unified['insulin'] = float(insulin) if pd.notna(insulin) and insulin > 0 else np.nan
        
        # Estimate HbA1c from fasting glucose (rough approximation)
        if unified['fasting_glucose'] > 0:
            unified['hba1c'] = (unified['fasting_glucose'] + 46.7) / 28.7
        
        # Estimate other values
        unified['heart_rate'] = np.random.randint(65, 85)
        unified['total_cholesterol'] = np.random.randint(180, 240)
        unified['max_heart_rate'] = 220 - unified['age']
        
        # Family history based on diabetes pedigree function
        if pd.notna(row['diabetes_pedigree']) and row['diabetes_pedigree'] > 0.5:
            unified['family_history_diabetes'] = 1
        
        # Pregnancies > 0 might indicate prior gestational diabetes risk
        if pd.notna(row['pregnancies']) and row['pregnancies'] > 4:
            unified['prior_diabetes'] = 0  # Risk factor, not prior diagnosis
        
        # Target: 0 = healthy, 1 = diabetes
        if row['outcome'] == 1:
            unified['diagnosis'] = 1  # diabetes
        else:
            unified['diagnosis'] = 0  # healthy
        
        unified['diagnosis_source'] = 'pima_diabetes'
        unified_rows.append(unified)
    
    return pd.DataFrame(unified_rows)


def map_hypertension(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map synthetic hypertension dataset to unified schema.
    Diagnosis: 0 = healthy, 2 = hypertension
    """
    unified_rows = []
    
    for _, row in df.iterrows():
        unified = create_empty_unified_row()
        unified['patient_id'] = str(uuid.uuid4())
        unified['age'] = int(row['age'])
        unified['sex'] = 'M' if row['sex'] == 1 else 'F'
        unified['bmi'] = float(row['bmi'])
        unified['systolic_bp'] = int(row['systolic_bp'])
        unified['diastolic_bp'] = int(row['diastolic_bp'])
        unified['heart_rate'] = int(row['heart_rate'])
        unified['fasting_glucose'] = int(row['glucose'])
        unified['total_cholesterol'] = int(row['cholesterol'])
        unified['smoking_status'] = int(row['smoking_status'])
        
        # Estimate other values
        unified['max_heart_rate'] = 220 - unified['age']
        unified['hba1c'] = (unified['fasting_glucose'] + 46.7) / 28.7
        
        # Target: hypertension = 1 → diagnosis = 2
        if row['hypertension'] == 1:
            unified['diagnosis'] = 2  # hypertension
        else:
            unified['diagnosis'] = 0  # healthy
        
        unified['diagnosis_source'] = 'hypertension_synthetic'
        unified_rows.append(unified)
    
    return pd.DataFrame(unified_rows)


def map_healthy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map synthetic healthy dataset to unified schema.
    Diagnosis: 0 = healthy
    """
    unified_rows = []
    
    for _, row in df.iterrows():
        unified = create_empty_unified_row()
        unified['patient_id'] = str(uuid.uuid4())
        unified['age'] = int(row['age'])
        unified['sex'] = 'M' if row['sex'] == 1 else 'F'
        unified['bmi'] = float(row['bmi'])
        unified['systolic_bp'] = int(row['systolic_bp'])
        unified['diastolic_bp'] = int(row['diastolic_bp'])
        unified['heart_rate'] = int(row['heart_rate'])
        unified['fasting_glucose'] = int(row['glucose'])
        unified['total_cholesterol'] = int(row['cholesterol'])
        unified['smoking_status'] = int(row['smoking_status'])
        
        # Healthy defaults
        unified['max_heart_rate'] = 220 - unified['age']
        unified['hba1c'] = 5.2  # Normal HbA1c
        unified['exercise_angina'] = 0
        unified['st_depression'] = 0.0
        unified['chest_pain_type'] = 4  # Asymptomatic
        unified['resting_ecg'] = 0  # Normal
        
        unified['diagnosis'] = 0  # healthy
        unified['diagnosis_source'] = 'healthy_synthetic'
        unified_rows.append(unified)
    
    return pd.DataFrame(unified_rows)


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using median for numeric, mode for categorical."""
    df = df.copy()
    
    # Numeric columns to impute
    numeric_cols = [
        'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
        'fasting_glucose', 'hba1c', 'insulin', 'total_cholesterol',
        'ldl_cholesterol', 'hdl_cholesterol', 'triglycerides',
        'max_heart_rate', 'st_depression'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            median_val = df[col].median()
            if pd.isna(median_val):
                # Use sensible defaults if all values are NaN
                defaults = {
                    'bmi': 25.0, 'systolic_bp': 120, 'diastolic_bp': 80,
                    'heart_rate': 72, 'fasting_glucose': 100, 'hba1c': 5.5,
                    'insulin': 10.0, 'total_cholesterol': 200, 'ldl_cholesterol': 100,
                    'hdl_cholesterol': 50, 'triglycerides': 150, 'max_heart_rate': 150,
                    'st_depression': 0.0
                }
                median_val = defaults.get(col, 0)
            df[col] = df[col].fillna(median_val)
    
    # Categorical/binary columns to impute with mode
    categorical_cols = [
        'chest_pain_type', 'resting_ecg', 'exercise_angina', 'st_slope',
        'smoking_status', 'family_history_cvd', 'family_history_diabetes',
        'prior_hypertension', 'prior_diabetes', 'prior_heart_disease',
        'on_bp_medication', 'on_diabetes_medication', 'on_cholesterol_medication'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
            else:
                df[col] = df[col].fillna(0)
    
    return df


def balance_classes(df: pd.DataFrame, target_per_class: int = 500) -> pd.DataFrame:
    """Balance dataset using undersampling for over-represented classes."""
    balanced_dfs = []
    
    for diagnosis in [0, 1, 2, 3]:
        class_df = df[df['diagnosis'] == diagnosis]
        n_samples = len(class_df)
        
        if n_samples == 0:
            print(f"  ⚠ No samples for class {diagnosis}, skipping")
            continue
        
        if n_samples > target_per_class:
            # Undersample
            class_df = resample(class_df, n_samples=target_per_class, random_state=42)
        elif n_samples < target_per_class:
            # Oversample with replacement
            class_df = resample(class_df, n_samples=target_per_class, replace=True, random_state=42)
        
        balanced_dfs.append(class_df)
        print(f"  • Class {diagnosis}: {n_samples} → {len(class_df)}")
    
    return pd.concat(balanced_dfs, ignore_index=True)


def add_encoded_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add encoded versions of categorical features for modeling."""
    df = df.copy()
    
    # Encode sex: M=0, F=1
    df['sex_encoded'] = (df['sex'] == 'F').astype(int)
    
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/validation/test sets, stratified by diagnosis."""
    # First split: separate out test set
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['diagnosis']
    )
    
    # Second split: separate train and validation
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=42, stratify=train_val['diagnosis']
    )
    
    return train, val, test


def create_lab_splits(df: pd.DataFrame, lab_ratio: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data into two lab subsets for federated learning simulation."""
    # Randomly split, maintaining class balance
    lab_a, lab_b = train_test_split(
        df, test_size=lab_ratio, random_state=42, stratify=df['diagnosis']
    )
    return lab_a, lab_b


def generate_metadata(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """Generate metadata.json with dataset statistics."""
    
    # Feature ranges
    numeric_features = [
        'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
        'fasting_glucose', 'hba1c', 'total_cholesterol', 'max_heart_rate', 'st_depression'
    ]
    
    feature_ranges = {}
    for col in numeric_features:
        if col in df.columns:
            feature_ranges[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
    
    # Reference ranges (clinical)
    reference_ranges = {
        'fasting_glucose': {
            'normal': [70, 100],
            'prediabetes': [100, 126],
            'diabetes': [126, 400]
        },
        'systolic_bp': {
            'normal': [90, 120],
            'elevated': [120, 130],
            'high_stage1': [130, 140],
            'high_stage2': [140, 180],
            'crisis': [180, 300]
        },
        'diastolic_bp': {
            'normal': [60, 80],
            'high_stage1': [80, 90],
            'high_stage2': [90, 120],
            'crisis': [120, 180]
        },
        'hba1c': {
            'normal': [4.0, 5.7],
            'prediabetes': [5.7, 6.5],
            'diabetes': [6.5, 14.0]
        },
        'total_cholesterol': {
            'desirable': [0, 200],
            'borderline': [200, 240],
            'high': [240, 400]
        },
        'bmi': {
            'underweight': [0, 18.5],
            'normal': [18.5, 25],
            'overweight': [25, 30],
            'obese': [30, 100]
        }
    }
    
    # Class distribution
    class_names = ['healthy', 'diabetes', 'hypertension', 'heart_disease']
    class_dist = {class_names[i]: int((df['diagnosis'] == i).sum()) for i in range(4)}
    
    metadata = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'validation_samples': len(val_df),
        'test_samples': len(test_df),
        'class_distribution': class_dist,
        'class_names': class_names,
        'feature_ranges': feature_ranges,
        'reference_ranges': reference_ranges,
        'feature_columns': MODEL_FEATURES,
        'created_at': pd.Timestamp.now().isoformat(),
        'sources': ['heart_uci', 'pima_diabetes', 'hypertension_synthetic', 'healthy_synthetic']
    }
    
    return metadata


def main():
    """Main dataset preparation pipeline."""
    print("=" * 60)
    print("MedSafe Dataset Preparation")
    print("=" * 60)
    
    # Load raw datasets
    print("\n📂 Loading raw datasets...")
    datasets = load_raw_datasets()
    
    if len(datasets) == 0:
        print("❌ No raw datasets found! Run download_datasets.py first.")
        sys.exit(1)
    
    # Map each dataset to unified schema
    print("\n🔄 Mapping datasets to unified schema...")
    unified_dfs = []
    
    if 'heart' in datasets:
        heart_unified = map_heart_disease(datasets['heart'])
        unified_dfs.append(heart_unified)
        print(f"  ✓ Heart disease: {len(heart_unified)} records")
    
    if 'diabetes' in datasets:
        diabetes_unified = map_diabetes(datasets['diabetes'])
        unified_dfs.append(diabetes_unified)
        print(f"  ✓ Diabetes: {len(diabetes_unified)} records")
    
    if 'hypertension' in datasets:
        hypertension_unified = map_hypertension(datasets['hypertension'])
        unified_dfs.append(hypertension_unified)
        print(f"  ✓ Hypertension: {len(hypertension_unified)} records")
    
    if 'healthy' in datasets:
        healthy_unified = map_healthy(datasets['healthy'])
        unified_dfs.append(healthy_unified)
        print(f"  ✓ Healthy: {len(healthy_unified)} records")
    
    # Combine all datasets
    combined_df = pd.concat(unified_dfs, ignore_index=True)
    print(f"\n📊 Combined dataset: {len(combined_df)} total records")
    
    # Show class distribution before balancing
    print("\n📈 Class distribution (before balancing):")
    for i, name in enumerate(['healthy', 'diabetes', 'hypertension', 'heart_disease']):
        count = (combined_df['diagnosis'] == i).sum()
        print(f"  • {name}: {count}")
    
    # Impute missing values
    print("\n🔧 Imputing missing values...")
    combined_df = impute_missing_values(combined_df)
    
    # Balance classes
    print("\n⚖️ Balancing classes...")
    balanced_df = balance_classes(combined_df, target_per_class=500)
    
    # Add encoded features
    balanced_df = add_encoded_features(balanced_df)
    
    # Split data
    print("\n✂️ Splitting data...")
    train_df, val_df, test_df = split_data(balanced_df)
    print(f"  • Train: {len(train_df)} ({len(train_df)/len(balanced_df)*100:.1f}%)")
    print(f"  • Validation: {len(val_df)} ({len(val_df)/len(balanced_df)*100:.1f}%)")
    print(f"  • Test: {len(test_df)} ({len(test_df)/len(balanced_df)*100:.1f}%)")
    
    # Create lab splits
    print("\n🏥 Creating lab splits...")
    lab_a_df, lab_b_df = create_lab_splits(train_df)
    print(f"  • Lab A: {len(lab_a_df)} records")
    print(f"  • Lab B: {len(lab_b_df)} records")
    
    # Save all datasets
    print("\n💾 Saving datasets...")
    train_df.to_csv(DATA_DIR / "combined_train.csv", index=False)
    val_df.to_csv(DATA_DIR / "combined_validation.csv", index=False)
    test_df.to_csv(DATA_DIR / "combined_test.csv", index=False)
    lab_a_df.to_csv(DATA_DIR / "lab_A_data.csv", index=False)
    lab_b_df.to_csv(DATA_DIR / "lab_B_data.csv", index=False)
    
    # Generate and save metadata
    print("\n📝 Generating metadata...")
    metadata = generate_metadata(balanced_df, train_df, val_df, test_df)
    with open(DATA_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print("=" * 60)
    print(f"\n📁 Files saved to: {DATA_DIR}")
    print("  • combined_train.csv")
    print("  • combined_validation.csv")
    print("  • combined_test.csv")
    print("  • lab_A_data.csv")
    print("  • lab_B_data.csv")
    print("  • metadata.json")
    print("\n🔜 Next step: Run train_baseline_model.py to train the model")


if __name__ == "__main__":
    main()
