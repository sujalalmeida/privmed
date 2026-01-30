"""
Download Real Medical Datasets for PrivMed Federated Learning

Downloads:
1. UCI Heart Disease Dataset
2. Pima Indians Diabetes Dataset
3. Synthetic Hypertension Data (based on Framingham-style features)

Usage:
    python download_datasets.py
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"


def ensure_directories():
    """Create necessary directories."""
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)
    print(f"✓ Created directories: {DATA_DIR}, {RAW_DIR}")


def download_uci_heart_disease():
    """
    Download UCI Heart Disease Dataset (Cleveland).
    
    Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, 
             oldpeak, slope, ca, thal, target
    Target: 0 = no disease, 1-4 = disease severity
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    output_path = RAW_DIR / "heart_disease_uci.csv"
    
    print("\n📥 Downloading UCI Heart Disease Dataset...")
    
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save raw data
        with open(output_path, 'w') as f:
            f.write(','.join(column_names) + '\n')
            f.write(response.text)
        
        # Load and clean
        df = pd.read_csv(output_path, na_values=['?'])
        
        # Convert target: 0 = healthy, 1+ = heart disease
        df['target'] = (df['target'] > 0).astype(int)
        
        # Save cleaned version
        df.to_csv(output_path, index=False)
        
        print(f"  ✓ Downloaded {len(df)} records to {output_path}")
        print(f"  • Healthy: {(df['target'] == 0).sum()}, Heart Disease: {(df['target'] == 1).sum()}")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error downloading UCI Heart Disease: {e}")
        return None


def download_pima_diabetes():
    """
    Download Pima Indians Diabetes Dataset.
    
    Columns: pregnancies, glucose, blood_pressure, skin_thickness, insulin, 
             bmi, diabetes_pedigree, age, outcome
    Target: 0 = no diabetes, 1 = diabetes
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    output_path = RAW_DIR / "pima_diabetes.csv"
    
    print("\n📥 Downloading Pima Indians Diabetes Dataset...")
    
    column_names = [
        'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
        'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome'
    ]
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save with headers
        with open(output_path, 'w') as f:
            f.write(','.join(column_names) + '\n')
            f.write(response.text)
        
        df = pd.read_csv(output_path)
        
        print(f"  ✓ Downloaded {len(df)} records to {output_path}")
        print(f"  • No Diabetes: {(df['outcome'] == 0).sum()}, Diabetes: {(df['outcome'] == 1).sum()}")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error downloading Pima Diabetes: {e}")
        return None


def generate_hypertension_data(n_samples: int = 800):
    """
    Generate synthetic hypertension dataset based on Framingham-style features.
    
    Since Framingham requires Kaggle API, we generate realistic synthetic data
    using the known risk factors for hypertension.
    """
    output_path = RAW_DIR / "hypertension_synthetic.csv"
    
    print("\n🔧 Generating Synthetic Hypertension Dataset...")
    
    np.random.seed(42)
    
    # Generate base demographics
    ages = np.random.randint(30, 80, n_samples)
    sexes = np.random.choice([0, 1], n_samples)  # 0=female, 1=male
    
    # BMI with realistic distribution
    bmis = np.random.normal(27, 5, n_samples)
    bmis = np.clip(bmis, 18, 45)
    
    # Blood pressure - key hypertension indicators
    # Higher for hypertensive individuals
    hypertensive = np.zeros(n_samples, dtype=int)
    
    # Systolic BP
    sys_bp = np.random.normal(120, 15, n_samples)
    
    # Diastolic BP
    dia_bp = np.random.normal(80, 10, n_samples)
    
    # Heart rate
    heart_rates = np.random.normal(75, 12, n_samples)
    
    # Glucose
    glucose = np.random.normal(100, 20, n_samples)
    
    # Cholesterol
    cholesterol = np.random.normal(200, 40, n_samples)
    
    # Smoking status (0, 1, 2)
    smoking = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
    
    # Apply hypertension logic based on risk factors
    for i in range(n_samples):
        risk_score = 0
        
        # Age factor
        if ages[i] > 55:
            risk_score += 2
        elif ages[i] > 45:
            risk_score += 1
        
        # BMI factor
        if bmis[i] > 30:
            risk_score += 2
        elif bmis[i] > 25:
            risk_score += 1
        
        # Male sex slightly higher risk
        if sexes[i] == 1:
            risk_score += 0.5
        
        # Smoking
        if smoking[i] == 2:
            risk_score += 1.5
        elif smoking[i] == 1:
            risk_score += 0.5
        
        # Determine if hypertensive
        prob = min(0.9, 0.1 + risk_score * 0.12)
        hypertensive[i] = 1 if np.random.random() < prob else 0
        
        # Adjust BP for hypertensive individuals
        if hypertensive[i] == 1:
            sys_bp[i] = np.random.normal(150, 15)
            dia_bp[i] = np.random.normal(95, 10)
    
    # Clip values to realistic ranges
    sys_bp = np.clip(sys_bp, 90, 200)
    dia_bp = np.clip(dia_bp, 50, 130)
    heart_rates = np.clip(heart_rates, 50, 120)
    glucose = np.clip(glucose, 60, 200)
    cholesterol = np.clip(cholesterol, 100, 350)
    
    df = pd.DataFrame({
        'age': ages.astype(int),
        'sex': sexes,
        'bmi': np.round(bmis, 1),
        'systolic_bp': np.round(sys_bp).astype(int),
        'diastolic_bp': np.round(dia_bp).astype(int),
        'heart_rate': np.round(heart_rates).astype(int),
        'glucose': np.round(glucose).astype(int),
        'cholesterol': np.round(cholesterol).astype(int),
        'smoking_status': smoking,
        'hypertension': hypertensive
    })
    
    df.to_csv(output_path, index=False)
    
    print(f"  ✓ Generated {len(df)} records to {output_path}")
    print(f"  • No Hypertension: {(df['hypertension'] == 0).sum()}, Hypertension: {(df['hypertension'] == 1).sum()}")
    
    return df


def generate_healthy_samples(n_samples: int = 500):
    """
    Generate synthetic healthy patient data.
    These are patients with normal vitals and no conditions.
    """
    output_path = RAW_DIR / "healthy_synthetic.csv"
    
    print("\n🔧 Generating Synthetic Healthy Patient Dataset...")
    
    np.random.seed(43)
    
    # Generate demographics - wider age range for healthy
    ages = np.random.randint(18, 70, n_samples)
    sexes = np.random.choice([0, 1], n_samples)  # 0=female, 1=male
    
    # BMI - mostly normal range
    bmis = np.random.normal(23, 3, n_samples)
    bmis = np.clip(bmis, 18.5, 27)
    
    # Normal blood pressure
    sys_bp = np.random.normal(115, 8, n_samples)
    dia_bp = np.random.normal(75, 5, n_samples)
    
    # Normal heart rate
    heart_rates = np.random.normal(70, 8, n_samples)
    
    # Normal glucose
    glucose = np.random.normal(90, 10, n_samples)
    
    # Normal cholesterol
    cholesterol = np.random.normal(180, 25, n_samples)
    
    # Mostly non-smokers
    smoking = np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1])
    
    # Clip to realistic ranges
    sys_bp = np.clip(sys_bp, 100, 130)
    dia_bp = np.clip(dia_bp, 60, 85)
    heart_rates = np.clip(heart_rates, 55, 85)
    glucose = np.clip(glucose, 70, 105)
    cholesterol = np.clip(cholesterol, 130, 210)
    
    df = pd.DataFrame({
        'age': ages.astype(int),
        'sex': sexes,
        'bmi': np.round(bmis, 1),
        'systolic_bp': np.round(sys_bp).astype(int),
        'diastolic_bp': np.round(dia_bp).astype(int),
        'heart_rate': np.round(heart_rates).astype(int),
        'glucose': np.round(glucose).astype(int),
        'cholesterol': np.round(cholesterol).astype(int),
        'smoking_status': smoking,
        'healthy': 1  # All healthy
    })
    
    df.to_csv(output_path, index=False)
    
    print(f"  ✓ Generated {len(df)} healthy records to {output_path}")
    
    return df


def main():
    """Main download function."""
    print("=" * 60)
    print("PrivMed Dataset Downloader")
    print("=" * 60)
    
    ensure_directories()
    
    # Download each dataset
    heart_df = download_uci_heart_disease()
    diabetes_df = download_pima_diabetes()
    hypertension_df = generate_hypertension_data(800)
    healthy_df = generate_healthy_samples(500)
    
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    datasets = [
        ("UCI Heart Disease", heart_df),
        ("Pima Diabetes", diabetes_df),
        ("Hypertension (Synthetic)", hypertension_df),
        ("Healthy (Synthetic)", healthy_df)
    ]
    
    for name, df in datasets:
        if df is not None:
            print(f"  ✓ {name}: {len(df)} records")
        else:
            print(f"  ✗ {name}: FAILED")
    
    print("\n📁 Raw data saved to:", RAW_DIR)
    print("\n🔜 Next step: Run prepare_dataset.py to create unified dataset")


if __name__ == "__main__":
    main()
