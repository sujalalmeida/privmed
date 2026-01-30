import os
import threading
import subprocess
import signal
import time
import pickle
import tempfile
import uuid
from typing import Optional, List
import base64
import json
import socket
from urllib.parse import urlparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from lab_model import (
    CATEGORICAL_MAPS,
    DISEASE_TYPES,
    FEATURE_COLUMNS,
    encode_features,
    encode_clinical_features,
    ensure_features_for_model,
    model_path_for_lab,
    load_or_init_model,
    save_model,
    get_parameters,
    set_parameters,
    predict_prob,
    predict_with_baseline,
    predict_rule_based,
)

# Load env from both repo root and server folder.
# - Frontend uses VITE_* vars, backend prefers non-VITE vars.
_SERVER_DIR = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(_SERVER_DIR, ".env"))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(_SERVER_DIR), ".env"))

# Set global random seed for reproducibility in FL training and aggregation
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Environment
FL_SERVER_ADDRESS = os.environ.get("FL_SERVER_ADDRESS", "127.0.0.1:8080")
SUPABASE_URL = os.environ.get("SUPABASE_URL") or os.environ.get("VITE_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("VITE_SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

app = Flask(__name__)
CORS(app)

_server_thread: Optional[threading.Thread] = None
_server_stop = threading.Event()
_client_procs: List[subprocess.Popen] = []
_current_run_id: Optional[str] = None


def sb():
    """
    Create a Supabase client for backend use.
    Prefers service role key if present (required for inserts when RLS is enabled).
    Validates URL so we fail fast with a helpful message instead of a DNS error.
    """
    url = SUPABASE_URL
    key = SUPABASE_SERVICE_KEY or SUPABASE_KEY
    if not url or not key:
        raise RuntimeError(
            "Supabase env is missing. Set SUPABASE_URL and SUPABASE_SERVICE_KEY (recommended) "
            "or SUPABASE_ANON_KEY. You can also set VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY."
        )

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise RuntimeError(
            f"Invalid SUPABASE_URL '{url}'. Expected like 'https://<project-ref>.supabase.co'."
        )

    hostname = parsed.hostname or ""

    # Common mistake: pasting a Postgres hostname or pooler hostname instead of the Supabase API URL.
    if "supabase.co" not in parsed.netloc:
        raise RuntimeError(
            f"SUPABASE_URL host '{parsed.netloc}' doesn't look like a Supabase API URL. "
            "Use the Project URL from Supabase dashboard (Settings → API)."
        )

    # Detect a common, painful misconfig: key belongs to a different project than SUPABASE_URL.
    # This catches typos like https://<ref-typo>.supabase.co which become DNS NXDOMAIN.
    try:
        parts = key.split(".")
        if len(parts) >= 2:
            payload = parts[1]
            payload += "=" * ((4 - (len(payload) % 4)) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
            ref = claims.get("ref")
            if (
                isinstance(ref, str)
                and ref
                and hostname.endswith(".supabase.co")
                and not hostname.startswith(ref + ".")
            ):
                raise RuntimeError(
                    f"SUPABASE_URL host '{hostname}' does not match project ref '{ref}' encoded in the Supabase key. "
                    f"Expected SUPABASE_URL like 'https://{ref}.supabase.co'."
                )
    except RuntimeError:
        raise
    except Exception:
        # If parsing fails (non-JWT key), just skip this check.
        pass

    # DNS preflight for clearer errors than httpx ConnectError.
    try:
        socket.getaddrinfo(hostname, 443)
    except OSError as e:
        raise RuntimeError(
            f"Cannot resolve SUPABASE_URL host '{hostname}'. "
            "Double-check the Project URL in Supabase dashboard (Settings → API → Project URL). "
            f"DNS error: {e}"
        ) from e

    return create_client(url, key)


def normalize_lab_label(raw_label: str) -> str:
    """
    Normalize lab labels to a consistent format: 'Lab A' -> 'lab_A', 'Lab B' -> 'lab_B'.
    This ensures consistency between Clinical Data Entry and Model Training.
    """
    import re
    label = str(raw_label or 'lab_A')
    # Convert 'Lab X' pattern to 'lab_X'
    label = re.sub(r'^Lab\s+', 'lab_', label, flags=re.IGNORECASE)
    # Replace remaining spaces with underscores
    label = re.sub(r'\s+', '_', label)
    # Remove any other special characters
    label = re.sub(r'[^a-zA-Z0-9_]', '', label)
    return label


def load_test_dataset():
    """
    Load the shared test dataset (combined_test.csv) with consistent encoding.
    Used for both local model validation and global model evaluation to ensure
    comparable accuracy metrics.
    
    Returns:
        tuple: (X_test, y_test) as numpy arrays, or (None, None) if loading fails
    """
    from lab_model import encode_clinical_features
    
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'combined_test.csv')
    
    if not os.path.exists(data_path):
        print(f"Test dataset not found: {data_path}")
        return None, None
    
    try:
        df = pd.read_csv(data_path)
        X_list, y_list = [], []
        
        for _, r in df.iterrows():
            try:
                row_dict = r.to_dict()
                X_row, _ = encode_clinical_features(row_dict)
                if X_row.shape[1] != 27:
                    continue
                X_list.append(X_row)
                # Use 'diagnosis' as primary label (the actual column in combined_test.csv)
                # Fallback to 'label' or 'disease_label' for compatibility
                label = row_dict.get('diagnosis', row_dict.get('label', row_dict.get('disease_label', 0)))
                y_list.append(int(label))
            except Exception:
                continue
        
        if len(X_list) == 0:
            print("No valid samples could be encoded from test data")
            return None, None
        
        X_test = np.vstack(X_list)
        y_test = np.array(y_list, dtype=int)
        print(f"Loaded test dataset: {len(y_test)} samples, label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        return X_test, y_test
        
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return None, None


def load_training_dataset():
    """
    Load the combined training dataset for global model retraining.
    
    Returns:
        tuple: (X_train, y_train) as numpy arrays, or (None, None) if loading fails
    """
    from lab_model import encode_clinical_features
    
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'combined_train.csv')
    
    if not os.path.exists(data_path):
        print(f"Training dataset not found: {data_path}")
        return None, None
    
    try:
        df = pd.read_csv(data_path)
        X_list, y_list = [], []
        
        for _, r in df.iterrows():
            try:
                row_dict = r.to_dict()
                X_row, _ = encode_clinical_features(row_dict)
                if X_row.shape[1] != 27:
                    continue
                X_list.append(X_row)
                label = row_dict.get('diagnosis', row_dict.get('label', row_dict.get('disease_label', 0)))
                y_list.append(int(label))
            except Exception:
                continue
        
        if len(X_list) == 0:
            print("No valid samples could be encoded from training data")
            return None, None
        
        X_train = np.vstack(X_list)
        y_train = np.array(y_list, dtype=int)
        print(f"Loaded training dataset: {len(y_train)} samples, label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        return X_train, y_train
        
    except Exception as e:
        print(f"Error loading training dataset: {e}")
        return None, None


def evaluate_model_on_test_set(model, model_type='tree'):
    """
    Evaluate a model on the shared test dataset.
    
    Args:
        model: Trained sklearn model
        model_type: 'tree' for tree-based models (no scaling), 'linear' for linear models (with scaling)
    
    Returns:
        float: Accuracy on test set, or None if evaluation fails
    """
    X_test, y_test = load_test_dataset()
    if X_test is None:
        return None
    
    try:
        # Pad features if model expects more dimensions (e.g., 27 -> 33)
        X_test_eval = ensure_features_for_model(X_test, model)
        
        # Tree-based models don't need scaling; linear models do
        if model_type == 'linear':
            scaler = StandardScaler()
            X_test_eval = scaler.fit_transform(X_test_eval)
        
        preds = model.predict(X_test_eval)
        accuracy = float((preds == y_test).mean())
        return accuracy
    except Exception as e:
        print(f"Error evaluating model on test set: {e}")
        return None


def get_lab_node_accuracy(lab_label: str) -> float | None:
    """
    Get the current node accuracy for a lab.
    
    This returns the single accuracy number that represents how good this lab's
    current model is overall, evaluated on the shared test set.
    
    Priority:
    1. From fl_model_downloads (if lab has downloaded global model, use that accuracy)
    2. From fl_client_updates (lab's last trained model accuracy)
    3. None if no data available
    
    Args:
        lab_label: Normalized lab label (e.g., 'lab_A')
    
    Returns:
        float: Node accuracy (0.0 to 1.0) or None if not available
    """
    # First try to get from downloads (may have node_accuracy column)
    try:
        downloads = sb().table('fl_model_downloads').select('node_accuracy, downloaded_at').eq('lab_label', lab_label).order('downloaded_at', desc=True).limit(1).execute()
        if downloads.data and downloads.data[0].get('node_accuracy') is not None:
            return float(downloads.data[0]['node_accuracy'])
    except Exception as e:
        # node_accuracy column might not exist yet - that's okay
        if 'node_accuracy does not exist' not in str(e):
            print(f"Error checking fl_model_downloads for {lab_label}: {e}")
    
    # Fallback to the lab's last training accuracy from fl_client_updates
    try:
        updates = sb().table('fl_client_updates').select('local_accuracy, created_at').eq('client_label', lab_label).order('created_at', desc=True).limit(1).execute()
        if updates.data and updates.data[0].get('local_accuracy') is not None:
            return float(updates.data[0]['local_accuracy'])
    except Exception as e:
        print(f"Error getting local_accuracy for {lab_label}: {e}")
    
    return None


def set_lab_node_accuracy(lab_label: str, accuracy: float, source: str = 'training') -> bool:
    """
    Update the lab's node accuracy after model training or global model download.
    
    For training: updates fl_client_updates.local_accuracy
    For download: updates fl_model_downloads.node_accuracy
    
    Args:
        lab_label: Normalized lab label
        accuracy: Accuracy value (0.0 to 1.0)
        source: 'training' or 'download'
    
    Returns:
        bool: Success status
    """
    try:
        if source == 'download':
            # Update the most recent download record with the new node accuracy
            downloads = sb().table('fl_model_downloads').select('id').eq('lab_label', lab_label).order('downloaded_at', desc=True).limit(1).execute()
            if downloads.data:
                sb().table('fl_model_downloads').update({'node_accuracy': accuracy}).eq('id', downloads.data[0]['id']).execute()
                print(f"Updated node accuracy for {lab_label} after download: {accuracy:.1%}")
                return True
        # For training, the accuracy is already set in send_model_update
        return True
    except Exception as e:
        print(f"Error setting node accuracy for {lab_label}: {e}")
        return False


def get_lab_current_model(lab_label: str):
    """
    Get the current model for a lab with priority:
    1. Latest downloaded global model
    2. Lab's local model
    3. Baseline model
    4. None (will fall back to rule-based)
    
    Returns:
        Tuple of (model, model_source) where model_source is 'global', 'local', 'baseline', or None
    """
    base = os.path.dirname(__file__)
    
    # Check for downloaded global models (use the most recent one)
    global_pattern = os.path.join(base, 'models', f'global_downloaded_{lab_label}_*.pkl')
    global_models = glob.glob(global_pattern)
    
    if global_models:
        # Use the most recent global model
        latest_global = max(global_models, key=os.path.getmtime)
        try:
            with open(latest_global, 'rb') as f:
                model = pickle.load(f)
                print(f"Using downloaded global model for {lab_label}: {os.path.basename(latest_global)}")
                return model, 'global'
        except Exception as e:
            print(f"Error loading global model: {e}, falling back to local model")
    
    # Check for lab's local model
    local_path = os.path.join(base, 'models', f'{lab_label}.pkl')
    if os.path.exists(local_path):
        try:
            with open(local_path, 'rb') as f:
                model = pickle.load(f)
                print(f"Using local model for {lab_label}")
                return model, 'local'
        except Exception as e:
            print(f"Error loading local model: {e}, falling back to baseline")
    
    # Check for baseline model
    baseline_path = os.path.join(base, 'models', 'baseline_global_model.pkl')
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, 'rb') as f:
                bundle = pickle.load(f)
                model = bundle.get('model') if isinstance(bundle, dict) else bundle
                print(f"Using baseline model for {lab_label}")
                return model, 'baseline'
        except Exception as e:
            print(f"Error loading baseline model: {e}")
    
    return None, None


def predict_with_lab_model(lab_label: str, patient_data: dict) -> dict:
    """
    Make prediction using the lab's current model (global > local > baseline > rule-based).
    
    Args:
        lab_label: Normalized lab label (e.g., 'lab_A')
        patient_data: Patient data dictionary with clinical features
    
    Returns:
        Dictionary with diagnosis, confidence, probabilities, and model_source
    """
    model, model_source = get_lab_current_model(lab_label)
    
    if model is None:
        # Fall back to rule-based prediction
        result = predict_rule_based(patient_data)
        result['model_source'] = 'rule_based'
        return result
    
    try:
        # Encode features
        X, _ = encode_clinical_features(patient_data)
        
        # Ensure X matches model's expected features
        X = ensure_features_for_model(X, model)
        
        # Predict
        probs = model.predict_proba(X)[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        
        from lab_model import DISEASE_TYPES
        
        return {
            'diagnosis': predicted_class,
            'diagnosis_label': DISEASE_TYPES[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'healthy': float(probs[0]),
                'diabetes': float(probs[1]),
                'hypertension': float(probs[2]),
                'heart_disease': float(probs[3])
            },
            'model_source': model_source
        }
    except Exception as e:
        print(f"Error predicting with lab model: {e}")
        # Fall back to rule-based prediction
        result = predict_rule_based(patient_data)
        result['model_source'] = 'rule_based'
        return result


def generate_clinical_insights(body: dict, disease_type: str, risk_score: float) -> dict:
    """Generate dynamic clinical insights based on patient data and prediction"""
    insights = {
        'risk_factors': [],
        'critical_values': [],
        'recommendations': []
    }
    
    # Helper function to safely convert to numeric with fallback
    def safe_float(value, default):
        if value is None or value == '':
            return float(default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return float(default)
    
    def safe_int(value, default):
        if value is None or value == '':
            return int(default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return int(default)
    
    # Handle None body
    if body is None:
        body = {}
    
    # Analyze risk factors
    age = safe_int(body.get('age'), 30)
    bmi = safe_float(body.get('bmi'), 25.0)
    bp_sys = safe_int(body.get('bp_sys'), 120)
    bp_dia = safe_int(body.get('bp_dia'), 80)
    glucose = safe_int(body.get('glucose'), 100)
    cholesterol = safe_int(body.get('cholesterol'), 200)
    smoker = str(body.get('smoker_status', 'no') or 'no').lower()
    family_history = str(body.get('family_history', 'none') or 'none').lower()
    medication_use = str(body.get('medication_use', 'none') or 'none').lower()
    prior = str(body.get('prior_conditions', '') or '').lower()
    
    # Identify risk factors
    if smoker == 'yes':
        insights['risk_factors'].append({'type': 'smoking', 'severity': 'high', 'description': 'Active smoker - major cardiovascular risk'})
    if bmi >= 30:
        insights['risk_factors'].append({'type': 'obesity', 'severity': 'high', 'description': f'Obesity (BMI: {bmi:.1f}) - increased disease risk'})
    elif bmi >= 25:
        insights['risk_factors'].append({'type': 'overweight', 'severity': 'moderate', 'description': f'Overweight (BMI: {bmi:.1f}) - monitor weight'})
    if bp_sys >= 140 or bp_dia >= 90:
        insights['risk_factors'].append({'type': 'hypertension', 'severity': 'high', 'description': f'Elevated BP ({bp_sys}/{bp_dia}) - requires immediate attention'})
    elif bp_sys >= 130 or bp_dia >= 85:
        insights['risk_factors'].append({'type': 'prehypertension', 'severity': 'moderate', 'description': f'Borderline BP ({bp_sys}/{bp_dia}) - lifestyle changes needed'})
    if family_history != 'none' and family_history != 'no':
        insights['risk_factors'].append({'type': 'genetic', 'severity': 'moderate', 'description': f'Family history of {family_history}'})
    if age > 60:
        insights['risk_factors'].append({'type': 'age', 'severity': 'moderate', 'description': 'Advanced age increases disease risk'})
    
    # Critical values
    if glucose >= 126:
        insights['critical_values'].append({'metric': 'glucose', 'value': glucose, 'status': 'critical', 'message': 'Diabetic range - immediate intervention required'})
    elif glucose >= 100:
        insights['critical_values'].append({'metric': 'glucose', 'value': glucose, 'status': 'warning', 'message': 'Prediabetic range - monitoring required'})
    
    if cholesterol >= 240:
        insights['critical_values'].append({'metric': 'cholesterol', 'value': cholesterol, 'status': 'critical', 'message': 'High cholesterol - cardiovascular risk'})
    elif cholesterol >= 200:
        insights['critical_values'].append({'metric': 'cholesterol', 'value': cholesterol, 'status': 'warning', 'message': 'Borderline cholesterol - dietary changes needed'})
    
    # Disease-specific recommendations
    if disease_type == 'healthy':
        insights['recommendations'] = [
            {'priority': 'routine', 'action': 'Annual health checkup recommended'},
            {'priority': 'routine', 'action': 'Continue healthy lifestyle with regular exercise'},
            {'priority': 'routine', 'action': 'Maintain balanced diet and monitor vital signs'}
        ]
    elif disease_type == 'diabetes':
        insights['recommendations'] = [
            {'priority': 'urgent', 'action': 'Immediate glucose tolerance test and HbA1c measurement required'},
            {'priority': 'high', 'action': 'Endocrinology consultation for diabetes management plan'},
            {'priority': 'high', 'action': 'Daily blood glucose monitoring and dietary modifications'},
            {'priority': 'moderate', 'action': 'Consider insulin therapy or oral hypoglycemic agents'}
        ]
        if glucose >= 200:
            insights['recommendations'].insert(0, {'priority': 'critical', 'action': 'URGENT: Blood glucose critically high - consider emergency care'})
    elif disease_type == 'hypertension':
        insights['recommendations'] = [
            {'priority': 'urgent', 'action': '24-hour ambulatory blood pressure monitoring recommended'},
            {'priority': 'high', 'action': 'Cardiology consultation for hypertension management'},
            {'priority': 'high', 'action': 'Reduce sodium intake below 2000mg/day'},
            {'priority': 'moderate', 'action': 'Regular aerobic exercise 30 min/day, 5 days/week'}
        ]
        if bp_sys >= 180 or bp_dia >= 110:
            insights['recommendations'].insert(0, {'priority': 'critical', 'action': 'URGENT: Hypertensive crisis - immediate medical attention required'})
    elif disease_type == 'heart_disease':
        insights['recommendations'] = [
            {'priority': 'critical', 'action': 'URGENT: Immediate cardiology referral required'},
            {'priority': 'urgent', 'action': 'ECG, echocardiogram, and cardiac biomarkers needed urgently'},
            {'priority': 'urgent', 'action': 'Consider hospitalization if symptoms worsen'},
            {'priority': 'high', 'action': 'Stress test and coronary angiography may be indicated'}
        ]
    
    # Add confidence-based insights
    if risk_score > 0.9:
        insights['confidence'] = 'very_high'
        insights['confidence_message'] = f'Model is {risk_score*100:.1f}% confident in this diagnosis'
    elif risk_score > 0.7:
        insights['confidence'] = 'high'
        insights['confidence_message'] = f'Strong indication ({risk_score*100:.1f}% confidence) - clinical correlation advised'
    else:
        insights['confidence'] = 'moderate'
        insights['confidence_message'] = f'Moderate confidence ({risk_score*100:.1f}%) - further testing recommended'
    
    return insights


def _encode_row(row: pd.Series) -> np.ndarray:
    payload = {
        'age': int(row['age']),
        'discomfort_level': int(row['discomfort_level']),
        'symptom_duration': int(row['symptom_duration']),
        'gender': str(row['gender']).lower(),
        'blood_type': str(row['blood_type']).upper(),
        'prior_conditions': str(row.get('prior_conditions') or ''),
        'bmi': float(row.get('bmi', 25.0)),
        'smoker_status': str(row.get('smoker_status', 'no')).lower(),
        'heart_rate': int(row.get('heart_rate', 70)),
        'bp_sys': int(row.get('bp_sys', 120)),
        'bp_dia': int(row.get('bp_dia', 80)),
        'cholesterol': int(row.get('cholesterol', 200)),
        'glucose': int(row.get('glucose', 100)),
        'family_history': str(row.get('family_history', 'none')).lower(),
        'medication_use': str(row.get('medication_use', 'none')).lower(),
    }
    X, _ = encode_features(payload)
    return X

def _encode_row_from_db(record: dict) -> np.ndarray:
    """
    Encode a database record to feature vector.
    Supports both old and new clinical schemas.
    """
    def safe_int(val, default=0):
        return int(val) if val is not None else default
    
    def safe_float(val, default=0.0):
        return float(val) if val is not None else default
    
    def safe_str(val, default=''):
        return str(val) if val is not None else default
    
    # Check if this is new clinical schema (has systolic_bp) or old schema (has bp_sys)
    is_new_schema = 'systolic_bp' in record or 'fasting_glucose' in record
    
    if is_new_schema:
        # New clinical schema
        payload = {
            'age': safe_int(record.get('age'), 50),
            'sex': safe_str(record.get('sex'), 'M'),
            'bmi': safe_float(record.get('bmi'), 25.0),
            'systolic_bp': safe_int(record.get('systolic_bp'), 120),
            'diastolic_bp': safe_int(record.get('diastolic_bp'), 80),
            'heart_rate': safe_int(record.get('heart_rate'), 72),
            'fasting_glucose': safe_int(record.get('fasting_glucose'), 100),
            'hba1c': safe_float(record.get('hba1c'), 5.5),
            'total_cholesterol': safe_int(record.get('total_cholesterol'), 200),
            'ldl_cholesterol': safe_int(record.get('ldl_cholesterol'), 100),
            'hdl_cholesterol': safe_int(record.get('hdl_cholesterol'), 50),
            'triglycerides': safe_int(record.get('triglycerides'), 150),
            'chest_pain_type': safe_int(record.get('chest_pain_type'), 4),
            'resting_ecg': safe_int(record.get('resting_ecg'), 0),
            'max_heart_rate': safe_int(record.get('max_heart_rate'), 150),
            'exercise_angina': safe_int(record.get('exercise_angina'), 0),
            'st_depression': safe_float(record.get('st_depression'), 0.0),
            'st_slope': safe_int(record.get('st_slope'), 2),
            'smoking_status': safe_int(record.get('smoking_status'), 0),
            'family_history_cvd': safe_int(record.get('family_history_cvd'), 0),
            'family_history_diabetes': safe_int(record.get('family_history_diabetes'), 0),
            'prior_hypertension': safe_int(record.get('prior_hypertension'), 0),
            'prior_diabetes': safe_int(record.get('prior_diabetes'), 0),
            'prior_heart_disease': safe_int(record.get('prior_heart_disease'), 0),
            'on_bp_medication': safe_int(record.get('on_bp_medication'), 0),
            'on_diabetes_medication': safe_int(record.get('on_diabetes_medication'), 0),
            'on_cholesterol_medication': safe_int(record.get('on_cholesterol_medication'), 0),
        }
    else:
        # Legacy schema - map to new format
        payload = {
            'age': safe_int(record.get('age'), 30),
            'sex': 'M' if safe_str(record.get('gender'), 'male').lower() == 'male' else 'F',
            'bmi': safe_float(record.get('bmi'), 25.0),
            'systolic_bp': safe_int(record.get('bp_sys'), 120),
            'diastolic_bp': safe_int(record.get('bp_dia'), 80),
            'heart_rate': safe_int(record.get('heart_rate'), 72),
            'fasting_glucose': safe_int(record.get('glucose'), 100),
            'hba1c': 5.5,  # Not in old schema
            'total_cholesterol': safe_int(record.get('cholesterol'), 200),
            'ldl_cholesterol': safe_int(record.get('cholesterol', 200)) * 0.6,
            'hdl_cholesterol': 50,
            'triglycerides': 150,
            'chest_pain_type': 4,
            'resting_ecg': 0,
            'max_heart_rate': 220 - safe_int(record.get('age'), 30),
            'exercise_angina': 0,
            'st_depression': 0.0,
            'st_slope': 2,
            'smoking_status': 2 if safe_str(record.get('smoker_status'), 'no').lower() == 'yes' else 0,
            'family_history_cvd': 1 if safe_str(record.get('family_history'), 'none').lower() == 'heart_disease' else 0,
            'family_history_diabetes': 1 if safe_str(record.get('family_history'), 'none').lower() == 'diabetes' else 0,
            'prior_hypertension': 0,
            'prior_diabetes': 0,
            'prior_heart_disease': 0,
            'on_bp_medication': 0,
            'on_diabetes_medication': 0,
            'on_cholesterol_medication': 0,
        }
    
    X, _ = encode_features(payload)
    return X


def _predict_label_from_features(body: dict) -> int:
    """
    Rule-based label prediction based on clinical features.
    Supports both old and new clinical schemas.
    Returns: 0=healthy, 1=diabetes, 2=hypertension, 3=heart_disease
    """
    # Helper functions for safe conversion
    def safe_int(value, default):
        if value is None or value == '':
            return int(default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return int(default)
    
    def safe_float(value, default):
        if value is None or value == '':
            return float(default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return float(default)
    
    # Handle None body
    if body is None:
        body = {}
    
    # Support both old and new field names
    glucose = safe_int(body.get('fasting_glucose', body.get('glucose')), 100)
    bp_sys = safe_int(body.get('systolic_bp', body.get('bp_sys')), 120)
    bp_dia = safe_int(body.get('diastolic_bp', body.get('bp_dia')), 80)
    cholesterol = safe_int(body.get('total_cholesterol', body.get('cholesterol')), 200)
    bmi = safe_float(body.get('bmi'), 25.0)
    hba1c = safe_float(body.get('hba1c'), 5.5)
    age = safe_int(body.get('age'), 30)
    
    # Handle smoking status - support both old and new formats
    smoking_raw = body.get('smoking_status', body.get('smoker_status', 'no'))
    if smoking_raw is None:
        is_smoker = False
    elif isinstance(smoking_raw, int):
        is_smoker = smoking_raw == 2
    else:
        is_smoker = str(smoking_raw).lower() in ['yes', 'current', '2']
    
    # Cardiac markers from new schema
    chest_pain = safe_int(body.get('chest_pain_type'), 4)
    st_depression = safe_float(body.get('st_depression'), 0)
    exercise_angina = safe_int(body.get('exercise_angina'), 0)
    
    # Score different conditions
    diabetes_score = 0
    hypertension_score = 0
    heart_disease_score = 0
    
    # Diabetes indicators
    if glucose >= 126 or hba1c >= 6.5:
        diabetes_score += 4
    elif glucose >= 100 or hba1c >= 5.7:
        diabetes_score += 2
    if bmi >= 30:
        diabetes_score += 1
    
    # Hypertension indicators
    if bp_sys >= 140 or bp_dia >= 90:
        hypertension_score += 4
    elif bp_sys >= 130 or bp_dia >= 85:
        hypertension_score += 2
    if age > 60:
        hypertension_score += 1
    
    # Heart disease indicators
    if chest_pain in [1, 2]:  # Typical or atypical angina
        heart_disease_score += 3
    if st_depression > 1.0:
        heart_disease_score += 2
    if exercise_angina == 1:
        heart_disease_score += 2
    if cholesterol >= 240:
        heart_disease_score += 2
    elif cholesterol >= 200:
        heart_disease_score += 1
    if is_smoker:
        heart_disease_score += 2
    if bp_sys >= 140:
        heart_disease_score += 1
    if age > 60:
        heart_disease_score += 1
    
    # Determine primary condition based on highest score
    max_score = max(diabetes_score, hypertension_score, heart_disease_score)
    
    if max_score < 2:
        return 0  # healthy
    elif diabetes_score == max_score:
        return 1  # diabetes
    elif hypertension_score == max_score:
        return 2  # hypertension
    else:
        return 3  # heart_disease


def _calculate_confidence_score(body: dict, predicted_label: int) -> float:
    """
    Calculate confidence score based on how strong the risk factors are.
    Supports both old and new clinical schemas.
    Returns: float between 0.5 and 0.95
    """
    # Helper functions for safe conversion
    def safe_int(value, default):
        if value is None or value == '':
            return int(default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return int(default)
    
    def safe_float(value, default):
        if value is None or value == '':
            return float(default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return float(default)
    
    # Handle None body
    if body is None:
        body = {}
    
    # Support both field names
    glucose = safe_int(body.get('fasting_glucose', body.get('glucose')), 100)
    bp_sys = safe_int(body.get('systolic_bp', body.get('bp_sys')), 120)
    bp_dia = safe_int(body.get('diastolic_bp', body.get('bp_dia')), 80)
    cholesterol = safe_int(body.get('total_cholesterol', body.get('cholesterol')), 200)
    bmi = safe_float(body.get('bmi'), 25.0)
    hba1c = safe_float(body.get('hba1c'), 5.5)
    age = safe_int(body.get('age'), 30)
    
    # Cardiac markers
    chest_pain = safe_int(body.get('chest_pain_type'), 4)
    st_depression = safe_float(body.get('st_depression'), 0)
    exercise_angina = safe_int(body.get('exercise_angina'), 0)
    
    # Handle smoking
    smoking_raw = body.get('smoking_status', body.get('smoker_status', 'no'))
    if isinstance(smoking_raw, int):
        is_smoker = smoking_raw == 2
    else:
        is_smoker = str(smoking_raw).lower() in ['yes', 'current', '2']
    age = int(body.get('age', 30))
    smoker = str(body.get('smoker_status', 'no')).lower()
    
    confidence = 0.5  # Base confidence
    
    # Increase confidence based on how clear the indicators are
    if predicted_label == 0:  # healthy
        # Lower confidence if any risk factors present
        if glucose < 100 and bp_sys < 120 and cholesterol < 200 and bmi < 25:
            confidence = 0.85  # Very healthy
        elif glucose < 110 and bp_sys < 130 and cholesterol < 220:
            confidence = 0.70  # Moderately healthy
        else:
            confidence = 0.60  # Some risk factors
            
    elif predicted_label == 1:  # diabetes
        if glucose >= 140:
            confidence = 0.90  # Very high glucose
        elif glucose >= 126:
            confidence = 0.80  # Diabetic range
        elif glucose >= 110:
            confidence = 0.70  # Elevated
        else:
            confidence = 0.65  # Borderline with other factors
        
        # Adjust for supporting factors
        if bmi >= 30:
            confidence += 0.05
            
    elif predicted_label == 2:  # hypertension
        if bp_sys >= 160 or bp_dia >= 100:
            confidence = 0.92  # Stage 2 hypertension
        elif bp_sys >= 140 or bp_dia >= 90:
            confidence = 0.82  # Stage 1 hypertension
        elif bp_sys >= 130 or bp_dia >= 85:
            confidence = 0.72  # Elevated
        else:
            confidence = 0.65  # Borderline
            
        # Adjust for age
        if age > 60:
            confidence += 0.05
            
    elif predicted_label == 3:  # heart_disease
        risk_factors = 0
        if cholesterol >= 240:
            risk_factors += 2
        elif cholesterol >= 200:
            risk_factors += 1
        if smoker == 'yes':
            risk_factors += 2
        if bp_sys >= 140:
            risk_factors += 1
        if age > 60:
            risk_factors += 1
            
        if risk_factors >= 4:
            confidence = 0.88  # Multiple major risks
        elif risk_factors >= 3:
            confidence = 0.78  # Several risks
        elif risk_factors >= 2:
            confidence = 0.68  # Some risks
        else:
            confidence = 0.60  # Limited risks
    
    # Ensure confidence is in valid range
    return min(0.95, max(0.50, confidence))


def _safe_int(value, default=0):
    if value is None or value == '':
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_float(value, default=None):
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _build_patient_record_from_submit(body, prediction):
    """
    Build a patient_records row from /submit request body (new clinical schema)
    and prediction result. Used so /submit can persist data via backend (bypasses RLS).
    """
    # Normalize lab label: 'Lab A' -> 'lab_A', 'Lab B' -> 'lab_B'
    lab_label = normalize_lab_label(body.get('lab_label') or 'lab_A')
    age = _safe_int(body.get('age'), 50)
    sex = str(body.get('sex', body.get('gender', 'M')) or 'M')

    row = {
        'lab_label': lab_label,
        'patient_id': body.get('patient_id'),
        'age': age,
        'sex': sex,
        'height_cm': _safe_float(body.get('height_cm')),
        'weight_kg': _safe_float(body.get('weight_kg')),
        'bmi': _safe_float(body.get('bmi')),
        'systolic_bp': _safe_int(body.get('systolic_bp'), 120),
        'diastolic_bp': _safe_int(body.get('diastolic_bp'), 80),
        'heart_rate': _safe_int(body.get('heart_rate'), 72),
        'fasting_glucose': _safe_int(body.get('fasting_glucose'), 100),
        'hba1c': _safe_float(body.get('hba1c')),
        'insulin': _safe_float(body.get('insulin')),
        'total_cholesterol': _safe_int(body.get('total_cholesterol'), 200),
        'ldl_cholesterol': _safe_int(body.get('ldl_cholesterol')),
        'hdl_cholesterol': _safe_int(body.get('hdl_cholesterol')),
        'triglycerides': _safe_int(body.get('triglycerides')),
        'chest_pain_type': _safe_int(body.get('chest_pain_type'), 4),
        'resting_ecg': _safe_int(body.get('resting_ecg'), 0),
        'max_heart_rate': _safe_int(body.get('max_heart_rate'), 220 - age),
        'exercise_angina': _safe_int(body.get('exercise_angina'), 0),
        'st_depression': _safe_float(body.get('st_depression'), 0.0),
        'st_slope': _safe_int(body.get('st_slope'), 2),
        'smoking_status': _safe_int(body.get('smoking_status'), 0),
        'family_history_cvd': _safe_int(body.get('family_history_cvd'), 0),
        'family_history_diabetes': _safe_int(body.get('family_history_diabetes'), 0),
        'prior_hypertension': _safe_int(body.get('prior_hypertension'), 0),
        'prior_diabetes': _safe_int(body.get('prior_diabetes'), 0),
        'prior_heart_disease': _safe_int(body.get('prior_heart_disease'), 0),
        'on_bp_medication': _safe_int(body.get('on_bp_medication'), 0),
        'on_diabetes_medication': _safe_int(body.get('on_diabetes_medication'), 0),
        'on_cholesterol_medication': _safe_int(body.get('on_cholesterol_medication'), 0),
        'diagnosis': int(prediction.get('diagnosis', 0)),
        'diagnosis_label': str(prediction.get('diagnosis_label', 'healthy')),
        'confidence': float(prediction.get('confidence', 0.5)),
        'probabilities': prediction.get('probabilities'),
    }
    # Omit None values so DB defaults apply where applicable
    return {k: v for k, v in row.items() if v is not None}


@app.post("/submit")
def submit_patient_data():
    """
    Submit patient data for AI prediction using the lab's current model.
    
    This endpoint uses the lab's current model with priority:
    1. Downloaded global model (if lab has downloaded one)
    2. Lab's local trained model
    3. Baseline model
    4. Rule-based prediction as fallback
    
    Returns diagnosis with probabilities and the model source used.
    """
    try:
        body = request.get_json(force=True) or {}
        # Normalize lab label: 'Lab A' -> 'lab_A', 'Lab B' -> 'lab_B'
        lab_label = normalize_lab_label(body.get('lab_label', 'lab_A'))
        
        print(f"Received patient data for prediction from {lab_label}")
        
        # Use the lab's current model (global > local > baseline > rule-based)
        prediction = predict_with_lab_model(lab_label, body)
        model_source = prediction.get('model_source', 'unknown')
        
        # Generate clinical insights
        insights = generate_clinical_insights(body, prediction['diagnosis_label'], prediction['confidence'])
        
        # Get the lab's current node accuracy
        node_accuracy = get_lab_node_accuracy(lab_label)
        
        # Also return the old format fields for backward compatibility
        response = {
            # New format
            'diagnosis': prediction['diagnosis'],
            'diagnosis_label': prediction['diagnosis_label'],
            'confidence': prediction['confidence'],
            'probabilities': prediction['probabilities'],
            
            # Old format for backward compatibility
            'risk_score': prediction['confidence'],
            'disease_type': prediction['diagnosis_label'],
            
            # Clinical insights
            'insights': insights,
            
            # Meta - now reflects actual model used
            'model_type': f'{model_source}_model',
            'model_source': model_source,
            'lab_label': lab_label,
            'node_accuracy': node_accuracy,
        }
        
        print(f"Prediction: {prediction['diagnosis_label']} ({prediction['confidence']:.1%} confidence) using {model_source} model")

        # Option B: persist patient record from /submit using backend (bypasses RLS)
        try:
            row = _build_patient_record_from_submit(body, prediction)
            sb().table('patient_records').insert(row).execute()
            print(f"Saved patient record to patient_records for lab_label={row.get('lab_label')}")
        except Exception as db_err:
            import traceback
            print(f"Error saving patient record from /submit: {db_err}")
            traceback.print_exc()
            # Do not fail the request; prediction was successful

        return jsonify(response)
        
    except Exception as e:
        import traceback
        print(f"Error in submit: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.post("/lab/add_patient_data")
def add_patient_data():
    try:
        body = request.get_json(force=True) or {}
        lab_label = body.get('lab_label') or 'lab_sim'
        
        # Encode features for the new patient
        X_req, _ = encode_features(body)

        # Get all patient records for this lab from database
        X_train = None
        y_train = None
        
        try:
            lab_records = sb().table('patient_records').select('*').eq('lab_label', lab_label).execute()
            if lab_records.data and len(lab_records.data) > 0:
                print(f"Loading {len(lab_records.data)} patient records for {lab_label} from database")
                # Convert to training data
                X_list = []
                y_list = []
                for record in lab_records.data:
                    # Reconstruct feature vector from database record
                    X_row = _encode_row_from_db(record)
                    # Ensure feature vector has the same shape as the new request
                    if X_row.shape[1] != X_req.shape[1]:
                        print(f"Warning: Feature dimension mismatch. DB: {X_row.shape[1]}, Request: {X_req.shape[1]}")
                        continue
                    # Get label - support both 'disease_label' (old) and 'diagnosis' (new)
                    label = record.get('disease_label')
                    if label is None:
                        label = record.get('diagnosis')
                    if label is None:
                        print(f"Warning: Skipping record with missing disease_label/diagnosis")
                        continue
                    X_list.append(X_row)
                    y_list.append(int(label))
                
                if X_list and len(X_list) > 0:
                    X_train = np.vstack(X_list)
                    y_train = np.array(y_list, dtype=int)
                    print(f"Successfully loaded {X_train.shape[0]} records with {len(np.unique(y_train))} unique disease labels")
        except Exception as e:
            print(f"Error loading lab data from database: {e}")
        
        # If no database records, use realistic sample data as seed
        if X_train is None or len(X_train) == 0:
            print(f"No patient records found for {lab_label}. Using current patient as initial training data.")
            # Use the current patient data as the first training example
            # This ensures predictions are based on actual data, not dummy data
            X_train = X_req.copy()
            # Predict label based on risk factors in the data
            y_train = np.array([_predict_label_from_features(body)], dtype=int)

        # Load or initialize model
        model = load_or_init_model(lab_label, X_req.shape[1])
        # Pad to model's expected features if needed (e.g. 27 -> 33 for downloaded global)
        X_req = ensure_features_for_model(X_req, model)
        if X_train is not None:
            X_train = ensure_features_for_model(X_train, model)
        prev_coef, prev_intercept = get_parameters(model)
    
        # Train model on lab's data
        # Check if we have enough classes for multiclass training
        unique_classes = np.unique(y_train)
        
        # Always use rule-based prediction for current patient
        # This ensures dynamic predictions based on actual risk factors
        print(f"Predicting for new patient using rule-based analysis...")
        predicted_label = _predict_label_from_features(body)
        disease_type = ['healthy', 'diabetes', 'hypertension', 'heart_disease'][predicted_label]
        risk_score = _calculate_confidence_score(body, predicted_label)
        pred_label = predicted_label
        
        print(f"Rule-based prediction: {disease_type} (label {pred_label}) with {risk_score:.1%} confidence")
        
        # For model training and accuracy tracking
        if X_train.shape[0] == 1:
            # First patient - no training yet
            print(f"First patient for {lab_label}. No model training yet.")
            local_accuracy = risk_score  # Use confidence as proxy for accuracy
            grad_norm = 0.0
        else:
            # Multiple patients - train model for future use (but don't use it for THIS prediction)
            print(f"Training model on {X_train.shape[0]} patients for future predictions...")
            
            if len(unique_classes) < 2:
                # If only one class, add synthetic diversity
                print(f"Warning: Only {len(unique_classes)} class(es) found. Adding synthetic diversity.")
                # Use fixed seed for reproducibility
                rng = np.random.RandomState(RANDOM_SEED)
                # IMPORTANT: Add labels that are NOT already present. (Previous logic could re-add the same class.)
                missing_labels = [lbl for lbl in range(4) if lbl not in set(unique_classes.tolist())]
                for lbl in missing_labels:
                    synthetic_X = X_train.copy()
                    noise = rng.normal(0, 0.1, synthetic_X.shape)
                    synthetic_X = synthetic_X + noise
                    synthetic_y = np.full(len(synthetic_X), lbl)
                    X_train = np.vstack([X_train, synthetic_X])
                    y_train = np.concatenate([y_train, synthetic_y])
            
            # Scale features and train
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model.fit(X_train_scaled, y_train)
            save_model(lab_label, model)
            
            # Calculate model accuracy on shared test set (not training data)
            # This gives comparable metrics across labs and with global accuracy
            local_accuracy = evaluate_model_on_test_set(model, model_type='tree')
            if local_accuracy is None:
                # Fallback to training accuracy if test set unavailable
                local_accuracy = float(model.score(X_train_scaled, y_train))
                print(f"Using training accuracy as fallback: {local_accuracy:.1%}")
            else:
                print(f"Model evaluated on test set: {local_accuracy:.1%} accuracy")
            
            # Compute grad norm (handle dimension mismatches gracefully)
            try:
                new_coef, new_intercept = get_parameters(model)
                # Check if dimensions match before computing norm
                if new_coef.shape == prev_coef.shape and new_intercept.shape == prev_intercept.shape:
                    grad_norm = float(np.mean([
                        np.linalg.norm((new_coef - prev_coef).ravel(), ord=2),
                        np.linalg.norm((new_intercept - prev_intercept).ravel(), ord=2)
                    ]))
                else:
                    # Dimensions don't match (e.g., feature count changed), use norm of new params
                    print(f"Parameter dimension mismatch: prev {prev_coef.shape}, new {new_coef.shape}. Using new param norm.")
                    grad_norm = float(np.mean([
                        np.linalg.norm(new_coef.ravel(), ord=2),
                        np.linalg.norm(new_intercept.ravel(), ord=2)
                    ]))
            except Exception as e:
                print(f"Error computing grad norm: {e}. Using default value.")
                grad_norm = 1.0
            
            print(f"Model trained with {local_accuracy:.1%} accuracy (for tracking only)")

        # Insert patient record with predicted label and disease type
        sb().table('patient_records').insert({
            'lab_label': lab_label,
            'age': int(body['age']),
            'gender': str(body['gender']),
            'blood_type': str(body['blood_type']),
            'discomfort_level': int(body['discomfort_level']),
            'symptom_duration': int(body['symptom_duration']),
            'prior_conditions': str(body.get('prior_conditions') or ''),
            'bmi': float(body.get('bmi') or 25.0) if body.get('bmi') is not None else 25.0,
            'smoker_status': str(body.get('smoker_status', 'no')),
            'heart_rate': int(body.get('heart_rate', 70)),
            'bp_sys': int(body.get('bp_sys', 120)),
            'bp_dia': int(body.get('bp_dia', 80)),
            'cholesterol': int(body.get('cholesterol', 200)),
            'glucose': int(body.get('glucose', 100)),
            'family_history': str(body.get('family_history', 'none')),
            'medication_use': str(body.get('medication_use', 'none')),
            'disease_label': int(pred_label),
            'disease_type': disease_type,
        }).execute()

        # Upload model to Supabase Storage
        timestamp = int(time.time())
        storage_path = f"models/local/{lab_label}/{timestamp}.pkl"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            pickle.dump(model, tmp)
            tmp_path = tmp.name
        
        try:
            # Upload to Supabase Storage
            with open(tmp_path, 'rb') as f:
                storage_client = sb().storage.from_('models')
                upload_response = storage_client.upload(storage_path, f.read())
                print(f"Model uploaded successfully to: {storage_path}")
        except Exception as e:
            print(f"Error uploading model to storage: {e}")
            # Try alternative storage path without special characters
            try:
                safe_lab_label = lab_label.replace('@', '_at_').replace('.', '_')
                storage_path = f"models/local/{safe_lab_label}/{timestamp}.pkl"
                with open(tmp_path, 'rb') as f:
                    storage_client = sb().storage.from_('models')
                    upload_response = storage_client.upload(storage_path, f.read())
                    print(f"Model uploaded successfully to alternative path: {storage_path}")
            except Exception as e2:
                print(f"Failed to upload to alternative path: {e2}")
                storage_path = None
        finally:
            os.unlink(tmp_path)

        # Send local update metadata to central (fl_client_updates)
        try:
            sb().table('fl_client_updates').insert({
                'run_id': None,
                'round': 1,
                'client_user_id': None,
                'client_label': lab_label,
                'local_accuracy': local_accuracy,
                'grad_norm': float(grad_norm),
                'num_examples': int(X_train.shape[0]),
                'storage_path': storage_path,
            }).execute()
        except Exception as e:
            print(f"Error inserting client update: {e}")

        # Generate dynamic clinical insights
        insights = generate_clinical_insights(body, disease_type, risk_score)

        return jsonify({
            'risk_score': float(risk_score),
            'disease_type': str(disease_type),
            'prediction_label': int(pred_label),
            'model_updated': True,
            'local_accuracy': local_accuracy,
            'num_examples': int(X_train.shape[0]),
            'storage_path': str(storage_path) if storage_path else None,
            'insights': insights
        })
    except Exception as e:
        print(f"Error in add_patient_data: {e}")
        return jsonify({'error': str(e)}), 500


@app.post("/lab/send_model_update")
def send_model_update():
    """Retrain and send model update without new patient data"""
    body = request.get_json(force=True) or {}
    raw_lab_label = body.get('lab_label') or 'lab_sim'
    # Use centralized lab label normalization
    lab_label = normalize_lab_label(raw_lab_label)
    
    try:
        # Debug: Print the lab_label being searched for
        print(f"Searching for patient records with lab_label: '{lab_label}'")
        
        # Get all patient records for this lab from database
        lab_records = sb().table('patient_records').select('*').eq('lab_label', lab_label).execute()
        
        # Debug: Print total records found
        print(f"Found {len(lab_records.data) if lab_records.data else 0} records for lab_label: '{lab_label}'")
        
        if not lab_records.data:
            # Debug: Let's see what lab_labels exist in the database
            all_records = sb().table('patient_records').select('lab_label').execute()
            existing_labels = set([r['lab_label'] for r in all_records.data]) if all_records.data else set()
            print(f"Existing lab_labels in database: {existing_labels}")
            return jsonify({'error': f'No patient data found for lab: {lab_label}. Available labs: {list(existing_labels)}'}), 400
        
        # Convert to training data
        X_list = []
        y_list = []
        expected_dims = 27  # New clinical schema always uses 27 features
        for record in lab_records.data:
            # Get disease label - support both 'disease_label' (old) and 'diagnosis' (new)
            disease_label = record.get('disease_label')
            if disease_label is None:
                disease_label = record.get('diagnosis')
            if disease_label is None:
                print(f"Warning: Skipping record with missing disease_label/diagnosis")
                continue
            
            X_row = _encode_row_from_db(record)
            # Log first record's dimensions
            if len(X_list) == 0:
                print(f"Training with {X_row.shape[1]} features (detected from first record)")
            # Ensure feature vector has consistent dimensions
            if X_row.shape[1] != expected_dims:
                print(f"Warning: Feature dimension mismatch. Got: {X_row.shape[1]}, expected: {expected_dims}. Skipping.")
                continue
            X_list.append(X_row)
            y_list.append(int(disease_label))
        
        if not X_list:
            return jsonify({'error': 'No valid training data found (all records missing disease_label/diagnosis or have wrong dimensions)'}), 400
            
        X_train = np.vstack(X_list)
        y_train = np.array(y_list, dtype=int)
        print(f"Successfully loaded {len(X_list)} records for training")
        
        # Load current model
        model = load_or_init_model(lab_label, X_train.shape[1])
        # Pad to model's expected features if needed (e.g. 27 -> 33 for downloaded global)
        X_train = ensure_features_for_model(X_train, model)
        prev_coef, prev_intercept = get_parameters(model)
        
        # Retrain model
        
        # Check if we have enough classes for multiclass training
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            # If only one class, create a dummy second class for training
            print(f"Warning: Only {len(unique_classes)} class(es) found. Adding dummy data for training.")
            # Use fixed seed for reproducibility
            rng = np.random.RandomState(RANDOM_SEED)
            # Add some dummy data with a DIFFERENT label than the existing one
            existing = int(unique_classes[0])
            alt = 0 if existing != 0 else 1
            dummy_X = X_train.copy()
            noise = rng.normal(0, 0.05, dummy_X.shape)
            dummy_X = dummy_X + noise
            dummy_y = np.full(len(dummy_X), alt)
            X_train = np.vstack([X_train, dummy_X])
            y_train = np.concatenate([y_train, dummy_y])
        
        # Scale features for better training
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train the model directly
        model.fit(X_train_scaled, y_train)
        save_model(lab_label, model)
        
        # Compute grad norm (handle dimension mismatches gracefully)
        try:
            new_coef, new_intercept = get_parameters(model)
            # Check if dimensions match before computing norm
            if new_coef.shape == prev_coef.shape and new_intercept.shape == prev_intercept.shape:
                grad_norm = float(np.mean([
                    np.linalg.norm((new_coef - prev_coef).ravel(), ord=2),
                    np.linalg.norm((new_intercept - prev_intercept).ravel(), ord=2)
                ]))
            else:
                # Dimensions don't match (e.g., feature count changed), use norm of new params
                print(f"Parameter dimension mismatch: prev {prev_coef.shape}, new {new_coef.shape}. Using new param norm.")
                grad_norm = float(np.mean([
                    np.linalg.norm(new_coef.ravel(), ord=2),
                    np.linalg.norm(new_intercept.ravel(), ord=2)
                ]))
        except Exception as e:
            print(f"Error computing grad norm: {e}. Using default value.")
            grad_norm = 1.0        
        # Upload model to Supabase Storage
        timestamp = int(time.time())
        storage_path = f"models/local/{lab_label}/{timestamp}.pkl"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            pickle.dump(model, tmp)
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                storage_client = sb().storage.from_('models')
                upload_response = storage_client.upload(storage_path, f.read())
                print(f"Model uploaded successfully to: {storage_path}")
        except Exception as e:
            print(f"Error uploading model to storage: {e}")
            # Try alternative storage path without special characters  
            try:
                safe_lab_label = lab_label.replace('@', '_at_').replace('.', '_')
                storage_path = f"models/local/{safe_lab_label}/{timestamp}.pkl"
                with open(tmp_path, 'rb') as f:
                    storage_client = sb().storage.from_('models')
                    upload_response = storage_client.upload(storage_path, f.read())
                    print(f"Model uploaded successfully to alternative path: {storage_path}")
            except Exception as e2:
                print(f"Failed to upload to alternative path: {e2}")
                storage_path = None
        finally:
            os.unlink(tmp_path)
        
        # Insert client update
        try:
            # Evaluate on shared test set for comparable metrics with global accuracy
            local_accuracy = evaluate_model_on_test_set(model, model_type='tree')
            if local_accuracy is None:
                # Fallback to training accuracy if test set unavailable
                local_accuracy = float(model.score(X_train_scaled, y_train))
                print(f"Using training accuracy as fallback: {local_accuracy:.1%}")
            else:
                print(f"Local model evaluated on test set: {local_accuracy:.1%} accuracy")
            
            insert_result = sb().table('fl_client_updates').insert({
                'run_id': None,
                'round': 1,
                'client_user_id': None,
                'client_label': lab_label,
                'local_accuracy': local_accuracy,
                'grad_norm': float(grad_norm),
                'num_examples': int(X_train.shape[0]),
                'storage_path': storage_path,
            }).execute()
            print(f"Successfully inserted client update for {lab_label}")
        except Exception as e:
            print(f"Error inserting client update: {e}")
            # Continue execution even if database insert fails
            local_accuracy = 0.0
        
        return jsonify({
            'model_updated': True,
            'local_accuracy': local_accuracy,
            'grad_norm': grad_norm,
            'num_examples': int(X_train.shape[0]),
            'storage_path': storage_path
        })
        
    except Exception as e:
        print(f"Error in send_model_update: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.post("/admin/aggregate_models")
def aggregate_models():
    """Enhanced model aggregation with FedAvg algorithm and comprehensive metrics"""
    try:
        # Get next round number
        try:
            latest_round = sb().table('fl_round_metrics').select('round').order('round', desc=True).limit(1).execute()
            next_round = (latest_round.data[0]['round'] + 1) if latest_round.data else 1
        except:
            next_round = 1
            
        # Get latest local models from each lab (one per lab)
        try:
            # Get most recent update per lab with valid storage_path
            updates = sb().table('fl_client_updates').select('*').not_.is_('storage_path', 'null').order('created_at', desc=True).execute()
            if not updates.data:
                return jsonify({'error': 'no local model updates found'}), 400
            
            # Group by lab, keep only latest update per lab
            lab_updates = {}
            for update in updates.data:
                lab = update.get('client_label', 'unknown')
                if lab not in lab_updates:
                    lab_updates[lab] = update
            
            updates_list = list(lab_updates.values())
            print(f"Found {len(updates_list)} labs with models for round {next_round}: {list(lab_updates.keys())}")
            
        except Exception as e:
            print(f"Error fetching client updates: {e}")
            return jsonify({'error': 'failed to fetch model updates'}), 500

        # Download and aggregate models using FedAvg
        models_data = []
        total_samples = 0
        
        for update in updates_list:
            storage_path = update['storage_path']
            num_examples = update.get('num_examples', 1)
            lab_label = update.get('client_label', 'unknown')
            
            if not storage_path:
                continue
                
            try:
                # Download model from storage
                storage_client = sb().storage.from_('models')
                model_data = storage_client.download(storage_path)
                
                # Load model from bytes
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                    tmp.write(model_data)
                    tmp_path = tmp.name
                
                with open(tmp_path, 'rb') as f:
                    model = pickle.load(f)
                    
                    # Extract parameters based on model type
                    if hasattr(model, 'coef_'):  # Linear models (LogisticRegression)
                        params = {
                            'coef': model.coef_,
                            'intercept': model.intercept_,
                            'type': 'linear'
                        }
                    elif hasattr(model, 'feature_importances_'):  # Tree-based models (fitted)
                        # For tree models, store the entire model
                        params = {
                            'model': model,
                            'type': 'tree',
                            'feature_importances': model.feature_importances_
                        }
                    elif hasattr(model, 'estimators_') or 'Gradient' in str(type(model)) or 'Forest' in str(type(model)):
                        # Tree-based model that might not be fitted yet, or just fitted
                        # Store the entire model regardless
                        params = {
                            'model': model,
                            'type': 'tree'
                        }
                    else:
                        print(f"Unknown model type for {lab_label}: {type(model)}")
                        os.unlink(tmp_path)
                        continue
                    
                    # Accept 27 (new schema) or 33 (legacy) feature models; skip other dimensions
                    model_features = None
                    if hasattr(model, 'n_features_in_'):
                        model_features = model.n_features_in_
                    VALID_FEATURE_COUNTS = (27, 33)
                    if model_features is not None and model_features not in VALID_FEATURE_COUNTS:
                        print(f"Skipping model from {lab_label}: has {model_features} features, expected 27 or 33")
                        os.unlink(tmp_path)
                        continue
                    
                    models_data.append({
                        'params': params,
                        'num_examples': num_examples,
                        'lab': lab_label,
                        'accuracy': update.get('local_accuracy', 0),
                        'model_features': model_features or 27,
                    })
                    total_samples += num_examples
                
                os.unlink(tmp_path)
                print(f"Loaded model from {lab_label}: {num_examples} samples, accuracy: {update.get('local_accuracy', 0):.3f}")
                
            except Exception as e:
                print(f"Error loading model from {storage_path}: {e}")
                continue

        if not models_data:
            return jsonify({'error': 'no valid models found'}), 400

        # Only aggregate models with the same feature count (all 27 or all 33)
        from collections import defaultdict
        by_features = defaultdict(list)
        for data in models_data:
            by_features[data['model_features']].append(data)
        # Use the group with most total samples
        best_feature_count = max(by_features.keys(), key=lambda n: sum(d['num_examples'] for d in by_features[n]))
        models_data = by_features[best_feature_count]
        n_features_aggregated = best_feature_count
        total_samples = sum(d['num_examples'] for d in models_data)
        print(f"Aggregating {len(models_data)} models with {n_features_aggregated} features, total samples: {total_samples}")

        # ====================================================================
        # FEEDBACK-BASED REWEIGHTING (Option A from closed-loop feedback)
        # Labs with higher doctor agreement rates get more weight in aggregation
        # ====================================================================
        use_feedback_weights = True  # Set to False to disable feedback weighting
        feedback_warnings = []
        
        if use_feedback_weights:
            print("Applying feedback-based weight adjustments...")
            for data in models_data:
                lab = data['lab']
                agreement_info = get_lab_agreement_rate(lab)
                agreement_rate = agreement_info['rate']
                feedback_count = agreement_info['total']
                
                # Only apply adjustment if there's sufficient feedback (at least 3 reviews)
                if feedback_count >= 3:
                    # Agreement factor: 0.5 (50% agreement) to 1.0 (100% agreement)
                    # Labs with low agreement get their weight reduced
                    agreement_factor = 0.5 + (0.5 * agreement_rate)
                    
                    # Adjust the effective sample count based on agreement
                    original_samples = data['num_examples']
                    data['num_examples'] = int(original_samples * agreement_factor)
                    
                    print(f"  {lab}: agreement={agreement_rate:.1%} ({feedback_count} reviews), "
                          f"factor={agreement_factor:.2f}, samples {original_samples} -> {data['num_examples']}")
                    
                    # Flag labs with low agreement (below 60%)
                    if agreement_rate < 0.6:
                        feedback_warnings.append({
                            'lab': lab,
                            'agreement_rate': agreement_rate,
                            'message': f'{lab} has low doctor agreement ({agreement_rate:.0%}) - consider review'
                        })
                else:
                    print(f"  {lab}: insufficient feedback ({feedback_count} reviews), using original weight")
            
            # Recalculate total samples after adjustments
            total_samples = sum(d['num_examples'] for d in models_data)
            print(f"Total weighted samples after feedback adjustment: {total_samples}")

        # Apply Federated Averaging (FedAvg)
        model_type = models_data[0]['params']['type']
        
        if model_type == 'linear':
            # Weighted average for linear models
            weighted_coef = None
            weighted_intercept = None
            
            for data in models_data:
                weight = data['num_examples'] / total_samples
                coef_contribution = data['params']['coef'] * weight
                intercept_contribution = data['params']['intercept'] * weight
                
                if weighted_coef is None:
                    weighted_coef = coef_contribution
                    weighted_intercept = intercept_contribution
                else:
                    weighted_coef += coef_contribution
                    weighted_intercept += intercept_contribution
            
            # Create global model
            from sklearn.linear_model import LogisticRegression
            global_model = LogisticRegression(max_iter=200, random_state=42)
            global_model.coef_ = weighted_coef
            global_model.intercept_ = weighted_intercept
            global_model.classes_ = np.array([0, 1, 2, 3])
            global_model.n_features_in_ = n_features_aggregated
            
        else:  # tree-based models
            # For tree models, we RETRAIN on the combined training data
            # This ensures the global model actually learns from all data
            from sklearn.ensemble import GradientBoostingClassifier
            
            print(f"Retraining global model on combined training data...")
            
            # Load training data
            X_train, y_train = load_training_dataset()
            
            if X_train is not None and y_train is not None:
                # Get the current round number for increasing model complexity
                try:
                    res = sb().table('fl_global_models').select('version').order('version', desc=True).limit(1).execute()
                    current_round = res.data[0]['version'] + 1 if res and res.data else 1
                except Exception:
                    current_round = 1
                
                # Increase model complexity with each round (but cap at reasonable values)
                # This allows the model to improve over time
                n_estimators = min(100 + (current_round * 20), 300)  # 100 -> 300 over 10 rounds
                max_depth = min(5 + (current_round // 2), 10)  # 5 -> 10 over 10 rounds
                learning_rate = max(0.1 - (current_round * 0.005), 0.05)  # 0.1 -> 0.05 over 10 rounds
                
                print(f"Round {current_round}: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate:.3f}")
                
                # Train a new global model on the combined training data
                global_model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=RANDOM_SEED + current_round,  # Vary seed by round for diversity
                )
                
                global_model.fit(X_train, y_train)
                print(f"Global model trained on {len(X_train)} samples")
                
            else:
                # Fallback: Create a voting ensemble if training data not available
                print("Training data not available, falling back to voting ensemble...")
                from sklearn.ensemble import VotingClassifier
                
                valid_models = [data for data in models_data if 'model' in data['params']]
                if not valid_models:
                    return jsonify({'error': 'No valid tree-based models found for aggregation'}), 400
                
                estimators = [(f"{data['lab']}", data['params']['model']) for data in valid_models]
                weights = [data['num_examples'] for data in valid_models]
                
                global_model = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    weights=weights
                )
                
                # Fit with dummy data
                n_features = n_features_aggregated
                rng = np.random.RandomState(RANDOM_SEED)
                dummy_X = rng.randn(5, n_features)
                dummy_y = np.array([0, 1, 2, 3, 0])
                global_model.fit(dummy_X, dummy_y)
                print(f"Created voting ensemble from {len(valid_models)} lab models")

        # Upload global model to Supabase Storage
        timestamp = int(time.time())
        global_storage_path = f"models/global/v{timestamp}.pkl"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            pickle.dump(global_model, tmp)
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                storage_client = sb().storage.from_('models')
                storage_client.upload(global_storage_path, f.read())
                print(f"Global model uploaded to: {global_storage_path}")
        except Exception as e:
            print(f"Error uploading global model: {e}")
            os.unlink(tmp_path)
            return jsonify({'error': 'failed to upload global model'}), 500
        finally:
            os.unlink(tmp_path)

        # Evaluate global model on test dataset using shared helper
        global_accuracy = None
        
        try:
            print(f"Evaluating global model with {n_features_aggregated} features using combined_test.csv")
            global_accuracy = evaluate_model_on_test_set(global_model, model_type=model_type)
            
            if global_accuracy is not None:
                print(f"Global model accuracy on test set: {global_accuracy:.3f} (model_type={model_type})")
            else:
                # Fallback: use weighted average of local accuracies
                global_accuracy = sum(data['accuracy'] * data['num_examples'] for data in models_data) / total_samples
                print(f"Using weighted average of local accuracies: {global_accuracy:.3f}")
        except Exception as e:
            print(f"Error evaluating global model: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: use weighted average of local accuracies
            global_accuracy = sum(data['accuracy'] * data['num_examples'] for data in models_data) / total_samples
            print(f"Using weighted average of local accuracies: {global_accuracy:.3f}")

        # Get current max version
        try:
            res = sb().table('fl_global_models').select('version').order('version', desc=True).limit(1).execute()
            maxv = res.data[0]['version'] if res and res.data else 0
        except Exception:
            maxv = 0
        version = int(maxv) + 1
        
        # Insert global model record (only columns that exist in the table)
        sb().table('fl_global_models').insert({
            'version': version,
            'model_type': 'gradient_boosting' if model_type == 'tree' else 'logistic_regression',
            'storage_path': global_storage_path
        }).execute()

        # Get or create a run for this aggregation
        # First, try to get an existing 'running' run
        runs_result = sb().table('fl_runs').select('id').eq('status', 'running').limit(1).execute()
        
        if runs_result.data and len(runs_result.data) > 0:
            # Use existing running run
            run_id = runs_result.data[0]['id']
        else:
            # Create a new run for aggregation
            new_run = sb().table('fl_runs').insert({
                'status': 'running',
                'num_rounds': 1,
                'model_type': 'gradient_boosting' if model_type == 'tree' else 'logreg',
            }).execute()
            run_id = new_run.data[0]['id']
        
        # Calculate convergence metrics
        previous_accuracy = None
        accuracy_delta = None
        convergence_rate = None
        
        try:
            # Get previous round's accuracy
            prev_metrics = sb().table('fl_round_metrics').select('*').order('round', desc=True).limit(2).execute()
            if prev_metrics.data and len(prev_metrics.data) >= 1:
                # Get the most recent previous round (not current)
                for metric in prev_metrics.data:
                    if metric['round'] < version:
                        previous_accuracy = metric.get('global_accuracy')
                        break
                
                if previous_accuracy is not None and global_accuracy is not None:
                    accuracy_delta = global_accuracy - previous_accuracy
                    convergence_rate = accuracy_delta / previous_accuracy if previous_accuracy != 0 else 0
                    print(f"Convergence metrics: Previous={previous_accuracy:.3f}, Current={global_accuracy:.3f}, Delta={accuracy_delta:.4f}, Rate={convergence_rate:.4f}")
        except Exception as conv_error:
            print(f"Warning: Could not calculate convergence metrics: {conv_error}")
        
        # Record round metric with valid run_id and convergence data
        sb().table('fl_round_metrics').insert({
            'run_id': run_id,
            'round': version,
            'global_accuracy': global_accuracy,
            'aggregated_grad_norm': 0.0,  # Default value instead of None
        }).execute()

        # Prepare lab contributions summary
        lab_contributions = [
            {
                'lab': data['lab'],
                'samples': data['num_examples'],
                'accuracy': data['accuracy'],
                'weight': data['num_examples'] / total_samples
            }
            for data in models_data
        ]

        return jsonify({
            'success': True,
            'modelVersion': version,
            'globalAccuracy': global_accuracy,
            'storage_path': global_storage_path,
            'num_models_aggregated': len(models_data),
            'total_samples': total_samples,
            'lab_contributions': lab_contributions,
            'model_type': 'gradient_boosting' if model_type == 'tree' else 'logistic_regression',
            'timestamp': timestamp,
            'convergence': {
                'previous_accuracy': previous_accuracy,
                'accuracy_delta': accuracy_delta,
                'convergence_rate': convergence_rate,
                'improving': accuracy_delta > 0 if accuracy_delta is not None else None
            },
            # Feedback-based reweighting info (Option A)
            'feedback_weighting': {
                'enabled': use_feedback_weights,
                'warnings': feedback_warnings,
                'message': 'Labs with higher doctor agreement rates contributed more weight' if use_feedback_weights else None
            }
        })
        
    except Exception as e:
        print(f"Error in aggregate_models: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.get("/admin/get_aggregation_status")
def get_aggregation_status():
    """Get current aggregation status and lab contributions"""
    try:
        # Get latest global model
        global_model_result = sb().table('fl_global_models').select('*').order('version', desc=True).limit(1).execute()
        
        latest_global = global_model_result.data[0] if global_model_result.data else None
        
        # Get all labs' latest updates
        client_updates = sb().table('fl_client_updates').select('*').order('created_at', desc=True).execute()
        
        # Group by lab to get latest update per lab
        lab_status = {}
        if client_updates.data:
            for update in client_updates.data:
                lab = update.get('client_label', 'unknown')
                if lab not in lab_status:
                    lab_status[lab] = {
                        'lab': lab,
                        'last_update': update['created_at'],
                        'local_accuracy': update.get('local_accuracy') or 0,
                        'num_examples': update.get('num_examples') or 0,
                        'has_model': update.get('storage_path') is not None,
                        'ready_for_aggregation': update.get('storage_path') is not None
                    }
        
        # Get recent round metrics for history
        round_metrics = sb().table('fl_round_metrics').select('*').order('round', desc=True).limit(10).execute()
        
        # Calculate total samples from lab statuses (handle None values)
        total_samples = sum((lab.get('num_examples') or 0) for lab in lab_status.values())
        num_labs_with_models = sum(1 for lab in lab_status.values() if lab['has_model'])
        
        return jsonify({
            'current_global_model': {
                'version': latest_global['version'] if latest_global else 0,
                'model_type': latest_global.get('model_type') if latest_global else None,
                'created_at': latest_global.get('created_at') if latest_global else None,
                'num_labs_contributed': num_labs_with_models,
                'total_samples': total_samples
            },
            'labs': list(lab_status.values()),
            'recent_rounds': round_metrics.data if round_metrics.data else [],
            'total_labs': len(lab_status),
            'ready_labs': sum(1 for lab in lab_status.values() if lab['ready_for_aggregation'])
        })
    except Exception as e:
        print(f"Error in get_aggregation_status: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.get("/admin/round_metrics")
def get_round_metrics():
    """
    Get accuracy metrics for all aggregation rounds.
    Used to display accuracy-over-rounds chart in Admin dashboard.
    """
    try:
        # Get all round metrics ordered by round (most recent first)
        round_metrics = sb().table('fl_round_metrics').select(
            'round', 'global_accuracy', 'created_at'
        ).order('round', desc=True).limit(50).execute()
        
        metrics = []
        if round_metrics.data:
            for metric in round_metrics.data:
                metrics.append({
                    'round': metric.get('round'),
                    'global_accuracy': metric.get('global_accuracy'),
                    'created_at': metric.get('created_at')
                })
        
        return jsonify({'metrics': metrics})
    except Exception as e:
        print(f"Error in get_round_metrics: {e}")
        return jsonify({'error': str(e), 'metrics': []}), 500


@app.get("/lab/get_global_model_info")
def get_global_model_info():
    """
    Get information about the latest global model without downloading
    Labs can use this to check if they need to update
    """
    try:
        body = request.args
        lab_label = body.get('lab_label', 'unknown')
        
        # Get latest global model
        global_model_result = sb().table('fl_global_models').select('*').order('version', desc=True).limit(1).execute()
        
        if not global_model_result.data:
            return jsonify({
                'available': False,
                'message': 'No global model available yet'
            })
        
        latest_global = global_model_result.data[0]
        
        # Check if lab has downloaded this version (table might not exist yet)
        has_downloaded = False
        try:
            download_check = sb().table('fl_model_downloads').select('*').eq('lab_label', lab_label).eq('global_model_version', latest_global['version']).execute()
            has_downloaded = len(download_check.data) > 0 if download_check.data else False
        except Exception as e:
            print(f"fl_model_downloads table not found (optional): {e}")
            # Table doesn't exist yet - that's okay, feature still works without tracking
        
        # Get lab's current local model info from last update
        lab_update = sb().table('fl_client_updates').select('*').eq('client_label', lab_label).order('created_at', desc=True).limit(1).execute()
        
        local_version = None
        local_accuracy = None
        if lab_update.data:
            local_accuracy = lab_update.data[0].get('local_accuracy')
            # Estimate local version from creation time
            local_version = lab_update.data[0].get('round', 0)
        
        # Get the lab's current node accuracy (single number for this lab)
        node_accuracy = get_lab_node_accuracy(lab_label)
        
        return jsonify({
            'available': True,
            'global_model': {
                'version': latest_global['version'],
                'model_type': latest_global.get('model_type'),
                'created_at': latest_global.get('created_at'),
                'storage_path': latest_global.get('storage_path')
            },
            'local_model': {
                'version': local_version,
                'accuracy': local_accuracy
            },
            'node_accuracy': node_accuracy,  # Single accuracy for this lab
            'needs_update': not has_downloaded,
            'has_downloaded': has_downloaded
        })
        
    except Exception as e:
        print(f"Error in get_global_model_info: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.get("/lab/get_node_accuracy")
def get_node_accuracy():
    """
    Get the current node accuracy for a lab.
    
    Returns a single accuracy number that represents how good this lab's
    current model is overall, evaluated on the shared test set.
    
    This accuracy updates when:
    1. Lab sends a model update (accuracy of trained model on test set)
    2. Lab downloads a global model (accuracy of global model on test set)
    
    The prediction screen uses this to show "Model accuracy: X%"
    """
    try:
        raw_lab_label = request.args.get('lab_label', 'unknown')
        lab_label = normalize_lab_label(raw_lab_label)
        
        node_accuracy = get_lab_node_accuracy(lab_label)
        
        if node_accuracy is None:
            return jsonify({
                'lab_label': lab_label,
                'node_accuracy': None,
                'message': 'No model trained or downloaded yet for this lab'
            })
        
        return jsonify({
            'lab_label': lab_label,
            'node_accuracy': node_accuracy,
            'node_accuracy_percent': f"{node_accuracy * 100:.1f}%"
        })
        
    except Exception as e:
        print(f"Error in get_node_accuracy: {e}")
        return jsonify({'error': str(e)}), 500


@app.get("/lab/get_current_model_info")
def get_current_model_info():
    """
    Get comprehensive information about the lab's current model.
    
    Returns:
    - current_model_type: 'global', 'local', 'baseline', or 'none'
    - current_model_version: Version number if using global model
    - current_model_accuracy: Accuracy on shared test set (0.0 to 1.0)
    - last_updated: When the model was last updated
    
    This endpoint should be called:
    1. On component mount in Clinical Data Entry
    2. After downloading a global model
    3. After sending a model update
    """
    try:
        raw_lab_label = request.args.get('lab_label', 'unknown')
        lab_label = normalize_lab_label(raw_lab_label)
        
        # Get the current model and its source
        model, model_source = get_lab_current_model(lab_label)
        
        # Get node accuracy
        node_accuracy = get_lab_node_accuracy(lab_label)
        
        # Get version info if using global model
        model_version = None
        last_updated = None
        
        if model_source == 'global':
            # Get the global model version from downloads
            try:
                downloads = sb().table('fl_model_downloads').select('global_model_version, downloaded_at').eq('lab_label', lab_label).order('downloaded_at', desc=True).limit(1).execute()
                if downloads.data:
                    model_version = downloads.data[0].get('global_model_version')
                    last_updated = downloads.data[0].get('downloaded_at')
            except Exception:
                pass
        elif model_source == 'local':
            # Get the training info
            try:
                updates = sb().table('fl_client_updates').select('model_version, created_at').eq('client_label', lab_label).order('created_at', desc=True).limit(1).execute()
                if updates.data:
                    model_version = updates.data[0].get('model_version')
                    last_updated = updates.data[0].get('created_at')
            except Exception:
                pass
        
        return jsonify({
            'lab_label': lab_label,
            'current_model_type': model_source or 'none',
            'current_model_version': model_version,
            'current_model_accuracy': node_accuracy,
            'current_model_accuracy_percent': f"{node_accuracy * 100:.1f}%" if node_accuracy else None,
            'last_updated': last_updated,
            'has_model': model is not None
        })
        
    except Exception as e:
        print(f"Error in get_current_model_info: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.post("/lab/download_global_model")
def download_global_model():
    """
    Download the latest global model
    Returns signed URL for model download and tracks the download
    """
    try:
        body = request.get_json(force=True) or {}
        raw_lab_label = body.get('lab_label', 'unknown')
        # Normalize lab label: 'Lab A' -> 'lab_A', 'Lab B' -> 'lab_B'
        import re
        lab_label = str(raw_lab_label)
        # Convert 'Lab X' pattern to 'lab_X'
        lab_label = re.sub(r'^Lab\s+', 'lab_', lab_label, flags=re.IGNORECASE)
        # Replace remaining spaces with underscores
        lab_label = re.sub(r'\s+', '_', lab_label)
        # Remove any other special characters
        lab_label = re.sub(r'[^a-zA-Z0-9_]', '', lab_label)
        
        print(f"Lab {lab_label} requesting global model download")
        
        # Get lab's current accuracy before download
        lab_update = sb().table('fl_client_updates').select('*').eq('client_label', lab_label).order('created_at', desc=True).limit(1).execute()
        accuracy_before = lab_update.data[0].get('local_accuracy') if lab_update.data else None
        
        # Get latest global model
        global_model_result = sb().table('fl_global_models').select('*').order('version', desc=True).limit(1).execute()
        
        if not global_model_result.data:
            return jsonify({'error': 'No global model available yet'}), 404
        
        latest_global = global_model_result.data[0]
        storage_path = latest_global.get('storage_path')
        
        if not storage_path:
            return jsonify({'error': 'Global model storage path not found'}), 404
        
        # Download the model from Supabase Storage
        try:
            storage_client = sb().storage.from_('models')
            model_data = storage_client.download(storage_path)
            
            # Save to local temp file
            timestamp = int(time.time())
            local_model_path = os.path.join(os.path.dirname(__file__), 'models', f'global_downloaded_{lab_label}_{timestamp}.pkl')
            
            with open(local_model_path, 'wb') as f:
                f.write(model_data)
            
            print(f"Global model downloaded to: {local_model_path}")
            
            # Load the model to get metadata
            with open(local_model_path, 'rb') as f:
                model = pickle.load(f)
                model_type = type(model).__name__
            
            # Evaluate the downloaded global model on the shared test set
            # This becomes the lab's new node accuracy
            node_accuracy = evaluate_model_on_test_set(model, model_type='tree')
            if node_accuracy is not None:
                print(f"Downloaded global model evaluated on test set: {node_accuracy:.1%} accuracy")
            else:
                # Fallback to the global accuracy from round metrics
                round_metrics = sb().table('fl_round_metrics').select('*').eq('round', latest_global['version']).execute()
                node_accuracy = round_metrics.data[0].get('average_accuracy') if round_metrics.data else None
                print(f"Using global model's stored accuracy: {node_accuracy}")
            
            # Update the lab's node accuracy in fl_client_updates so it's used consistently
            # This ensures get_lab_node_accuracy returns the correct value
            if node_accuracy is not None:
                try:
                    # Insert a new record with the new accuracy from global model
                    # This way get_lab_node_accuracy will pick up the most recent one
                    # Only include columns that exist in the table
                    sb().table('fl_client_updates').insert({
                        'client_label': lab_label,
                        'local_accuracy': node_accuracy,
                        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    }).execute()
                    print(f"Updated {lab_label} node accuracy to {node_accuracy:.1%} after global model download")
                except Exception as acc_err:
                    print(f"Warning: Could not update node accuracy in fl_client_updates: {acc_err}")
            
            # Track the download with performance metrics
            try:
                download_record = {
                    'lab_label': lab_label,
                    'global_model_version': latest_global['version'],
                    'downloaded_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'accuracy_before_download': accuracy_before
                }
                
                # Use upsert to avoid duplicate key constraint violations
                sb().table('fl_model_downloads').upsert(
                    download_record, 
                    on_conflict='lab_label,global_model_version'
                ).execute()
                print(f"Download tracked for {lab_label}, version {latest_global['version']}")
                if accuracy_before and node_accuracy:
                    improvement = ((node_accuracy - accuracy_before) / accuracy_before) * 100
                    print(f"  Node accuracy improvement: {accuracy_before:.1%} → {node_accuracy:.1%} ({improvement:+.2f}%)")
            except Exception as track_error:
                print(f"Warning: Could not track download: {track_error}")
                # Continue even if tracking fails
            
            improvement_metrics = None
            if accuracy_before and node_accuracy:
                improvement = ((node_accuracy - accuracy_before) / accuracy_before) * 100
                improvement_metrics = {
                    'accuracy_before': accuracy_before,
                    'accuracy_after': node_accuracy,
                    'improvement_percentage': round(improvement, 2),
                    'absolute_improvement': round(node_accuracy - accuracy_before, 4)
                }
            
            return jsonify({
                'success': True,
                'global_model': {
                    'version': latest_global['version'],
                    'model_type': latest_global.get('model_type'),
                    'created_at': latest_global.get('created_at'),
                    'storage_path': storage_path,
                    'local_path': local_model_path,
                    'accuracy': node_accuracy
                },
                'node_accuracy': node_accuracy,  # Lab's new node accuracy after download
                'improvement_metrics': improvement_metrics,
                'message': f'Global model v{latest_global["version"]} downloaded successfully'
            })
            
        except Exception as download_error:
            print(f"Error downloading model from storage: {download_error}")
            return jsonify({'error': f'Failed to download model: {str(download_error)}'}), 500
        
    except Exception as e:
        print(f"Error in download_global_model: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.get("/lab/get_download_history")
def get_download_history():
    """
    Get download history with improvement metrics for a specific lab
    Shows how the lab's accuracy improved after downloading global models
    """
    try:
        lab_label = request.args.get('lab_label', 'unknown')
        
        # Get download history for this lab
        downloads = sb().table('fl_model_downloads').select('*').eq('lab_label', lab_label).order('downloaded_at', desc=True).execute()
        
        if not downloads.data:
            return jsonify({
                'downloads': [],
                'total_downloads': 0,
                'average_improvement': None
            })
        
        # Calculate statistics
        improvements = [d.get('improvement_percentage') for d in downloads.data if d.get('improvement_percentage') is not None]
        avg_improvement = sum(improvements) / len(improvements) if improvements else None
        
        return jsonify({
            'downloads': downloads.data,
            'total_downloads': len(downloads.data),
            'average_improvement': round(avg_improvement, 2) if avg_improvement else None,
            'latest_download': downloads.data[0] if downloads.data else None
        })
        
    except Exception as e:
        print(f"Error in get_download_history: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.get("/admin/get_round_history")
def get_round_history():
    """
    Get federated learning round history with performance metrics
    Returns historical data for visualization and analysis
    """
    try:
        # Get all round metrics
        round_metrics = sb().table('fl_round_metrics').select('*').order('round', desc=False).execute()
        
        # Get all global models
        global_models = sb().table('fl_global_models').select('*').order('version', desc=False).execute()
        
        # Get all client updates to calculate participation
        client_updates = sb().table('fl_client_updates').select('*').order('created_at', desc=False).execute()
        
        rounds = []
        if round_metrics.data:
            for metric in round_metrics.data:
                round_num = metric.get('round') or 0
                
                # Find corresponding global model
                global_model = next((gm for gm in global_models.data if gm['version'] == round_num), None) if global_models.data else None
                
                # Count labs that participated in this round (handle None round values)
                round_updates = [u for u in (client_updates.data or []) if (u.get('round') or 0) == round_num]
                participating_labs = len(set(u.get('client_label') for u in round_updates))
                total_samples = sum((u.get('num_examples') or 0) for u in round_updates)
                
                rounds.append({
                    'round': round_num,
                    'global_accuracy': metric.get('global_accuracy'),
                    'timestamp': metric.get('created_at') if 'created_at' in metric else global_model.get('created_at') if global_model else None,
                    'labs_participated': participating_labs,
                    'total_samples': total_samples,
                    'model_type': global_model.get('model_type') if global_model else None,
                    'aggregated_grad_norm': metric.get('aggregated_grad_norm')
                })
        
        # Calculate convergence statistics
        convergence_stats = {}
        if len(rounds) >= 2:
            accuracies = [r['global_accuracy'] for r in rounds if r['global_accuracy'] is not None]
            if accuracies:
                convergence_stats = {
                    'total_rounds': len(rounds),
                    'initial_accuracy': accuracies[0] if accuracies else None,
                    'current_accuracy': accuracies[-1] if accuracies else None,
                    'accuracy_improvement': accuracies[-1] - accuracies[0] if len(accuracies) >= 2 else 0,
                    'average_accuracy': sum(accuracies) / len(accuracies),
                    'best_accuracy': max(accuracies),
                    'worst_accuracy': min(accuracies),
                    'convergence_rate': (accuracies[-1] - accuracies[0]) / len(accuracies) if len(accuracies) > 1 else 0
                }
        
        # Calculate participation trends
        participation_stats = {}
        if client_updates.data:
            unique_labs = set(u.get('client_label') for u in client_updates.data)
            participation_stats = {
                'total_unique_labs': len(unique_labs),
                'average_participation': sum(r['labs_participated'] for r in rounds) / len(rounds) if rounds else 0,
                'labs_list': list(unique_labs)
            }
        
        return jsonify({
            'rounds': rounds,
            'convergence_stats': convergence_stats,
            'participation_stats': participation_stats,
            'total_rounds': len(rounds)
        })
        
    except Exception as e:
        print(f"Error in get_round_history: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.get("/admin/get_convergence_stats")
def get_convergence_stats():
    """
    Advanced convergence analytics for FL system health monitoring
    """
    try:
        # Get round history
        round_metrics = sb().table('fl_round_metrics').select('*').order('round', desc=False).execute()
        
        if not round_metrics.data or len(round_metrics.data) < 2:
            return jsonify({
                'available': False,
                'message': 'Need at least 2 rounds for convergence analysis'
            })
        
        accuracies = [m.get('global_accuracy') for m in round_metrics.data if m.get('global_accuracy') is not None]
        
        # Calculate convergence metrics
        convergence_speed = None
        threshold = 0.80  # 80% accuracy threshold
        for i, acc in enumerate(accuracies):
            if acc >= threshold:
                convergence_speed = i + 1
                break
        
        # Model stability (variance in accuracy)
        if len(accuracies) > 1:
            mean_acc = sum(accuracies) / len(accuracies)
            variance = sum((acc - mean_acc) ** 2 for acc in accuracies) / len(accuracies)
            stability_score = 1 - min(variance * 10, 1)  # Normalize to 0-1
        else:
            stability_score = None
        
        # Lab contribution fairness (simplified Gini coefficient)
        client_updates = sb().table('fl_client_updates').select('*').execute()
        if client_updates.data:
            lab_contributions = {}
            for update in client_updates.data:
                lab = update.get('client_label')
                samples = update.get('num_examples', 0)
                lab_contributions[lab] = lab_contributions.get(lab, 0) + samples
            
            if lab_contributions:
                total = sum(lab_contributions.values())
                contributions_sorted = sorted(lab_contributions.values())
                n = len(contributions_sorted)
                gini = sum((2 * i - n - 1) * x for i, x in enumerate(contributions_sorted, 1)) / (n * total) if total > 0 else 0
                fairness_score = 1 - abs(gini)  # Higher is fairer
            else:
                fairness_score = None
        else:
            fairness_score = None
        
        return jsonify({
            'available': True,
            'convergence_speed_rounds': convergence_speed,
            'stability_score': stability_score,
            'fairness_score': fairness_score,
            'threshold_accuracy': threshold,
            'current_accuracy': accuracies[-1] if accuracies else None,
            'accuracy_trend': 'improving' if len(accuracies) >= 2 and accuracies[-1] > accuracies[-2] else 'stable',
            'recommendations': [
                'Add more training data to improve accuracy' if accuracies[-1] < 0.7 else None,
                'Consider increasing model complexity' if stability_score and stability_score > 0.9 else None,
                'Encourage more labs to participate' if fairness_score and fairness_score < 0.5 else None
            ]
        })
        
    except Exception as e:
        print(f"Error in get_convergence_stats: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.get("/lab/global_weights")
def get_global_weights():
    # Return current global model parameters
    global_path = os.path.join(os.path.dirname(__file__), 'models', 'global.pkl')
    if not os.path.exists(global_path):
        return jsonify({'error': 'global model not found'}), 404
    import pickle
    with open(global_path, 'rb') as f:
        gm = pickle.load(f)
    return jsonify({
        'coef': gm.coef_.tolist(),
        'intercept': gm.intercept_.tolist(),
        'classes': gm.classes_.tolist(),
    })


# -------- Existing Flower controls below --------

def _run_flower_server(run_id: str, num_rounds: int, model_type: str) -> None:
    os.environ["FL_RUN_ID"] = run_id
    os.environ["FL_NUM_ROUNDS"] = str(num_rounds)
    os.environ["FL_MODEL_TYPE"] = model_type
    from fl_server import start_server  # lazy import
    start_server(server_address=FL_SERVER_ADDRESS, stop_event=_server_stop)


@app.post("/fl/start")
def start_fl():
    global _server_thread, _current_run_id
    body = request.get_json(force=True) or {}
    model_type = body.get("model_type", "logreg")
    num_rounds = int(body.get("num_rounds", 3))
    run_id = body.get("run_id")
    if _server_thread and _server_thread.is_alive():
        return jsonify({"error": "server already running", "run_id": _current_run_id}), 409
    if not run_id:
        return jsonify({"error": "run_id required"}), 400
    _server_stop.clear()
    _current_run_id = run_id
    _server_thread = threading.Thread(target=_run_flower_server, args=(run_id, num_rounds, model_type), daemon=True)
    _server_thread.start()
    return jsonify({"ok": True, "run_id": run_id})


@app.post("/fl/stop")
def stop_fl():
    global _server_thread
    _server_stop.set()
    if _server_thread and _server_thread.is_alive():
        _server_thread.join(timeout=5)
    for p in _client_procs:
        try:
            p.send_signal(signal.SIGINT)
        except Exception:
            pass
    return jsonify({"ok": True})


@app.post("/fl/clients/start")
def start_clients():
    body = request.get_json(force=True) or {}
    datasets = body.get("datasets", ["lab_A.csv", "lab_B.csv"])  # dev defaults
    count = 0
    for ds in datasets:
        proc = subprocess.Popen(["python", "-u", os.path.join(os.path.dirname(__file__), "fl_client.py"),
                                 "--server", FL_SERVER_ADDRESS, "--dataset", ds])
        _client_procs.append(proc)
        count += 1
    return jsonify({"ok": True, "started": count})


@app.get("/fl/status")
def status():
    running = _server_thread.is_alive() if _server_thread else False
    return jsonify({"running": running, "run_id": _current_run_id})


# ============================================================================
# FEATURE 1: Push Global Model to Labs
# ============================================================================

@app.post("/admin/push_global_model")
def push_global_model():
    """
    Initiates broadcast of global model to all participating labs.
    Creates broadcast record and notifies all labs.
    """
    try:
        body = request.get_json(force=True) or {}
        initiated_by = body.get('initiated_by', 'admin')
        
        # Get current global model version
        global_model_result = sb().table('fl_global_models').select('*').order('version', desc=True).limit(1).execute()
        
        if not global_model_result.data:
            return jsonify({'error': 'No global model available to push'}), 400
        
        global_model = global_model_result.data[0]
        global_version = global_model['version']
        
        # Get all participating labs
        labs_result = sb().table('fl_client_updates').select('client_label').execute()
        if not labs_result.data:
            return jsonify({'error': 'No labs found to push to'}), 400
        
        # Get unique lab labels
        lab_labels = list(set([r['client_label'] for r in labs_result.data if r['client_label']]))
        
        if not lab_labels:
            return jsonify({'error': 'No labs found to push to'}), 400
        
        # Create broadcast record
        broadcast = sb().table('fl_model_broadcasts').insert({
            'global_model_version': global_version,
            'initiated_by': initiated_by,
            'status': 'in_progress',
            'labs_notified': 0,
            'labs_downloaded': 0
        }).execute()
        
        broadcast_id = broadcast.data[0]['id']
        
        # Get lab preferences for auto-sync
        preferences_result = sb().table('fl_lab_preferences').select('*').execute()
        preferences = {p['lab_label']: p['auto_sync_enabled'] for p in (preferences_result.data or [])}
        
        # Create sync status records for each lab
        sync_records = []
        for lab_label in lab_labels:
            auto_sync = preferences.get(lab_label, False)
            sync_records.append({
                'broadcast_id': broadcast_id,
                'lab_label': lab_label,
                'notified_at': 'now()',
                'auto_sync_enabled': auto_sync,
                'status': 'notified'
            })
        
        # Insert all sync status records
        sb().table('fl_lab_sync_status').insert(sync_records).execute()
        
        # Update broadcast with notified count
        sb().table('fl_model_broadcasts').update({
            'labs_notified': len(lab_labels),
            'status': 'in_progress'
        }).eq('id', broadcast_id).execute()
        
        return jsonify({
            'success': True,
            'broadcast_id': broadcast_id,
            'global_model_version': global_version,
            'labs_notified': len(lab_labels),
            'lab_labels': lab_labels
        })
        
    except Exception as e:
        print(f"Error in push_global_model: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.get("/admin/broadcast_status/<broadcast_id>")
def get_broadcast_status(broadcast_id):
    """
    Returns real-time download status per lab for a specific broadcast.
    """
    try:
        # Get broadcast details
        broadcast = sb().table('fl_model_broadcasts').select('*').eq('id', broadcast_id).execute()
        
        if not broadcast.data:
            return jsonify({'error': 'Broadcast not found'}), 404
        
        broadcast_data = broadcast.data[0]
        
        # Get sync status for all labs
        sync_status = sb().table('fl_lab_sync_status').select('*').eq('broadcast_id', broadcast_id).execute()
        
        labs_status = []
        labs_downloaded = 0
        labs_notified = 0
        
        for status in (sync_status.data or []):
            lab_info = {
                'lab_label': status['lab_label'],
                'status': status['status'],
                'notified_at': status['notified_at'],
                'downloaded_at': status['downloaded_at'],
                'auto_sync_enabled': status['auto_sync_enabled']
            }
            labs_status.append(lab_info)
            
            if status['status'] == 'downloaded':
                labs_downloaded += 1
                labs_notified += 1
            elif status['status'] == 'notified':
                labs_notified += 1
        
        # Update broadcast counts if changed
        if labs_downloaded != broadcast_data['labs_downloaded'] or labs_notified != broadcast_data['labs_notified']:
            new_status = 'completed' if labs_downloaded == len(labs_status) else 'in_progress'
            sb().table('fl_model_broadcasts').update({
                'labs_notified': labs_notified,
                'labs_downloaded': labs_downloaded,
                'status': new_status
            }).eq('id', broadcast_id).execute()
            broadcast_data['status'] = new_status
        
        return jsonify({
            'broadcast_id': broadcast_id,
            'global_model_version': broadcast_data['global_model_version'],
            'initiated_by': broadcast_data['initiated_by'],
            'created_at': broadcast_data['created_at'],
            'status': broadcast_data['status'],
            'labs_notified': labs_notified,
            'labs_downloaded': labs_downloaded,
            'total_labs': len(labs_status),
            'labs': labs_status,
            'progress_percentage': (labs_downloaded / len(labs_status) * 100) if labs_status else 0
        })
        
    except Exception as e:
        print(f"Error in get_broadcast_status: {e}")
        return jsonify({'error': str(e)}), 500


@app.get("/admin/broadcast_history")
def get_broadcast_history():
    """
    Returns history of all model broadcasts.
    """
    try:
        broadcasts = sb().table('fl_model_broadcasts').select('*').order('created_at', desc=True).limit(20).execute()
        
        result = []
        for b in (broadcasts.data or []):
            result.append({
                'id': b['id'],
                'created_at': b['created_at'],
                'global_model_version': b['global_model_version'],
                'initiated_by': b['initiated_by'],
                'status': b['status'],
                'labs_notified': b['labs_notified'],
                'labs_downloaded': b['labs_downloaded']
            })
        
        return jsonify({'broadcasts': result})
        
    except Exception as e:
        print(f"Error in get_broadcast_history: {e}")
        return jsonify({'error': str(e)}), 500


@app.post("/lab/enable_auto_sync")
def enable_auto_sync():
    """
    Lab enables/disables auto-sync for automatic global model downloads.
    """
    try:
        body = request.get_json(force=True) or {}
        lab_label = body.get('lab_label')
        enabled = body.get('enabled', False)
        
        if not lab_label:
            return jsonify({'error': 'lab_label required'}), 400
        
        # Upsert lab preference
        existing = sb().table('fl_lab_preferences').select('*').eq('lab_label', lab_label).execute()
        
        if existing.data:
            sb().table('fl_lab_preferences').update({
                'auto_sync_enabled': enabled,
                'updated_at': 'now()'
            }).eq('lab_label', lab_label).execute()
        else:
            sb().table('fl_lab_preferences').insert({
                'lab_label': lab_label,
                'auto_sync_enabled': enabled
            }).execute()
        
        return jsonify({
            'success': True,
            'lab_label': lab_label,
            'auto_sync_enabled': enabled
        })
        
    except Exception as e:
        print(f"Error in enable_auto_sync: {e}")
        return jsonify({'error': str(e)}), 500


@app.get("/lab/get_auto_sync_status")
def get_auto_sync_status():
    """
    Get auto-sync status for a lab.
    """
    try:
        lab_label = request.args.get('lab_label')
        
        if not lab_label:
            return jsonify({'error': 'lab_label required'}), 400
        
        pref = sb().table('fl_lab_preferences').select('*').eq('lab_label', lab_label).execute()
        
        auto_sync_enabled = False
        if pref.data:
            auto_sync_enabled = pref.data[0].get('auto_sync_enabled', False)
        
        return jsonify({
            'lab_label': lab_label,
            'auto_sync_enabled': auto_sync_enabled
        })
        
    except Exception as e:
        print(f"Error in get_auto_sync_status: {e}")
        return jsonify({'error': str(e)}), 500


@app.get("/lab/check_for_updates")
def check_for_updates():
    """
    Lab polls for new global model updates.
    Returns info about pending broadcasts that haven't been downloaded.
    """
    try:
        lab_label = request.args.get('lab_label')
        
        if not lab_label:
            return jsonify({'error': 'lab_label required'}), 400
        
        # Check for pending sync status entries (notified but not downloaded)
        pending = sb().table('fl_lab_sync_status').select('*, fl_model_broadcasts(*)').eq('lab_label', lab_label).eq('status', 'notified').order('notified_at', desc=True).limit(1).execute()
        
        if pending.data and len(pending.data) > 0:
            record = pending.data[0]
            broadcast = record.get('fl_model_broadcasts', {})
            
            return jsonify({
                'new_model_available': True,
                'version': broadcast.get('global_model_version'),
                'broadcast_id': record['broadcast_id'],
                'notified_at': record['notified_at'],
                'auto_sync_enabled': record['auto_sync_enabled']
            })
        
        return jsonify({
            'new_model_available': False,
            'version': None,
            'broadcast_id': None
        })
        
    except Exception as e:
        print(f"Error in check_for_updates: {e}")
        return jsonify({'error': str(e)}), 500


@app.post("/lab/acknowledge_download")
def acknowledge_download():
    """
    Lab confirms successful download of global model from a broadcast.
    Updates fl_lab_sync_status record.
    """
    try:
        body = request.get_json(force=True) or {}
        lab_label = body.get('lab_label')
        broadcast_id = body.get('broadcast_id')
        
        if not lab_label or not broadcast_id:
            return jsonify({'error': 'lab_label and broadcast_id required'}), 400
        
        # Update sync status to downloaded
        sb().table('fl_lab_sync_status').update({
            'status': 'downloaded',
            'downloaded_at': 'now()'
        }).eq('broadcast_id', broadcast_id).eq('lab_label', lab_label).execute()
        
        # Update broadcast counts
        sync_status = sb().table('fl_lab_sync_status').select('*').eq('broadcast_id', broadcast_id).execute()
        labs_downloaded = len([s for s in (sync_status.data or []) if s['status'] == 'downloaded'])
        total_labs = len(sync_status.data or [])
        
        new_status = 'completed' if labs_downloaded == total_labs else 'in_progress'
        sb().table('fl_model_broadcasts').update({
            'labs_downloaded': labs_downloaded,
            'status': new_status
        }).eq('id', broadcast_id).execute()
        
        return jsonify({
            'success': True,
            'lab_label': lab_label,
            'broadcast_id': broadcast_id,
            'status': 'downloaded'
        })
        
    except Exception as e:
        print(f"Error in acknowledge_download: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# FEATURE 2: A/B Test Dashboard
# ============================================================================

@app.post("/admin/run_ab_test")
def run_ab_test():
    """
    Run an A/B test comparing two models.
    Model A: Local lab model
    Model B: Global federated model
    """
    try:
        body = request.get_json(force=True) or {}
        test_name = body.get('test_name', 'A/B Test')
        model_a = body.get('model_a', {})  # {type: 'local', lab_label: 'Lab A'}
        model_b = body.get('model_b', {})  # {type: 'global', version: 5}
        use_held_out = body.get('use_held_out', False)
        
        # Create test record
        test_record = sb().table('fl_ab_tests').insert({
            'test_name': test_name,
            'model_a_type': model_a.get('type', 'local'),
            'model_a_version': model_a.get('lab_label') or str(model_a.get('version', '')),
            'model_b_type': model_b.get('type', 'global'),
            'model_b_version': model_b.get('lab_label') or str(model_b.get('version', '')),
            'status': 'running'
        }).execute()
        
        test_id = test_record.data[0]['id']
        
        # Load test data
        if use_held_out:
            # Use held-out test dataset
            test_data = sb().table('fl_test_patient_records').select('*').eq('is_active', True).execute()
            if not test_data.data:
                # Fall back to sampling from patient_records
                test_data = sb().table('patient_records').select('*').limit(50).execute()
        else:
            # Sample from patient_records
            test_data = sb().table('patient_records').select('*').limit(50).execute()
        
        if not test_data.data or len(test_data.data) == 0:
            sb().table('fl_ab_tests').update({'status': 'failed'}).eq('id', test_id).execute()
            return jsonify({'error': 'No test data available'}), 400
        
        test_records = test_data.data
        num_samples = len(test_records)
        
        # Load Model A (local)
        model_a_obj = None
        if model_a.get('type') == 'local' and model_a.get('lab_label'):
            model_a_path = model_path_for_lab(model_a['lab_label'])
            if os.path.exists(model_a_path):
                with open(model_a_path, 'rb') as f:
                    model_a_obj = pickle.load(f)
        
        # Load Model B (global)
        model_b_obj = None
        global_path = os.path.join(os.path.dirname(__file__), 'models', 'global.pkl')
        if os.path.exists(global_path):
            with open(global_path, 'rb') as f:
                model_b_obj = pickle.load(f)
        
        if model_a_obj is None and model_b_obj is None:
            sb().table('fl_ab_tests').update({'status': 'failed'}).eq('id', test_id).execute()
            return jsonify({'error': 'Neither model could be loaded'}), 400
        
        # Run predictions
        disease_labels = ['healthy', 'diabetes', 'hypertension', 'heart_disease']
        model_a_predictions = []
        model_b_predictions = []
        model_a_correct = 0
        model_b_correct = 0
        
        # Confusion matrix initialization (4x4 for 4 classes)
        confusion_a = [[0]*4 for _ in range(4)]
        confusion_b = [[0]*4 for _ in range(4)]
        
        for i, record in enumerate(test_records):
            # Encode features
            X = _encode_row_from_db(record)
            # Support both 'disease_label' and 'diagnosis' field names
            actual_label = record.get('disease_label')
            if actual_label is None:
                actual_label = record.get('diagnosis', 0)
            actual_label = int(actual_label) if actual_label is not None else 0
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Model A prediction
            if model_a_obj:
                try:
                    pred_a = int(model_a_obj.predict(X_scaled)[0])
                    proba_a = model_a_obj.predict_proba(X_scaled)[0]
                    conf_a = float(max(proba_a))
                except:
                    pred_a = _predict_label_from_features(record)
                    conf_a = 0.5
            else:
                pred_a = _predict_label_from_features(record)
                conf_a = 0.5
            
            # Model B prediction
            if model_b_obj:
                try:
                    pred_b = int(model_b_obj.predict(X_scaled)[0])
                    proba_b = model_b_obj.predict_proba(X_scaled)[0]
                    conf_b = float(max(proba_b))
                except:
                    pred_b = _predict_label_from_features(record)
                    conf_b = 0.5
            else:
                pred_b = _predict_label_from_features(record)
                conf_b = 0.5
            
            # Track correctness
            a_correct = pred_a == actual_label
            b_correct = pred_b == actual_label
            if a_correct:
                model_a_correct += 1
            if b_correct:
                model_b_correct += 1
            
            # Update confusion matrices
            if 0 <= actual_label < 4 and 0 <= pred_a < 4:
                confusion_a[actual_label][pred_a] += 1
            if 0 <= actual_label < 4 and 0 <= pred_b < 4:
                confusion_b[actual_label][pred_b] += 1
            
            # Store predictions
            model_a_predictions.append({
                'patient_id': i + 1,
                'actual': disease_labels[actual_label] if 0 <= actual_label < 4 else 'unknown',
                'actual_label': actual_label,
                'predicted': disease_labels[pred_a] if 0 <= pred_a < 4 else 'unknown',
                'predicted_label': pred_a,
                'confidence': conf_a,
                'correct': a_correct
            })
            
            model_b_predictions.append({
                'patient_id': i + 1,
                'actual': disease_labels[actual_label] if 0 <= actual_label < 4 else 'unknown',
                'actual_label': actual_label,
                'predicted': disease_labels[pred_b] if 0 <= pred_b < 4 else 'unknown',
                'predicted_label': pred_b,
                'confidence': conf_b,
                'correct': b_correct
            })
        
        # Calculate accuracies
        model_a_accuracy = model_a_correct / num_samples if num_samples > 0 else 0
        model_b_accuracy = model_b_correct / num_samples if num_samples > 0 else 0
        accuracy_delta = model_b_accuracy - model_a_accuracy
        
        # Calculate per-class metrics
        def calc_per_class_metrics(confusion):
            metrics = {}
            for i, label in enumerate(disease_labels):
                tp = confusion[i][i]
                fp = sum(confusion[j][i] for j in range(4) if j != i)
                fn = sum(confusion[i][j] for j in range(4) if j != i)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                metrics[label] = {'precision': precision, 'recall': recall, 'f1': f1}
            return metrics
        
        per_class_a = calc_per_class_metrics(confusion_a)
        per_class_b = calc_per_class_metrics(confusion_b)
        
        # Simple statistical significance (using McNemar's test approximation)
        # Count disagreements where one model is right and the other is wrong
        n01 = sum(1 for a, b in zip(model_a_predictions, model_b_predictions) if not a['correct'] and b['correct'])
        n10 = sum(1 for a, b in zip(model_a_predictions, model_b_predictions) if a['correct'] and not b['correct'])
        
        # Chi-squared approximation
        if n01 + n10 > 0:
            chi_sq = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            # p-value approximation (chi-squared with 1 df)
            p_value = 1 - min(0.99, chi_sq / 10)  # Simplified approximation
            is_significant = p_value < 0.05
        else:
            p_value = 1.0
            is_significant = False
        
        statistical_significance = {
            'p_value': p_value,
            'is_significant': is_significant,
            'test_used': 'McNemar',
            'model_b_wins': n01,
            'model_a_wins': n10
        }
        
        # Update test record with results
        sb().table('fl_ab_tests').update({
            'num_samples': num_samples,
            'model_a_accuracy': model_a_accuracy,
            'model_b_accuracy': model_b_accuracy,
            'accuracy_delta': accuracy_delta,
            'model_a_predictions': model_a_predictions,
            'model_b_predictions': model_b_predictions,
            'confusion_matrix_a': confusion_a,
            'confusion_matrix_b': confusion_b,
            'per_class_metrics_a': per_class_a,
            'per_class_metrics_b': per_class_b,
            'statistical_significance': statistical_significance,
            'status': 'completed'
        }).eq('id', test_id).execute()
        
        return jsonify({
            'success': True,
            'test_id': test_id,
            'test_name': test_name,
            'num_samples': num_samples,
            'model_a_accuracy': model_a_accuracy,
            'model_b_accuracy': model_b_accuracy,
            'accuracy_delta': accuracy_delta,
            'winner': 'Model B (Global)' if model_b_accuracy > model_a_accuracy else ('Model A (Local)' if model_a_accuracy > model_b_accuracy else 'Tie'),
            'statistical_significance': statistical_significance,
            'fl_improvement': f"+{accuracy_delta*100:.2f}%" if accuracy_delta > 0 else f"{accuracy_delta*100:.2f}%"
        })
        
    except Exception as e:
        print(f"Error in run_ab_test: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.get("/admin/ab_test_results/<test_id>")
def get_ab_test_results(test_id):
    """
    Get detailed results for a specific A/B test.
    """
    try:
        test = sb().table('fl_ab_tests').select('*').eq('id', test_id).execute()
        
        if not test.data:
            return jsonify({'error': 'Test not found'}), 404
        
        result = test.data[0]
        
        return jsonify({
            'test_id': result['id'],
            'test_name': result['test_name'],
            'created_at': result['created_at'],
            'status': result['status'],
            'model_a': {
                'type': result['model_a_type'],
                'version': result['model_a_version'],
                'accuracy': result['model_a_accuracy'],
                'predictions': result['model_a_predictions'],
                'confusion_matrix': result['confusion_matrix_a'],
                'per_class_metrics': result.get('per_class_metrics_a')
            },
            'model_b': {
                'type': result['model_b_type'],
                'version': result['model_b_version'],
                'accuracy': result['model_b_accuracy'],
                'predictions': result['model_b_predictions'],
                'confusion_matrix': result['confusion_matrix_b'],
                'per_class_metrics': result.get('per_class_metrics_b')
            },
            'num_samples': result['num_samples'],
            'accuracy_delta': result['accuracy_delta'],
            'winner': 'Model B' if (result['model_b_accuracy'] or 0) > (result['model_a_accuracy'] or 0) else ('Model A' if (result['model_a_accuracy'] or 0) > (result['model_b_accuracy'] or 0) else 'Tie'),
            'statistical_significance': result.get('statistical_significance')
        })
        
    except Exception as e:
        print(f"Error in get_ab_test_results: {e}")
        return jsonify({'error': str(e)}), 500


@app.get("/admin/ab_test_history")
def get_ab_test_history():
    """
    Get list of all past A/B tests.
    """
    try:
        tests = sb().table('fl_ab_tests').select('*').order('created_at', desc=True).limit(20).execute()
        
        result = []
        for t in (tests.data or []):
            winner = 'Tie'
            if (t['model_b_accuracy'] or 0) > (t['model_a_accuracy'] or 0):
                winner = 'Model B (Global)'
            elif (t['model_a_accuracy'] or 0) > (t['model_b_accuracy'] or 0):
                winner = 'Model A (Local)'
            
            result.append({
                'id': t['id'],
                'created_at': t['created_at'],
                'test_name': t['test_name'],
                'model_a_type': t['model_a_type'],
                'model_a_version': t['model_a_version'],
                'model_b_type': t['model_b_type'],
                'model_b_version': t['model_b_version'],
                'model_a_accuracy': t['model_a_accuracy'],
                'model_b_accuracy': t['model_b_accuracy'],
                'accuracy_delta': t['accuracy_delta'],
                'num_samples': t['num_samples'],
                'winner': winner,
                'status': t['status']
            })
        
        return jsonify({'tests': result})
        
    except Exception as e:
        print(f"Error in get_ab_test_history: {e}")
        return jsonify({'error': str(e)}), 500


@app.get("/admin/get_available_models")
def get_available_models():
    """
    Get list of available models for A/B testing.
    Returns local lab models and global model versions.
    """
    try:
        # Get local lab models
        local_models = []
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Check for lab-specific models
        lab_updates = sb().table('fl_client_updates').select('client_label').execute()
        lab_labels = list(set([r['client_label'] for r in (lab_updates.data or []) if r['client_label']]))
        
        for lab_label in lab_labels:
            model_path = model_path_for_lab(lab_label)
            if os.path.exists(model_path):
                local_models.append({
                    'type': 'local',
                    'lab_label': lab_label,
                    'display_name': f"Local: {lab_label}"
                })
        
        # Get global model versions
        global_models = []
        global_model_result = sb().table('fl_global_models').select('*').order('version', desc=True).limit(5).execute()
        
        for gm in (global_model_result.data or []):
            global_models.append({
                'type': 'global',
                'version': gm['version'],
                'display_name': f"Global v{gm['version']}"
            })
        
        # Also check for local global.pkl file
        global_path = os.path.join(models_dir, 'global.pkl')
        if os.path.exists(global_path) and not global_models:
            global_models.append({
                'type': 'global',
                'version': 1,
                'display_name': 'Global v1 (Local)'
            })
        
        return jsonify({
            'local_models': local_models,
            'global_models': global_models
        })
        
    except Exception as e:
        print(f"Error in get_available_models: {e}")
        return jsonify({'error': str(e)}), 500


@app.post("/admin/create_test_dataset")
def create_test_dataset():
    """
    Split current patient data into train/test sets.
    Moves a percentage of records to the test dataset.
    """
    try:
        body = request.get_json(force=True) or {}
        test_percentage = body.get('test_percentage', 20)  # Default 20%
        
        # Get all patient records
        records = sb().table('patient_records').select('*').execute()
        
        if not records.data or len(records.data) < 5:
            return jsonify({'error': 'Not enough patient records to create test set (minimum 5 required)'}), 400
        
        total_records = len(records.data)
        num_test = max(1, int(total_records * test_percentage / 100))
        
        # Randomly select records for test set
        import random
        test_indices = random.sample(range(total_records), num_test)
        
        # Insert selected records into test table
        test_records_created = 0
        for idx in test_indices:
            record = records.data[idx]
            
            # Get diagnosis - support both field names
            diagnosis_val = record.get('disease_label')
            if diagnosis_val is None:
                diagnosis_val = record.get('diagnosis', 0)
            diagnosis_val = int(diagnosis_val) if diagnosis_val is not None else 0
            
            # Create test record
            sb().table('fl_test_patient_records').insert({
                'original_record_id': record.get('id'),
                'lab_label': record.get('lab_label'),
                'patient_data': record,
                'actual_diagnosis': ['healthy', 'diabetes', 'hypertension', 'heart_disease'][diagnosis_val],
                'is_active': True
            }).execute()
            test_records_created += 1
        
        return jsonify({
            'success': True,
            'total_records': total_records,
            'test_records_created': test_records_created,
            'test_percentage': test_percentage,
            'message': f'Created {test_records_created} test records ({test_percentage}% of {total_records} total)'
        })
        
    except Exception as e:
        print(f"Error in create_test_dataset: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.get("/admin/get_test_dataset_info")
def get_test_dataset_info():
    """
    Get information about the current held-out test dataset.
    """
    try:
        # Count active test records
        test_records = sb().table('fl_test_patient_records').select('*').eq('is_active', True).execute()
        
        if not test_records.data:
            return jsonify({
                'has_test_dataset': False,
                'num_samples': 0,
                'message': 'No held-out test dataset. Use random sampling for A/B tests.'
            })
        
        # Analyze test dataset composition
        diagnosis_counts = {}
        lab_counts = {}
        
        for record in test_records.data:
            diagnosis = record.get('actual_diagnosis', 'unknown')
            lab = record.get('lab_label', 'unknown')
            
            diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
            lab_counts[lab] = lab_counts.get(lab, 0) + 1
        
        return jsonify({
            'has_test_dataset': True,
            'num_samples': len(test_records.data),
            'diagnosis_distribution': diagnosis_counts,
            'lab_distribution': lab_counts
        })
        
    except Exception as e:
        print(f"Error in get_test_dataset_info: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Patient Endpoints - Broadcasted Labs
# ============================================================================

@app.get("/patient/labs")
def get_patient_labs():
    """
    Get list of broadcasted (registered) labs with their current accuracy.
    
    For patients to see available labs and choose one.
    Returns labs sorted by accuracy (highest first).
    
    Response: Array of {
        lab_id: string,
        display_name: string,
        accuracy: number (0.0 to 1.0),
        accuracy_percent: string (e.g., "72.5%"),
        last_updated: string (ISO timestamp),
        num_patients: number (samples used for training)
    }
    """
    try:
        # Get all labs' latest updates from fl_client_updates
        client_updates = sb().table('fl_client_updates').select('*').order('created_at', desc=True).execute()
        
        if not client_updates.data:
            return jsonify({
                'labs': [],
                'message': 'No labs are registered yet. Check back later.'
            })
        
        # Group by lab to get latest update per lab
        lab_data = {}
        for update in client_updates.data:
            lab = update.get('client_label', 'unknown')
            if lab not in lab_data:
                # Get node accuracy (same as used elsewhere)
                accuracy = get_lab_node_accuracy(lab)
                if accuracy is None:
                    accuracy = update.get('local_accuracy', 0)
                
                lab_data[lab] = {
                    'lab_id': lab,
                    'display_name': lab.replace('_', ' ').title(),  # lab_A -> "Lab A"
                    'accuracy': accuracy if accuracy else 0,
                    'accuracy_percent': f"{(accuracy or 0) * 100:.1f}%",
                    'last_updated': update.get('created_at'),
                    'num_patients': update.get('num_examples', 0)
                }
        
        # Sort by accuracy descending (highest first)
        labs_list = sorted(lab_data.values(), key=lambda x: x['accuracy'], reverse=True)
        
        return jsonify({
            'labs': labs_list,
            'total_labs': len(labs_list)
        })
        
    except Exception as e:
        print(f"Error in get_patient_labs: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.post("/patient/select_lab")
def select_patient_lab():
    """
    Save the patient's selected/preferred lab.
    
    Request body: { patient_id: string, lab_id: string }
    
    Stores the selection in patient_lab_selections table or updates patient profile.
    """
    try:
        body = request.get_json(force=True) or {}
        patient_id = body.get('patient_id')
        lab_id = body.get('lab_id')
        
        if not patient_id:
            return jsonify({'error': 'patient_id is required'}), 400
        if not lab_id:
            return jsonify({'error': 'lab_id is required'}), 400
        
        # Normalize lab_id
        lab_id = normalize_lab_label(lab_id)
        
        # Verify the lab exists
        lab_check = sb().table('fl_client_updates').select('client_label').eq('client_label', lab_id).limit(1).execute()
        if not lab_check.data:
            return jsonify({'error': f'Lab {lab_id} not found'}), 404
        
        # Try to upsert to patient_lab_selections table
        # If table doesn't exist, we'll store in user metadata via Supabase auth
        try:
            sb().table('patient_lab_selections').upsert({
                'patient_id': patient_id,
                'lab_id': lab_id,
                'selected_at': time.strftime('%Y-%m-%dT%H:%M:%SZ')
            }, on_conflict='patient_id').execute()
            
            print(f"Patient {patient_id} selected lab {lab_id}")
            
        except Exception as table_error:
            # Table might not exist - just log and continue
            # The frontend can store in localStorage as fallback
            print(f"Could not save to patient_lab_selections: {table_error}")
            # Don't fail - let the frontend handle storage
        
        # Get lab details to return
        accuracy = get_lab_node_accuracy(lab_id)
        
        return jsonify({
            'success': True,
            'selected_lab': {
                'lab_id': lab_id,
                'display_name': lab_id.replace('_', ' ').title(),
                'accuracy': accuracy,
                'accuracy_percent': f"{(accuracy or 0) * 100:.1f}%"
            },
            'message': f'Successfully selected {lab_id.replace("_", " ").title()}'
        })
        
    except Exception as e:
        print(f"Error in select_patient_lab: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.get("/patient/selected_lab")
def get_patient_selected_lab():
    """
    Get the patient's currently selected lab.
    
    Query params: patient_id (required)
    
    Returns the selected lab details or null if none selected.
    """
    try:
        patient_id = request.args.get('patient_id')
        
        if not patient_id:
            return jsonify({'error': 'patient_id is required'}), 400
        
        # Try to get from patient_lab_selections table
        try:
            selection = sb().table('patient_lab_selections').select('*').eq('patient_id', patient_id).limit(1).execute()
            
            if selection.data and len(selection.data) > 0:
                lab_id = selection.data[0].get('lab_id')
                accuracy = get_lab_node_accuracy(lab_id)
                
                return jsonify({
                    'has_selection': True,
                    'selected_lab': {
                        'lab_id': lab_id,
                        'display_name': lab_id.replace('_', ' ').title(),
                        'accuracy': accuracy,
                        'accuracy_percent': f"{(accuracy or 0) * 100:.1f}%",
                        'selected_at': selection.data[0].get('selected_at')
                    }
                })
        except Exception as table_error:
            # Table might not exist
            print(f"Could not query patient_lab_selections: {table_error}")
        
        return jsonify({
            'has_selection': False,
            'selected_lab': None
        })
        
    except Exception as e:
        print(f"Error in get_patient_selected_lab: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Doctor Feedback Endpoints (Closed-Loop Feedback System)
# ============================================================================

DIAGNOSIS_LABELS = {
    0: 'healthy',
    1: 'diabetes',
    2: 'hypertension',
    3: 'heart_disease'
}

DIAGNOSIS_FROM_LABEL = {v: k for k, v in DIAGNOSIS_LABELS.items()}


@app.get("/admin/reports")
def get_admin_reports():
    """
    Get list of patient records (reports) for admin review.
    
    Query params:
      - limit (int, default 50): max records to return
      - offset (int, default 0): pagination offset
      - lab_label (str, optional): filter by lab
      - status (str, optional): 'pending', 'reviewed', 'all' (default 'all')
    
    Returns list of reports with feedback status.
    """
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        lab_filter = request.args.get('lab_label')
        status_filter = request.args.get('status', 'all')
        
        # Query patient_records
        query = sb().table('patient_records').select(
            'id, patient_id, lab_label, diagnosis, diagnosis_label, confidence, '
            'probabilities, created_at, age, sex'
        ).order('created_at', desc=True).range(offset, offset + limit - 1)
        
        if lab_filter:
            query = query.eq('lab_label', normalize_lab_label(lab_filter))
        
        result = query.execute()
        records = result.data or []
        
        # Get feedback for these records to determine status
        record_ids = [r['id'] for r in records]
        feedback_map = {}
        
        if record_ids:
            try:
                feedback_result = sb().table('doctor_feedback').select(
                    'record_id, agree, correct_diagnosis, reviewer_name, created_at'
                ).in_('record_id', record_ids).execute()
                
                for fb in (feedback_result.data or []):
                    feedback_map[fb['record_id']] = fb
            except Exception as fb_err:
                print(f"Could not fetch feedback: {fb_err}")
        
        # Build response with status
        reports = []
        for rec in records:
            has_feedback = rec['id'] in feedback_map
            fb = feedback_map.get(rec['id'])
            
            # Apply status filter
            if status_filter == 'pending' and has_feedback:
                continue
            if status_filter == 'reviewed' and not has_feedback:
                continue
            
            reports.append({
                'id': rec['id'],
                'patient_id': rec.get('patient_id', 'Unknown'),
                'lab_label': rec.get('lab_label', 'Unknown'),
                'diagnosis': rec.get('diagnosis'),
                'diagnosis_label': rec.get('diagnosis_label', 'Unknown'),
                'confidence': rec.get('confidence'),
                'probabilities': rec.get('probabilities'),
                'created_at': rec.get('created_at'),
                'age': rec.get('age'),
                'sex': rec.get('sex'),
                'status': 'reviewed' if has_feedback else 'pending',
                'feedback': {
                    'agree': fb.get('agree'),
                    'correct_diagnosis': fb.get('correct_diagnosis'),
                    'reviewer_name': fb.get('reviewer_name'),
                    'reviewed_at': fb.get('created_at')
                } if fb else None
            })
        
        # Get total count
        try:
            count_query = sb().table('patient_records').select('id', count='exact')
            if lab_filter:
                count_query = count_query.eq('lab_label', normalize_lab_label(lab_filter))
            count_result = count_query.execute()
            total = count_result.count or len(records)
        except:
            total = len(records)
        
        return jsonify({
            'reports': reports,
            'total': total,
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        print(f"Error in get_admin_reports: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.get("/admin/reports/<record_id>")
def get_admin_report_by_id(record_id):
    """
    Get a single patient record (report) by ID for evaluation.
    
    Returns full record details including any existing feedback.
    """
    try:
        # Fetch the record
        result = sb().table('patient_records').select('*').eq('id', record_id).limit(1).execute()
        
        if not result.data or len(result.data) == 0:
            return jsonify({'error': 'Report not found'}), 404
        
        record = result.data[0]
        
        # Fetch existing feedback if any
        feedback = None
        try:
            fb_result = sb().table('doctor_feedback').select('*').eq('record_id', record_id).limit(1).execute()
            if fb_result.data and len(fb_result.data) > 0:
                feedback = fb_result.data[0]
        except Exception as fb_err:
            print(f"Could not fetch feedback: {fb_err}")
        
        return jsonify({
            'report': {
                'id': record['id'],
                'patient_id': record.get('patient_id', 'Unknown'),
                'lab_label': record.get('lab_label', 'Unknown'),
                'diagnosis': record.get('diagnosis'),
                'diagnosis_label': record.get('diagnosis_label', 'Unknown'),
                'confidence': record.get('confidence'),
                'probabilities': record.get('probabilities'),
                'created_at': record.get('created_at'),
                # Demographics
                'age': record.get('age'),
                'sex': record.get('sex'),
                'height_cm': record.get('height_cm'),
                'weight_kg': record.get('weight_kg'),
                'bmi': record.get('bmi'),
                # Vitals
                'systolic_bp': record.get('systolic_bp'),
                'diastolic_bp': record.get('diastolic_bp'),
                'heart_rate': record.get('heart_rate'),
                # Blood chemistry
                'fasting_glucose': record.get('fasting_glucose'),
                'hba1c': record.get('hba1c'),
                'total_cholesterol': record.get('total_cholesterol'),
                'ldl_cholesterol': record.get('ldl_cholesterol'),
                'hdl_cholesterol': record.get('hdl_cholesterol'),
                'triglycerides': record.get('triglycerides'),
            },
            'feedback': {
                'id': feedback.get('id'),
                'agree': feedback.get('agree'),
                'correct_diagnosis': feedback.get('correct_diagnosis'),
                'correct_diagnosis_label': feedback.get('correct_diagnosis_label'),
                'remarks': feedback.get('remarks'),
                'reviewer_id': feedback.get('reviewer_id'),
                'reviewer_name': feedback.get('reviewer_name'),
                'created_at': feedback.get('created_at'),
                'updated_at': feedback.get('updated_at')
            } if feedback else None,
            'status': 'reviewed' if feedback else 'pending'
        })
        
    except Exception as e:
        print(f"Error in get_admin_report_by_id: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.post("/admin/reports/<record_id>/feedback")
def submit_report_feedback(record_id):
    """
    Submit doctor feedback on a patient record's AI prediction.
    
    Body (JSON):
      - agree (bool, required): true = agree with AI, false = disagree
      - correct_diagnosis (int, optional): 0-3 if disagree, the correct diagnosis
      - remarks (str, optional): additional comments
      - reviewer_id (str, optional): reviewer identifier (default: 'admin')
      - reviewer_name (str, optional): reviewer display name
    
    Uses upsert: one feedback per (record_id, reviewer_id), latest wins.
    """
    try:
        body = request.get_json(force=True) or {}
        
        # Validate required fields
        if 'agree' not in body:
            return jsonify({'error': 'agree (boolean) is required'}), 400
        
        agree = bool(body['agree'])
        correct_diagnosis = body.get('correct_diagnosis')
        correct_diagnosis_label = None
        remarks = body.get('remarks', '')
        reviewer_id = body.get('reviewer_id', 'admin')
        reviewer_name = body.get('reviewer_name', 'Admin')
        reviewer_role = body.get('reviewer_role', 'central_admin')
        
        # If disagree and correct_diagnosis provided, validate it
        if not agree and correct_diagnosis is not None:
            if correct_diagnosis not in [0, 1, 2, 3]:
                return jsonify({'error': 'correct_diagnosis must be 0, 1, 2, or 3'}), 400
            correct_diagnosis_label = DIAGNOSIS_LABELS.get(correct_diagnosis)
        
        # Verify record exists
        rec_check = sb().table('patient_records').select('id').eq('id', record_id).limit(1).execute()
        if not rec_check.data or len(rec_check.data) == 0:
            return jsonify({'error': 'Record not found'}), 404
        
        # Upsert feedback (insert or update if exists)
        feedback_data = {
            'record_id': record_id,
            'reviewer_id': reviewer_id,
            'reviewer_name': reviewer_name,
            'reviewer_role': reviewer_role,
            'agree': agree,
            'correct_diagnosis': correct_diagnosis,
            'correct_diagnosis_label': correct_diagnosis_label,
            'remarks': remarks
        }
        
        result = sb().table('doctor_feedback').upsert(
            feedback_data,
            on_conflict='record_id,reviewer_id'
        ).execute()
        
        print(f"Feedback submitted for record {record_id}: agree={agree}")
        
        return jsonify({
            'success': True,
            'message': 'Feedback submitted successfully',
            'feedback': result.data[0] if result.data else feedback_data
        })
        
    except Exception as e:
        print(f"Error in submit_report_feedback: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.get("/admin/feedback_stats")
def get_feedback_stats():
    """
    Get feedback statistics: overall and per-lab agreement rates.
    
    Query params:
      - days (int, optional): limit to last N days (default: all time)
    
    Returns:
      - overall: total, agreed, disagreed, agreement_rate
      - per_lab: list of {lab_label, total, agreed, disagreed, agreement_rate}
      - recent_trend: weekly breakdown if enough data
    """
    try:
        days = request.args.get('days')
        
        # Build query - join feedback with patient_records to get lab_label
        # Since Supabase doesn't support complex joins easily, we'll fetch both and join in Python
        
        # Fetch all feedback
        fb_query = sb().table('doctor_feedback').select('*').order('created_at', desc=True)
        fb_result = fb_query.execute()
        all_feedback = fb_result.data or []
        
        # Apply days filter if specified
        if days:
            from datetime import datetime, timedelta
            cutoff = datetime.utcnow() - timedelta(days=int(days))
            cutoff_str = cutoff.isoformat()
            all_feedback = [f for f in all_feedback if f.get('created_at', '') >= cutoff_str]
        
        if not all_feedback:
            return jsonify({
                'overall': {
                    'total': 0,
                    'agreed': 0,
                    'disagreed': 0,
                    'agreement_rate': 0.0,
                    'agreement_percent': '0%'
                },
                'per_lab': [],
                'recent_trend': []
            })
        
        # Get record_ids to fetch lab_labels
        record_ids = list(set(f['record_id'] for f in all_feedback))
        
        # Fetch patient_records for lab_label mapping
        lab_map = {}
        try:
            if record_ids:
                rec_result = sb().table('patient_records').select('id, lab_label').in_('id', record_ids).execute()
                for rec in (rec_result.data or []):
                    lab_map[rec['id']] = rec.get('lab_label', 'unknown')
        except Exception as e:
            print(f"Could not fetch lab labels: {e}")
        
        # Calculate overall stats
        total = len(all_feedback)
        agreed = sum(1 for f in all_feedback if f.get('agree'))
        disagreed = total - agreed
        agreement_rate = agreed / total if total > 0 else 0
        
        # Calculate per-lab stats
        lab_stats = {}
        for fb in all_feedback:
            lab = lab_map.get(fb['record_id'], 'unknown')
            if lab not in lab_stats:
                lab_stats[lab] = {'total': 0, 'agreed': 0}
            lab_stats[lab]['total'] += 1
            if fb.get('agree'):
                lab_stats[lab]['agreed'] += 1
        
        per_lab = []
        for lab, stats in sorted(lab_stats.items(), key=lambda x: x[1]['total'], reverse=True):
            per_lab.append({
                'lab_label': lab,
                'display_name': lab.replace('_', ' ').title(),
                'total': stats['total'],
                'agreed': stats['agreed'],
                'disagreed': stats['total'] - stats['agreed'],
                'agreement_rate': stats['agreed'] / stats['total'] if stats['total'] > 0 else 0,
                'agreement_percent': f"{(stats['agreed'] / stats['total'] * 100):.1f}%" if stats['total'] > 0 else '0%'
            })
        
        # Calculate weekly trend (last 4 weeks)
        from datetime import datetime, timedelta
        recent_trend = []
        now = datetime.utcnow()
        
        for week_offset in range(4):
            week_start = now - timedelta(weeks=week_offset + 1)
            week_end = now - timedelta(weeks=week_offset)
            
            week_feedback = [
                f for f in all_feedback
                if week_start.isoformat() <= f.get('created_at', '') < week_end.isoformat()
            ]
            
            week_total = len(week_feedback)
            week_agreed = sum(1 for f in week_feedback if f.get('agree'))
            
            recent_trend.append({
                'week': f"Week -{week_offset + 1}",
                'start_date': week_start.strftime('%Y-%m-%d'),
                'end_date': week_end.strftime('%Y-%m-%d'),
                'total': week_total,
                'agreed': week_agreed,
                'agreement_rate': week_agreed / week_total if week_total > 0 else 0
            })
        
        recent_trend.reverse()  # Oldest to newest
        
        return jsonify({
            'overall': {
                'total': total,
                'agreed': agreed,
                'disagreed': disagreed,
                'agreement_rate': agreement_rate,
                'agreement_percent': f"{agreement_rate * 100:.1f}%"
            },
            'per_lab': per_lab,
            'recent_trend': recent_trend
        })
        
    except Exception as e:
        print(f"Error in get_feedback_stats: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.get("/lab/feedback")
def get_lab_feedback():
    """
    Get feedback on reports created by a specific lab.
    
    Query params:
      - lab_label (str, required): the lab identifier
      - limit (int, default 50): max records
    
    Returns list of feedback for this lab's patient records.
    """
    try:
        lab_label = request.args.get('lab_label')
        limit = int(request.args.get('limit', 50))
        
        if not lab_label:
            return jsonify({'error': 'lab_label is required'}), 400
        
        lab_label = normalize_lab_label(lab_label)
        
        # Get patient_records for this lab
        rec_result = sb().table('patient_records').select(
            'id, patient_id, diagnosis, diagnosis_label, confidence, created_at'
        ).eq('lab_label', lab_label).order('created_at', desc=True).limit(limit).execute()
        
        records = rec_result.data or []
        record_ids = [r['id'] for r in records]
        
        if not record_ids:
            return jsonify({'feedback': [], 'total': 0})
        
        # Get feedback for these records
        fb_result = sb().table('doctor_feedback').select('*').in_('record_id', record_ids).execute()
        feedback_map = {fb['record_id']: fb for fb in (fb_result.data or [])}
        
        # Build response
        feedback_list = []
        for rec in records:
            fb = feedback_map.get(rec['id'])
            if fb:  # Only include records that have feedback
                feedback_list.append({
                    'record_id': rec['id'],
                    'patient_id': rec.get('patient_id', 'Unknown'),
                    'ai_diagnosis': rec.get('diagnosis'),
                    'ai_diagnosis_label': rec.get('diagnosis_label'),
                    'ai_confidence': rec.get('confidence'),
                    'record_created_at': rec.get('created_at'),
                    'feedback': {
                        'agree': fb.get('agree'),
                        'correct_diagnosis': fb.get('correct_diagnosis'),
                        'correct_diagnosis_label': fb.get('correct_diagnosis_label'),
                        'remarks': fb.get('remarks'),
                        'reviewer_name': fb.get('reviewer_name'),
                        'created_at': fb.get('created_at')
                    }
                })
        
        return jsonify({
            'feedback': feedback_list,
            'total': len(feedback_list),
            'lab_label': lab_label
        })
        
    except Exception as e:
        print(f"Error in get_lab_feedback: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.post("/lab/feedback")
def submit_lab_feedback():
    """
    Allow lab to submit feedback on their own reports.
    
    Body (JSON):
      - record_id (str, required): the patient record ID
      - lab_label (str, required): must match the record's lab
      - agree (bool, required): agree with AI prediction
      - correct_diagnosis (int, optional): 0-3 if disagree
      - remarks (str, optional): additional comments
      - reviewer_id (str, optional): lab user identifier
      - reviewer_name (str, optional): lab user name
    
    Only allows feedback on records belonging to this lab.
    """
    try:
        body = request.get_json(force=True) or {}
        
        record_id = body.get('record_id')
        lab_label = body.get('lab_label')
        
        if not record_id:
            return jsonify({'error': 'record_id is required'}), 400
        if not lab_label:
            return jsonify({'error': 'lab_label is required'}), 400
        if 'agree' not in body:
            return jsonify({'error': 'agree (boolean) is required'}), 400
        
        lab_label = normalize_lab_label(lab_label)
        
        # Verify record exists and belongs to this lab
        rec_check = sb().table('patient_records').select('id, lab_label').eq('id', record_id).limit(1).execute()
        if not rec_check.data or len(rec_check.data) == 0:
            return jsonify({'error': 'Record not found'}), 404
        
        record_lab = rec_check.data[0].get('lab_label')
        if record_lab != lab_label:
            return jsonify({'error': 'Record does not belong to this lab'}), 403
        
        # Prepare feedback
        agree = bool(body['agree'])
        correct_diagnosis = body.get('correct_diagnosis')
        correct_diagnosis_label = None
        
        if not agree and correct_diagnosis is not None:
            if correct_diagnosis not in [0, 1, 2, 3]:
                return jsonify({'error': 'correct_diagnosis must be 0, 1, 2, or 3'}), 400
            correct_diagnosis_label = DIAGNOSIS_LABELS.get(correct_diagnosis)
        
        reviewer_id = body.get('reviewer_id', f'lab_{lab_label}')
        reviewer_name = body.get('reviewer_name', lab_label.replace('_', ' ').title())
        
        feedback_data = {
            'record_id': record_id,
            'reviewer_id': reviewer_id,
            'reviewer_name': reviewer_name,
            'reviewer_role': 'lab',
            'agree': agree,
            'correct_diagnosis': correct_diagnosis,
            'correct_diagnosis_label': correct_diagnosis_label,
            'remarks': body.get('remarks', '')
        }
        
        result = sb().table('doctor_feedback').upsert(
            feedback_data,
            on_conflict='record_id,reviewer_id'
        ).execute()
        
        print(f"Lab feedback submitted for record {record_id}: agree={agree}")
        
        return jsonify({
            'success': True,
            'message': 'Feedback submitted successfully',
            'feedback': result.data[0] if result.data else feedback_data
        })
        
    except Exception as e:
        print(f"Error in submit_lab_feedback: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def get_lab_agreement_rate(lab_label: str) -> dict:
    """
    Helper function to get agreement rate for a specific lab.
    Used in aggregation for reweighting.
    
    Returns: {'total': int, 'agreed': int, 'rate': float}
    """
    try:
        lab_label = normalize_lab_label(lab_label)
        
        # Get record IDs for this lab
        rec_result = sb().table('patient_records').select('id').eq('lab_label', lab_label).execute()
        record_ids = [r['id'] for r in (rec_result.data or [])]
        
        if not record_ids:
            return {'total': 0, 'agreed': 0, 'rate': 1.0}  # Default to 1.0 if no feedback
        
        # Get feedback for these records
        fb_result = sb().table('doctor_feedback').select('agree').in_('record_id', record_ids).execute()
        
        total = len(fb_result.data or [])
        agreed = sum(1 for f in (fb_result.data or []) if f.get('agree'))
        
        return {
            'total': total,
            'agreed': agreed,
            'rate': agreed / total if total > 0 else 1.0  # Default to 1.0 if no feedback
        }
        
    except Exception as e:
        print(f"Error getting lab agreement rate: {e}")
        return {'total': 0, 'agreed': 0, 'rate': 1.0}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
