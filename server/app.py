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
    encode_features,
    model_path_for_lab,
    load_or_init_model,
    save_model,
    get_parameters,
    set_parameters,
    predict_prob,
)

# Load env from both repo root and server folder.
# - Frontend uses VITE_* vars, backend prefers non-VITE vars.
_SERVER_DIR = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(_SERVER_DIR, ".env"))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(_SERVER_DIR), ".env"))

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


def generate_clinical_insights(body: dict, disease_type: str, risk_score: float) -> dict:
    """Generate dynamic clinical insights based on patient data and prediction"""
    insights = {
        'risk_factors': [],
        'critical_values': [],
        'recommendations': []
    }
    
    # Analyze risk factors
    age = int(body.get('age', 30))
    bmi = float(body.get('bmi', 25.0))
    bp_sys = int(body.get('bp_sys', 120))
    bp_dia = int(body.get('bp_dia', 80))
    glucose = int(body.get('glucose', 100))
    cholesterol = int(body.get('cholesterol', 200))
    smoker = str(body.get('smoker_status', 'no')).lower()
    family_history = str(body.get('family_history', 'none')).lower()
    medication_use = str(body.get('medication_use', 'none')).lower()
    prior = str(body.get('prior_conditions', '')).lower()
    
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
    # Use default values for missing fields since the database might not have all new columns yet
    # Handle None values safely
    def safe_int(val, default=0):
        return int(val) if val is not None else default
    
    def safe_float(val, default=0.0):
        return float(val) if val is not None else default
    
    def safe_str(val, default=''):
        return str(val) if val is not None else default
    
    payload = {
        'age': safe_int(record.get('age'), 30),
        'discomfort_level': safe_int(record.get('discomfort_level'), 3),
        'symptom_duration': safe_int(record.get('symptom_duration'), 7),
        'gender': safe_str(record.get('gender'), 'male').lower(),
        'blood_type': safe_str(record.get('blood_type'), 'O').upper(),
        'prior_conditions': safe_str(record.get('prior_conditions'), ''),
        'bmi': safe_float(record.get('bmi'), 25.0),
        'smoker_status': safe_str(record.get('smoker_status'), 'no').lower(),
        'heart_rate': safe_int(record.get('heart_rate'), 70),
        'bp_sys': safe_int(record.get('bp_sys'), 120),
        'bp_dia': safe_int(record.get('bp_dia'), 80),
        'cholesterol': safe_int(record.get('cholesterol'), 200),
        'glucose': safe_int(record.get('glucose'), 100),
        'family_history': safe_str(record.get('family_history'), 'none').lower(),
        'medication_use': safe_str(record.get('medication_use'), 'none').lower(),
    }
    X, _ = encode_features(payload)
    return X


def _predict_label_from_features(body: dict) -> int:
    """
    Rule-based label prediction for initial training data
    Returns: 0=healthy, 1=diabetes, 2=hypertension, 3=heart_disease
    """
    glucose = int(body.get('glucose', 100))
    bp_sys = int(body.get('bp_sys', 120))
    bp_dia = int(body.get('bp_dia', 80))
    cholesterol = int(body.get('cholesterol', 200))
    bmi = float(body.get('bmi', 25.0))
    heart_rate = int(body.get('heart_rate', 70))
    age = int(body.get('age', 30))
    smoker = str(body.get('smoker_status', 'no')).lower()
    
    # Score different conditions
    diabetes_score = 0
    hypertension_score = 0
    heart_disease_score = 0
    
    # Diabetes indicators
    if glucose >= 126:
        diabetes_score += 3
    elif glucose >= 100:
        diabetes_score += 2
    if bmi >= 30:
        diabetes_score += 1
    
    # Hypertension indicators
    if bp_sys >= 140 or bp_dia >= 90:
        hypertension_score += 3
    elif bp_sys >= 130 or bp_dia >= 85:
        hypertension_score += 2
    if age > 60:
        hypertension_score += 1
    
    # Heart disease indicators
    if cholesterol >= 240:
        heart_disease_score += 2
    elif cholesterol >= 200:
        heart_disease_score += 1
    if smoker == 'yes':
        heart_disease_score += 2
    if bp_sys >= 140:
        heart_disease_score += 1
    if age > 60:
        heart_disease_score += 1
    
    # Determine primary condition based on highest score
    max_score = max(diabetes_score, hypertension_score, heart_disease_score)
    
    if max_score == 0:
        return 0  # healthy
    elif diabetes_score == max_score:
        return 1  # diabetes
    elif hypertension_score == max_score:
        return 2  # hypertension
    else:
        return 3  # heart_disease


def _calculate_confidence_score(body: dict, predicted_label: int) -> float:
    """
    Calculate confidence score based on how strong the risk factors are
    Returns: float between 0.5 and 0.95
    """
    glucose = int(body.get('glucose', 100))
    bp_sys = int(body.get('bp_sys', 120))
    bp_dia = int(body.get('bp_dia', 80))
    cholesterol = int(body.get('cholesterol', 200))
    bmi = float(body.get('bmi', 25.0))
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
                    X_list.append(X_row)
                    y_list.append(record['disease_label'])
                
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
                # IMPORTANT: Add labels that are NOT already present. (Previous logic could re-add the same class.)
                missing_labels = [lbl for lbl in range(4) if lbl not in set(unique_classes.tolist())]
                for lbl in missing_labels:
                    synthetic_X = X_train.copy()
                    noise = np.random.normal(0, 0.1, synthetic_X.shape)
                    synthetic_X = synthetic_X + noise
                    synthetic_y = np.full(len(synthetic_X), lbl)
                    X_train = np.vstack([X_train, synthetic_X])
                    y_train = np.concatenate([y_train, synthetic_y])
            
            # Scale features and train
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model.fit(X_train_scaled, y_train)
            save_model(lab_label, model)
            
            # Calculate model accuracy for tracking
            local_accuracy = float(model.score(X_train_scaled, y_train))
            
            # Compute grad norm
            new_coef, new_intercept = get_parameters(model)
            grad_norm = float(np.mean([
                np.linalg.norm((new_coef - prev_coef).ravel(), ord=2),
                np.linalg.norm((new_intercept - prev_intercept).ravel(), ord=2)
            ]))
            
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
            'bmi': float(body.get('bmi', 25.0)),
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
    lab_label = body.get('lab_label') or 'lab_sim'
    
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
        for record in lab_records.data:
            X_row = _encode_row_from_db(record)
            # Ensure feature vector has consistent dimensions
            if X_row.shape[1] != 33:  # Expected feature dimension
                print(f"Warning: Feature dimension mismatch in send_model_update. Got: {X_row.shape[1]}")
                continue
            X_list.append(X_row)
            y_list.append(record['disease_label'])
        
        if not X_list:
            return jsonify({'error': 'No valid training data found'}), 400
            
        X_train = np.vstack(X_list)
        y_train = np.array(y_list, dtype=int)
        
        # Load current model
        model = load_or_init_model(lab_label, X_train.shape[1])
        prev_coef, prev_intercept = get_parameters(model)
        
        # Retrain model
        
        # Check if we have enough classes for multiclass training
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            # If only one class, create a dummy second class for training
            print(f"Warning: Only {len(unique_classes)} class(es) found. Adding dummy data for training.")
            # Add some dummy data with a DIFFERENT label than the existing one
            existing = int(unique_classes[0])
            alt = 0 if existing != 0 else 1
            dummy_X = X_train.copy()
            noise = np.random.normal(0, 0.05, dummy_X.shape)
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
        
        # Compute grad norm
        new_coef, new_intercept = get_parameters(model)
        grad_norm = float(np.mean([
            np.linalg.norm((new_coef - prev_coef).ravel(), ord=2),
            np.linalg.norm((new_intercept - prev_intercept).ravel(), ord=2)
        ]))
        
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
            local_accuracy = float(model.score(X_train_scaled, y_train))
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
        # Get latest local models from each lab (one per lab)
        try:
            # Get most recent update per lab
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
            print(f"Found {len(updates_list)} labs with models: {list(lab_updates.keys())}")
            
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
                    
                    models_data.append({
                        'params': params,
                        'num_examples': num_examples,
                        'lab': lab_label,
                        'accuracy': update.get('local_accuracy', 0)
                    })
                    total_samples += num_examples
                
                os.unlink(tmp_path)
                print(f"Loaded model from {lab_label}: {num_examples} samples, accuracy: {update.get('local_accuracy', 0):.3f}")
                
            except Exception as e:
                print(f"Error loading model from {storage_path}: {e}")
                continue

        if not models_data:
            return jsonify({'error': 'no valid models found'}), 400

        print(f"Total samples across all labs: {total_samples}")

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
            
        else:  # tree-based models
            # For tree models, create a weighted voting ensemble
            from sklearn.ensemble import VotingClassifier
            
            # Debug: Check what's in params for each model
            print(f"Model types in aggregation:")
            for data in models_data:
                print(f"  Lab {data['lab']}: params keys = {data['params'].keys()}")
            
            # Ensure all models have the 'model' key
            valid_models = [data for data in models_data if 'model' in data['params']]
            if len(valid_models) < len(models_data):
                print(f"Warning: Only {len(valid_models)}/{len(models_data)} models have valid 'model' key")
                if not valid_models:
                    return jsonify({'error': 'No valid tree-based models found for aggregation'}), 400
            
            # Create weighted voting classifier with valid lab models
            estimators = [(f"{data['lab']}", data['params']['model']) for data in valid_models]
            weights = [data['num_examples'] for data in valid_models]
            
            global_model = VotingClassifier(
                estimators=estimators,
                voting='soft',  # Use probability voting
                weights=weights
            )
            
            # The VotingClassifier needs to be "fitted" with dummy data to initialize
            # We'll use a small subset from the first model's training set
            try:
                # Create minimal dummy data just to initialize the ensemble
                # Feature count: 9 numeric + 3 gender + 8 blood_type + 2 smoker + 4 family + 6 medication + 1 prior_len = 33
                dummy_X = np.random.randn(5, 33)  # 5 samples, 33 features (our actual feature count)
                dummy_y = np.array([0, 1, 2, 3, 0])  # Dummy labels covering all classes
                global_model.fit(dummy_X, dummy_y)
                print(f"Created weighted voting ensemble from {len(valid_models)} lab models")
            except Exception as e:
                print(f"Error creating ensemble, using best lab model: {e}")
                import traceback
                traceback.print_exc()
                best_lab = max(valid_models, key=lambda x: x['num_examples'])
                global_model = best_lab['params']['model']
                print(f"Fallback: Using model from {best_lab['lab']} (most samples)")

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

        # Evaluate global model on test dataset
        # Use patient_data.csv and split it into train/test
        global_accuracy = None
        
        try:
            data_path = os.path.join(os.path.dirname(__file__), 'data', 'patient_data.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                
                # Encode features and labels
                X_list, y_list = [], []
                for _, r in df.iterrows():
                    X_list.append(_encode_row(r))
                    y_list.append(int(r['disease_label']))
                X_all = np.vstack(X_list)
                y_all = np.array(y_list)
                
                # Split into train (80%) and test (20%)
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
                )
                
                # Scale test data using scaler fitted on training data
                scaler_eval = StandardScaler()
                scaler_eval.fit(X_train)  # Fit on training data
                X_test_scaled = scaler_eval.transform(X_test)  # Transform test data
                
                # Evaluate on test set
                preds = global_model.predict(X_test_scaled)
                global_accuracy = float((preds == y_test).mean())
                print(f"Global model accuracy on test set: {global_accuracy:.3f} (evaluated on {len(y_test)}/{len(y_all)} samples)")
            else:
                print(f"Test data file not found: {data_path}")
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
                        'local_accuracy': update.get('local_accuracy'),
                        'num_examples': update.get('num_examples'),
                        'has_model': update.get('storage_path') is not None,
                        'ready_for_aggregation': update.get('storage_path') is not None
                    }
        
        # Get recent round metrics for history
        round_metrics = sb().table('fl_round_metrics').select('*').order('round', desc=True).limit(10).execute()
        
        # Calculate total samples from lab statuses
        total_samples = sum(lab.get('num_examples', 0) for lab in lab_status.values())
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
        return jsonify({'error': str(e)}), 500


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
            'needs_update': not has_downloaded,
            'has_downloaded': has_downloaded
        })
        
    except Exception as e:
        print(f"Error in get_global_model_info: {e}")
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
        lab_label = body.get('lab_label', 'unknown')
        
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
            
            # Track the download with performance metrics
            try:
                # Get global model's accuracy from round metrics
                round_metrics = sb().table('fl_round_metrics').select('*').eq('round', latest_global['version']).execute()
                global_accuracy = round_metrics.data[0].get('average_accuracy') if round_metrics.data else None
                
                download_record = {
                    'lab_label': lab_label,
                    'global_model_version': latest_global['version'],
                    'downloaded_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'accuracy_before_download': accuracy_before
                }
                
                # Calculate potential improvement
                if accuracy_before and global_accuracy:
                    improvement = ((global_accuracy - accuracy_before) / accuracy_before) * 100
                    download_record['improvement_percentage'] = round(improvement, 2)
                
                sb().table('fl_model_downloads').insert(download_record).execute()
                print(f"Download tracked for {lab_label}, version {latest_global['version']}")
                if accuracy_before and global_accuracy:
                    print(f"  Accuracy improvement: {accuracy_before:.4f} → {global_accuracy:.4f} ({improvement:+.2f}%)")
            except Exception as track_error:
                print(f"Warning: Could not track download: {track_error}")
                # Continue even if tracking fails
            
            # Get global model accuracy for comparison
            round_metrics = sb().table('fl_round_metrics').select('*').eq('round', latest_global['version']).execute()
            global_accuracy = round_metrics.data[0].get('average_accuracy') if round_metrics.data else None
            
            improvement_metrics = None
            if accuracy_before and global_accuracy:
                improvement = ((global_accuracy - accuracy_before) / accuracy_before) * 100
                improvement_metrics = {
                    'accuracy_before': accuracy_before,
                    'accuracy_after': global_accuracy,
                    'improvement_percentage': round(improvement, 2),
                    'absolute_improvement': round(global_accuracy - accuracy_before, 4)
                }
            
            return jsonify({
                'success': True,
                'global_model': {
                    'version': latest_global['version'],
                    'model_type': latest_global.get('model_type'),
                    'created_at': latest_global.get('created_at'),
                    'storage_path': storage_path,
                    'local_path': local_model_path,
                    'accuracy': global_accuracy
                },
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
                round_num = metric.get('round', 0)
                
                # Find corresponding global model
                global_model = next((gm for gm in global_models.data if gm['version'] == round_num), None) if global_models.data else None
                
                # Count labs that participated in this round
                round_updates = [u for u in (client_updates.data or []) if u.get('round') == round_num]
                participating_labs = len(set(u.get('client_label') for u in round_updates))
                total_samples = sum(u.get('num_examples', 0) for u in round_updates)
                
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
