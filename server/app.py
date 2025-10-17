import os
import threading
import subprocess
import signal
import time
import pickle
import tempfile
import uuid
from typing import Optional, List
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

# Load .env from this folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Environment
FL_SERVER_ADDRESS = os.environ.get("FL_SERVER_ADDRESS", "127.0.0.1:8080")
SUPABASE_URL = os.environ.get("SUPABASE_URL") or os.environ.get("VITE_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("VITE_SUPABASE_ANON_KEY")

app = Flask(__name__)
CORS(app)

_server_thread: Optional[threading.Thread] = None
_server_stop = threading.Event()
_client_procs: List[subprocess.Popen] = []
_current_run_id: Optional[str] = None


def sb():
    assert SUPABASE_URL and SUPABASE_KEY, "Supabase env is missing"
    return create_client(SUPABASE_URL, SUPABASE_KEY)


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


@app.post("/lab/add_patient_data")
def add_patient_data():
    try:
        body = request.get_json(force=True) or {}
        lab_label = body.get('lab_label') or 'lab_sim'
        
        # Encode features for the new patient
        X_req, _ = encode_features(body)

        # Get all patient records for this lab from database
        try:
            lab_records = sb().table('patient_records').select('*').eq('lab_label', lab_label).execute()
            if lab_records.data:
                # Convert to training data
                X_list = []
                y_list = []
                for record in lab_records.data:
                    # Reconstruct feature vector from database record
                    X_row = _encode_row_from_db(record)
                    # Ensure feature vector has the same shape as the new request
                    if X_row.shape[1] != X_req.shape[1]:
                        print(f"Warning: Feature dimension mismatch. DB: {X_row.shape[1]}, Request: {X_req.shape[1]}")
                        # Skip this record if dimensions don't match
                        continue
                    X_list.append(X_row)
                    y_list.append(record['disease_label'])
                
                if X_list:
                    X_train = np.vstack(X_list)
                    y_train = np.array(y_list, dtype=int)
                else:
                    # No valid records, fall back to sample data
                    X_train = np.zeros((1, X_req.shape[1]))
                    y_train = np.array([0], dtype=int)
            else:
                # Fallback to sample CSV if no lab data
                data_path = os.path.join(os.path.dirname(__file__), 'data', 'patient_data.csv')
                df = pd.read_csv(data_path)
                X_list = []
                for _, r in df.iterrows():
                    X_list.append(_encode_row(r))
                X_train = np.vstack(X_list) if X_list else np.zeros((1, X_req.shape[1]))
                y_train = df['disease_label'].to_numpy(dtype=int) if 'disease_label' in df.columns else np.zeros((X_train.shape[0],), dtype=int)
        except Exception as e:
            print(f"Error loading lab data: {e}")
            # Fallback to sample data
            data_path = os.path.join(os.path.dirname(__file__), 'data', 'patient_data.csv')
            df = pd.read_csv(data_path)
            X_list = []
            for _, r in df.iterrows():
                X_list.append(_encode_row(r))
            X_train = np.vstack(X_list) if X_list else np.zeros((1, X_req.shape[1]))
            y_train = df['disease_label'].to_numpy(dtype=int) if 'disease_label' in df.columns else np.zeros((X_train.shape[0],), dtype=int)

        # Load or initialize model
        model = load_or_init_model(lab_label, X_req.shape[1])
        prev_coef, prev_intercept = get_parameters(model)
    
        # Train model on lab's data
        
        # Check if we have enough classes for multiclass training
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            # If only one class, create a dummy second class for training
            print(f"Warning: Only {len(unique_classes)} class(es) found. Adding dummy data for training.")
            # Add some dummy data with different labels
            dummy_X = X_train.copy()
            dummy_y = np.full(len(dummy_X), 1)  # Create class 1
            X_train = np.vstack([X_train, dummy_X])
            y_train = np.concatenate([y_train, dummy_y])
        
        # Scale features for better training
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train the model directly (don't create a fresh one, use the loaded/initialized model)
        model.fit(X_train_scaled, y_train)
        save_model(lab_label, model)

        # Predict on the new patient (scale the input)
        X_req_scaled = scaler.transform(X_req)
        risk_score, disease_type = predict_prob(model, X_req_scaled)
        pred_label = np.argmax(model.predict_proba(X_req_scaled)[0])

        # Compute grad norm approximation vs previous params
        new_coef, new_intercept = get_parameters(model)
        grad_norm = float(np.mean([
            np.linalg.norm((new_coef - prev_coef).ravel(), ord=2),
            np.linalg.norm((new_intercept - prev_intercept).ravel(), ord=2)
        ]))

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
        local_accuracy = float(model.score(X_train_scaled, y_train))
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
            # Add some dummy data with different labels
            dummy_X = X_train.copy()
            dummy_y = np.full(len(dummy_X), 1)  # Create class 1
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
        
        # Record round metric with valid run_id
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
            'timestamp': timestamp
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
