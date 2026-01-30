import os
import pickle
import glob
import json
import requests
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Use Gradient Boosting for better performance (you can switch to RandomForest if needed)
MODEL_TYPE = 'gradient_boosting'  # Options: 'logistic', 'random_forest', 'gradient_boosting'

# Disease types: 0=healthy, 1=diabetes, 2=hypertension, 3=heart_disease
DISEASE_TYPES = ['healthy', 'diabetes', 'hypertension', 'heart_disease']

# Unified feature columns for the new clinical schema
# Total: 27 features (13 numerical + 14 categorical/binary)
FEATURE_COLUMNS = [
    # Numerical features (13)
    'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'fasting_glucose', 'hba1c', 'total_cholesterol', 'ldl_cholesterol',
    'hdl_cholesterol', 'triglycerides', 'max_heart_rate', 'st_depression',
    # Categorical/binary features (14)
    'sex_encoded',          # M=0, F=1
    'chest_pain_type',      # 1-4
    'resting_ecg',          # 0-2
    'exercise_angina',      # 0-1
    'st_slope',             # 1-3
    'smoking_status',       # 0-2
    'family_history_cvd',   # 0-1
    'family_history_diabetes',  # 0-1
    'prior_hypertension',   # 0-1
    'prior_diabetes',       # 0-1
    'prior_heart_disease',  # 0-1
    'on_bp_medication',     # 0-1
    'on_diabetes_medication',   # 0-1
    'on_cholesterol_medication' # 0-1
]

# Legacy categorical maps (for backward compatibility)
CATEGORICAL_MAPS = {
    'gender': ['male', 'female', 'other'],
    'blood_type': ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-'],
    'smoker_status': ['yes', 'no'],
    'family_history': ['none', 'diabetes', 'hypertension', 'heart_disease'],
    'medication_use': ['none', 'metformin', 'insulin', 'lisinopril', 'amlodipine', 'atorvastatin']
}


def encode_features(payload: Dict, use_legacy: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode patient data into feature vector for model prediction.
    
    Supports both new clinical schema and legacy format for backward compatibility.
    
    Args:
        payload: Patient data dictionary
        use_legacy: If True, use old encoding format (for existing data)
    
    Returns:
        Tuple of (X feature array, empty array for compatibility)
    """
    # Detect which format the payload is using
    is_new_format = 'systolic_bp' in payload or 'fasting_glucose' in payload
    
    if is_new_format or not use_legacy:
        return encode_clinical_features(payload)
    else:
        return encode_legacy_features(payload)


def encode_clinical_features(payload: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode features using the new unified clinical schema.
    
    Expected payload keys match FEATURE_COLUMNS with some aliases:
    - 'sex' or 'gender': 'M'/'F' or 'male'/'female'
    - 'bp_sys' or 'systolic_bp': systolic blood pressure
    - 'bp_dia' or 'diastolic_bp': diastolic blood pressure
    - 'glucose' or 'fasting_glucose': fasting glucose level
    - 'cholesterol' or 'total_cholesterol': total cholesterol
    """
    # Handle None payload
    if payload is None:
        payload = {}
    
    # Helper function to safely convert to float with fallback
    def safe_float(value, default):
        if value is None or value == '':
            return float(default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return float(default)
    
    # Helper function to safely convert to int with fallback
    def safe_int(value, default):
        if value is None or value == '':
            return int(default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return int(default)
    
    # Extract numerical features with defaults
    age = safe_float(payload.get('age'), 50)
    bmi = safe_float(payload.get('bmi'), 25.0)
    
    # Blood pressure (support both naming conventions)
    systolic_bp = safe_float(payload.get('systolic_bp', payload.get('bp_sys')), 120)
    diastolic_bp = safe_float(payload.get('diastolic_bp', payload.get('bp_dia')), 80)
    
    heart_rate = safe_float(payload.get('heart_rate'), 72)
    
    # Glucose (support both naming conventions)
    fasting_glucose = safe_float(payload.get('fasting_glucose', payload.get('glucose')), 100)
    
    # HbA1c - estimate from glucose if not provided
    hba1c = safe_float(payload.get('hba1c'), (fasting_glucose + 46.7) / 28.7)
    
    # Cholesterol (support both naming conventions)
    total_cholesterol = safe_float(payload.get('total_cholesterol', payload.get('cholesterol')), 200)
    ldl_cholesterol = safe_float(payload.get('ldl_cholesterol'), total_cholesterol * 0.6)
    hdl_cholesterol = safe_float(payload.get('hdl_cholesterol'), 50)
    triglycerides = safe_float(payload.get('triglycerides'), 150)
    
    # Cardiac markers
    max_heart_rate = safe_float(payload.get('max_heart_rate'), 220 - age)
    st_depression = safe_float(payload.get('st_depression'), 0.0)
    
    # Sex encoding: M=0, F=1
    sex = str(payload.get('sex', payload.get('gender', 'M')) or 'M').upper()
    sex_encoded = 1 if sex in ['F', 'FEMALE'] else 0
    
    # Categorical features
    chest_pain_type = safe_int(payload.get('chest_pain_type'), 4)  # 4 = asymptomatic
    resting_ecg = safe_int(payload.get('resting_ecg'), 0)  # 0 = normal
    exercise_angina = safe_int(payload.get('exercise_angina'), 0)
    st_slope = safe_int(payload.get('st_slope'), 2)  # 2 = flat
    
    # Smoking status: 0=never, 1=former, 2=current
    smoking_raw = payload.get('smoking_status', 0)
    if smoking_raw is None:
        smoking_status = 0
    elif isinstance(smoking_raw, str):
        smoking_status = {'never': 0, 'no': 0, 'former': 1, 'yes': 2, 'current': 2}.get(smoking_raw.lower(), 0)
    else:
        smoking_status = safe_int(smoking_raw, 0)
    
    # Medical history (binary)
    family_history_cvd = safe_int(payload.get('family_history_cvd'), 0)
    family_history_diabetes = safe_int(payload.get('family_history_diabetes'), 0)
    prior_hypertension = safe_int(payload.get('prior_hypertension'), 0)
    prior_diabetes = safe_int(payload.get('prior_diabetes'), 0)
    prior_heart_disease = safe_int(payload.get('prior_heart_disease'), 0)
    
    # Current medications (binary)
    on_bp_medication = safe_int(payload.get('on_bp_medication'), 0)
    on_diabetes_medication = safe_int(payload.get('on_diabetes_medication'), 0)
    on_cholesterol_medication = safe_int(payload.get('on_cholesterol_medication'), 0)
    
    # Build feature vector in correct order
    X = np.array([
        age, bmi, systolic_bp, diastolic_bp, heart_rate,
        fasting_glucose, hba1c, total_cholesterol, ldl_cholesterol,
        hdl_cholesterol, triglycerides, max_heart_rate, st_depression,
        sex_encoded, chest_pain_type, resting_ecg, exercise_angina,
        st_slope, smoking_status, family_history_cvd, family_history_diabetes,
        prior_hypertension, prior_diabetes, prior_heart_disease,
        on_bp_medication, on_diabetes_medication, on_cholesterol_medication
    ], dtype=float).reshape(1, -1)
    
    return X, np.array([])


def encode_legacy_features(payload: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy feature encoding for backward compatibility with old form data.
    Maps old fields to new clinical schema as best as possible.
    """
    # Map legacy fields to new schema
    age = int(payload.get('age', 50))
    bmi = float(payload.get('bmi', 25.0))
    heart_rate = int(payload.get('heart_rate', 72))
    bp_sys = int(payload.get('bp_sys', 120))
    bp_dia = int(payload.get('bp_dia', 80))
    cholesterol = int(payload.get('cholesterol', 200))
    glucose = int(payload.get('glucose', 100))
    
    # Encode gender
    gender = str(payload.get('gender', 'male')).lower()
    sex_encoded = 1 if gender == 'female' else 0
    
    # Map smoker status
    smoker = str(payload.get('smoker_status', 'no')).lower()
    smoking_status = 2 if smoker == 'yes' else 0
    
    # Map family history
    family = str(payload.get('family_history', 'none')).lower()
    family_history_cvd = 1 if family == 'heart_disease' else 0
    family_history_diabetes = 1 if family == 'diabetes' else 0
    
    # Map medication use
    medication = str(payload.get('medication_use', 'none')).lower()
    on_bp_medication = 1 if medication in ['lisinopril', 'amlodipine'] else 0
    on_diabetes_medication = 1 if medication in ['metformin', 'insulin'] else 0
    on_cholesterol_medication = 1 if medication == 'atorvastatin' else 0
    
    # Map prior conditions
    prior = str(payload.get('prior_conditions', '')).lower()
    prior_hypertension = 1 if 'hypertension' in prior else 0
    prior_diabetes = 1 if 'diabetes' in prior else 0
    prior_heart_disease = 1 if 'heart' in prior else 0
    
    # Estimate other values
    hba1c = (glucose + 46.7) / 28.7
    max_heart_rate = 220 - age
    
    # Build feature vector
    X = np.array([
        age, bmi, bp_sys, bp_dia, heart_rate,
        glucose, hba1c, cholesterol, cholesterol * 0.6,  # Estimate LDL
        50, 150, max_heart_rate, 0.0,  # HDL, triglycerides, max_hr, st_depression
        sex_encoded, 4, 0, 0,  # chest_pain, resting_ecg, exercise_angina
        2, smoking_status, family_history_cvd, family_history_diabetes,
        prior_hypertension, prior_diabetes, prior_heart_disease,
        on_bp_medication, on_diabetes_medication, on_cholesterol_medication
    ], dtype=float).reshape(1, -1)
    
    return X, np.array([])


def model_path_for_lab(lab_label: str) -> str:
    base = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f'{lab_label}.pkl')


def load_or_init_model(lab_label: str, n_features: int):
    """
    Load existing model with priority:
    1. Latest downloaded global model (best performance)
    2. Lab's local model
    3. Create new model
    """
    # Check for downloaded global models (use the most recent one)
    base = os.path.dirname(__file__)
    global_pattern = os.path.join(base, 'models', f'global_downloaded_{lab_label}_*.pkl')
    global_models = glob.glob(global_pattern)
    
    if global_models:
        # Use the most recent global model
        latest_global = max(global_models, key=os.path.getmtime)
        print(f"Loading global model for {lab_label}: {latest_global}")
        try:
            with open(latest_global, 'rb') as f:
                model = pickle.load(f)
                # Also copy it to the lab's local path for persistence
                local_path = model_path_for_lab(lab_label)
                with open(local_path, 'wb') as lf:
                    pickle.dump(model, lf)
                return model
        except Exception as e:
            print(f"Error loading global model: {e}, falling back to local model")
    
    # Check for lab's local model
    path = model_path_for_lab(lab_label)
    if os.path.exists(path):
        print(f"Loading local model for {lab_label}: {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    # Create new model based on configured type
    print(f"Creating new {MODEL_TYPE} model for {lab_label}")
    if MODEL_TYPE == 'random_forest':
        m = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif MODEL_TYPE == 'gradient_boosting':
        m = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    else:  # Default to logistic regression
        m = LogisticRegression(max_iter=200, random_state=42)
    
    # Initialize for tree-based models (they don't need coef_ initialization)
    if hasattr(m, 'coef_'):
        m.coef_ = np.zeros((4, n_features))
        m.intercept_ = np.zeros((4,))
    m.classes_ = np.array([0, 1, 2, 3])
    return m


def save_model(lab_label: str, model) -> None:
    path = model_path_for_lab(lab_label)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def get_parameters(model):
    """Get model parameters - works for both linear and tree-based models"""
    if hasattr(model, 'coef_'):
        return [model.coef_.astype(np.float64), model.intercept_.astype(np.float64)]
    elif hasattr(model, 'estimators_'):
        # For tree-based models, return feature importances as pseudo-parameters
        if hasattr(model, 'feature_importances_'):
            n_features = len(model.feature_importances_)
            return [model.feature_importances_.astype(np.float64), np.zeros(4)]
    # Return zeros with appropriate dimensions (27 features for new schema)
    return [np.zeros(27), np.zeros(4)]


def set_parameters(model, params) -> None:
    """Set model parameters - works for both linear and tree-based models"""
    coef, intercept = params
    if hasattr(model, 'coef_'):
        model.coef_ = np.array(coef)
        model.intercept_ = np.array(intercept)
    # Tree-based models can't have parameters set directly, they need retraining
    model.classes_ = np.array([0, 1, 2, 3])


def get_model_n_features(model) -> Optional[int]:
    """Return expected number of features for a fitted model (27 or 33)."""
    if hasattr(model, 'n_features_in_') and model.n_features_in_ is not None:
        return int(model.n_features_in_)
    if hasattr(model, 'coef_') and model.coef_ is not None:
        return int(model.coef_.shape[1])
    return None


def ensure_features_for_model(X: np.ndarray, model) -> np.ndarray:
    """
    Ensure X has the same number of features as the model expects.
    Pads with zeros if X has fewer (e.g. 27-feature encoding with 33-feature model).
    """
    n_want = get_model_n_features(model)
    if n_want is None:
        return X
    n_have = X.shape[1] if X.ndim >= 2 else X.shape[0]
    if n_have >= n_want:
        return X[:, :n_want] if X.ndim >= 2 else X[:n_want].reshape(1, -1)
    pad = np.zeros((X.shape[0], n_want - n_have), dtype=X.dtype)
    return np.hstack([X, pad])


def predict_prob(model, X: np.ndarray) -> Tuple[float, str]:
    """Returns (risk_score, disease_type)"""
    X = ensure_features_for_model(X, model)
    probs = model.predict_proba(X)[0]
    max_class = np.argmax(probs)
    risk_score = float(probs[max_class])
    disease_type = DISEASE_TYPES[max_class]
    return risk_score, disease_type


def load_baseline_model() -> Optional[Dict]:
    """
    Load the pre-trained baseline global model with scaler.
    
    Returns:
        Dictionary with 'model', 'scaler', 'features', 'class_names' or None if not found
    """
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, 'models', 'baseline_global_model.pkl')
    
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                bundle = pickle.load(f)
            return bundle
        except Exception as e:
            print(f"Error loading baseline model: {e}")
            return None
    return None


def predict_with_baseline(patient_data: Dict) -> Dict:
    """
    Make prediction using the trained baseline model.
    
    Args:
        patient_data: Patient data dictionary with clinical features
    
    Returns:
        Dictionary with diagnosis, confidence, and probabilities
    """
    bundle = load_baseline_model()
    
    if bundle is None:
        # Fall back to rule-based prediction if no model
        return predict_rule_based(patient_data)
    
    model = bundle['model']
    scaler = bundle.get('scaler')
    
    # Encode features
    X, _ = encode_features(patient_data)
    
    # Scale if scaler is available
    if scaler is not None:
        X = scaler.transform(X)
    
    # Ensure X matches model's expected features (e.g. pad 27 -> 33 if needed)
    X = ensure_features_for_model(X, model)
    
    # Predict
    probs = model.predict_proba(X)[0]
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    
    return {
        'diagnosis': predicted_class,
        'diagnosis_label': DISEASE_TYPES[predicted_class],
        'confidence': confidence,
        'probabilities': {
            'healthy': float(probs[0]),
            'diabetes': float(probs[1]),
            'hypertension': float(probs[2]),
            'heart_disease': float(probs[3])
        }
    }


def predict_rule_based(patient_data: Dict) -> Dict:
    """
    Fallback rule-based prediction when no ML model is available.
    
    Uses clinical thresholds to determine most likely condition.
    """
    # Extract values with defaults
    glucose = float(patient_data.get('fasting_glucose', patient_data.get('glucose', 100)))
    systolic_bp = float(patient_data.get('systolic_bp', patient_data.get('bp_sys', 120)))
    diastolic_bp = float(patient_data.get('diastolic_bp', patient_data.get('bp_dia', 80)))
    cholesterol = float(patient_data.get('total_cholesterol', patient_data.get('cholesterol', 200)))
    hba1c = float(patient_data.get('hba1c', 5.5))
    chest_pain = int(patient_data.get('chest_pain_type', 4))
    st_depression = float(patient_data.get('st_depression', 0))
    exercise_angina = int(patient_data.get('exercise_angina', 0))
    
    # Initialize probabilities
    probs = [0.25, 0.25, 0.25, 0.25]  # Base equal probability
    
    # Diabetes indicators
    if glucose >= 126 or hba1c >= 6.5:
        probs[1] += 0.4
    elif glucose >= 100 or hba1c >= 5.7:
        probs[1] += 0.2
    
    # Hypertension indicators
    if systolic_bp >= 140 or diastolic_bp >= 90:
        probs[2] += 0.4
    elif systolic_bp >= 130 or diastolic_bp >= 85:
        probs[2] += 0.2
    
    # Heart disease indicators
    if chest_pain in [1, 2]:  # Typical or atypical angina
        probs[3] += 0.3
    if st_depression > 1.0:
        probs[3] += 0.2
    if exercise_angina == 1:
        probs[3] += 0.2
    if cholesterol >= 240:
        probs[3] += 0.1
    
    # Healthy indicators (decrease disease probabilities)
    if glucose < 100 and systolic_bp < 120 and cholesterol < 200:
        probs[0] += 0.3
    
    # Normalize probabilities
    total = sum(probs)
    probs = [p / total for p in probs]
    
    predicted_class = int(np.argmax(probs))
    
    return {
        'diagnosis': predicted_class,
        'diagnosis_label': DISEASE_TYPES[predicted_class],
        'confidence': float(probs[predicted_class]),
        'probabilities': {
            'healthy': float(probs[0]),
            'diabetes': float(probs[1]),
            'hypertension': float(probs[2]),
            'heart_disease': float(probs[3])
        }
    }
