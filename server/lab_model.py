import os
import pickle
from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# Use Gradient Boosting for better performance (you can switch to RandomForest if needed)
MODEL_TYPE = 'gradient_boosting'  # Options: 'logistic', 'random_forest', 'gradient_boosting'

CATEGORICAL_MAPS = {
    'gender': ['male', 'female', 'other'],
    'blood_type': ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-'],
    'smoker_status': ['yes', 'no'],
    'family_history': ['none', 'diabetes', 'hypertension', 'heart_disease'],
    'medication_use': ['none', 'metformin', 'insulin', 'lisinopril', 'amlodipine', 'atorvastatin']
}

# Disease types: 0=healthy, 1=diabetes, 2=hypertension, 3=heart_disease
DISEASE_TYPES = ['healthy', 'diabetes', 'hypertension', 'heart_disease']


def encode_features(payload: Dict) -> Tuple[np.ndarray, np.ndarray]:
    # X vector order: age, discomfort_level, symptom_duration, bmi, heart_rate, bp_sys, bp_dia, cholesterol, glucose,
    # gender one-hot (3), blood_type one-hot (8), smoker one-hot (2), family_history one-hot (4), medication one-hot (6), prior_conditions_len
    age = int(payload['age'])
    discomfort = int(payload['discomfort_level'])
    duration = int(payload['symptom_duration'])
    bmi = float(payload.get('bmi', 25.0))
    heart_rate = int(payload.get('heart_rate', 70))
    bp_sys = int(payload.get('bp_sys', 120))
    bp_dia = int(payload.get('bp_dia', 80))
    cholesterol = int(payload.get('cholesterol', 200))
    glucose = int(payload.get('glucose', 100))
    
    gender = str(payload['gender']).lower()
    blood = str(payload['blood_type']).upper()
    smoker = str(payload.get('smoker_status', 'no')).lower()
    family = str(payload.get('family_history', 'none')).lower()
    medication = str(payload.get('medication_use', 'none')).lower()
    prior = str(payload.get('prior_conditions') or '')
    prior_len = min(len(prior), 50)

    gender_oh = [1 if gender == g else 0 for g in CATEGORICAL_MAPS['gender']]
    blood_oh = [1 if blood == b else 0 for b in CATEGORICAL_MAPS['blood_type']]
    smoker_oh = [1 if smoker == s else 0 for s in CATEGORICAL_MAPS['smoker_status']]
    family_oh = [1 if family == f else 0 for f in CATEGORICAL_MAPS['family_history']]
    medication_oh = [1 if medication == m else 0 for m in CATEGORICAL_MAPS['medication_use']]

    X = np.array([
        age, discomfort, duration, bmi, heart_rate, bp_sys, bp_dia, cholesterol, glucose,
        *gender_oh, *blood_oh, *smoker_oh, *family_oh, *medication_oh, prior_len
    ], dtype=float).reshape(1, -1)
    return X, np.array([])


def model_path_for_lab(lab_label: str) -> str:
    base = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f'{lab_label}.pkl')


def load_or_init_model(lab_label: str, n_features: int):
    """Load existing model or create a new one based on MODEL_TYPE"""
    path = model_path_for_lab(lab_label)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    # Create model based on configured type
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
            return [model.feature_importances_.astype(np.float64), np.zeros(4)]
    return [np.zeros((4, 33)), np.zeros(4)]


def set_parameters(model, params) -> None:
    """Set model parameters - works for both linear and tree-based models"""
    coef, intercept = params
    if hasattr(model, 'coef_'):
        model.coef_ = np.array(coef)
        model.intercept_ = np.array(intercept)
    # Tree-based models can't have parameters set directly, they need retraining
    model.classes_ = np.array([0, 1, 2, 3])


def predict_prob(model, X: np.ndarray) -> Tuple[float, str]:
    """Returns (risk_score, disease_type)"""
    probs = model.predict_proba(X)[0]
    max_class = np.argmax(probs)
    risk_score = float(probs[max_class])
    disease_type = DISEASE_TYPES[max_class]
    return risk_score, disease_type
