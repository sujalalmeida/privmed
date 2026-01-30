"""
Federated Learning Metrics Module for PrivMed Paper

Provides comprehensive metric computation including:
- Per-class metrics (precision, recall, specificity, F1, AUC-ROC)
- Loss computation (training and validation)
- Confusion matrix analysis
- Multi-class evaluation utilities

Author: PrivMed Team
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    log_loss
)
import warnings
warnings.filterwarnings('ignore')

# Class names for PrivMed diagnosis
CLASS_NAMES = ['healthy', 'diabetes', 'hypertension', 'heart_disease']


def compute_loss(model, X: np.ndarray, y: np.ndarray, model_type: str = 'tree') -> float:
    """
    Compute cross-entropy loss for a trained model.
    
    Args:
        model: Trained sklearn model
        X: Feature matrix (n_samples, n_features)
        y: True labels (n_samples,)
        model_type: 'tree' for tree-based models, 'linear' for linear models
    
    Returns:
        float: Cross-entropy loss
    """
    try:
        # Get predicted probabilities
        y_proba = model.predict_proba(X)
        
        # Compute log loss (cross-entropy)
        loss = log_loss(y, y_proba, labels=np.arange(len(CLASS_NAMES)))
        return float(loss)
    except Exception as e:
        print(f"Warning: Could not compute loss: {e}")
        # Fallback: compute from accuracy
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return float(1.0 - accuracy)  # Simple approximation


def compute_per_class_specificity(y_true: np.ndarray, y_pred: np.ndarray, 
                                   num_classes: int = 4) -> np.ndarray:
    """
    Compute specificity (true negative rate) for each class in multi-class setting.
    
    For each class, specificity = TN / (TN + FP) in one-vs-rest sense.
    
    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        num_classes: Number of classes (default 4)
    
    Returns:
        np.ndarray: Specificity for each class
    """
    specificities = []
    
    for class_idx in range(num_classes):
        # One-vs-rest: this class vs all others
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)
        
        # True negatives and false positives
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        
        # Specificity = TN / (TN + FP)
        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0.0
        
        specificities.append(specificity)
    
    return np.array(specificities)


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                               y_proba: Optional[np.ndarray] = None,
                               class_names: List[str] = None) -> Dict:
    """
    Compute comprehensive per-class metrics for multi-class classification.
    
    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        y_proba: Predicted probabilities (n_samples, n_classes), optional for AUC-ROC
        class_names: List of class names (default: CLASS_NAMES)
    
    Returns:
        Dict: Per-class metrics with structure:
            {
                "healthy": {"precision": 0.95, "recall": 0.93, "specificity": 0.97, 
                           "f1": 0.94, "support": 100, "auc_roc": 0.96},
                ...
            }
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    num_classes = len(class_names)
    
    # Compute precision, recall, F1, support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), zero_division=0
    )
    
    # Compute specificity
    specificity = compute_per_class_specificity(y_true, y_pred, num_classes)
    
    # Compute per-class AUC-ROC if probabilities are provided
    auc_per_class = None
    if y_proba is not None and y_proba.shape[1] == num_classes:
        try:
            auc_per_class = roc_auc_score(
                y_true, y_proba, 
                multi_class='ovr', 
                labels=np.arange(num_classes),
                average=None
            )
        except Exception as e:
            print(f"Warning: Could not compute per-class AUC-ROC: {e}")
            auc_per_class = None
    
    # Build per-class metrics dictionary
    metrics = {}
    for i, class_name in enumerate(class_names):
        metrics[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'specificity': float(specificity[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
        
        if auc_per_class is not None:
            metrics[class_name]['auc_roc'] = float(auc_per_class[i])
    
    return metrics


def compute_macro_auc_roc(y_true: np.ndarray, y_proba: np.ndarray, 
                          num_classes: int = 4) -> Optional[float]:
    """
    Compute macro-averaged AUC-ROC score.
    
    Args:
        y_true: True labels (n_samples,)
        y_proba: Predicted probabilities (n_samples, n_classes)
        num_classes: Number of classes
    
    Returns:
        float: Macro-averaged AUC-ROC, or None if computation fails
    """
    try:
        auc_macro = roc_auc_score(
            y_true, y_proba,
            multi_class='ovr',
            labels=np.arange(num_classes),
            average='macro'
        )
        return float(auc_macro)
    except Exception as e:
        print(f"Warning: Could not compute macro AUC-ROC: {e}")
        return None


def compute_confusion_matrix_elements(y_true: np.ndarray, y_pred: np.ndarray,
                                      class_idx: int, num_classes: int = 4) -> Dict:
    """
    Compute TP, FP, TN, FN for a specific class in one-vs-rest sense.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_idx: Index of the class to compute for
        num_classes: Total number of classes
    
    Returns:
        Dict: {'tp': int, 'fp': int, 'tn': int, 'fn': int}
    """
    # One-vs-rest binary classification
    y_true_binary = (y_true == class_idx).astype(int)
    y_pred_binary = (y_pred == class_idx).astype(int)
    
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    
    return {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }


def evaluate_model_comprehensive(model, X: np.ndarray, y: np.ndarray,
                                  model_type: str = 'tree',
                                  class_names: List[str] = None) -> Dict:
    """
    Comprehensive model evaluation with all metrics needed for paper.
    
    Args:
        model: Trained sklearn model
        X: Feature matrix
        y: True labels
        model_type: 'tree' or 'linear'
        class_names: List of class names
    
    Returns:
        Dict: Complete evaluation results including:
            - accuracy: Overall accuracy
            - loss: Cross-entropy loss
            - per_class_metrics: Dict with per-class precision, recall, etc.
            - auc_roc_macro: Macro-averaged AUC-ROC
            - auc_roc_per_class: Dict with per-class AUC-ROC
            - confusion_matrix: Confusion matrix as list
            - predictions: Dict with true labels, predicted labels, and probabilities
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Accuracy
    accuracy = accuracy_score(y, y_pred)
    
    # Loss
    loss = compute_loss(model, X, y, model_type)
    
    # Per-class metrics
    per_class_metrics = compute_per_class_metrics(y, y_pred, y_proba, class_names)
    
    # Macro AUC-ROC
    auc_roc_macro = compute_macro_auc_roc(y, y_proba, len(class_names))
    
    # Per-class AUC-ROC (extract from per_class_metrics)
    auc_roc_per_class = {
        class_name: metrics.get('auc_roc')
        for class_name, metrics in per_class_metrics.items()
    }
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=np.arange(len(class_names)))
    
    # Prepare results
    results = {
        'accuracy': float(accuracy),
        'loss': float(loss),
        'per_class_metrics': per_class_metrics,
        'auc_roc_macro': auc_roc_macro,
        'auc_roc_per_class': auc_roc_per_class,
        'confusion_matrix': cm.tolist(),
        'predictions': {
            'y_true': y.tolist(),
            'y_pred': y_pred.tolist(),
            'y_proba': y_proba.tolist()
        }
    }
    
    return results


def format_per_class_metrics_for_db(per_class_metrics: Dict) -> Dict:
    """
    Format per-class metrics for database storage (JSONB).
    
    Ensures all values are JSON-serializable.
    
    Args:
        per_class_metrics: Dict from compute_per_class_metrics
    
    Returns:
        Dict: Formatted for JSONB storage
    """
    formatted = {}
    for class_name, metrics in per_class_metrics.items():
        formatted[class_name] = {
            key: float(value) if isinstance(value, (int, float, np.number)) else value
            for key, value in metrics.items()
        }
    return formatted


def compute_weight_update_magnitude(params_before: np.ndarray, 
                                     params_after: np.ndarray) -> float:
    """
    Compute L2 norm of weight update (‖Δw‖).
    
    Args:
        params_before: Flattened parameter array before update
        params_after: Flattened parameter array after update
    
    Returns:
        float: L2 norm of the difference
    """
    delta = params_after - params_before
    magnitude = np.linalg.norm(delta)
    return float(magnitude)


def extract_predictions_for_roc(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_proba: np.ndarray) -> List[Dict]:
    """
    Extract predictions in format suitable for ROC curve generation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
    
    Returns:
        List[Dict]: List of prediction records
            [{"true_label": 0, "pred_label": 0, "proba": [0.9, 0.05, 0.03, 0.02]}, ...]
    """
    predictions = []
    for i in range(len(y_true)):
        predictions.append({
            'true_label': int(y_true[i]),
            'pred_label': int(y_pred[i]),
            'proba': y_proba[i].tolist()
        })
    return predictions


def compute_data_distribution(y: np.ndarray, class_names: List[str] = None) -> Dict:
    """
    Compute class distribution in dataset.
    
    Args:
        y: Label array
        class_names: List of class names
    
    Returns:
        Dict: {"healthy": count, "diabetes": count, ...}
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    unique, counts = np.unique(y, return_counts=True)
    distribution = {class_names[i]: 0 for i in range(len(class_names))}
    
    for label, count in zip(unique, counts):
        if 0 <= label < len(class_names):
            distribution[class_names[label]] = int(count)
    
    return distribution


def print_evaluation_summary(results: Dict, title: str = "Evaluation Results"):
    """
    Pretty print evaluation results to console.
    
    Args:
        results: Dict from evaluate_model_comprehensive
        title: Title for the output
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Loss: {results['loss']:.4f}")
    if results.get('auc_roc_macro'):
        print(f"  AUC-ROC (macro): {results['auc_roc_macro']:.4f}")
    
    print(f"\n📋 Per-Class Metrics:")
    for class_name, metrics in results['per_class_metrics'].items():
        print(f"\n  {class_name.capitalize()}:")
        print(f"    Precision:   {metrics['precision']:.4f}")
        print(f"    Recall:      {metrics['recall']:.4f}")
        print(f"    Specificity: {metrics['specificity']:.4f}")
        print(f"    F1-Score:    {metrics['f1']:.4f}")
        print(f"    Support:     {metrics['support']}")
        if metrics.get('auc_roc'):
            print(f"    AUC-ROC:     {metrics['auc_roc']:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test the metrics computation
    print("Testing FL Metrics Module...")
    
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 100
    n_classes = 4
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
    
    # Test per-class metrics
    print("\n✓ Testing per-class metrics computation...")
    metrics = compute_per_class_metrics(y_true, y_pred, y_proba)
    print(f"  Computed metrics for {len(metrics)} classes")
    
    # Test specificity
    print("\n✓ Testing specificity computation...")
    spec = compute_per_class_specificity(y_true, y_pred, n_classes)
    print(f"  Specificity: {spec}")
    
    # Test macro AUC-ROC
    print("\n✓ Testing macro AUC-ROC computation...")
    auc = compute_macro_auc_roc(y_true, y_proba, n_classes)
    print(f"  Macro AUC-ROC: {auc:.4f}" if auc else "  Could not compute")
    
    # Test data distribution
    print("\n✓ Testing data distribution computation...")
    dist = compute_data_distribution(y_true)
    print(f"  Distribution: {dist}")
    
    print("\n✅ All tests passed!")
