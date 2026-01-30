"""
Federated Learning Pipeline Logging Module

Provides functions to log comprehensive FL metrics during training and aggregation.
Integrates with app.py to store per-round, per-lab, and global metrics.

Author: MedSafe Team
"""

import json
import time
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from fl_metrics import (
    evaluate_model_comprehensive,
    format_per_class_metrics_for_db,
    compute_loss,
    compute_data_distribution,
    compute_weight_update_magnitude,
    extract_predictions_for_roc
)


class FLMetricsLogger:
    """
    Centralized logging for federated learning experiments.
    
    Handles logging to Supabase tables:
    - fl_experiment_log
    - fl_round_detailed_metrics
    - fl_per_class_performance
    - fl_lab_data_distribution
    - fl_model_predictions
    """
    
    def __init__(self, supabase_client, experiment_id: Optional[str] = None):
        """
        Initialize metrics logger.
        
        Args:
            supabase_client: Supabase client instance
            experiment_id: Unique experiment identifier (auto-generated if None)
        """
        self.sb = supabase_client
        self.experiment_id = experiment_id or f"fl_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_round = 0
        
        print(f"📊 FLMetricsLogger initialized for experiment: {self.experiment_id}")
    
    def log_lab_update(self, round_num: int, lab_label: str, 
                       local_accuracy: float, num_examples: int,
                       grad_norm: float, model=None, X_train=None, y_train=None,
                       X_val=None, y_val=None, training_time: float = None):
        """
        Log metrics when a lab sends a model update.
        
        Args:
            round_num: Current federated round
            lab_label: Lab identifier (e.g., 'lab_A')
            local_accuracy: Local model accuracy on test set
            num_examples: Number of training examples
            grad_norm: Gradient norm or weight update magnitude
            model: Trained model (optional, for computing loss)
            X_train: Training features (optional, for computing loss)
            y_train: Training labels (optional, for computing loss)
            X_val: Validation features (optional, for computing val loss)
            y_val: Validation labels (optional, for computing val loss)
            training_time: Training time in seconds (optional)
        """
        try:
            # Compute local training loss if data provided
            local_train_loss = None
            local_val_loss = None
            
            if model is not None and X_train is not None and y_train is not None:
                try:
                    local_train_loss = compute_loss(model, X_train, y_train, model_type='tree')
                except Exception as e:
                    print(f"Warning: Could not compute training loss for {lab_label}: {e}")
            
            if model is not None and X_val is not None and y_val is not None:
                try:
                    local_val_loss = compute_loss(model, X_val, y_val, model_type='tree')
                except Exception as e:
                    print(f"Warning: Could not compute validation loss for {lab_label}: {e}")
            
            # Log to fl_round_detailed_metrics
            record = {
                'experiment_id': self.experiment_id,
                'round': round_num,
                'lab_label': lab_label,
                'is_global': False,
                'local_accuracy': local_accuracy,
                'local_train_loss': local_train_loss,
                'local_val_loss': local_val_loss,
                'num_examples': num_examples,
                'grad_norm': grad_norm,
                'weight_update_magnitude': grad_norm,  # Same as grad_norm for now
                'training_time_seconds': training_time
            }
            
            self.sb.table('fl_round_detailed_metrics').insert(record).execute()
            print(f"  ✓ Logged lab update: {lab_label} (round {round_num})")
            
        except Exception as e:
            print(f"Error logging lab update for {lab_label}: {e}")
    
    def log_global_aggregation(self, round_num: int, global_model, 
                               X_test, y_test, X_train=None, y_train=None,
                               aggregation_method: str = 'FedAvg',
                               aggregated_grad_norm: float = 0.0,
                               participating_labs: List[str] = None,
                               store_predictions: bool = True):
        """
        Log metrics after global model aggregation.
        
        Args:
            round_num: Current federated round
            global_model: Aggregated global model
            X_test: Test features
            y_test: Test labels
            X_train: Training features (optional, for computing training loss)
            y_train: Training labels (optional, for computing training loss)
            aggregation_method: Method used for aggregation (default: 'FedAvg')
            aggregated_grad_norm: Aggregated gradient norm
            participating_labs: List of labs that participated
            store_predictions: Whether to store raw predictions for ROC curves
        """
        try:
            print(f"\n📊 Logging global aggregation for round {round_num}...")
            
            # Comprehensive evaluation on test set
            evaluation = evaluate_model_comprehensive(
                global_model, X_test, y_test,
                model_type='tree',
                class_names=['healthy', 'diabetes', 'hypertension', 'heart_disease']
            )
            
            global_accuracy = evaluation['accuracy']
            global_val_loss = evaluation['loss']
            
            # Compute training loss if training data provided
            global_train_loss = None
            if X_train is not None and y_train is not None:
                try:
                    global_train_loss = compute_loss(global_model, X_train, y_train, model_type='tree')
                except Exception as e:
                    print(f"Warning: Could not compute global training loss: {e}")
            
            # Log to fl_round_detailed_metrics (global row)
            global_record = {
                'experiment_id': self.experiment_id,
                'round': round_num,
                'lab_label': None,
                'is_global': True,
                'global_accuracy': global_accuracy,
                'global_train_loss': global_train_loss,
                'global_val_loss': global_val_loss,
                'num_examples': len(y_test),
                'grad_norm': aggregated_grad_norm,
                'aggregation_method': aggregation_method
            }
            
            self.sb.table('fl_round_detailed_metrics').insert(global_record).execute()
            print(f"  ✓ Logged global metrics (accuracy: {global_accuracy:.4f})")
            
            # Log per-class performance
            for class_name, metrics in evaluation['per_class_metrics'].items():
                class_record = {
                    'experiment_id': self.experiment_id,
                    'round': round_num,
                    'model_type': 'federated',
                    'lab_label': None,
                    'class_name': class_name,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'specificity': metrics['specificity'],
                    'f1_score': metrics['f1'],
                    'support': metrics['support'],
                    'auc_roc': metrics.get('auc_roc')
                }
                self.sb.table('fl_per_class_performance').insert(class_record).execute()
            
            print(f"  ✓ Logged per-class metrics for {len(evaluation['per_class_metrics'])} classes")
            
            # Store predictions for ROC curves (optional, can be large)
            if store_predictions:
                predictions_data = extract_predictions_for_roc(
                    np.array(evaluation['predictions']['y_true']),
                    np.array(evaluation['predictions']['y_pred']),
                    np.array(evaluation['predictions']['y_proba'])
                )
                
                # Limit to reasonable size (e.g., sample if too large)
                if len(predictions_data) > 10000:
                    import random
                    predictions_data = random.sample(predictions_data, 10000)
                    print(f"  ⚠️  Sampled predictions to 10000 for storage")
                
                predictions_record = {
                    'experiment_id': self.experiment_id,
                    'round': round_num,
                    'model_type': 'federated',
                    'predictions': predictions_data,
                    'total_predictions': len(predictions_data),
                    'dataset_type': 'test'
                }
                
                self.sb.table('fl_model_predictions').insert(predictions_record).execute()
                print(f"  ✓ Logged {len(predictions_data)} predictions")
            
            # Update current round
            self.current_round = round_num
            
            return evaluation
            
        except Exception as e:
            print(f"Error logging global aggregation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def log_lab_data_distribution(self, round_num: int, lab_label: str,
                                   y_train: np.ndarray):
        """
        Log data distribution for a lab.
        
        Args:
            round_num: Current federated round
            lab_label: Lab identifier
            y_train: Training labels for the lab
        """
        try:
            distribution = compute_data_distribution(y_train)
            
            record = {
                'experiment_id': self.experiment_id,
                'round': round_num,
                'lab_label': lab_label,
                'total_samples': len(y_train),
                'samples_per_class': distribution
            }
            
            self.sb.table('fl_lab_data_distribution').insert(record).execute()
            print(f"  ✓ Logged data distribution for {lab_label}")
            
        except Exception as e:
            print(f"Error logging data distribution for {lab_label}: {e}")
    
    def finalize_experiment(self, centralized_accuracy: Optional[float] = None,
                           federated_accuracy: Optional[float] = None,
                           federated_he_accuracy: Optional[float] = None,
                           total_rounds: Optional[int] = None,
                           num_clients: Optional[int] = None,
                           notes: Optional[str] = None):
        """
        Finalize experiment by logging summary to fl_experiment_log.
        
        Args:
            centralized_accuracy: Centralized baseline accuracy
            federated_accuracy: Final federated model accuracy
            federated_he_accuracy: Federated + HE accuracy (if applicable)
            total_rounds: Total number of FL rounds
            num_clients: Number of participating clients
            notes: Additional notes about the experiment
        """
        try:
            # Get final round metrics if not provided
            if federated_accuracy is None:
                final_metrics = self.sb.table('fl_round_detailed_metrics') \
                    .select('global_accuracy, global_val_loss, per_class_metrics') \
                    .eq('experiment_id', self.experiment_id) \
                    .eq('is_global', True) \
                    .order('round', desc=True) \
                    .limit(1) \
                    .execute()
                
                if final_metrics.data:
                    federated_accuracy = final_metrics.data[0].get('global_accuracy')
            
            # Get centralized baseline if not provided
            if centralized_accuracy is None:
                baseline = self.sb.table('fl_centralized_baselines') \
                    .select('accuracy') \
                    .eq('experiment_id', self.experiment_id) \
                    .limit(1) \
                    .execute()
                
                if baseline.data:
                    centralized_accuracy = baseline.data[0].get('accuracy')
            
            # Get per-class metrics from final round
            per_class_perf = self.sb.table('fl_per_class_performance') \
                .select('*') \
                .eq('experiment_id', self.experiment_id) \
                .eq('model_type', 'federated') \
                .eq('round', self.current_round) \
                .is_('lab_label', 'null') \
                .execute()
            
            per_class_metrics = {}
            auc_roc_per_class = {}
            
            for row in per_class_perf.data:
                class_name = row['class_name']
                per_class_metrics[class_name] = {
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'specificity': row['specificity'],
                    'f1': row['f1_score'],
                    'support': row['support']
                }
                if row.get('auc_roc'):
                    auc_roc_per_class[class_name] = row['auc_roc']
            
            # Calculate macro AUC-ROC
            auc_roc_macro = None
            if auc_roc_per_class:
                auc_roc_macro = np.mean(list(auc_roc_per_class.values()))
            
            # Create experiment log record
            experiment_record = {
                'experiment_id': self.experiment_id,
                'experiment_name': f"FL Experiment {self.experiment_id}",
                'centralized_accuracy': centralized_accuracy,
                'federated_accuracy': federated_accuracy,
                'federated_he_accuracy': federated_he_accuracy,
                'per_class_metrics': per_class_metrics if per_class_metrics else None,
                'auc_roc_macro': auc_roc_macro,
                'auc_roc_per_class': auc_roc_per_class if auc_roc_per_class else None,
                'total_rounds': total_rounds or self.current_round,
                'num_clients': num_clients,
                'model_type': 'gradient_boosting',
                'random_seed': 42,
                'notes': notes
            }
            
            self.sb.table('fl_experiment_log').upsert(
                experiment_record,
                on_conflict='experiment_id'
            ).execute()
            
            print(f"\n✅ Experiment finalized: {self.experiment_id}")
            print(f"   Centralized: {centralized_accuracy:.4f}" if centralized_accuracy else "   Centralized: N/A")
            print(f"   Federated:   {federated_accuracy:.4f}" if federated_accuracy else "   Federated: N/A")
            print(f"   Total Rounds: {total_rounds or self.current_round}")
            
        except Exception as e:
            print(f"Error finalizing experiment: {e}")
            import traceback
            traceback.print_exc()


def create_metrics_logger(supabase_client, experiment_id: Optional[str] = None) -> FLMetricsLogger:
    """
    Factory function to create a metrics logger.
    
    Args:
        supabase_client: Supabase client instance
        experiment_id: Optional experiment ID
    
    Returns:
        FLMetricsLogger instance
    """
    return FLMetricsLogger(supabase_client, experiment_id)


# Helper functions for backward compatibility with app.py

def log_client_update_metrics(sb, experiment_id: str, round_num: int,
                               lab_label: str, local_accuracy: float,
                               num_examples: int, grad_norm: float):
    """
    Quick helper to log client update metrics.
    
    Compatible with existing app.py code.
    """
    try:
        record = {
            'experiment_id': experiment_id,
            'round': round_num,
            'lab_label': lab_label,
            'is_global': False,
            'local_accuracy': local_accuracy,
            'num_examples': num_examples,
            'grad_norm': grad_norm,
            'weight_update_magnitude': grad_norm
        }
        sb.table('fl_round_detailed_metrics').insert(record).execute()
    except Exception as e:
        print(f"Warning: Could not log client update metrics: {e}")


def log_global_round_metrics(sb, experiment_id: str, round_num: int,
                             global_accuracy: float, global_loss: float = None,
                             aggregated_grad_norm: float = 0.0):
    """
    Quick helper to log global round metrics.
    
    Compatible with existing app.py code.
    """
    try:
        record = {
            'experiment_id': experiment_id,
            'round': round_num,
            'is_global': True,
            'global_accuracy': global_accuracy,
            'global_val_loss': global_loss,
            'grad_norm': aggregated_grad_norm
        }
        sb.table('fl_round_detailed_metrics').insert(record).execute()
    except Exception as e:
        print(f"Warning: Could not log global round metrics: {e}")


if __name__ == "__main__":
    print("FL Pipeline Logging Module")
    print("=" * 60)
    print("This module provides comprehensive logging for FL experiments.")
    print("\nUsage:")
    print("  from fl_logging import FLMetricsLogger, create_metrics_logger")
    print("  logger = create_metrics_logger(supabase_client, 'exp_001')")
    print("  logger.log_lab_update(...)")
    print("  logger.log_global_aggregation(...)")
    print("  logger.finalize_experiment(...)")
