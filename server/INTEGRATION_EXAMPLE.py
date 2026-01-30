"""
Quick Integration Example for FL Metrics Logging in app.py

This file shows how to integrate comprehensive metrics logging into your
existing federated learning pipeline in app.py.

Copy and adapt these snippets into your aggregation and training endpoints.
"""

# ============================================================================
# STEP 1: Add imports at the top of app.py
# ============================================================================

from fl_logging import FLMetricsLogger, log_client_update_metrics, log_global_round_metrics
from fl_metrics import evaluate_model_comprehensive, compute_loss

# ============================================================================
# STEP 2: Initialize logger (once per experiment)
# ============================================================================

# Option A: Global variable (simple but not ideal for concurrent experiments)
_metrics_logger = None
_current_experiment_id = None

def get_or_create_metrics_logger(experiment_id=None):
    """Get or create a metrics logger for the current experiment."""
    global _metrics_logger, _current_experiment_id
    
    if experiment_id is None:
        from datetime import datetime
        experiment_id = f"fl_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if _metrics_logger is None or _current_experiment_id != experiment_id:
        _metrics_logger = FLMetricsLogger(sb(), experiment_id)
        _current_experiment_id = experiment_id
    
    return _metrics_logger

# ============================================================================
# STEP 3: In your lab training endpoint (e.g., /train_local_model)
# ============================================================================

@app.route('/train_local_model', methods=['POST'])
def train_local_model():
    """Train a local model at a lab with comprehensive metrics logging."""
    data = request.json
    lab_label = data.get('lab_label', 'lab_unknown')
    
    # ... existing training code ...
    
    # Evaluate on test set
    local_accuracy = evaluate_model_on_test_set(model, model_type='tree')
    
    # Compute gradient norm (if not already computed)
    # grad_norm = np.linalg.norm(get_parameters(model))
    
    # Get metrics logger
    logger = get_or_create_metrics_logger()
    
    # Log lab update with comprehensive metrics
    logger.log_lab_update(
        round_num=current_round,
        lab_label=lab_label,
        local_accuracy=local_accuracy,
        num_examples=len(X_train),
        grad_norm=grad_norm,
        model=model,              # Optional: for loss computation
        X_train=X_train,          # Optional: for training loss
        y_train=y_train,          # Optional: for training loss
        X_val=X_test,             # Optional: use test set as validation
        y_val=y_test,             # Optional
        training_time=training_time
    )
    
    # Log lab data distribution (do this once per lab, not every round)
    # logger.log_lab_data_distribution(
    #     round_num=current_round,
    #     lab_label=lab_label,
    #     y_train=y_train
    # )
    
    return jsonify({'status': 'success', 'accuracy': local_accuracy})

# ============================================================================
# STEP 4: In your aggregation endpoint (e.g., /aggregate_models)
# ============================================================================

@app.route('/aggregate_models', methods=['POST'])
def aggregate_models():
    """Aggregate lab models into global model with comprehensive logging."""
    data = request.json
    
    # ... existing aggregation code ...
    # Assume you have: global_model, current_round, participating_labs
    
    # Load test dataset for evaluation
    X_test, y_test = load_test_dataset()
    
    # Load training dataset for global training loss (optional)
    X_train, y_train = load_training_dataset()
    
    # Get metrics logger
    logger = get_or_create_metrics_logger()
    
    # Log global aggregation with comprehensive evaluation
    evaluation = logger.log_global_aggregation(
        round_num=current_round,
        global_model=global_model,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,              # Optional: for global training loss
        y_train=y_train,              # Optional
        aggregation_method='FedAvg',
        aggregated_grad_norm=aggregated_grad_norm,
        participating_labs=participating_labs,
        store_predictions=True        # Store predictions for ROC curves
    )
    
    # Use evaluation results
    global_accuracy = evaluation['accuracy'] if evaluation else None
    
    return jsonify({
        'status': 'success',
        'global_accuracy': global_accuracy,
        'round': current_round
    })

# ============================================================================
# STEP 5: Finalize experiment when FL rounds complete
# ============================================================================

@app.route('/finalize_experiment', methods=['POST'])
def finalize_experiment():
    """Finalize the current FL experiment."""
    data = request.json
    
    # Get centralized baseline accuracy (from database or file)
    baseline_result = sb().table('fl_centralized_baselines') \
        .select('accuracy') \
        .order('created_at', desc=True) \
        .limit(1) \
        .execute()
    
    centralized_accuracy = baseline_result.data[0]['accuracy'] if baseline_result.data else None
    
    # Get final federated accuracy (from last round)
    final_metrics = sb().table('fl_round_detailed_metrics') \
        .select('global_accuracy') \
        .eq('is_global', True) \
        .order('round', desc=True) \
        .limit(1) \
        .execute()
    
    federated_accuracy = final_metrics.data[0]['global_accuracy'] if final_metrics.data else None
    
    # Get logger
    logger = get_or_create_metrics_logger()
    
    # Finalize experiment
    logger.finalize_experiment(
        centralized_accuracy=centralized_accuracy,
        federated_accuracy=federated_accuracy,
        federated_he_accuracy=None,  # Not implemented yet
        total_rounds=data.get('total_rounds'),
        num_clients=data.get('num_clients'),
        notes=data.get('notes', 'FL experiment completed')
    )
    
    return jsonify({'status': 'success', 'experiment_id': logger.experiment_id})

# ============================================================================
# STEP 6: Alternative - Minimal integration without full logger
# ============================================================================

# If you prefer not to use the full FLMetricsLogger, use these helpers:

# When a lab sends an update:
def log_lab_update_simple(round_num, lab_label, local_accuracy, num_examples, grad_norm):
    """Simple helper to log lab update."""
    experiment_id = "your_experiment_id"  # Get from somewhere
    log_client_update_metrics(
        sb(), experiment_id, round_num,
        lab_label, local_accuracy, num_examples, grad_norm
    )

# When aggregating:
def log_aggregation_simple(round_num, global_accuracy, global_loss):
    """Simple helper to log global aggregation."""
    experiment_id = "your_experiment_id"  # Get from somewhere
    log_global_round_metrics(
        sb(), experiment_id, round_num,
        global_accuracy, global_loss
    )

# ============================================================================
# STEP 7: Example - Full integration in existing aggregate endpoint
# ============================================================================

@app.route('/api/fl/aggregate', methods=['POST'])
def fl_aggregate_full_example():
    """
    Full example of aggregate endpoint with metrics logging.
    
    Assumes you have:
    - Client updates in fl_client_updates table
    - Test dataset available via load_test_dataset()
    """
    try:
        # Get client updates
        updates = sb().table('fl_client_updates') \
            .select('*') \
            .not_.is_('storage_path', 'null') \
            .order('created_at', desc=True) \
            .execute()
        
        if not updates.data:
            return jsonify({'error': 'no updates to aggregate'}), 400
        
        # Load and aggregate models (existing code)
        models_data = []
        for update in updates.data:
            # Load model from storage
            model = load_model_from_storage(update['storage_path'])
            models_data.append({
                'lab': update['client_label'],
                'model': model,
                'num_examples': update['num_examples'],
                'accuracy': update['local_accuracy']
            })
        
        # FedAvg aggregation
        global_model = fedavg_aggregate(models_data)
        
        # Determine round number
        current_round = get_next_round_number()
        
        # Load test dataset
        X_test, y_test = load_test_dataset()
        
        if X_test is None:
            return jsonify({'error': 'test dataset not available'}), 500
        
        # METRICS LOGGING STARTS HERE
        # ============================
        
        # Initialize logger
        logger = get_or_create_metrics_logger("my_experiment_001")
        
        # Log lab updates (if not already logged in training)
        for data in models_data:
            logger.log_lab_update(
                round_num=current_round,
                lab_label=data['lab'],
                local_accuracy=data['accuracy'],
                num_examples=data['num_examples'],
                grad_norm=0.0  # Or compute from model parameters
            )
        
        # Log global aggregation with comprehensive evaluation
        evaluation = logger.log_global_aggregation(
            round_num=current_round,
            global_model=global_model,
            X_test=X_test,
            y_test=y_test,
            aggregation_method='FedAvg',
            participating_labs=[d['lab'] for d in models_data],
            store_predictions=(current_round % 5 == 0)  # Store every 5 rounds
        )
        
        # METRICS LOGGING ENDS HERE
        # =========================
        
        # Save global model (existing code)
        save_global_model(global_model, current_round)
        
        # Return results
        return jsonify({
            'status': 'success',
            'round': current_round,
            'global_accuracy': evaluation['accuracy'] if evaluation else None,
            'participating_labs': len(models_data)
        })
        
    except Exception as e:
        print(f"Error in aggregation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================================
# NOTES
# ============================================================================

"""
Key Points:

1. Import fl_logging and fl_metrics at the top of app.py

2. Initialize FLMetricsLogger once per experiment:
   - Store in global variable, or
   - Pass through request context, or
   - Create factory function

3. Log lab updates when labs train:
   - Call logger.log_lab_update() with model, data, and metrics
   - Optionally log data distribution once per lab

4. Log global aggregation after FedAvg:
   - Call logger.log_global_aggregation() with global model and test data
   - This computes per-class metrics, loss, AUC-ROC automatically
   - Optionally store predictions for ROC curves

5. Finalize experiment when complete:
   - Call logger.finalize_experiment() to create summary record
   - Links centralized baseline with federated results

6. Export metrics:
   - Run python scripts/export_paper_metrics.py --all
   - Generates CSV/JSON files ready for plotting

Benefits:
- Automatic per-class metrics computation
- Loss tracking (training and validation)
- AUC-ROC and specificity
- Raw predictions for custom ROC curves
- Data imbalance tracking
- Ready for paper figures with minimal post-processing
"""
