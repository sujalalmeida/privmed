import { useState, useEffect } from 'react';
import { Brain, Users, TrendingUp, Database, RefreshCw, Zap, CheckCircle, AlertCircle, Clock, Send } from 'lucide-react';
import PushModelModal from './PushModelModal';
import BroadcastHistory from './BroadcastHistory';

interface LabStatus {
  lab: string;
  last_update: string;
  local_accuracy: number | null;
  num_examples: number | null;
  has_model: boolean;
  ready_for_aggregation: boolean;
}

interface AggregationResult {
  success: boolean;
  modelVersion: number;
  globalAccuracy: number | null;
  num_models_aggregated: number;
  total_samples: number;
  lab_contributions: Array<{
    lab: string;
    samples: number;
    accuracy: number;
    weight: number;
  }>;
  model_type: string;
}

interface AggregationStatus {
  current_global_model: {
    version: number;
    model_type: string | null;
    created_at: string | null;
    num_labs_contributed: number;
    total_samples: number;
  };
  labs: LabStatus[];
  recent_rounds: any[];
  total_labs: number;
  ready_labs: number;
}

export default function ModelAggregation() {
  const [serverUrl, setServerUrl] = useState<string>('http://127.0.0.1:5001');
  const [isAggregating, setIsAggregating] = useState(false);
  const [status, setStatus] = useState<AggregationStatus | null>(null);
  const [lastResult, setLastResult] = useState<AggregationResult | null>(null);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');
  const [showPushModal, setShowPushModal] = useState(false);

  // Load aggregation status
  const loadStatus = async () => {
    try {
      const resp = await fetch(`${serverUrl}/admin/get_aggregation_status`);
      if (resp.ok) {
        const data = await resp.json();
        setStatus(data);
      }
    } catch (error) {
      console.error('Error loading status:', error);
    }
  };

  useEffect(() => {
    loadStatus();
    // Refresh status every 10 seconds
    const interval = setInterval(loadStatus, 10000);
    return () => clearInterval(interval);
  }, [serverUrl]);

  const handleAggregate = async () => {
    setIsAggregating(true);
    setError('');
    setSuccess('');
    setLastResult(null);
    
    try {
      const resp = await fetch(`${serverUrl}/admin/aggregate_models`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!resp.ok) {
        const errorData = await resp.json();
        throw new Error(errorData.error || 'Aggregation failed');
      }
      
      const data: AggregationResult = await resp.json();
      setLastResult(data);
      setSuccess(`Successfully created global model v${data.modelVersion}!`);
      
      // Refresh status
      setTimeout(loadStatus, 1000);
      
    } catch (err: any) {
      setError(err.message || 'Failed to aggregate models');
    } finally {
      setIsAggregating(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card p-5 border-l-3 border-l-primary-500">
          <div className="flex items-center justify-between mb-2">
            <div className="w-9 h-9 bg-primary-50 rounded-lg flex items-center justify-center">
              <Brain className="w-5 h-5 text-primary-500" />
            </div>
            <span className="text-xs font-medium text-neutral-500 uppercase">Global Model</span>
          </div>
          <div className="text-2xl font-semibold text-neutral-900">v{status?.current_global_model.version || 0}</div>
          <div className="text-xs text-neutral-500 mt-1">
            {status?.current_global_model.model_type || 'Not created'}
          </div>
        </div>

        <div className="card p-5 border-l-3 border-l-success-500">
          <div className="flex items-center justify-between mb-2">
            <div className="w-9 h-9 bg-success-50 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-success-500" />
            </div>
            <span className="text-xs font-medium text-neutral-500 uppercase">Accuracy</span>
          </div>
          <div className="text-2xl font-semibold text-neutral-900">
            {lastResult?.globalAccuracy 
              ? `${(lastResult.globalAccuracy * 100).toFixed(1)}%`
              : status?.recent_rounds[0]?.global_accuracy
              ? `${(status.recent_rounds[0].global_accuracy * 100).toFixed(1)}%`
              : 'N/A'}
          </div>
          <div className="text-xs text-neutral-500 mt-1">Last aggregation</div>
        </div>

        <div className="card p-5 border-l-3 border-l-heart-disease">
          <div className="flex items-center justify-between mb-2">
            <div className="w-9 h-9 bg-purple-50 rounded-lg flex items-center justify-center">
              <Users className="w-5 h-5 text-heart-disease" />
            </div>
            <span className="text-xs font-medium text-neutral-500 uppercase">Labs</span>
          </div>
          <div className="text-2xl font-semibold text-neutral-900">{status?.ready_labs || 0}/{status?.total_labs || 0}</div>
          <div className="text-xs text-neutral-500 mt-1">Ready for aggregation</div>
        </div>

        <div className="card p-5 border-l-3 border-l-warning-500">
          <div className="flex items-center justify-between mb-2">
            <div className="w-9 h-9 bg-warning-50 rounded-lg flex items-center justify-center">
              <Database className="w-5 h-5 text-warning-500" />
            </div>
            <span className="text-xs font-medium text-neutral-500 uppercase">Samples</span>
          </div>
          <div className="text-2xl font-semibold text-neutral-900">
            {status?.current_global_model.total_samples || lastResult?.total_samples || 0}
          </div>
          <div className="text-xs text-neutral-500 mt-1">Total training data</div>
        </div>
      </div>

      {/* Aggregation Control Panel */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-base font-semibold text-neutral-900">Model Aggregation Control</h3>
            <p className="text-sm text-neutral-500 mt-1">
              Federated Averaging (FedAvg) — Combine local models into global knowledge
            </p>
          </div>
          <button
            onClick={loadStatus}
            className="btn-ghost !p-2"
            title="Refresh status"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>

        {error && (
          <div className="alert-error mb-4">
            <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {success && (
          <div className="alert-success mb-4">
            <CheckCircle className="w-4 h-4 mr-2 flex-shrink-0" />
            <span>{success}</span>
          </div>
        )}

        <div className="flex items-center gap-4">
          <button
            onClick={handleAggregate}
            disabled={isAggregating || (status?.ready_labs || 0) === 0}
            className="btn-primary"
          >
            {isAggregating ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Aggregating Models...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 mr-2" />
                Aggregate Models Now
              </>
            )}
          </button>

          <button
            onClick={() => setShowPushModal(true)}
            disabled={(status?.current_global_model.version || 0) === 0}
            className="btn-primary bg-success-500 hover:bg-success-600"
          >
            <Send className="w-4 h-4 mr-2" />
            Push to All Labs
          </button>

          <div className="text-sm text-neutral-500">
            <input
              type="text"
              value={serverUrl}
              onChange={(e) => setServerUrl(e.target.value)}
              className="form-input !py-2 !text-sm w-48"
              placeholder="Server URL"
            />
          </div>
        </div>

        {(status?.ready_labs || 0) === 0 && (
          <div className="alert-warning mt-4">
            <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
            No labs are ready for aggregation. Labs must submit patient data and train local models first.
          </div>
        )}
      </div>

      {/* Participating Labs */}
      <div className="card p-6">
        <h3 className="text-base font-semibold text-neutral-900 mb-4 flex items-center">
          <Users className="w-4 h-4 mr-2 text-primary-500" />
          Participating Labs
        </h3>
        
        {!status || status.labs.length === 0 ? (
          <div className="text-center py-12 text-neutral-500">
            <Brain className="w-12 h-12 mx-auto mb-3 text-neutral-300" />
            <p className="text-sm">No lab updates yet. Waiting for labs to submit patient data...</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {status.labs.map((lab) => (
              <div
                key={lab.lab}
                className={`border rounded-lg p-4 transition-all ${
                  lab.ready_for_aggregation
                    ? 'border-success-200 bg-success-50'
                    : 'border-neutral-200 bg-neutral-50'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-medium text-neutral-900 flex items-center text-sm">
                    {lab.ready_for_aggregation ? (
                      <CheckCircle className="w-4 h-4 mr-2 text-success-500" />
                    ) : (
                      <Clock className="w-4 h-4 mr-2 text-neutral-400" />
                    )}
                    {lab.lab}
                  </h4>
                  {lab.ready_for_aggregation && (
                    <span className="badge-success">
                      Ready
                    </span>
                  )}
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between items-center">
                    <span className="text-neutral-500">Local Accuracy:</span>
                    <span className="font-medium text-neutral-900">
                      {lab.local_accuracy ? `${(lab.local_accuracy * 100).toFixed(1)}%` : 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-neutral-500">Training Samples:</span>
                    <span className="font-medium text-neutral-900">
                      {lab.num_examples || 0}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-neutral-500">Last Update:</span>
                    <span className="text-xs text-neutral-500">
                      {new Date(lab.last_update).toLocaleString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Aggregation Results */}
      {lastResult && (
        <div className="card p-6 bg-primary-50 border-primary-200">
          <h3 className="text-base font-semibold text-neutral-900 mb-4 flex items-center">
            <CheckCircle className="w-4 h-4 mr-2 text-success-500" />
            Latest Aggregation Results
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white rounded-lg p-4 border border-neutral-200">
              <h4 className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-3">Performance Metrics</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-neutral-600">Global Model Version:</span>
                  <span className="font-semibold text-primary-500">v{lastResult.modelVersion}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Global Accuracy:</span>
                  <span className="font-semibold text-success-500">
                    {lastResult.globalAccuracy ? `${(lastResult.globalAccuracy * 100).toFixed(2)}%` : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Labs Aggregated:</span>
                  <span className="font-semibold text-primary-500">{lastResult.num_models_aggregated}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Total Samples:</span>
                  <span className="font-semibold text-heart-disease">{lastResult.total_samples}</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border border-neutral-200">
              <h4 className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-3">Lab Contributions</h4>
              <div className="space-y-2">
                {lastResult.lab_contributions.map((contrib) => (
                  <div key={contrib.lab} className="flex items-center justify-between p-2 bg-neutral-50 rounded-md">
                    <div className="flex items-center">
                      <div className="w-2 h-2 bg-primary-500 rounded-full mr-2"></div>
                      <span className="text-sm font-medium text-neutral-900">{contrib.lab}</span>
                    </div>
                    <div className="flex items-center gap-3 text-xs">
                      <span className="text-neutral-500">{contrib.samples} samples</span>
                      <span className="font-medium text-primary-500">
                        {(contrib.weight * 100).toFixed(1)}% weight
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Broadcast History */}
      <BroadcastHistory serverUrl={serverUrl} />

      {/* Push Model Modal */}
      <PushModelModal
        isOpen={showPushModal}
        onClose={() => setShowPushModal(false)}
        serverUrl={serverUrl}
        globalModelVersion={status?.current_global_model.version || 0}
      />
    </div>
  );
}
