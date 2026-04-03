import { useState, useEffect } from 'react';
import { Brain, Users, TrendingUp, Database, RefreshCw, Zap, CheckCircle, AlertCircle, Clock, Send, BarChart3 } from 'lucide-react';
import PushModelModal from './PushModelModal';
import BroadcastHistory from './BroadcastHistory';

interface LabStatus {
  lab: string;
  last_update: string;
  local_accuracy: number | null;
  local_test_accuracy: number | null;
  num_examples: number | null;
  effective_samples?: number | null;
  approval_rate?: number | null;
  has_model: boolean;
  ready_for_aggregation: boolean;
}

interface AggregationResult {
  success: boolean;
  modelVersion: number;
  globalAccuracy: number | null;
  globalValidationAccuracy?: number | null;
  num_models_aggregated: number;
  total_samples: number;
  total_effective_samples?: number;
  lab_contributions: Array<{
    lab: string;
    samples: number;
    effective_samples?: number;
    local_test_accuracy?: number;
    approval_rate?: number;
    doctor_weight_multiplier?: number;
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
    global_validation_accuracy?: number | null;
  };
  labs: LabStatus[];
  recent_rounds: any[];
  total_labs: number;
  ready_labs: number;
}

interface RoundMetric {
  round: number;
  global_accuracy: number;
  global_validation_accuracy?: number;
  created_at: string;
}

export default function ModelAggregation() {
  const [serverUrl, setServerUrl] = useState<string>('http://127.0.0.1:5001');
  const [isAggregating, setIsAggregating] = useState(false);
  const [status, setStatus] = useState<AggregationStatus | null>(null);
  const [lastResult, setLastResult] = useState<AggregationResult | null>(null);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');
  const [showPushModal, setShowPushModal] = useState(false);
  const [roundMetrics, setRoundMetrics] = useState<RoundMetric[]>([]);

  // Load round metrics (accuracy over rounds)
  const loadRoundMetrics = async () => {
    try {
      const resp = await fetch(`${serverUrl}/admin/round_metrics`);
      if (resp.ok) {
        const data = await resp.json();
        setRoundMetrics(data.metrics || []);
      }
    } catch (error) {
      console.error('Error loading round metrics:', error);
    }
  };

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
    loadRoundMetrics();
    // Refresh status every 10 seconds
    const interval = setInterval(() => {
      loadStatus();
      loadRoundMetrics();
    }, 10000);
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
      setSuccess(
        data.globalValidationAccuracy !== null && data.globalValidationAccuracy !== undefined
          ? `Successfully created global weights v${data.modelVersion} and recorded Global Validation Accuracy.`
          : `Successfully aggregated encrypted global weights v${data.modelVersion}. Waiting for lab-side validation reporting.`
      );
      
      // Refresh status and metrics
      setTimeout(() => {
        loadStatus();
        loadRoundMetrics();
      }, 1000);
      
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
            <span className="text-xs font-medium text-neutral-500 uppercase">Validation</span>
          </div>
          <div className="text-2xl font-semibold text-neutral-900">
            {lastResult?.globalValidationAccuracy !== undefined && lastResult?.globalValidationAccuracy !== null
              ? `${(lastResult.globalValidationAccuracy * 100).toFixed(1)}%`
              : status?.current_global_model.global_validation_accuracy !== null && status?.current_global_model.global_validation_accuracy !== undefined
              ? `${(status.current_global_model.global_validation_accuracy * 100).toFixed(1)}%`
              : status?.recent_rounds[0]?.global_validation_accuracy
              ? `${(status.recent_rounds[0].global_validation_accuracy * 100).toFixed(1)}%`
              : status?.recent_rounds[0]?.global_accuracy
              ? `${(status.recent_rounds[0].global_accuracy * 100).toFixed(1)}%`
              : 'N/A'}
          </div>
          <div className="text-xs text-neutral-500 mt-1">Global Validation Accuracy</div>
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
              Federated Averaging (FedAvg) — aggregate lab weight updates, then evaluate the aggregated model on the central validation split
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
        
        {/* Homomorphic Encryption Status */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
          <div className="flex items-center">
            <svg className="w-4 h-4 text-blue-600 mr-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
            <p className="text-xs font-medium text-blue-800">
              Secure Aggregation: All model updates are homomorphically encrypted. Aggregation is performed on encrypted weights.
            </p>
          </div>
          <p className="text-xs text-blue-700 mt-2">
            FedAvg uses each lab&apos;s training sample count only as the weighting coefficient for encrypted weight averaging. The samples themselves are never uploaded during aggregation.
          </p>
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
                Aggregating And Validating...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 mr-2" />
                Aggregate Weight Updates Now
              </>
            )}
          </button>

          <button
            onClick={() => setShowPushModal(true)}
            disabled={(status?.current_global_model.version || 0) === 0}
            className="btn-primary bg-success-500 hover:bg-success-600"
          >
            <Send className="w-4 h-4 mr-2" />
            Distribute Global Weights To Labs
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
                    <span className="text-neutral-500">Local Test Accuracy:</span>
                    <span className="font-medium text-neutral-900">
                      {lab.local_test_accuracy !== null && lab.local_test_accuracy !== undefined ? `${(lab.local_test_accuracy * 100).toFixed(1)}%` : 'N/A'}
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
              <h4 className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-3">Validation Metrics</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-neutral-600">Global Model Version:</span>
                  <span className="font-semibold text-primary-500">v{lastResult.modelVersion}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Global Validation Accuracy:</span>
                  <span className="font-semibold text-success-500">
                    {lastResult.globalValidationAccuracy !== undefined && lastResult.globalValidationAccuracy !== null
                      ? `${(lastResult.globalValidationAccuracy * 100).toFixed(2)}%`
                      : lastResult.globalAccuracy !== null && lastResult.globalAccuracy !== undefined
                      ? `${(lastResult.globalAccuracy * 100).toFixed(2)}%`
                      : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Labs Aggregated:</span>
                  <span className="font-semibold text-primary-500">{lastResult.num_models_aggregated}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Total Train Samples:</span>
                  <span className="font-semibold text-heart-disease">{lastResult.total_samples}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Total Effective Samples:</span>
                  <span className="font-semibold text-primary-500">{lastResult.total_effective_samples ?? lastResult.total_samples}</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border border-neutral-200">
              <h4 className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-3">Lab Contributions</h4>
              <div className="space-y-2">
                {lastResult.lab_contributions.map((contrib) => (
                  <div key={contrib.lab} className="p-3 bg-neutral-50 rounded-md border border-neutral-200">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-primary-500 rounded-full mr-2"></div>
                        <span className="text-sm font-medium text-neutral-900">{contrib.lab}</span>
                      </div>
                      <span className="font-medium text-primary-500">
                        {(contrib.weight * 100).toFixed(1)}% aggregation weight
                      </span>
                    </div>
                    <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="text-neutral-500">Raw train samples:</span>{' '}
                        <span className="font-medium text-neutral-900">{contrib.samples}</span>
                      </div>
                      <div>
                        <span className="text-neutral-500">Effective samples:</span>{' '}
                        <span className="font-medium text-neutral-900">
                          {contrib.effective_samples ?? contrib.samples}
                        </span>
                      </div>
                      <div>
                        <span className="text-neutral-500">Doctor approval:</span>{' '}
                        <span className="font-medium text-neutral-900">
                          {contrib.approval_rate !== undefined && contrib.approval_rate !== null
                            ? `${(contrib.approval_rate * 100).toFixed(1)}%`
                            : 'N/A'}
                        </span>
                      </div>
                      <div>
                        <span className="text-neutral-500">Weight multiplier:</span>{' '}
                        <span className={contrib.doctor_weight_multiplier && contrib.doctor_weight_multiplier > 1 ? 'font-medium text-success-600' : 'font-medium text-neutral-900'}>
                          {(contrib.doctor_weight_multiplier ?? 1).toFixed(1)}x
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Accuracy Over Rounds - Visual Timeline */}
      {roundMetrics.length > 0 && (
        <div className="card p-6">
          <h3 className="text-base font-semibold text-neutral-900 mb-4 flex items-center">
            <BarChart3 className="w-4 h-4 mr-2 text-primary-500" />
            Global Validation Accuracy Over Aggregation Rounds
          </h3>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-neutral-200">
              <thead className="bg-neutral-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Round</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Global Validation Accuracy</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Change</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Date</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Trend</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-neutral-200">
                {roundMetrics.map((metric, index) => {
                  const prevMetric = index < roundMetrics.length - 1 ? roundMetrics[index + 1] : null;
                  const change = prevMetric && metric.global_accuracy && prevMetric.global_accuracy
                    ? ((metric.global_accuracy - prevMetric.global_accuracy) * 100).toFixed(2)
                    : null;
                  const isPositive = change !== null && parseFloat(change) > 0;
                  const isNegative = change !== null && parseFloat(change) < 0;
                  
                  return (
                    <tr key={metric.round} className="hover:bg-neutral-50">
                      <td className="px-4 py-3 whitespace-nowrap">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-800">
                          v{metric.round}
                        </span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <span className="text-sm font-semibold text-neutral-900">
                          {metric.global_accuracy 
                            ? `${(metric.global_accuracy * 100).toFixed(2)}%`
                            : 'N/A'}
                        </span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        {change !== null ? (
                          <span className={`text-sm font-medium ${
                            isPositive ? 'text-success-600' : isNegative ? 'text-error-600' : 'text-neutral-500'
                          }`}>
                            {isPositive ? '+' : ''}{change}%
                          </span>
                        ) : (
                          <span className="text-sm text-neutral-400">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-neutral-500">
                        {new Date(metric.created_at).toLocaleDateString('en-US', {
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit'
                        })}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        {isPositive ? (
                          <TrendingUp className="w-4 h-4 text-success-500" />
                        ) : isNegative ? (
                          <TrendingUp className="w-4 h-4 text-error-500 transform rotate-180" />
                        ) : (
                          <span className="w-4 h-4 inline-block text-center text-neutral-400">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          
          {/* Simple visual accuracy bar chart */}
          <div className="mt-6">
            <h4 className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-3">Accuracy Trend</h4>
            <div className="flex items-end gap-2 h-24">
              {roundMetrics.slice().reverse().map((metric) => {
                const height = metric.global_accuracy ? metric.global_accuracy * 100 : 0;
                return (
                  <div key={metric.round} className="flex flex-col items-center flex-1 max-w-16">
                    <div 
                      className="w-full bg-primary-500 rounded-t transition-all duration-300"
                      style={{ height: `${height}%` }}
                      title={`v${metric.round}: ${height.toFixed(1)}%`}
                    />
                    <span className="text-xs text-neutral-500 mt-1">v{metric.round}</span>
                  </div>
                );
              })}
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
