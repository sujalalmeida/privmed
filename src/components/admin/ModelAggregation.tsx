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
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg shadow-lg p-6 text-white">
          <div className="flex items-center justify-between mb-2">
            <Brain className="w-8 h-8 opacity-80" />
            <span className="text-sm font-medium opacity-90">Global Model</span>
          </div>
          <div className="text-3xl font-bold">v{status?.current_global_model.version || 0}</div>
          <div className="text-xs opacity-75 mt-1">
            {status?.current_global_model.model_type || 'Not created'}
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg shadow-lg p-6 text-white">
          <div className="flex items-center justify-between mb-2">
            <TrendingUp className="w-8 h-8 opacity-80" />
            <span className="text-sm font-medium opacity-90">Accuracy</span>
          </div>
          <div className="text-3xl font-bold">
            {lastResult?.globalAccuracy 
              ? `${(lastResult.globalAccuracy * 100).toFixed(1)}%`
              : status?.recent_rounds[0]?.global_accuracy
              ? `${(status.recent_rounds[0].global_accuracy * 100).toFixed(1)}%`
              : 'N/A'}
          </div>
          <div className="text-xs opacity-75 mt-1">Last aggregation</div>
        </div>

        <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg shadow-lg p-6 text-white">
          <div className="flex items-center justify-between mb-2">
            <Users className="w-8 h-8 opacity-80" />
            <span className="text-sm font-medium opacity-90">Labs</span>
          </div>
          <div className="text-3xl font-bold">{status?.ready_labs || 0}/{status?.total_labs || 0}</div>
          <div className="text-xs opacity-75 mt-1">Ready for aggregation</div>
        </div>

        <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-lg shadow-lg p-6 text-white">
          <div className="flex items-center justify-between mb-2">
            <Database className="w-8 h-8 opacity-80" />
            <span className="text-sm font-medium opacity-90">Samples</span>
          </div>
          <div className="text-3xl font-bold">
            {status?.current_global_model.total_samples || lastResult?.total_samples || 0}
          </div>
          <div className="text-xs opacity-75 mt-1">Total training data</div>
        </div>
      </div>

      {/* Aggregation Control Panel */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-xl font-bold text-gray-900">Model Aggregation Control</h3>
            <p className="text-sm text-gray-600 mt-1">
              Federated Averaging (FedAvg) - Combine local models into global knowledge
            </p>
          </div>
          <button
            onClick={loadStatus}
            className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            title="Refresh status"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>

        {error && (
          <div className="mb-4 p-4 bg-red-50 border-l-4 border-red-500 text-red-700 flex items-start">
            <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {success && (
          <div className="mb-4 p-4 bg-green-50 border-l-4 border-green-500 text-green-700 flex items-start">
            <CheckCircle className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5" />
            <span>{success}</span>
          </div>
        )}

        <div className="flex items-center gap-4">
          <button
            onClick={handleAggregate}
            disabled={isAggregating || (status?.ready_labs || 0) === 0}
            className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 px-6 rounded-lg font-semibold
                     hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 
                     disabled:cursor-not-allowed transition-all duration-200 flex items-center shadow-lg"
          >
            {isAggregating ? (
              <>
                <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                Aggregating Models...
              </>
            ) : (
              <>
                <Zap className="w-5 h-5 mr-2" />
                Aggregate Models Now
              </>
            )}
          </button>

          <button
            onClick={() => setShowPushModal(true)}
            disabled={(status?.current_global_model.version || 0) === 0}
            className="bg-gradient-to-r from-green-600 to-emerald-600 text-white py-3 px-6 rounded-lg font-semibold
                     hover:from-green-700 hover:to-emerald-700 disabled:from-gray-400 disabled:to-gray-500 
                     disabled:cursor-not-allowed transition-all duration-200 flex items-center shadow-lg"
          >
            <Send className="w-5 h-5 mr-2" />
            Push to All Labs
          </button>

          <div className="text-sm text-gray-600">
            <div className="flex items-center">
              <input
                type="text"
                value={serverUrl}
                onChange={(e) => setServerUrl(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm"
                placeholder="Server URL"
              />
            </div>
          </div>
        </div>

        {(status?.ready_labs || 0) === 0 && (
          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-yellow-800 text-sm">
            ⚠️ No labs are ready for aggregation. Labs must submit patient data and train local models first.
          </div>
        )}
      </div>

      {/* Participating Labs */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
          <Users className="w-5 h-5 mr-2 text-blue-600" />
          Participating Labs
        </h3>
        
        {!status || status.labs.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <Brain className="w-16 h-16 mx-auto mb-4 text-gray-300" />
            <p className="text-sm">No lab updates yet. Waiting for labs to submit patient data...</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {status.labs.map((lab) => (
              <div
                key={lab.lab}
                className={`border-2 rounded-lg p-4 transition-all ${
                  lab.ready_for_aggregation
                    ? 'border-green-300 bg-green-50'
                    : 'border-gray-200 bg-gray-50'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-semibold text-gray-900 flex items-center">
                    {lab.ready_for_aggregation ? (
                      <CheckCircle className="w-4 h-4 mr-2 text-green-600" />
                    ) : (
                      <Clock className="w-4 h-4 mr-2 text-gray-400" />
                    )}
                    {lab.lab}
                  </h4>
                  {lab.ready_for_aggregation && (
                    <span className="px-2 py-1 bg-green-600 text-white text-xs rounded-full font-medium">
                      Ready
                    </span>
                  )}
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Local Accuracy:</span>
                    <span className="font-semibold text-gray-900">
                      {lab.local_accuracy ? `${(lab.local_accuracy * 100).toFixed(1)}%` : 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Training Samples:</span>
                    <span className="font-semibold text-gray-900">
                      {lab.num_examples || 0}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Last Update:</span>
                    <span className="text-xs text-gray-500">
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
        <div className="bg-gradient-to-r from-indigo-50 to-blue-50 border-2 border-indigo-200 rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-bold text-indigo-900 mb-4 flex items-center">
            <CheckCircle className="w-5 h-5 mr-2 text-green-600" />
            Latest Aggregation Results
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg p-4 shadow">
              <h4 className="text-sm font-semibold text-gray-700 mb-3">Performance Metrics</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Global Model Version:</span>
                  <span className="font-bold text-indigo-600">v{lastResult.modelVersion}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Global Accuracy:</span>
                  <span className="font-bold text-green-600">
                    {lastResult.globalAccuracy ? `${(lastResult.globalAccuracy * 100).toFixed(2)}%` : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Labs Aggregated:</span>
                  <span className="font-bold text-blue-600">{lastResult.num_models_aggregated}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Samples:</span>
                  <span className="font-bold text-purple-600">{lastResult.total_samples}</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 shadow">
              <h4 className="text-sm font-semibold text-gray-700 mb-3">Lab Contributions</h4>
              <div className="space-y-2">
                {lastResult.lab_contributions.map((contrib) => (
                  <div key={contrib.lab} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <div className="flex items-center">
                      <div className="w-2 h-2 bg-indigo-600 rounded-full mr-2"></div>
                      <span className="text-sm font-medium text-gray-900">{contrib.lab}</span>
                    </div>
                    <div className="flex items-center gap-3 text-xs">
                      <span className="text-gray-600">{contrib.samples} samples</span>
                      <span className="font-semibold text-indigo-600">
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
