import { useState, useEffect } from 'react';
import { 
  FlaskConical, Play, CheckCircle, XCircle, TrendingUp, TrendingDown, 
  BarChart3, Table, Filter, RefreshCw, Clock, AlertCircle,
  Database, Beaker
} from 'lucide-react';

interface Model {
  type: string;
  lab_label?: string;
  version?: number;
  display_name: string;
}

interface Prediction {
  patient_id: number;
  actual: string;
  actual_label: number;
  predicted: string;
  predicted_label: number;
  confidence: number;
  correct: boolean;
}

interface ABTestResult {
  id: string;
  created_at: string;
  test_name: string;
  model_a_type: string;
  model_a_version: string;
  model_b_type: string;
  model_b_version: string;
  model_a_accuracy: number;
  model_b_accuracy: number;
  accuracy_delta: number;
  num_samples: number;
  winner: string;
  status: string;
  model_a_predictions?: Prediction[];
  model_b_predictions?: Prediction[];
  confusion_matrix_a?: number[][];
  confusion_matrix_b?: number[][];
  per_class_metrics_a?: Record<string, { precision: number; recall: number; f1: number }>;
  per_class_metrics_b?: Record<string, { precision: number; recall: number; f1: number }>;
  statistical_significance?: {
    p_value: number;
    is_significant: boolean;
    test_used: string;
    model_b_wins: number;
    model_a_wins: number;
  };
}

export default function ABTestDashboard() {
  const serverUrl = 'http://127.0.0.1:5001';
  
  // Test configuration state
  const [localModels, setLocalModels] = useState<Model[]>([]);
  const [globalModels, setGlobalModels] = useState<Model[]>([]);
  const [selectedModelA, setSelectedModelA] = useState<string>('');
  const [selectedModelB, setSelectedModelB] = useState<string>('');
  const [testName, setTestName] = useState('');
  const [useHeldOut, setUseHeldOut] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  
  // Results state
  const [currentResult, setCurrentResult] = useState<ABTestResult | null>(null);
  const [testHistory, setTestHistory] = useState<ABTestResult[]>([]);
  const [showDisagreementsOnly, setShowDisagreementsOnly] = useState(false);
  
  // Test dataset info
  const [testDatasetInfo, setTestDatasetInfo] = useState<any>(null);
  const [isCreatingTestSet, setIsCreatingTestSet] = useState(false);
  
  // UI state
  const [error, setError] = useState<string>('');
  const [activeTab, setActiveTab] = useState<'configure' | 'results' | 'history'>('configure');

  const diseaseLabels = ['healthy', 'diabetes', 'hypertension', 'heart_disease'];

  useEffect(() => {
    loadAvailableModels();
    loadTestHistory();
    loadTestDatasetInfo();
  }, []);

  const loadAvailableModels = async () => {
    try {
      const response = await fetch(`${serverUrl}/admin/get_available_models`);
      if (response.ok) {
        const data = await response.json();
        setLocalModels(data.local_models || []);
        setGlobalModels(data.global_models || []);
      }
    } catch (err) {
      console.error('Error loading models:', err);
    }
  };

  const loadTestHistory = async () => {
    try {
      const response = await fetch(`${serverUrl}/admin/ab_test_history`);
      if (response.ok) {
        const data = await response.json();
        setTestHistory(data.tests || []);
      }
    } catch (err) {
      console.error('Error loading test history:', err);
    }
  };

  const loadTestDatasetInfo = async () => {
    try {
      const response = await fetch(`${serverUrl}/admin/get_test_dataset_info`);
      if (response.ok) {
        const data = await response.json();
        setTestDatasetInfo(data);
      }
    } catch (err) {
      console.error('Error loading test dataset info:', err);
    }
  };

  const handleRunTest = async () => {
    if (!selectedModelA || !selectedModelB) {
      setError('Please select both Model A and Model B');
      return;
    }

    setIsRunning(true);
    setError('');
    setCurrentResult(null);

    try {
      // Parse selected models
      const modelAData = selectedModelA.startsWith('local:') 
        ? { type: 'local', lab_label: selectedModelA.replace('local:', '') }
        : { type: 'global', version: parseInt(selectedModelA.replace('global:', '')) };
      
      const modelBData = selectedModelB.startsWith('local:')
        ? { type: 'local', lab_label: selectedModelB.replace('local:', '') }
        : { type: 'global', version: parseInt(selectedModelB.replace('global:', '')) };

      const response = await fetch(`${serverUrl}/admin/run_ab_test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          test_name: testName || `${modelAData.type === 'local' ? modelAData.lab_label : 'Global v' + modelAData.version} vs ${modelBData.type === 'local' ? modelBData.lab_label : 'Global v' + modelBData.version}`,
          model_a: modelAData,
          model_b: modelBData,
          use_held_out: useHeldOut
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to run A/B test');
      }

      const data = await response.json();
      
      // Load full results
      const resultsResponse = await fetch(`${serverUrl}/admin/ab_test_results/${data.test_id}`);
      if (resultsResponse.ok) {
        const fullResults = await resultsResponse.json();
        setCurrentResult(fullResults);
      }

      setActiveTab('results');
      loadTestHistory();

    } catch (err: any) {
      setError(err.message || 'Failed to run A/B test');
    } finally {
      setIsRunning(false);
    }
  };

  const handleCreateTestDataset = async () => {
    setIsCreatingTestSet(true);
    try {
      const response = await fetch(`${serverUrl}/admin/create_test_dataset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ test_percentage: 20 })
      });

      if (response.ok) {
        loadTestDatasetInfo();
      }
    } catch (err) {
      console.error('Error creating test dataset:', err);
    } finally {
      setIsCreatingTestSet(false);
    }
  };

  const loadHistoryTestDetails = async (testId: string) => {
    try {
      const response = await fetch(`${serverUrl}/admin/ab_test_results/${testId}`);
      if (response.ok) {
        const data = await response.json();
        setCurrentResult(data);
        setActiveTab('results');
      }
    } catch (err) {
      console.error('Error loading test details:', err);
    }
  };

  const renderConfusionMatrix = (matrix: number[][] | undefined, title: string, colorClass: string) => {
    if (!matrix) return null;

    return (
      <div className="bg-white rounded-lg p-4 shadow">
        <h4 className="text-sm font-semibold text-gray-700 mb-3">{title}</h4>
        <div className="overflow-x-auto">
          <table className="text-xs">
            <thead>
              <tr>
                <th className="p-1"></th>
                {diseaseLabels.map(label => (
                  <th key={label} className="p-1 text-center font-medium text-gray-600 truncate max-w-[60px]">
                    {label.slice(0, 4)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, i) => (
                <tr key={i}>
                  <td className="p-1 font-medium text-gray-600 truncate max-w-[60px]">
                    {diseaseLabels[i].slice(0, 4)}
                  </td>
                  {row.map((cell, j) => (
                    <td
                      key={j}
                      className={`p-1 text-center w-10 h-10 ${
                        i === j
                          ? `${colorClass} text-white font-bold`
                          : cell > 0
                          ? 'bg-red-100 text-red-800'
                          : 'bg-gray-50 text-gray-400'
                      }`}
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-500 mt-2">Rows: Actual | Cols: Predicted</p>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 flex items-center">
            <FlaskConical className="w-7 h-7 mr-3 text-purple-600" />
            A/B Test Dashboard
          </h2>
          <p className="text-gray-600 mt-1">
            Compare local vs. global federated models to measure FL improvement
          </p>
        </div>
        <button
          onClick={loadAvailableModels}
          className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <RefreshCw className="w-5 h-5" />
        </button>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'configure', label: 'Configure Test', icon: Beaker },
            { id: 'results', label: 'Results', icon: BarChart3 },
            { id: 'history', label: 'Test History', icon: Clock }
          ].map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-purple-500 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="w-5 h-5 mr-2" />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border-l-4 border-red-500 text-red-700 flex items-start">
          <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* Configure Tab */}
      {activeTab === 'configure' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Test Configuration */}
          <div className="lg:col-span-2 bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4">Test Configuration</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Test Name</label>
                <input
                  type="text"
                  value={testName}
                  onChange={(e) => setTestName(e.target.value)}
                  placeholder="e.g., Lab A vs Global v5"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Model A (Baseline)
                  </label>
                  <select
                    value={selectedModelA}
                    onChange={(e) => setSelectedModelA(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  >
                    <option value="">Select Model A...</option>
                    <optgroup label="Local Lab Models">
                      {localModels.map((model) => (
                        <option key={`local:${model.lab_label}`} value={`local:${model.lab_label}`}>
                          {model.display_name}
                        </option>
                      ))}
                    </optgroup>
                    <optgroup label="Global Models">
                      {globalModels.map((model) => (
                        <option key={`global:${model.version}`} value={`global:${model.version}`}>
                          {model.display_name}
                        </option>
                      ))}
                    </optgroup>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Model B (Challenger)
                  </label>
                  <select
                    value={selectedModelB}
                    onChange={(e) => setSelectedModelB(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  >
                    <option value="">Select Model B...</option>
                    <optgroup label="Global Models">
                      {globalModels.map((model) => (
                        <option key={`global:${model.version}`} value={`global:${model.version}`}>
                          {model.display_name}
                        </option>
                      ))}
                    </optgroup>
                    <optgroup label="Local Lab Models">
                      {localModels.map((model) => (
                        <option key={`local:${model.lab_label}`} value={`local:${model.lab_label}`}>
                          {model.display_name}
                        </option>
                      ))}
                    </optgroup>
                  </select>
                </div>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="useHeldOut"
                  checked={useHeldOut}
                  onChange={(e) => setUseHeldOut(e.target.checked)}
                  className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
                />
                <label htmlFor="useHeldOut" className="ml-2 text-sm text-gray-700">
                  Use held-out test dataset (recommended for unbiased comparison)
                </label>
              </div>

              <button
                onClick={handleRunTest}
                disabled={isRunning || !selectedModelA || !selectedModelB}
                className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white py-3 px-6 rounded-lg font-semibold
                         hover:from-purple-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 
                         disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center shadow-lg"
              >
                {isRunning ? (
                  <>
                    <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                    Running A/B Test...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5 mr-2" />
                    Run A/B Test
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Test Dataset Info */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
              <Database className="w-5 h-5 mr-2 text-blue-600" />
              Test Dataset
            </h3>
            
            {testDatasetInfo?.has_test_dataset ? (
              <div className="space-y-4">
                <div className="p-4 bg-green-50 rounded-lg">
                  <div className="flex items-center text-green-700 font-medium mb-2">
                    <CheckCircle className="w-5 h-5 mr-2" />
                    Held-out Dataset Ready
                  </div>
                  <p className="text-sm text-green-600">
                    {testDatasetInfo.num_samples} test samples available
                  </p>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Diagnosis Distribution</h4>
                  <div className="space-y-1">
                    {Object.entries(testDatasetInfo.diagnosis_distribution || {}).map(([diagnosis, count]) => (
                      <div key={diagnosis} className="flex justify-between text-sm">
                        <span className="text-gray-600 capitalize">{diagnosis}</span>
                        <span className="font-medium">{count as number}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-4">
                <Database className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                <p className="text-gray-600 mb-4">No held-out test dataset</p>
                <button
                  onClick={handleCreateTestDataset}
                  disabled={isCreatingTestSet}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 text-sm font-medium"
                >
                  {isCreatingTestSet ? 'Creating...' : 'Create Test Dataset (20%)'}
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Results Tab */}
      {activeTab === 'results' && currentResult && (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg shadow-lg p-6 text-white">
              <div className="text-sm opacity-90 mb-1">Model A ({currentResult.model_a_type})</div>
              <div className="text-3xl font-bold">
                {((currentResult.model_a_accuracy || 0) * 100).toFixed(1)}%
              </div>
              <div className="text-sm opacity-75 mt-1">{currentResult.model_a_version}</div>
            </div>

            <div className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg shadow-lg p-6 text-white">
              <div className="text-sm opacity-90 mb-1">Model B ({currentResult.model_b_type})</div>
              <div className="text-3xl font-bold">
                {((currentResult.model_b_accuracy || 0) * 100).toFixed(1)}%
              </div>
              <div className="text-sm opacity-75 mt-1">{currentResult.model_b_version}</div>
            </div>

            <div className={`rounded-lg shadow-lg p-6 text-white ${
              (currentResult.accuracy_delta || 0) > 0
                ? 'bg-gradient-to-br from-purple-500 to-indigo-600'
                : (currentResult.accuracy_delta || 0) < 0
                ? 'bg-gradient-to-br from-orange-500 to-red-600'
                : 'bg-gradient-to-br from-gray-500 to-gray-600'
            }`}>
              <div className="text-sm opacity-90 mb-1">FL Improvement</div>
              <div className="text-3xl font-bold flex items-center">
                {(currentResult.accuracy_delta || 0) > 0 ? (
                  <TrendingUp className="w-8 h-8 mr-2" />
                ) : (currentResult.accuracy_delta || 0) < 0 ? (
                  <TrendingDown className="w-8 h-8 mr-2" />
                ) : null}
                {(currentResult.accuracy_delta || 0) > 0 ? '+' : ''}
                {((currentResult.accuracy_delta || 0) * 100).toFixed(2)}%
              </div>
              <div className="text-sm opacity-75 mt-1">
                {currentResult.statistical_significance?.is_significant 
                  ? `p < 0.05 (Significant)` 
                  : `p = ${currentResult.statistical_significance?.p_value?.toFixed(3) || 'N/A'}`}
              </div>
            </div>
          </div>

          {/* Winner Banner */}
          <div className={`p-4 rounded-lg border-2 ${
            currentResult.winner?.includes('Model B') || currentResult.winner?.includes('Global')
              ? 'bg-green-50 border-green-300'
              : currentResult.winner?.includes('Model A')
              ? 'bg-blue-50 border-blue-300'
              : 'bg-gray-50 border-gray-300'
          }`}>
            <div className="flex items-center justify-center text-lg font-semibold">
              {currentResult.winner?.includes('Model B') || currentResult.winner?.includes('Global') ? (
                <>
                  <CheckCircle className="w-6 h-6 mr-2 text-green-600" />
                  <span className="text-green-700">
                    🎉 Global Federated Model Wins! FL improves accuracy by {((currentResult.accuracy_delta || 0) * 100).toFixed(2)}%
                  </span>
                </>
              ) : currentResult.winner?.includes('Model A') ? (
                <>
                  <AlertCircle className="w-6 h-6 mr-2 text-blue-600" />
                  <span className="text-blue-700">
                    Local model performs better. Consider more FL rounds or data.
                  </span>
                </>
              ) : (
                <span className="text-gray-700">Models perform equally</span>
              )}
            </div>
          </div>

          {/* Confusion Matrices */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {renderConfusionMatrix(currentResult.confusion_matrix_a, 'Model A Confusion Matrix', 'bg-blue-500')}
            {renderConfusionMatrix(currentResult.confusion_matrix_b, 'Model B Confusion Matrix', 'bg-green-500')}
          </div>

          {/* Per-Patient Comparison Table */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-gray-900 flex items-center">
                <Table className="w-5 h-5 mr-2 text-gray-600" />
                Per-Patient Comparison
              </h3>
              <label className="flex items-center text-sm text-gray-600">
                <input
                  type="checkbox"
                  checked={showDisagreementsOnly}
                  onChange={(e) => setShowDisagreementsOnly(e.target.checked)}
                  className="mr-2 rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                />
                <Filter className="w-4 h-4 mr-1" />
                Show disagreements only
              </label>
            </div>

            <div className="overflow-x-auto max-h-96">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50 sticky top-0">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Patient</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actual</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model A</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model B</th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">A ✓</th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">B ✓</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {(currentResult.model_a_predictions || [])
                    .filter((predA, idx) => {
                      if (!showDisagreementsOnly) return true;
                      const predB = currentResult.model_b_predictions?.[idx];
                      return predA.correct !== predB?.correct;
                    })
                    .slice(0, 50)
                    .map((predA, idx) => {
                      const predB = currentResult.model_b_predictions?.[idx];
                      const isDisagreement = predA.correct !== predB?.correct;
                      
                      return (
                        <tr 
                          key={predA.patient_id} 
                          className={`${isDisagreement ? 'bg-yellow-50' : ''} hover:bg-gray-50`}
                        >
                          <td className="px-4 py-2 text-sm text-gray-900">#{predA.patient_id}</td>
                          <td className="px-4 py-2 text-sm font-medium text-gray-900 capitalize">{predA.actual}</td>
                          <td className="px-4 py-2 text-sm">
                            <span className={`capitalize ${predA.correct ? 'text-green-600' : 'text-red-600'}`}>
                              {predA.predicted}
                            </span>
                            <span className="text-gray-400 text-xs ml-1">
                              ({(predA.confidence * 100).toFixed(0)}%)
                            </span>
                          </td>
                          <td className="px-4 py-2 text-sm">
                            <span className={`capitalize ${predB?.correct ? 'text-green-600' : 'text-red-600'}`}>
                              {predB?.predicted}
                            </span>
                            <span className="text-gray-400 text-xs ml-1">
                              ({((predB?.confidence || 0) * 100).toFixed(0)}%)
                            </span>
                          </td>
                          <td className="px-4 py-2 text-center">
                            {predA.correct ? (
                              <CheckCircle className="w-5 h-5 text-green-500 mx-auto" />
                            ) : (
                              <XCircle className="w-5 h-5 text-red-400 mx-auto" />
                            )}
                          </td>
                          <td className="px-4 py-2 text-center">
                            {predB?.correct ? (
                              <CheckCircle className="w-5 h-5 text-green-500 mx-auto" />
                            ) : (
                              <XCircle className="w-5 h-5 text-red-400 mx-auto" />
                            )}
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Showing {Math.min(50, (currentResult.model_a_predictions || []).filter((predA, idx) => {
                if (!showDisagreementsOnly) return true;
                const predB = currentResult.model_b_predictions?.[idx];
                return predA.correct !== predB?.correct;
              }).length)} of {currentResult.num_samples} patients
            </p>
          </div>
        </div>
      )}

      {activeTab === 'results' && !currentResult && (
        <div className="text-center py-12 bg-white rounded-lg shadow">
          <BarChart3 className="w-16 h-16 mx-auto mb-4 text-gray-300" />
          <p className="text-gray-600">No test results yet</p>
          <p className="text-sm text-gray-500 mt-1">Run an A/B test to see results here</p>
        </div>
      )}

      {/* History Tab */}
      {activeTab === 'history' && (
        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-bold text-gray-900">Test History</h3>
          </div>
          
          {testHistory.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <Clock className="w-12 h-12 mx-auto mb-3 text-gray-300" />
              <p>No A/B tests run yet</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Test Name</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model A</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model B</th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase">Winner</th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase">Δ Accuracy</th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase">Samples</th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase">Action</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {testHistory.map((test) => (
                    <tr key={test.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {new Date(test.created_at).toLocaleDateString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {test.test_name}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {test.model_a_type}: {test.model_a_version}
                        <span className="ml-2 text-blue-600">
                          {((test.model_a_accuracy || 0) * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {test.model_b_type}: {test.model_b_version}
                        <span className="ml-2 text-green-600">
                          {((test.model_b_accuracy || 0) * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                          test.winner?.includes('Model B') || test.winner?.includes('Global')
                            ? 'bg-green-100 text-green-800'
                            : test.winner?.includes('Model A')
                            ? 'bg-blue-100 text-blue-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}>
                          {test.winner}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center text-sm">
                        <span className={`font-medium ${
                          (test.accuracy_delta || 0) > 0 ? 'text-green-600' : 
                          (test.accuracy_delta || 0) < 0 ? 'text-red-600' : 'text-gray-600'
                        }`}>
                          {(test.accuracy_delta || 0) > 0 ? '+' : ''}
                          {((test.accuracy_delta || 0) * 100).toFixed(2)}%
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-gray-500">
                        {test.num_samples}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <button
                          onClick={() => loadHistoryTestDetails(test.id)}
                          className="text-purple-600 hover:text-purple-800 text-sm font-medium"
                        >
                          View Details
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
