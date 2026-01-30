

interface PredictionResult {
  diagnosis: number;
  diagnosis_label: string;
  confidence: number;
  probabilities: {
    healthy: number;
    diabetes: number;
    hypertension: number;
    heart_disease: number;
  };
}

interface ModelInfo {
  current_model_type: 'global' | 'local' | 'baseline' | 'none';
  current_model_version: number | null;
  current_model_accuracy: number | null;
  current_model_accuracy_percent: string | null;
  last_updated: string | null;
  has_model: boolean;
}

interface DiagnosisResultProps {
  result: PredictionResult | null;
  isLoading: boolean;
  nodeAccuracy?: number | null;
  modelInfo?: ModelInfo | null;
}

const DIAGNOSIS_COLORS: Record<string, string> = {
  healthy: 'bg-green-100 border-green-500 text-green-800',
  diabetes: 'bg-amber-100 border-amber-500 text-amber-800',
  hypertension: 'bg-orange-100 border-orange-500 text-orange-800',
  heart_disease: 'bg-red-100 border-red-500 text-red-800',
};

const DIAGNOSIS_ICONS: Record<string, string> = {
  healthy: '✓',
  diabetes: '⚠️',
  hypertension: '⚡',
  heart_disease: '❤️',
};

const DIAGNOSIS_DESCRIPTIONS: Record<string, string> = {
  healthy: 'No significant abnormalities detected. Continue routine monitoring.',
  diabetes: 'Elevated glucose markers suggest diabetes. Consider confirmatory testing and lifestyle modifications.',
  hypertension: 'Blood pressure readings indicate hypertension. Recommend monitoring and potential treatment.',
  heart_disease: 'Cardiac markers suggest cardiovascular concerns. Further cardiac workup recommended.',
};

const MODEL_TYPE_LABELS: Record<string, string> = {
  global: 'Federated Global Model',
  local: 'Local Model',
  baseline: 'Baseline Model',
  none: 'Rule-Based',
};

export default function DiagnosisResult({ result, isLoading, nodeAccuracy, modelInfo }: DiagnosisResultProps) {
  // Render model accuracy badge
  const renderModelAccuracyBadge = () => {
    if (nodeAccuracy === null && nodeAccuracy === undefined) return null;
    
    const modelType = modelInfo?.current_model_type || 'unknown';
    const modelLabel = MODEL_TYPE_LABELS[modelType] || modelType;
    
    return (
      <div className="flex flex-col items-end gap-1">
        <div className="flex items-center gap-2 bg-primary-50 px-3 py-1.5 rounded-lg">
          <span className="text-sm text-primary-600 font-medium">Model Accuracy:</span>
          <span className="text-lg font-bold text-primary-700">
            {nodeAccuracy !== null && nodeAccuracy !== undefined 
              ? `${(nodeAccuracy * 100).toFixed(1)}%`
              : 'N/A'}
          </span>
        </div>
        <span className="text-xs text-gray-500">{modelLabel}</span>
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="clinical-section diagnosis-result">
        <h3>AI Prediction</h3>
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600"></div>
          <span className="ml-4 text-gray-600">Analyzing patient data...</span>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="clinical-section diagnosis-result">
        <div className="flex justify-between items-center mb-4">
          <h3>AI Prediction</h3>
          {renderModelAccuracyBadge()}
        </div>
        <div className="text-center py-8 text-gray-500">
          <p>Complete the form and submit to get AI-powered diagnosis prediction</p>
        </div>
      </div>
    );
  }

  const diagnosisKey = result.diagnosis_label.toLowerCase().replace(' ', '_');
  const colorClass = DIAGNOSIS_COLORS[diagnosisKey] || DIAGNOSIS_COLORS.healthy;
  const icon = DIAGNOSIS_ICONS[diagnosisKey] || '?';
  const description = DIAGNOSIS_DESCRIPTIONS[diagnosisKey] || '';

  return (
    <div className="clinical-section diagnosis-result">
      <div className="flex justify-between items-center mb-4">
        <h3 className="mb-0">AI Prediction Result</h3>
        {renderModelAccuracyBadge()}
      </div>
      
      {/* Main diagnosis card */}
      <div className={`border-l-4 ${colorClass} p-4 rounded-r-lg mb-6`}>
        <div className="flex items-center justify-between">
          <div>
            <span className="text-2xl mr-2">{icon}</span>
            <span className="text-xl font-semibold capitalize">
              {result.diagnosis_label.replace('_', ' ')}
            </span>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-500">Prediction Confidence</div>
            <div className="text-2xl font-bold">
              {(result.confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>
        <p className="mt-2 text-sm opacity-80">{description}</p>
      </div>
      
      {/* Probability breakdown */}
      <div className="mb-6">
        <h4 className="text-sm font-medium text-gray-700 mb-3">Probability Distribution</h4>
        <div className="space-y-2">
          {Object.entries(result.probabilities)
            .sort(([, a], [, b]) => b - a)
            .map(([condition, probability]) => (
              <div key={condition} className="flex items-center">
                <span className="w-28 text-sm text-gray-600 capitalize">
                  {condition.replace('_', ' ')}
                </span>
                <div className="flex-1 mx-3 bg-gray-200 rounded-full h-4 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      condition === diagnosisKey ? 'bg-blue-600' : 'bg-gray-400'
                    }`}
                    style={{ width: `${probability * 100}%` }}
                  />
                </div>
                <span className="w-16 text-right text-sm font-medium">
                  {(probability * 100).toFixed(1)}%
                </span>
              </div>
            ))}
        </div>
      </div>
      
      {/* Disclaimer */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
        <p className="text-xs text-yellow-800">
          <strong>⚠️ Clinical Decision Support:</strong> This AI prediction is for informational 
          purposes only and should not replace professional medical judgment. Always verify 
          findings with appropriate diagnostic tests and clinical evaluation.
        </p>
      </div>
    </div>
  );
}
