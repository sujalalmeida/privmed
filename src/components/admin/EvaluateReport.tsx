import { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { ArrowLeft, Lock, Brain, Send, ThumbsUp, ThumbsDown, AlertCircle } from 'lucide-react';

interface EvaluateReportProps {
  reportId: string;
  onClose: () => void;
}

interface ReportData {
  id: string;
  patient_id: string;
  lab_label: string;
  diagnosis: number;
  diagnosis_label: string;
  confidence: number;
  probabilities: number[] | null;
  created_at: string;
  age: number | null;
  sex: string | null;
  height_cm: number | null;
  weight_kg: number | null;
  bmi: number | null;
  systolic_bp: number | null;
  diastolic_bp: number | null;
  heart_rate: number | null;
  fasting_glucose: number | null;
  hba1c: number | null;
  total_cholesterol: number | null;
  ldl_cholesterol: number | null;
  hdl_cholesterol: number | null;
  triglycerides: number | null;
}

interface FeedbackData {
  id: string;
  agree: boolean;
  correct_diagnosis: number | null;
  correct_diagnosis_label: string | null;
  remarks: string | null;
  reviewer_name: string | null;
  created_at: string;
}

const DIAGNOSIS_OPTIONS = [
  { value: 0, label: 'Healthy' },
  { value: 1, label: 'Diabetes' },
  { value: 2, label: 'Hypertension' },
  { value: 3, label: 'Heart Disease' },
];

const API_BASE = 'http://localhost:5001';

export default function EvaluateReport({ reportId, onClose }: EvaluateReportProps) {
  const { user } = useAuth();
  const [report, setReport] = useState<ReportData | null>(null);
  const [existingFeedback, setExistingFeedback] = useState<FeedbackData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Feedback form state
  const [agree, setAgree] = useState<boolean | null>(null);
  const [correctDiagnosis, setCorrectDiagnosis] = useState<number | null>(null);
  const [remarks, setRemarks] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);

  useEffect(() => {
    const fetchReport = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await fetch(`${API_BASE}/admin/reports/${reportId}`);
        
        if (!response.ok) {
          throw new Error('Failed to fetch report');
        }
        
        const data = await response.json();
        setReport(data.report);
        
        if (data.feedback) {
          setExistingFeedback(data.feedback);
          setAgree(data.feedback.agree);
          setCorrectDiagnosis(data.feedback.correct_diagnosis);
          setRemarks(data.feedback.remarks || '');
        }
      } catch (err) {
        console.error('Error fetching report:', err);
        setError(err instanceof Error ? err.message : 'Failed to load report');
      } finally {
        setLoading(false);
      }
    };
    
    fetchReport();
  }, [reportId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (agree === null) {
      setError('Please select whether you agree or disagree with the AI prediction');
      return;
    }

    setIsSending(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/admin/reports/${reportId}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          agree,
          correct_diagnosis: !agree ? correctDiagnosis : null,
          remarks,
          reviewer_id: user?.id || 'admin',
          reviewer_name: user?.fullName || 'Admin',
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || 'Failed to submit feedback');
      }

      setSubmitSuccess(true);
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (err) {
      console.error('Error submitting feedback:', err);
      setError(err instanceof Error ? err.message : 'Failed to submit feedback');
    } finally {
      setIsSending(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-gray-600">Loading report...</span>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <p className="text-red-600">Report not found</p>
        <button onClick={onClose} className="mt-4 text-blue-600 hover:text-blue-800">
          Go Back
        </button>
      </div>
    );
  }

  const probabilities = report.probabilities || [];

  return (
    <div className="space-y-6">
      <div className="flex items-center">
        <button
          onClick={onClose}
          className="flex items-center text-gray-600 hover:text-gray-900 mr-4"
        >
          <ArrowLeft className="w-5 h-5 mr-1" />
          Back
        </button>
        <h2 className="text-2xl font-bold text-gray-900">Evaluate Report</h2>
        {existingFeedback && (
          <span className="ml-4 px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
            Previously Reviewed
          </span>
        )}
      </div>

      {submitSuccess && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <p className="text-green-800 font-medium">✓ Feedback submitted successfully!</p>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center">
          <AlertCircle className="w-5 h-5 text-red-600 mr-2" />
          <p className="text-red-800">{error}</p>
        </div>
      )}

      <div className="bg-white rounded-lg shadow p-6">
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Report Information</h3>
            <span className="flex items-center text-sm text-blue-600">
              <Lock className="w-4 h-4 mr-1" />
              Encrypted Data
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Patient ID:</span>
              <span className="ml-2 font-medium text-gray-900">{report.patient_id || 'Unknown'}</span>
            </div>
            <div>
              <span className="text-gray-600">Lab:</span>
              <span className="ml-2 font-medium text-gray-900">
                {report.lab_label?.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) || 'Unknown'}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Date:</span>
              <span className="ml-2 font-medium text-gray-900">
                {report.created_at ? new Date(report.created_at).toLocaleDateString() : 'N/A'}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Age/Sex:</span>
              <span className="ml-2 font-medium text-gray-900">
                {report.age || 'N/A'} / {report.sex || 'N/A'}
              </span>
            </div>
          </div>

          {/* Clinical Data Summary */}
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-2">Clinical Data</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              {report.systolic_bp && report.diastolic_bp && (
                <div>
                  <span className="text-gray-500">BP:</span>
                  <span className="ml-1 font-medium">{report.systolic_bp}/{report.diastolic_bp} mmHg</span>
                </div>
              )}
              {report.heart_rate && (
                <div>
                  <span className="text-gray-500">HR:</span>
                  <span className="ml-1 font-medium">{report.heart_rate} bpm</span>
                </div>
              )}
              {report.fasting_glucose && (
                <div>
                  <span className="text-gray-500">Glucose:</span>
                  <span className="ml-1 font-medium">{report.fasting_glucose} mg/dL</span>
                </div>
              )}
              {report.hba1c && (
                <div>
                  <span className="text-gray-500">HbA1c:</span>
                  <span className="ml-1 font-medium">{report.hba1c}%</span>
                </div>
              )}
              {report.total_cholesterol && (
                <div>
                  <span className="text-gray-500">Cholesterol:</span>
                  <span className="ml-1 font-medium">{report.total_cholesterol} mg/dL</span>
                </div>
              )}
              {report.bmi && (
                <div>
                  <span className="text-gray-500">BMI:</span>
                  <span className="ml-1 font-medium">{report.bmi.toFixed(1)}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* AI Prediction Section */}
        <div className="mb-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center mb-3">
            <Brain className="w-5 h-5 text-blue-600 mr-2" />
            <h4 className="font-semibold text-blue-900">AI Prediction</h4>
          </div>
          
          <div className="flex items-center justify-between mb-4">
            <div>
              <span className="text-2xl font-bold text-blue-900 capitalize">
                {report.diagnosis_label?.replace('_', ' ') || 'Unknown'}
              </span>
              <span className="ml-3 text-lg text-blue-700">
                ({((report.confidence || 0) * 100).toFixed(1)}% confidence)
              </span>
            </div>
          </div>

          {probabilities.length > 0 && (
            <div className="space-y-2">
              <p className="text-sm text-blue-800 font-medium">Probability Distribution:</p>
              {DIAGNOSIS_OPTIONS.map((opt, idx) => (
                <div key={opt.value} className="flex items-center">
                  <span className="text-sm text-blue-800 w-28">{opt.label}</span>
                  <div className="flex-1 bg-blue-200 rounded-full h-2 mr-3">
                    <div
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${(probabilities[idx] || 0) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium text-blue-900 w-16 text-right">
                    {((probabilities[idx] || 0) * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Feedback Form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Do you agree with the AI prediction?
            </label>
            <div className="flex space-x-4">
              <button
                type="button"
                onClick={() => setAgree(true)}
                className={`flex-1 flex items-center justify-center py-3 px-4 rounded-lg border-2 transition-colors ${
                  agree === true
                    ? 'border-green-500 bg-green-50 text-green-700'
                    : 'border-gray-300 hover:border-green-300 text-gray-700'
                }`}
              >
                <ThumbsUp className="w-5 h-5 mr-2" />
                Agree
              </button>
              <button
                type="button"
                onClick={() => setAgree(false)}
                className={`flex-1 flex items-center justify-center py-3 px-4 rounded-lg border-2 transition-colors ${
                  agree === false
                    ? 'border-red-500 bg-red-50 text-red-700'
                    : 'border-gray-300 hover:border-red-300 text-gray-700'
                }`}
              >
                <ThumbsDown className="w-5 h-5 mr-2" />
                Disagree
              </button>
            </div>
          </div>

          {agree === false && (
            <div className="animate-in slide-in-from-top-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                What is the correct diagnosis?
              </label>
              <select
                value={correctDiagnosis ?? ''}
                onChange={(e) => setCorrectDiagnosis(e.target.value ? parseInt(e.target.value) : null)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Select correct diagnosis (optional)</option>
                {DIAGNOSIS_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Remarks (optional)
            </label>
            <textarea
              value={remarks}
              onChange={(e) => setRemarks(e.target.value)}
              placeholder="Enter any additional comments or clinical notes"
              rows={3}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <p className="text-sm text-yellow-800">
              Your feedback helps improve the AI model and ensures quality of predictions across the federated learning network.
            </p>
          </div>

          <div className="flex space-x-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 bg-gray-200 text-gray-700 py-3 px-4 rounded-lg hover:bg-gray-300 transition-colors font-medium"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSending || agree === null}
              className="flex-1 bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed font-medium flex items-center justify-center"
            >
              <Send className="w-5 h-5 mr-2" />
              {isSending ? 'Submitting...' : existingFeedback ? 'Update Feedback' : 'Submit Feedback'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
