import { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { MessageSquare, Calendar, FileText, ThumbsUp, ThumbsDown, RefreshCw, AlertCircle } from 'lucide-react';

interface FeedbackItem {
  record_id: string;
  patient_id: string;
  ai_diagnosis: number;
  ai_diagnosis_label: string;
  ai_confidence: number;
  record_created_at: string;
  feedback: {
    agree: boolean;
    correct_diagnosis: number | null;
    correct_diagnosis_label: string | null;
    remarks: string | null;
    reviewer_name: string | null;
    created_at: string;
  };
}

const API_BASE = 'http://localhost:5001';

export default function DoctorFeedback() {
  const { user } = useAuth();
  const [feedback, setFeedback] = useState<FeedbackItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Get lab label from user context or default
  const labLabel = user?.labName?.toLowerCase().includes('lab a') ? 'lab_A' : 
                   user?.labName?.toLowerCase().includes('lab b') ? 'lab_B' : 'lab_A';

  const fetchFeedback = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE}/lab/feedback?lab_label=${labLabel}&limit=50`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch feedback');
      }
      
      const data = await response.json();
      setFeedback(data.feedback || []);
    } catch (err) {
      console.error('Error fetching feedback:', err);
      setError(err instanceof Error ? err.message : 'Failed to load feedback');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFeedback();
  }, [labLabel]);

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div>
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Doctor Feedback</h2>
            <p className="text-sm text-gray-600 mt-1">View feedback from central admin on your predictions</p>
          </div>
          <button
            onClick={fetchFeedback}
            disabled={loading}
            className="flex items-center px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 text-sm"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {error && (
          <div className="px-6 py-4 bg-red-50 border-b border-red-200 flex items-center">
            <AlertCircle className="w-5 h-5 text-red-600 mr-2" />
            <p className="text-red-700 text-sm">{error}</p>
          </div>
        )}

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Patient ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  AI Prediction
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Confidence
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Doctor Verdict
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Correct Diagnosis
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Reviewed By
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {loading ? (
                <tr>
                  <td colSpan={7} className="px-6 py-12 text-center text-gray-500">
                    <RefreshCw className="w-8 h-8 text-gray-400 mx-auto mb-4 animate-spin" />
                    Loading feedback...
                  </td>
                </tr>
              ) : feedback.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-6 py-12 text-center text-gray-500">
                    <MessageSquare className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    No feedback received yet
                  </td>
                </tr>
              ) : (
                feedback.map((item) => (
                  <tr key={item.record_id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <FileText className="w-4 h-4 text-gray-400 mr-2" />
                        <span className="text-sm font-medium text-gray-900">
                          {item.patient_id || 'Unknown'}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm text-gray-900 capitalize">
                        {item.ai_diagnosis_label?.replace('_', ' ') || 'Unknown'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`text-sm font-medium ${getConfidenceColor(item.ai_confidence || 0)}`}>
                        {((item.ai_confidence || 0) * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {item.feedback.agree ? (
                        <span className="flex items-center text-green-600">
                          <ThumbsUp className="w-4 h-4 mr-1" />
                          <span className="text-sm font-medium">Agreed</span>
                        </span>
                      ) : (
                        <span className="flex items-center text-red-600">
                          <ThumbsDown className="w-4 h-4 mr-1" />
                          <span className="text-sm font-medium">Disagreed</span>
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {item.feedback.correct_diagnosis_label ? (
                        <span className="text-sm text-gray-900 capitalize">
                          {item.feedback.correct_diagnosis_label.replace('_', ' ')}
                        </span>
                      ) : (
                        <span className="text-sm text-gray-400">-</span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                        {item.feedback.reviewer_name || 'Admin'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <Calendar className="w-4 h-4 text-gray-400 mr-2" />
                        <span className="text-sm text-gray-500">
                          {item.feedback.created_at 
                            ? new Date(item.feedback.created_at).toLocaleDateString() 
                            : 'N/A'}
                        </span>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Summary Stats */}
      {feedback.length > 0 && (
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Feedback</p>
                <p className="text-2xl font-bold text-gray-900">{feedback.length}</p>
              </div>
              <MessageSquare className="w-10 h-10 text-blue-600" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Agreed</p>
                <p className="text-2xl font-bold text-green-600">
                  {feedback.filter(f => f.feedback.agree).length}
                </p>
              </div>
              <ThumbsUp className="w-10 h-10 text-green-600" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Agreement Rate</p>
                <p className="text-2xl font-bold text-blue-600">
                  {feedback.length > 0 
                    ? `${((feedback.filter(f => f.feedback.agree).length / feedback.length) * 100).toFixed(0)}%`
                    : 'N/A'}
                </p>
              </div>
              <Calendar className="w-10 h-10 text-blue-600" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
