import { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { getModelUpdatesByLab, addModelUpdate, getAllModelUpdates } from '../../utils/mockData';
import { Brain, Upload, TrendingUp, CheckCircle } from 'lucide-react';

export default function FederatedLearning() {
  const { user } = useAuth();
  const [isSending, setIsSending] = useState(false);
  const [sendSuccess, setSendSuccess] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);

  const labUpdates = user ? getModelUpdatesByLab(user.id) : [];
  const allUpdates = getAllModelUpdates();
  const aggregatedUpdates = allUpdates.filter(u => u.isAggregated);
  const latestGlobalAccuracy = aggregatedUpdates.length > 0
    ? Math.max(...aggregatedUpdates.map(u => u.accuracy))
    : 0;

  const handleSendUpdate = async () => {
    if (!user) return;

    setIsSending(true);
    setSendError(null);

    try {
      // Use the same lab_label logic as PatientDataCollection
      const labLabel = user?.labName || user?.email || 'lab_sim';

      const response = await fetch('http://127.0.0.1:5001/lab/send_model_update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lab_label: labLabel,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to send model update');
      }

      const data = await response.json();
      
      // Add to mock data for UI display
      const version = `v1.2.${labUpdates.length + 1}`;
      addModelUpdate({
        labId: user.id,
        labName: user.labName || 'Lab',
        modelVersion: version,
        accuracy: data.local_accuracy || 0.92,
        isAggregated: false,
      });

      setSendSuccess(true);
      setTimeout(() => setSendSuccess(false), 3000);
    } catch (error) {
      setSendError(error instanceof Error ? error.message : 'Failed to send model update');
      setTimeout(() => setSendError(null), 5000);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-gray-600">Local Model Updates</p>
              <p className="text-2xl font-bold text-gray-900">{labUpdates.length}</p>
            </div>
            <Brain className="w-10 h-10 text-blue-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-gray-600">Latest Local Accuracy</p>
              <p className="text-2xl font-bold text-green-600">
                {labUpdates.length > 0
                  ? `${(labUpdates[labUpdates.length - 1].accuracy * 100).toFixed(1)}%`
                  : 'N/A'}
              </p>
            </div>
            <TrendingUp className="w-10 h-10 text-green-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-gray-600">Global Model Accuracy</p>
              <p className="text-2xl font-bold text-blue-600">
                {latestGlobalAccuracy > 0
                  ? `${(latestGlobalAccuracy * 100).toFixed(1)}%`
                  : 'N/A'}
              </p>
            </div>
            <CheckCircle className="w-10 h-10 text-blue-600" />
          </div>
        </div>
      </div>

      <div className="max-w-md mx-auto">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Send Model Update</h3>
          <p className="text-sm text-gray-600 mb-4">
            Send encrypted model gradients/weights to the central server for federated aggregation. Your local data never leaves the lab.
          </p>

          {sendSuccess && (
            <div className="flex items-center p-3 bg-green-50 border border-green-200 rounded-lg text-green-800 text-sm mb-4">
              <CheckCircle className="w-4 h-4 mr-2" />
              Model update sent successfully!
            </div>
          )}

          {sendError && (
            <div className="flex items-center p-3 bg-red-50 border border-red-200 rounded-lg text-red-800 text-sm mb-4">
              <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              {sendError}
            </div>
          )}

          <button
            onClick={handleSendUpdate}
            disabled={isSending}
            className="w-full bg-green-600 text-white py-3 px-4 rounded-lg hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed font-medium flex items-center justify-center"
          >
            <Upload className="w-5 h-5 mr-2" />
            {isSending ? 'Sending Update...' : 'Send Model Update'}
          </button>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Model Update History</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Version
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Accuracy
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {labUpdates.length === 0 ? (
                <tr>
                  <td colSpan={4} className="px-6 py-12 text-center text-gray-500">
                    <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    No model updates yet
                  </td>
                </tr>
              ) : (
                labUpdates.map((update) => (
                  <tr key={update.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {update.modelVersion}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {(update.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                        update.isAggregated
                          ? 'bg-green-100 text-green-800'
                          : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {update.isAggregated ? 'Aggregated' : 'Pending'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(update.createdAt).toLocaleDateString()}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
