import { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { getAllReports, addDiagnosis, updateReportStatus } from '../../utils/mockData';
import { ArrowLeft, Lock, Brain, Send } from 'lucide-react';

interface EvaluateReportProps {
  reportId: string;
  onClose: () => void;
}

export default function EvaluateReport({ reportId, onClose }: EvaluateReportProps) {
  const { user } = useAuth();
  const report = getAllReports().find(r => r.id === reportId);
  const [diagnosisResult, setDiagnosisResult] = useState('');
  const [doctorRemarks, setDoctorRemarks] = useState('');
  const [isSending, setIsSending] = useState(false);

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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user) return;

    setIsSending(true);

    await new Promise(resolve => setTimeout(resolve, 1500));

    addDiagnosis({
      reportId: report.id,
      doctorId: user.id,
      doctorName: user.fullName,
      encryptedFeedback: 'encrypted_' + Date.now(),
      diagnosisResult,
      confidenceScore: report.aiPrediction?.[0]?.confidence || 0.85,
    });

    updateReportStatus(report.id, 'diagnosed');

    setIsSending(false);
    onClose();
  };

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
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Report Information</h3>
            <span className="flex items-center text-sm text-blue-600">
              <Lock className="w-4 h-4 mr-1" />
              Encrypted Data
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Anonymous ID:</span>
              <span className="ml-2 font-medium text-gray-900">{report.patientName}</span>
            </div>
            <div>
              <span className="text-gray-600">Test Type:</span>
              <span className="ml-2 font-medium text-gray-900">{report.testType}</span>
            </div>
            <div>
              <span className="text-gray-600">Lab:</span>
              <span className="ml-2 font-medium text-gray-900">{report.labName}</span>
            </div>
            <div>
              <span className="text-gray-600">Date:</span>
              <span className="ml-2 font-medium text-gray-900">
                {new Date(report.createdAt).toLocaleDateString()}
              </span>
            </div>
          </div>
        </div>

        {report.aiPrediction && report.aiPrediction.length > 0 && (
          <div className="mb-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center mb-3">
              <Brain className="w-5 h-5 text-blue-600 mr-2" />
              <h4 className="font-semibold text-blue-900">AI Model Predictions</h4>
            </div>
            <div className="space-y-2">
              {report.aiPrediction.map((pred, idx) => (
                <div key={idx} className="flex items-center justify-between">
                  <span className="text-sm text-blue-800">{pred.disease}</span>
                  <div className="flex items-center">
                    <div className="w-32 bg-blue-200 rounded-full h-2 mr-3">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${pred.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium text-blue-900">
                      {(pred.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Diagnosis Result
            </label>
            <input
              type="text"
              value={diagnosisResult}
              onChange={(e) => setDiagnosisResult(e.target.value)}
              placeholder="Enter diagnosis (e.g., Type 2 Diabetes detected)"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Doctor's Remarks
            </label>
            <textarea
              value={doctorRemarks}
              onChange={(e) => setDoctorRemarks(e.target.value)}
              placeholder="Enter your clinical remarks and recommendations"
              rows={4}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
            />
          </div>

          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <p className="text-sm text-yellow-800">
              Your feedback will be encrypted before being sent back to the lab and patient, maintaining complete privacy.
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
              disabled={isSending}
              className="flex-1 bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed font-medium flex items-center justify-center"
            >
              <Send className="w-5 h-5 mr-2" />
              {isSending ? 'Encrypting & Sending...' : 'Encrypt & Send Feedback'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
