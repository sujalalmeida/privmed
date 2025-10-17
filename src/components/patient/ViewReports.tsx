import { useAuth } from '../../contexts/AuthContext';
import { getReportsByPatient, getDiagnosisByReportId } from '../../utils/mockData';
import { FileText, Download, Clock, CheckCircle, AlertCircle } from 'lucide-react';

export default function ViewReports() {
  const { user } = useAuth();
  const reports = user ? getReportsByPatient(user.id) : [];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'diagnosed':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'under_review':
        return <Clock className="w-5 h-5 text-yellow-600" />;
      default:
        return <AlertCircle className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'diagnosed':
        return 'Diagnosed';
      case 'under_review':
        return 'Under Review';
      default:
        return 'Pending';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'diagnosed':
        return 'bg-green-100 text-green-800';
      case 'under_review':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div>
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">My Medical Reports</h2>
        </div>

        <div className="divide-y divide-gray-200">
          {reports.length === 0 ? (
            <div className="px-6 py-12 text-center">
              <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No reports uploaded yet</p>
            </div>
          ) : (
            reports.map((report) => {
              const diagnosis = getDiagnosisByReportId(report.id);
              return (
                <div key={report.id} className="px-6 py-4 hover:bg-gray-50 transition-colors">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center mb-2">
                        <h3 className="text-lg font-medium text-gray-900">{report.testType}</h3>
                        <span className={`ml-3 px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(report.status)}`}>
                          {getStatusText(report.status)}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-4 text-sm text-gray-600 mb-3">
                        <div>
                          <span className="font-medium">File:</span> {report.fileName}
                        </div>
                        <div>
                          <span className="font-medium">Date:</span> {new Date(report.createdAt).toLocaleDateString()}
                        </div>
                        <div>
                          <span className="font-medium">Lab:</span> {report.labName}
                        </div>
                        <div className="flex items-center">
                          {getStatusIcon(report.status)}
                          <span className="ml-2">{getStatusText(report.status)}</span>
                        </div>
                      </div>

                      {report.aiPrediction && report.aiPrediction.length > 0 && (
                        <div className="mb-3">
                          <p className="text-sm font-medium text-gray-700 mb-1">AI Predictions:</p>
                          <div className="flex flex-wrap gap-2">
                            {report.aiPrediction.map((pred, idx) => (
                              <span
                                key={idx}
                                className="px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-xs"
                              >
                                {pred.disease}: {(pred.confidence * 100).toFixed(0)}%
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {diagnosis && (
                        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                          <p className="text-sm font-medium text-green-900 mb-1">Diagnosis Result:</p>
                          <p className="text-sm text-green-800">{diagnosis.diagnosisResult}</p>
                          <p className="text-xs text-green-600 mt-2">
                            By {diagnosis.doctorName} on {new Date(diagnosis.createdAt).toLocaleDateString()}
                          </p>
                        </div>
                      )}
                    </div>

                    {report.status === 'diagnosed' && (
                      <button className="ml-4 flex items-center px-4 py-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors">
                        <Download className="w-4 h-4 mr-2" />
                        Download
                      </button>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
