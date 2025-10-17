import { useState } from 'react';
import { getAllReports, updateReportStatus } from '../../utils/mockData';
import { FileText, Eye } from 'lucide-react';
import EvaluateReport from './EvaluateReport';

export default function IncomingReports() {
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null);
  const reports = getAllReports();

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

  const handleEvaluate = (reportId: string) => {
    updateReportStatus(reportId, 'under_review');
    setSelectedReportId(reportId);
  };

  const handleClose = () => {
    setSelectedReportId(null);
  };

  if (selectedReportId) {
    return <EvaluateReport reportId={selectedReportId} onClose={handleClose} />;
  }

  return (
    <div>
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">Incoming Medical Reports</h2>
          <p className="text-sm text-gray-600 mt-1">Anonymized patient reports from all labs</p>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Anonymous ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Lab
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Test Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Action
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {reports.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-6 py-12 text-center text-gray-500">
                    <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    No reports available
                  </td>
                </tr>
              ) : (
                reports.map((report) => (
                  <tr key={report.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm font-medium text-gray-900">
                        {report.patientName}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {report.labName}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <FileText className="w-4 h-4 text-gray-400 mr-2" />
                        <span className="text-sm text-gray-900">{report.testType}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(report.createdAt).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(report.status)}`}>
                        {getStatusText(report.status)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <button
                        onClick={() => handleEvaluate(report.id)}
                        className="flex items-center text-blue-600 hover:text-blue-800 font-medium"
                      >
                        <Eye className="w-4 h-4 mr-1" />
                        Evaluate
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Reports</p>
              <p className="text-2xl font-bold text-gray-900">{reports.length}</p>
            </div>
            <FileText className="w-10 h-10 text-blue-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Pending Review</p>
              <p className="text-2xl font-bold text-gray-600">
                {reports.filter(r => r.status === 'pending').length}
              </p>
            </div>
            <FileText className="w-10 h-10 text-gray-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Diagnosed</p>
              <p className="text-2xl font-bold text-green-600">
                {reports.filter(r => r.status === 'diagnosed').length}
              </p>
            </div>
            <FileText className="w-10 h-10 text-green-600" />
          </div>
        </div>
      </div>
    </div>
  );
}
