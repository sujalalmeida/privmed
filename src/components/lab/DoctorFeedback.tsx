import { useAuth } from '../../contexts/AuthContext';
import { getReportsByLab, getDiagnosisByReportId } from '../../utils/mockData';
import { MessageSquare, Calendar, FileText } from 'lucide-react';

export default function DoctorFeedback() {
  const { user } = useAuth();
  const reports = user ? getReportsByLab(user.id).filter(r => r.status === 'diagnosed') : [];

  return (
    <div>
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">Doctor Feedback</h2>
          <p className="text-sm text-gray-600 mt-1">View diagnosis results from central lab doctors</p>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Patient ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Test Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Diagnosis Result
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Doctor Remarks
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {reports.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-6 py-12 text-center text-gray-500">
                    <MessageSquare className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    No feedback received yet
                  </td>
                </tr>
              ) : (
                reports.map((report) => {
                  const diagnosis = getDiagnosisByReportId(report.id);
                  return (
                    <tr key={report.id} className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <FileText className="w-4 h-4 text-gray-400 mr-2" />
                          <span className="text-sm font-medium text-gray-900">
                            {report.patientName}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {report.testType}
                      </td>
                      <td className="px-6 py-4">
                        <span className="text-sm text-gray-900">
                          {diagnosis?.diagnosisResult || 'N/A'}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-xs font-medium">
                          {diagnosis ? `By ${diagnosis.doctorName}` : 'N/A'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <Calendar className="w-4 h-4 text-gray-400 mr-2" />
                          <span className="text-sm text-gray-500">
                            {diagnosis ? new Date(diagnosis.createdAt).toLocaleDateString() : 'N/A'}
                          </span>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
