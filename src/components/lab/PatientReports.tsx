import { useCallback, useEffect, useState } from 'react';
import { Shield, RefreshCw, FileText } from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';

const API_BASE = 'http://localhost:5001';

interface PrivateReport {
  id: string;
  created_at: string;
  lab_label: string;
  patient_id_hash: string;
  prediction: string;
  confidence: number;
  clinical_reasoning: string;
  decrypted_numeric_fields: Record<string, number>;
}

export default function PatientReports() {
  const { user } = useAuth();
  const [reports, setReports] = useState<PrivateReport[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchReports = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const labLabel = user?.labName || 'Lab A';
      const response = await fetch(`${API_BASE}/lab/private_reports?lab_label=${encodeURIComponent(labLabel)}&limit=50`);
      if (!response.ok) {
        throw new Error('Failed to load private encrypted reports');
      }
      const data = await response.json();
      setReports(data.reports || []);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : 'Failed to load private encrypted reports');
    } finally {
      setLoading(false);
    }
  }, [user?.labName]);

  useEffect(() => {
    fetchReports();
  }, [fetchReports]);

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Private Encrypted Reports</h2>
            <p className="text-sm text-gray-600 mt-1">
              Your lab decrypts these patient values locally. The global dashboard never shows these individual measurements.
            </p>
          </div>
          <button
            onClick={fetchReports}
            disabled={loading}
            className="flex items-center px-4 py-2 border border-gray-300 rounded-lg text-sm hover:bg-gray-50 disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            {error}
          </div>
        )}
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start">
          <Shield className="w-5 h-5 text-blue-700 mr-3 mt-0.5" />
          <div className="text-sm text-blue-900">
            <p className="font-medium">Lab-only view</p>
            <p className="mt-1">
              Numerical clinical fields are decrypted only for this lab workflow. Global admin users only see aggregated statistics.
            </p>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="bg-white rounded-lg shadow p-10 text-center text-gray-500">
          Loading encrypted patient reports...
        </div>
      ) : reports.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-10 text-center text-gray-500">
          <FileText className="w-10 h-10 mx-auto mb-3 text-gray-400" />
          No encrypted global reports have been submitted by this lab yet.
        </div>
      ) : (
        <div className="space-y-4">
          {reports.map((report) => (
            <div key={report.id} className="bg-white rounded-lg shadow p-6">
              <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 capitalize">
                    {report.prediction.replace('_', ' ')}
                  </h3>
                  <p className="text-sm text-gray-600">
                    Patient hash: {report.patient_id_hash.slice(0, 16)}... | Confidence: {(report.confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-sm text-gray-500">
                  {new Date(report.created_at).toLocaleString()}
                </div>
              </div>
              <p className="mt-3 text-sm text-gray-700">{report.clinical_reasoning}</p>
              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                {Object.entries(report.decrypted_numeric_fields).map(([field, value]) => (
                  <div key={field} className="border border-gray-200 rounded-lg p-3">
                    <p className="text-xs uppercase tracking-wide text-gray-500">{field.replace(/_/g, ' ')}</p>
                    <p className="text-lg font-semibold text-gray-900">{value}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
