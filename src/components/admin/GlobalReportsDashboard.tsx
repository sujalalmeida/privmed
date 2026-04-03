import { useCallback, useEffect, useState } from 'react';
import { RefreshCw, Shield, BarChart3, Activity } from 'lucide-react';

const API_BASE = 'http://localhost:5001';

interface AggregateStats {
  averages: Record<string, number>;
  counts: Record<string, number>;
  age_band: string | null;
}

interface SummaryResponse {
  total_reports: number;
  average_confidence: number | null;
  prediction_prevalence: Record<string, { count: number; share: number }>;
  recent_reports: Array<{
    id: string;
    created_at: string;
    lab_label: string;
    patient_id_hash: string;
    prediction: string;
    confidence: number;
    clinical_reasoning: string;
    encrypted: boolean;
    encryption_scheme: string;
  }>;
  aggregate_stats: AggregateStats | null;
  evaluated_by_lab: string;
}

export default function GlobalReportsDashboard() {
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [aggregating, setAggregating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSummary = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE}/admin/global_reports/summary?limit=100`);
      if (!response.ok) {
        throw new Error('Failed to load encrypted report summary');
      }
      const data = await response.json();
      setSummary(data);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : 'Failed to load encrypted report summary');
    } finally {
      setLoading(false);
    }
  }, []);

  const runAggregation = async () => {
    try {
      setAggregating(true);
      setError(null);
      const response = await fetch(`${API_BASE}/admin/global_reports/aggregate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: 1000 }),
      });
      if (!response.ok) {
        throw new Error('Failed to aggregate encrypted report statistics');
      }
      const data = await response.json();
      setSummary(data);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : 'Failed to aggregate encrypted report statistics');
    } finally {
      setAggregating(false);
    }
  };

  useEffect(() => {
    fetchSummary();
  }, [fetchSummary]);

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Encrypted Global Reports</h2>
            <p className="text-sm text-gray-600 mt-1">
              Sensitive clinical values stay homomorphically encrypted. This dashboard shows metadata and aggregate-only summaries.
            </p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={fetchSummary}
              disabled={loading}
              className="flex items-center px-4 py-2 border border-gray-300 rounded-lg text-sm hover:bg-gray-50 disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
            <button
              onClick={runAggregation}
              disabled={aggregating}
              className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:bg-blue-300"
            >
              <BarChart3 className={`w-4 h-4 mr-2 ${aggregating ? 'animate-pulse' : ''}`} />
              {aggregating ? 'Aggregating...' : 'Aggregate Encrypted Stats'}
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            {error}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-sm text-gray-600">Encrypted Reports</p>
          <p className="text-2xl font-bold text-gray-900">{summary?.total_reports ?? 0}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-sm text-gray-600">Avg Confidence</p>
          <p className="text-2xl font-bold text-blue-700">
            {summary?.average_confidence != null ? `${(summary.average_confidence * 100).toFixed(1)}%` : 'N/A'}
          </p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-sm text-gray-600">Evaluated By</p>
          <p className="text-2xl font-bold text-gray-900">{summary?.evaluated_by_lab?.replace('_', ' ') || 'lab A'}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-sm text-gray-600">Age Band</p>
          <p className="text-2xl font-bold text-emerald-700">{summary?.aggregate_stats?.age_band || 'N/A'}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center mb-4">
            <Shield className="w-5 h-5 text-blue-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Prediction Prevalence</h3>
          </div>
          <div className="space-y-3">
            {summary && Object.keys(summary.prediction_prevalence).length > 0 ? (
              Object.entries(summary.prediction_prevalence).map(([label, value]) => (
                <div key={label} className="flex items-center justify-between">
                  <span className="capitalize text-gray-700">{label.replace('_', ' ')}</span>
                  <span className="font-medium text-gray-900">
                    {value.count} ({(value.share * 100).toFixed(1)}%)
                  </span>
                </div>
              ))
            ) : (
              <p className="text-sm text-gray-500">No encrypted reports submitted yet.</p>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center mb-4">
            <Activity className="w-5 h-5 text-emerald-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Aggregate Clinical Statistics</h3>
          </div>
          <div className="space-y-3">
            {summary?.aggregate_stats ? (
              Object.entries(summary.aggregate_stats.averages).map(([field, value]) => (
                <div key={field} className="flex items-center justify-between">
                  <span className="text-gray-700 capitalize">{field.replace(/_/g, ' ')}</span>
                  <span className="font-medium text-gray-900">{value}</span>
                </div>
              ))
            ) : (
              <p className="text-sm text-gray-500">Run encrypted aggregation to view summary statistics.</p>
            )}
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Recent Encrypted Report Metadata</h3>
          <p className="text-sm text-gray-600 mt-1">
            Only de-identified metadata is visible globally. Individual vitals and lab values stay hidden.
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Patient Hash</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Lab</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Prediction</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Reasoning</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Encryption</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {!summary || summary.recent_reports.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-6 py-10 text-center text-gray-500">
                    No encrypted global reports yet.
                  </td>
                </tr>
              ) : (
                summary.recent_reports.map((report) => (
                  <tr key={report.id}>
                    <td className="px-6 py-4 text-sm text-gray-900">{report.patient_id_hash.slice(0, 12)}...</td>
                    <td className="px-6 py-4 text-sm text-gray-900">{report.lab_label.replace('_', ' ')}</td>
                    <td className="px-6 py-4 text-sm text-gray-900 capitalize">{report.prediction.replace('_', ' ')}</td>
                    <td className="px-6 py-4 text-sm text-gray-900">{(report.confidence * 100).toFixed(1)}%</td>
                    <td className="px-6 py-4 text-sm text-gray-600 max-w-md">{report.clinical_reasoning}</td>
                    <td className="px-6 py-4 text-sm text-emerald-700 font-medium">{report.encryption_scheme}</td>
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
