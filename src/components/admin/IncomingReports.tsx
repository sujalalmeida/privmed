import { useState, useEffect, useCallback } from 'react';
import { FileText, Eye, RefreshCw, CheckCircle, Clock, AlertCircle, Shield, BarChart3 } from 'lucide-react';
import EvaluateReport from './EvaluateReport';

interface Report {
  id: string;
  patient_id_hash: string;
  lab_label: string;
  diagnosis_label: string;
  confidence: number;
  clinical_reasoning: string;
  created_at: string;
  encrypted: boolean;
  encryption_scheme: string;
  status: 'pending' | 'reviewed';
  feedback: {
    agree: boolean;
    correct_diagnosis: number | null;
    correct_diagnosis_label?: string | null;
    reviewer_name: string | null;
    reviewed_at: string | null;
  } | null;
}

interface EncryptedSummary {
  total_reports: number;
  average_confidence: number | null;
  prediction_prevalence: Record<string, { count: number; share: number }>;
  aggregate_stats: {
    averages: Record<string, number>;
    counts: Record<string, number>;
    age_band: string | null;
  } | null;
  evaluated_by_lab: string;
}

const API_BASE = 'http://localhost:5001';

export default function IncomingReports() {
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null);
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [summaryLoading, setSummaryLoading] = useState(true);
  const [summary, setSummary] = useState<EncryptedSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<'all' | 'pending' | 'reviewed'>('all');

  const fetchReports = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE}/admin/reports?status=${statusFilter}&limit=100`);
      if (!response.ok) {
        throw new Error('Failed to fetch reports');
      }
      const data = await response.json();
      setReports(data.reports || []);
    } catch (err) {
      console.error('Error fetching reports:', err);
      setError(err instanceof Error ? err.message : 'Failed to load reports');
    } finally {
      setLoading(false);
    }
  }, [statusFilter]);

  const fetchEncryptedSummary = useCallback(async () => {
    try {
      setSummaryLoading(true);
      const response = await fetch(`${API_BASE}/admin/global_reports/summary?limit=100`);
      if (!response.ok) {
        throw new Error('Failed to fetch encrypted report summary');
      }
      const data = await response.json();
      setSummary(data);
    } catch (err) {
      console.error('Error fetching encrypted summary:', err);
      setError(err instanceof Error ? err.message : 'Failed to load encrypted report summary');
    } finally {
      setSummaryLoading(false);
    }
  }, []);

  const refreshAll = useCallback(async () => {
    await Promise.all([fetchReports(), fetchEncryptedSummary()]);
  }, [fetchReports, fetchEncryptedSummary]);

  const runEncryptedAggregation = async () => {
    try {
      setSummaryLoading(true);
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
      console.error('Error aggregating encrypted summary:', err);
      setError(err instanceof Error ? err.message : 'Failed to aggregate encrypted report statistics');
    } finally {
      setSummaryLoading(false);
    }
  };

  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'reviewed':
        return 'bg-green-100 text-green-800';
      case 'pending':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'reviewed':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'pending':
        return <Clock className="w-4 h-4 text-yellow-600" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-600" />;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const handleEvaluate = (reportId: string) => {
    setSelectedReportId(reportId);
  };

  const handleClose = () => {
    setSelectedReportId(null);
    refreshAll();
  };

  if (selectedReportId) {
    return <EvaluateReport reportId={selectedReportId} onClose={handleClose} />;
  }

  const pendingCount = reports.filter(r => r.status === 'pending').length;
  const reviewedCount = reports.filter(r => r.status === 'reviewed').length;
  const agreedCount = reports.filter(r => r.feedback?.agree === true).length;

  return (
    <div>
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Incoming Medical Reports</h2>
            <p className="text-sm text-gray-600 mt-1">De-identified encrypted reports with AI predictions for review</p>
          </div>
          <div className="flex items-center space-x-4">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as 'all' | 'pending' | 'reviewed')}
              className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Reports</option>
              <option value="pending">Pending Review</option>
              <option value="reviewed">Reviewed</option>
            </select>
            <button
              onClick={refreshAll}
              disabled={loading || summaryLoading}
              className="flex items-center px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 text-sm"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${(loading || summaryLoading) ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        {error && (
          <div className="px-6 py-4 bg-red-50 border-b border-red-200">
            <p className="text-red-700 text-sm">{error}</p>
          </div>
        )}

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Patient Hash</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Lab</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">AI Prediction</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence / Reasoning</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Encryption</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Action</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {loading ? (
                <tr>
                  <td colSpan={7} className="px-6 py-12 text-center text-gray-500">
                    <RefreshCw className="w-8 h-8 text-gray-400 mx-auto mb-4 animate-spin" />
                    Loading reports...
                  </td>
                </tr>
              ) : reports.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-6 py-12 text-center text-gray-500">
                    <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    No reports available
                  </td>
                </tr>
              ) : (
                reports.map((report) => (
                  <tr key={report.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {report.patient_id_hash ? `${report.patient_id_hash.slice(0, 12)}...` : 'Unknown'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {report.lab_label?.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) || 'Unknown'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 capitalize">
                      {report.diagnosis_label?.replace('_', ' ') || 'Unknown'}
                    </td>
                    <td className="px-6 py-4 text-sm">
                      <div>
                        <span className={`font-medium ${getConfidenceColor(report.confidence || 0)}`}>
                          {((report.confidence || 0) * 100).toFixed(1)}%
                        </span>
                        <p className="text-xs text-gray-500 mt-1 max-w-xs truncate">
                          {report.clinical_reasoning || 'Encrypted report metadata only'}
                        </p>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className="flex items-center text-emerald-700 font-medium">
                        <Shield className="w-4 h-4 mr-2" />
                        {report.encryption_scheme || 'HE-CKKS'}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className="flex items-center">
                        {getStatusIcon(report.status)}
                        <span className={`ml-2 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(report.status)}`}>
                          {report.status === 'reviewed' ? 'Reviewed' : 'Pending'}
                        </span>
                        {report.feedback && (
                          <span className={`ml-2 text-xs ${report.feedback.agree ? 'text-green-600' : 'text-red-600'}`}>
                            ({report.feedback.agree ? 'Agreed' : 'Disagreed'})
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <button
                        onClick={() => handleEvaluate(report.id)}
                        className="flex items-center text-blue-600 hover:text-blue-800 font-medium"
                      >
                        <Eye className="w-4 h-4 mr-1" />
                        {report.status === 'reviewed' ? 'View' : 'Evaluate'}
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-3 bg-blue-50 border border-blue-200 rounded-lg p-4">
        <p className="text-sm text-blue-900">
          Individual patient measurements remain homomorphically encrypted on the admin side. Review uses only de-identified metadata, AI prediction output, and encrypted cohort summaries.
        </p>
      </div>

      <div className="mt-6 bg-white rounded-lg shadow p-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <div className="flex items-center">
              <Shield className="w-5 h-5 text-blue-600 mr-2" />
              <h3 className="text-lg font-semibold text-gray-900">Homomorphically Encrypted Report Summary</h3>
            </div>
            <p className="text-sm text-gray-600 mt-1">
              Summary metrics are derived from encrypted computation without exposing raw patient values.
            </p>
          </div>
          <button
            onClick={runEncryptedAggregation}
            disabled={summaryLoading}
            className="flex items-center px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:bg-emerald-300 text-sm"
          >
            <BarChart3 className="w-4 h-4 mr-2" />
            {summaryLoading ? 'Computing...' : 'Recompute Encrypted Stats'}
          </button>
        </div>

        <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="rounded-lg border border-blue-100 bg-blue-50 p-4">
            <p className="text-sm text-blue-700">Encrypted Reports</p>
            <p className="text-2xl font-bold text-blue-900">{summary?.total_reports ?? 0}</p>
          </div>
          <div className="rounded-lg border border-emerald-100 bg-emerald-50 p-4">
            <p className="text-sm text-emerald-700">Average Confidence</p>
            <p className="text-2xl font-bold text-emerald-900">
              {summary?.average_confidence != null ? `${(summary.average_confidence * 100).toFixed(1)}%` : 'N/A'}
            </p>
          </div>
          <div className="rounded-lg border border-amber-100 bg-amber-50 p-4">
            <p className="text-sm text-amber-700">Age Band</p>
            <p className="text-2xl font-bold text-amber-900">{summary?.aggregate_stats?.age_band || 'N/A'}</p>
          </div>
          <div className="rounded-lg border border-purple-100 bg-purple-50 p-4">
            <p className="text-sm text-purple-700">Aggregate Evaluator</p>
            <p className="text-2xl font-bold text-purple-900">{(summary?.evaluated_by_lab || 'lab_A').replace('_', ' ')}</p>
          </div>
        </div>

        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">Disease Prevalence</h4>
            <div className="space-y-2">
              {summary && Object.keys(summary.prediction_prevalence).length > 0 ? (
                Object.entries(summary.prediction_prevalence).map(([label, value]) => (
                  <div key={label} className="flex items-center justify-between border border-gray-200 rounded-lg px-4 py-3">
                    <span className="capitalize text-gray-700">{label.replace('_', ' ')}</span>
                    <span className="font-medium text-gray-900">{value.count} ({(value.share * 100).toFixed(1)}%)</span>
                  </div>
                ))
              ) : (
                <p className="text-sm text-gray-500">No encrypted reports submitted yet.</p>
              )}
            </div>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">Encrypted Aggregate Metrics</h4>
            <div className="space-y-2">
              {summary?.aggregate_stats ? (
                Object.entries(summary.aggregate_stats.averages).map(([field, value]) => (
                  <div key={field} className="flex items-center justify-between border border-gray-200 rounded-lg px-4 py-3">
                    <span className="capitalize text-gray-700">{field.replace(/_/g, ' ')}</span>
                    <span className="font-medium text-gray-900">{value}</span>
                  </div>
                ))
              ) : (
                <p className="text-sm text-gray-500">Run encrypted computation to populate aggregate metrics.</p>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-6">
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
              <p className="text-2xl font-bold text-yellow-600">{pendingCount}</p>
            </div>
            <Clock className="w-10 h-10 text-yellow-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Reviewed</p>
              <p className="text-2xl font-bold text-green-600">{reviewedCount}</p>
            </div>
            <CheckCircle className="w-10 h-10 text-green-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Agreed</p>
              <p className="text-2xl font-bold text-blue-600">{agreedCount}</p>
            </div>
            <Shield className="w-10 h-10 text-blue-600" />
          </div>
        </div>
      </div>
    </div>
  );
}
