import { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Users, ThumbsUp, ThumbsDown, RefreshCw, AlertTriangle } from 'lucide-react';

interface OverallStats {
  total: number;
  agreed: number;
  disagreed: number;
  agreement_rate: number;
  agreement_percent: string;
}

interface LabStats {
  lab_label: string;
  display_name: string;
  total: number;
  agreed: number;
  disagreed: number;
  agreement_rate: number;
  agreement_percent: string;
}

interface WeeklyTrend {
  week: string;
  start_date: string;
  end_date: string;
  total: number;
  agreed: number;
  agreement_rate: number;
}

interface FeedbackStatsData {
  overall: OverallStats;
  per_lab: LabStats[];
  recent_trend: WeeklyTrend[];
}

const API_BASE = 'http://localhost:5001';

export default function FeedbackStats() {
  const [stats, setStats] = useState<FeedbackStatsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE}/admin/feedback_stats`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch feedback statistics');
      }
      
      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error('Error fetching feedback stats:', err);
      setError(err instanceof Error ? err.message : 'Failed to load statistics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  const getAgreementColor = (rate: number) => {
    if (rate >= 0.9) return 'text-green-600';
    if (rate >= 0.7) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getAgreementBgColor = (rate: number) => {
    if (rate >= 0.9) return 'bg-green-100';
    if (rate >= 0.7) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="w-8 h-8 text-gray-400 animate-spin" />
          <span className="ml-3 text-gray-600">Loading feedback statistics...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="text-center py-12">
          <AlertTriangle className="w-12 h-12 text-red-400 mx-auto mb-4" />
          <p className="text-red-600">{error}</p>
          <button
            onClick={fetchStats}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!stats) {
    return null;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Doctor Feedback Statistics</h2>
          <p className="text-sm text-gray-600 mt-1">Agreement rates between doctors and AI predictions</p>
        </div>
        <button
          onClick={fetchStats}
          disabled={loading}
          className="flex items-center px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 text-sm"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Overall Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Reviews</p>
              <p className="text-3xl font-bold text-gray-900">{stats.overall.total}</p>
            </div>
            <Users className="w-10 h-10 text-blue-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Agreed with AI</p>
              <p className="text-3xl font-bold text-green-600">{stats.overall.agreed}</p>
            </div>
            <ThumbsUp className="w-10 h-10 text-green-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Disagreed</p>
              <p className="text-3xl font-bold text-red-600">{stats.overall.disagreed}</p>
            </div>
            <ThumbsDown className="w-10 h-10 text-red-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Agreement Rate</p>
              <p className={`text-3xl font-bold ${getAgreementColor(stats.overall.agreement_rate)}`}>
                {stats.overall.agreement_percent}
              </p>
            </div>
            <TrendingUp className={`w-10 h-10 ${getAgreementColor(stats.overall.agreement_rate)}`} />
          </div>
        </div>
      </div>

      {/* Per-Lab Statistics */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <div className="flex items-center">
            <BarChart3 className="w-5 h-5 text-gray-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Per-Lab Agreement Rates</h3>
          </div>
          <p className="text-sm text-gray-600 mt-1">
            Labs with low agreement rates may need additional review
          </p>
        </div>

        {stats.per_lab.length === 0 ? (
          <div className="px-6 py-12 text-center text-gray-500">
            No feedback data available yet
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Lab
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Total Reviews
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Agreed
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Disagreed
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Agreement Rate
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {stats.per_lab.map((lab) => (
                  <tr key={lab.lab_label} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm font-medium text-gray-900">{lab.display_name}</span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {lab.total}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm text-green-600 font-medium">{lab.agreed}</span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm text-red-600 font-medium">{lab.disagreed}</span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="w-24 bg-gray-200 rounded-full h-2 mr-3">
                          <div
                            className={`h-2 rounded-full ${
                              lab.agreement_rate >= 0.9 ? 'bg-green-500' :
                              lab.agreement_rate >= 0.7 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${lab.agreement_rate * 100}%` }}
                          />
                        </div>
                        <span className={`text-sm font-medium ${getAgreementColor(lab.agreement_rate)}`}>
                          {lab.agreement_percent}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {lab.agreement_rate < 0.6 ? (
                        <span className="flex items-center text-xs text-red-600">
                          <AlertTriangle className="w-4 h-4 mr-1" />
                          Needs Review
                        </span>
                      ) : lab.agreement_rate < 0.8 ? (
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getAgreementBgColor(lab.agreement_rate)} ${getAgreementColor(lab.agreement_rate)}`}>
                          Moderate
                        </span>
                      ) : (
                        <span className="px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                          Good
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Weekly Trend */}
      {stats.recent_trend.length > 0 && stats.recent_trend.some(w => w.total > 0) && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center mb-4">
            <TrendingUp className="w-5 h-5 text-gray-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Weekly Trend (Last 4 Weeks)</h3>
          </div>

          <div className="grid grid-cols-4 gap-4">
            {stats.recent_trend.map((week, idx) => (
              <div key={idx} className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-xs text-gray-500 mb-1">{week.start_date}</p>
                <p className="text-sm font-medium text-gray-700">{week.week}</p>
                <p className={`text-2xl font-bold mt-2 ${getAgreementColor(week.agreement_rate)}`}>
                  {week.total > 0 ? `${(week.agreement_rate * 100).toFixed(0)}%` : 'N/A'}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  {week.total} reviews
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Aggregation Impact Notice */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start">
          <BarChart3 className="w-5 h-5 text-blue-600 mt-0.5 mr-3" />
          <div>
            <h4 className="font-medium text-blue-900">Aggregation Weighting</h4>
            <p className="text-sm text-blue-800 mt-1">
              Lab agreement rates are used to weight contributions during model aggregation.
              Labs with higher agreement rates have more influence on the global model.
              Labs with agreement rates below 60% are flagged for review.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
