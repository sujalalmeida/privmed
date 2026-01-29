import { useState, useEffect } from 'react';
import { TrendingUp, Activity, Users, Database, AlertCircle, CheckCircle2 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface Round {
  round: number;
  global_accuracy: number | null;
  timestamp: string;
  labs_participated: number;
  total_samples: number;
  model_type: string | null;
}

interface ConvergenceStats {
  total_rounds: number;
  initial_accuracy: number | null;
  current_accuracy: number | null;
  accuracy_improvement: number;
  average_accuracy: number;
  best_accuracy: number;
  worst_accuracy: number;
  convergence_rate: number;
}

interface ParticipationStats {
  total_unique_labs: number;
  average_participation: number;
  labs_list: string[];
}

export default function RoundHistory() {
  const [rounds, setRounds] = useState<Round[]>([]);
  const [convergenceStats, setConvergenceStats] = useState<ConvergenceStats | null>(null);
  const [participationStats, setParticipationStats] = useState<ParticipationStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const serverUrl = 'http://localhost:5001';

  const loadRoundHistory = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${serverUrl}/admin/get_round_history`);
      const data = await response.json();

      if (response.ok) {
        setRounds(data.rounds || []);
        setConvergenceStats(data.convergence_stats || null);
        setParticipationStats(data.participation_stats || null);
        setError('');
      } else {
        setError(data.error || 'Failed to load round history');
      }
    } catch (err) {
      setError('Network error: Could not fetch round history');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadRoundHistory();
    // Auto-refresh every 30 seconds
    const interval = setInterval(loadRoundHistory, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading && rounds.length === 0) {
    return (
      <div className="card p-6">
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-6 w-6 border-2 border-primary-500 border-t-transparent"></div>
          <span className="ml-3 text-neutral-500 text-sm">Loading round history...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-6">
        <div className="alert-error">
          <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
          <span>{error}</span>
        </div>
      </div>
    );
  }

  if (rounds.length === 0) {
    return (
      <div className="card p-6">
        <h2 className="text-base font-semibold text-neutral-900 mb-4">Federated Learning Rounds</h2>
        <div className="text-center py-12 text-neutral-500">
          <Activity className="w-12 h-12 mx-auto mb-3 text-neutral-300" />
          <p className="text-sm">No federated learning rounds completed yet.</p>
          <p className="text-xs text-neutral-400 mt-1">Aggregate models to start tracking FL performance.</p>
        </div>
      </div>
    );
  }

  // Prepare chart data
  const chartData = rounds.map(r => ({
    round: r.round,
    accuracy: r.global_accuracy ? (r.global_accuracy * 100).toFixed(2) : null,
    labs: r.labs_participated,
    samples: r.total_samples
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-neutral-900 mb-1">Federated Learning Performance</h2>
        <p className="text-sm text-neutral-500">Track model convergence and participation across FL rounds</p>
      </div>

      {/* Key Metrics */}
      {convergenceStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="card p-5 border-l-3 border-l-primary-500">
            <div className="flex items-center justify-between mb-2">
              <div className="w-9 h-9 bg-primary-50 rounded-lg flex items-center justify-center">
                <Activity className="w-5 h-5 text-primary-500" />
              </div>
              <span className="text-xs font-medium text-neutral-500 uppercase">Total Rounds</span>
            </div>
            <div className="text-2xl font-semibold text-neutral-900">{convergenceStats.total_rounds}</div>
            <div className="text-xs text-neutral-500 mt-1">Aggregations completed</div>
          </div>

          <div className="card p-5 border-l-3 border-l-success-500">
            <div className="flex items-center justify-between mb-2">
              <div className="w-9 h-9 bg-success-50 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-success-500" />
              </div>
              <span className="text-xs font-medium text-neutral-500 uppercase">Current Accuracy</span>
            </div>
            <div className="text-2xl font-semibold text-neutral-900">
              {convergenceStats.current_accuracy 
                ? `${(convergenceStats.current_accuracy * 100).toFixed(1)}%`
                : 'N/A'}
            </div>
            <div className="text-xs text-neutral-500 mt-1">
              {convergenceStats.accuracy_improvement > 0 && (
                <span className="flex items-center text-success-500">
                  <TrendingUp className="w-3 h-3 mr-1" />
                  +{(convergenceStats.accuracy_improvement * 100).toFixed(2)}% from start
                </span>
              )}
            </div>
          </div>

          <div className="card p-5 border-l-3 border-l-heart-disease">
            <div className="flex items-center justify-between mb-2">
              <div className="w-9 h-9 bg-purple-50 rounded-lg flex items-center justify-center">
                <Users className="w-5 h-5 text-heart-disease" />
              </div>
              <span className="text-xs font-medium text-neutral-500 uppercase">Avg Participation</span>
            </div>
            <div className="text-2xl font-semibold text-neutral-900">
              {participationStats?.average_participation.toFixed(1) || '0'} labs
            </div>
            <div className="text-xs text-neutral-500 mt-1">
              {participationStats?.total_unique_labs || 0} total unique labs
            </div>
          </div>

          <div className="card p-5 border-l-3 border-l-warning-500">
            <div className="flex items-center justify-between mb-2">
              <div className="w-9 h-9 bg-warning-50 rounded-lg flex items-center justify-center">
                <Database className="w-5 h-5 text-warning-500" />
              </div>
              <span className="text-xs font-medium text-neutral-500 uppercase">Best Accuracy</span>
            </div>
            <div className="text-2xl font-semibold text-neutral-900">
              {convergenceStats.best_accuracy 
                ? `${(convergenceStats.best_accuracy * 100).toFixed(1)}%`
                : 'N/A'}
            </div>
            <div className="text-xs text-neutral-500 mt-1">Peak performance</div>
          </div>
        </div>
      )}

      {/* Performance Chart */}
      <div className="card p-6">
        <h3 className="text-base font-semibold text-neutral-900 mb-4">Accuracy Progression</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E2E8EA" />
            <XAxis 
              dataKey="round" 
              label={{ value: 'Round Number', position: 'insideBottom', offset: -5 }}
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }}
              domain={[0, 100]}
              tick={{ fontSize: 12 }}
            />
            <Tooltip 
              content={({ active, payload }: any) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="card p-3 text-sm">
                      <p className="font-medium text-neutral-900">Round {data.round}</p>
                      <p className="text-success-500">Accuracy: {data.accuracy}%</p>
                      <p className="text-primary-500">Labs: {data.labs}</p>
                      <p className="text-warning-500">Samples: {data.samples}</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="accuracy" 
              stroke="#0F6E66" 
              strokeWidth={2}
              dot={{ fill: '#0F6E66', r: 4 }}
              name="Global Accuracy (%)"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Round History Table */}
      <div className="card overflow-hidden">
        <div className="px-6 py-4 border-b border-neutral-200">
          <h3 className="text-base font-semibold text-neutral-900">Round History</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-neutral-200">
            <thead className="bg-neutral-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wide">
                  Round
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wide">
                  Timestamp
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wide">
                  Global Accuracy
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wide">
                  Labs Participated
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wide">
                  Total Samples
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wide">
                  Model Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wide">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-neutral-200">
              {rounds.slice().reverse().map((round) => (
                <tr key={round.round} className="hover:bg-neutral-50 transition-colors">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="font-medium text-primary-500">#{round.round}</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-500">
                    {round.timestamp ? new Date(round.timestamp).toLocaleString() : 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-sm font-medium text-success-500">
                      {round.global_accuracy 
                        ? `${(round.global_accuracy * 100).toFixed(2)}%`
                        : 'N/A'}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="badge-neutral">
                      <Users className="w-3 h-3 mr-1" />
                      {round.labs_participated}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-900">
                    {round.total_samples}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-500">
                    {round.model_type || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="badge-success">
                      <CheckCircle2 className="w-3 h-3 mr-1" />
                      Completed
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
