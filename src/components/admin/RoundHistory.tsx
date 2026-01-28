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
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-gray-600">Loading round history...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center text-red-600">
          <AlertCircle className="w-5 h-5 mr-2" />
          <span>{error}</span>
        </div>
      </div>
    );
  }

  if (rounds.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Federated Learning Rounds</h2>
        <div className="text-center py-12 text-gray-500">
          <Activity className="w-16 h-16 mx-auto mb-4 text-gray-300" />
          <p>No federated learning rounds completed yet.</p>
          <p className="text-sm mt-2">Aggregate models to start tracking FL performance.</p>
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
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Federated Learning Performance</h2>
        <p className="text-gray-600">Track model convergence and participation across FL rounds</p>
      </div>

      {/* Key Metrics */}
      {convergenceStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg shadow-lg p-6 text-white">
            <div className="flex items-center justify-between mb-2">
              <Activity className="w-8 h-8 opacity-80" />
              <span className="text-sm font-medium opacity-90">Total Rounds</span>
            </div>
            <div className="text-3xl font-bold">{convergenceStats.total_rounds}</div>
            <div className="text-xs opacity-75 mt-1">Aggregations completed</div>
          </div>

          <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg shadow-lg p-6 text-white">
            <div className="flex items-center justify-between mb-2">
              <TrendingUp className="w-8 h-8 opacity-80" />
              <span className="text-sm font-medium opacity-90">Current Accuracy</span>
            </div>
            <div className="text-3xl font-bold">
              {convergenceStats.current_accuracy 
                ? `${(convergenceStats.current_accuracy * 100).toFixed(1)}%`
                : 'N/A'}
            </div>
            <div className="text-xs opacity-75 mt-1">
              {convergenceStats.accuracy_improvement > 0 && (
                <span className="flex items-center">
                  <TrendingUp className="w-3 h-3 mr-1" />
                  +{(convergenceStats.accuracy_improvement * 100).toFixed(2)}% from start
                </span>
              )}
            </div>
          </div>

          <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg shadow-lg p-6 text-white">
            <div className="flex items-center justify-between mb-2">
              <Users className="w-8 h-8 opacity-80" />
              <span className="text-sm font-medium opacity-90">Avg Participation</span>
            </div>
            <div className="text-3xl font-bold">
              {participationStats?.average_participation.toFixed(1) || '0'} labs
            </div>
            <div className="text-xs opacity-75 mt-1">
              {participationStats?.total_unique_labs || 0} total unique labs
            </div>
          </div>

          <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-lg shadow-lg p-6 text-white">
            <div className="flex items-center justify-between mb-2">
              <Database className="w-8 h-8 opacity-80" />
              <span className="text-sm font-medium opacity-90">Best Accuracy</span>
            </div>
            <div className="text-3xl font-bold">
              {convergenceStats.best_accuracy 
                ? `${(convergenceStats.best_accuracy * 100).toFixed(1)}%`
                : 'N/A'}
            </div>
            <div className="text-xs opacity-75 mt-1">Peak performance</div>
          </div>
        </div>
      )}

      {/* Performance Chart */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Accuracy Progression</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="round" 
              label={{ value: 'Round Number', position: 'insideBottom', offset: -5 }}
            />
            <YAxis 
              label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }}
              domain={[0, 100]}
            />
            <Tooltip 
              content={({ active, payload }: any) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-3">
                      <p className="font-semibold">Round {data.round}</p>
                      <p className="text-green-600">Accuracy: {data.accuracy}%</p>
                      <p className="text-blue-600">Labs: {data.labs}</p>
                      <p className="text-orange-600">Samples: {data.samples}</p>
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
              stroke="#10b981" 
              strokeWidth={3}
              dot={{ fill: '#10b981', r: 5 }}
              name="Global Accuracy (%)"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Round History Table */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Round History</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Round
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Timestamp
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Global Accuracy
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Labs Participated
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Total Samples
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Model Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {rounds.slice().reverse().map((round) => (
                <tr key={round.round} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="font-semibold text-indigo-600">#{round.round}</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {round.timestamp ? new Date(round.timestamp).toLocaleString() : 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-sm font-semibold text-green-600">
                      {round.global_accuracy 
                        ? `${(round.global_accuracy * 100).toFixed(2)}%`
                        : 'N/A'}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      <Users className="w-3 h-3 mr-1" />
                      {round.labs_participated}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {round.total_samples}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {round.model_type || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
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
