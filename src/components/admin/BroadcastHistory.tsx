import { useState, useEffect } from 'react';
import { History, CheckCircle, Clock, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';

interface Broadcast {
  id: string;
  created_at: string;
  global_model_version: number;
  initiated_by: string;
  status: string;
  labs_notified: number;
  labs_downloaded: number;
}

interface BroadcastHistoryProps {
  serverUrl: string;
}

export default function BroadcastHistory({ serverUrl }: BroadcastHistoryProps) {
  const [broadcasts, setBroadcasts] = useState<Broadcast[]>([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [labDetails, setLabDetails] = useState<Record<string, any>>({});

  useEffect(() => {
    loadHistory();
  }, [serverUrl]);

  const loadHistory = async () => {
    try {
      const response = await fetch(`${serverUrl}/admin/broadcast_history`);
      if (response.ok) {
        const data = await response.json();
        setBroadcasts(data.broadcasts || []);
      }
    } catch (error) {
      console.error('Error loading broadcast history:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleExpand = async (broadcastId: string) => {
    if (expanded === broadcastId) {
      setExpanded(null);
    } else {
      setExpanded(broadcastId);
      
      // Load lab details if not already loaded
      if (!labDetails[broadcastId]) {
        try {
          const response = await fetch(`${serverUrl}/admin/broadcast_status/${broadcastId}`);
          if (response.ok) {
            const data = await response.json();
            setLabDetails(prev => ({
              ...prev,
              [broadcastId]: data.labs
            }));
          }
        } catch (error) {
          console.error('Error loading broadcast details:', error);
        }
      }
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
            <CheckCircle className="w-3 h-3 mr-1" />
            Completed
          </span>
        );
      case 'in_progress':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
            <Clock className="w-3 h-3 mr-1 animate-pulse" />
            In Progress
          </span>
        );
      case 'failed':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
            <AlertCircle className="w-3 h-3 mr-1" />
            Failed
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
            {status}
          </span>
        );
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-200 rounded w-1/4"></div>
          <div className="h-20 bg-gray-100 rounded"></div>
          <div className="h-20 bg-gray-100 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-gray-900 flex items-center">
          <History className="w-5 h-5 mr-2 text-blue-600" />
          Model Broadcast History
        </h3>
        <button
          onClick={loadHistory}
          className="text-sm text-blue-600 hover:text-blue-800 font-medium"
        >
          Refresh
        </button>
      </div>

      {broadcasts.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          <History className="w-12 h-12 mx-auto mb-3 text-gray-300" />
          <p>No model broadcasts yet</p>
          <p className="text-sm mt-1">Push your first global model to see history here</p>
        </div>
      ) : (
        <div className="space-y-3">
          {broadcasts.map((broadcast) => (
            <div
              key={broadcast.id}
              className="border rounded-lg overflow-hidden transition-all hover:shadow-md"
            >
              {/* Broadcast Summary Row */}
              <div
                onClick={() => toggleExpand(broadcast.id)}
                className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-50"
              >
                <div className="flex items-center space-x-4">
                  <div className="flex-shrink-0">
                    <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center text-white font-bold">
                      v{broadcast.global_model_version}
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-medium text-gray-900">
                        Global Model v{broadcast.global_model_version}
                      </span>
                      {getStatusBadge(broadcast.status)}
                    </div>
                    <p className="text-sm text-gray-500">
                      {new Date(broadcast.created_at).toLocaleString()}
                      {broadcast.initiated_by && ` • by ${broadcast.initiated_by}`}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <div className="text-sm font-medium text-gray-900">
                      {broadcast.labs_downloaded}/{broadcast.labs_notified} downloaded
                    </div>
                    <div className="w-24 bg-gray-200 rounded-full h-2 mt-1">
                      <div
                        className="bg-green-500 h-2 rounded-full"
                        style={{ 
                          width: `${broadcast.labs_notified > 0 
                            ? (broadcast.labs_downloaded / broadcast.labs_notified) * 100 
                            : 0}%` 
                        }}
                      />
                    </div>
                  </div>
                  {expanded === broadcast.id ? (
                    <ChevronUp className="w-5 h-5 text-gray-400" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-gray-400" />
                  )}
                </div>
              </div>

              {/* Expanded Lab Details */}
              {expanded === broadcast.id && (
                <div className="border-t bg-gray-50 p-4">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3">Lab Download Status</h4>
                  {labDetails[broadcast.id] ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      {labDetails[broadcast.id].map((lab: any) => (
                        <div
                          key={lab.lab_label}
                          className={`flex items-center justify-between p-2 rounded border ${
                            lab.status === 'downloaded'
                              ? 'bg-green-50 border-green-200'
                              : 'bg-white border-gray-200'
                          }`}
                        >
                          <span className="font-medium text-sm">{lab.lab_label}</span>
                          <div className="flex items-center space-x-2">
                            {lab.auto_sync_enabled && (
                              <span className="text-xs text-blue-600">Auto</span>
                            )}
                            {lab.status === 'downloaded' ? (
                              <CheckCircle className="w-4 h-4 text-green-500" />
                            ) : (
                              <Clock className="w-4 h-4 text-orange-500" />
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-4 text-gray-500">
                      <div className="animate-spin w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full mx-auto"></div>
                      <p className="text-sm mt-2">Loading details...</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
