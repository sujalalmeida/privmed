import { useState, useEffect } from 'react';
import { X, Send, CheckCircle, Clock, AlertCircle, Loader2 } from 'lucide-react';

interface Lab {
  lab_label: string;
  status: string;
  notified_at: string | null;
  downloaded_at: string | null;
  auto_sync_enabled: boolean;
}

interface PushModelModalProps {
  isOpen: boolean;
  onClose: () => void;
  serverUrl: string;
  globalModelVersion: number;
}

export default function PushModelModal({ isOpen, onClose, serverUrl, globalModelVersion }: PushModelModalProps) {
  const [isPushing, setIsPushing] = useState(false);
  const [broadcastId, setBroadcastId] = useState<string | null>(null);
  const [labs, setLabs] = useState<Lab[]>([]);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState(false);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);

  const handlePush = async () => {
    setIsPushing(true);
    setError('');
    setSuccess(false);
    
    try {
      const response = await fetch(`${serverUrl}/admin/push_global_model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ initiated_by: 'admin' })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to push model');
      }
      
      const data = await response.json();
      setBroadcastId(data.broadcast_id);
      
      // Initialize labs from response
      const initialLabs = data.lab_labels.map((label: string) => ({
        lab_label: label,
        status: 'notified',
        notified_at: new Date().toISOString(),
        downloaded_at: null,
        auto_sync_enabled: false
      }));
      setLabs(initialLabs);
      
      // Start polling for status
      const interval = setInterval(() => {
        pollStatus(data.broadcast_id);
      }, 2000);
      setPollingInterval(interval);
      
    } catch (err: any) {
      setError(err.message || 'Failed to push model');
    } finally {
      setIsPushing(false);
    }
  };

  const pollStatus = async (bId: string) => {
    try {
      const response = await fetch(`${serverUrl}/admin/broadcast_status/${bId}`);
      if (response.ok) {
        const data = await response.json();
        setLabs(data.labs);
        
        // Check if all labs have downloaded
        if (data.status === 'completed') {
          setSuccess(true);
          if (pollingInterval) {
            clearInterval(pollingInterval);
            setPollingInterval(null);
          }
        }
      }
    } catch (err) {
      console.error('Error polling status:', err);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'downloaded':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'notified':
        return <Clock className="w-5 h-5 text-orange-500" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'downloaded':
        return 'Downloaded ✓';
      case 'notified':
        return 'Pending...';
      case 'failed':
        return 'Failed';
      default:
        return 'Waiting';
    }
  };

  const downloadedCount = labs.filter(l => l.status === 'downloaded').length;
  const progressPercentage = labs.length > 0 ? (downloadedCount / labs.length) * 100 : 0;

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-4 flex items-center justify-between">
          <div className="text-white">
            <h2 className="text-xl font-bold">Push Global Model</h2>
            <p className="text-sm text-blue-100">Broadcast v{globalModelVersion} to all labs</p>
          </div>
          <button
            onClick={onClose}
            className="text-white/80 hover:text-white transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {error && (
            <div className="mb-4 p-4 bg-red-50 border-l-4 border-red-500 text-red-700 flex items-start">
              <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {success && (
            <div className="mb-4 p-4 bg-green-50 border-l-4 border-green-500 text-green-700 flex items-start">
              <CheckCircle className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5" />
              <span>Global model pushed successfully to all labs!</span>
            </div>
          )}

          {!broadcastId ? (
            // Pre-push state
            <div className="text-center py-6">
              <Send className="w-16 h-16 mx-auto mb-4 text-blue-500" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                Ready to Push Global Model v{globalModelVersion}
              </h3>
              <p className="text-gray-600 mb-6">
                This will notify all participating labs that a new global model is available for download.
              </p>
              <button
                onClick={handlePush}
                disabled={isPushing}
                className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 px-8 rounded-lg font-semibold
                         hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 
                         disabled:cursor-not-allowed transition-all duration-200 flex items-center mx-auto"
              >
                {isPushing ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Pushing...
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5 mr-2" />
                    Push to All Labs
                  </>
                )}
              </button>
            </div>
          ) : (
            // Post-push state with progress
            <div>
              {/* Progress Bar */}
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Download Progress</span>
                  <span className="text-sm font-bold text-blue-600">
                    {downloadedCount}/{labs.length} labs
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-indigo-500 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${progressPercentage}%` }}
                  />
                </div>
              </div>

              {/* Lab Status List */}
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {labs.map((lab) => (
                  <div
                    key={lab.lab_label}
                    className={`flex items-center justify-between p-3 rounded-lg border-2 transition-all ${
                      lab.status === 'downloaded'
                        ? 'border-green-200 bg-green-50'
                        : lab.status === 'notified'
                        ? 'border-orange-200 bg-orange-50'
                        : 'border-gray-200 bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center">
                      {getStatusIcon(lab.status)}
                      <span className="ml-3 font-medium text-gray-900">{lab.lab_label}</span>
                      {lab.auto_sync_enabled && (
                        <span className="ml-2 px-2 py-0.5 text-xs bg-blue-100 text-blue-700 rounded-full">
                          Auto-sync
                        </span>
                      )}
                    </div>
                    <span className={`text-sm font-medium ${
                      lab.status === 'downloaded' ? 'text-green-600' :
                      lab.status === 'notified' ? 'text-orange-600' : 'text-gray-500'
                    }`}>
                      {getStatusText(lab.status)}
                    </span>
                  </div>
                ))}
              </div>

              {labs.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  <Loader2 className="w-8 h-8 mx-auto mb-2 animate-spin" />
                  <p>Notifying labs...</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="bg-gray-50 px-6 py-4 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-700 hover:text-gray-900 font-medium transition-colors"
          >
            {success ? 'Done' : 'Close'}
          </button>
        </div>
      </div>
    </div>
  );
}
