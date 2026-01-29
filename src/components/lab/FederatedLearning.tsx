import { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { getModelUpdatesByLab, addModelUpdate, getAllModelUpdates } from '../../utils/mockData';
import { Brain, Upload, TrendingUp, CheckCircle, Download, RefreshCw, AlertCircle, Bell, BellOff } from 'lucide-react';

interface GlobalModelInfo {
  available: boolean;
  global_model?: {
    version: number;
    model_type: string;
    created_at: string;
  };
  local_model?: {
    version: number | null;
    accuracy: number | null;
  };
  needs_update: boolean;
  has_downloaded: boolean;
}

interface ImprovementMetrics {
  accuracy_before: number;
  accuracy_after: number;
  improvement_percentage: number;
  absolute_improvement: number;
}

export default function FederatedLearning() {
  const { user } = useAuth();
  const [isSending, setIsSending] = useState(false);
  const [sendSuccess, setSendSuccess] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  
  // Global model download state
  const [globalModelInfo, setGlobalModelInfo] = useState<GlobalModelInfo | null>(null);
  const [isCheckingGlobal, setIsCheckingGlobal] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadSuccess, setDownloadSuccess] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [improvementMetrics, setImprovementMetrics] = useState<ImprovementMetrics | null>(null);
  
  // Auto-sync state
  const [autoSyncEnabled, setAutoSyncEnabled] = useState(false);
  const [isTogglingAutoSync, setIsTogglingAutoSync] = useState(false);
  const [pendingUpdate, setPendingUpdate] = useState<{ version: number; broadcast_id: string } | null>(null);
  const [showUpdateNotification, setShowUpdateNotification] = useState(false);
  
  const serverUrl = 'http://localhost:5001';

  const labUpdates = user ? getModelUpdatesByLab(user.id) : [];
  const allUpdates = getAllModelUpdates();
  const aggregatedUpdates = allUpdates.filter(u => u.isAggregated);
  const latestGlobalAccuracy = aggregatedUpdates.length > 0
    ? Math.max(...aggregatedUpdates.map(u => u.accuracy))
    : 0;

  // Check for global model updates
  const checkGlobalModel = async () => {
    if (!user) return;
    
    setIsCheckingGlobal(true);
    try {
      const labLabel = user?.labName || user?.email || 'lab_sim';
      const response = await fetch(`${serverUrl}/lab/get_global_model_info?lab_label=${encodeURIComponent(labLabel)}`);
      
      if (response.ok) {
        const data = await response.json();
        setGlobalModelInfo(data);
      }
    } catch (error) {
      console.error('Error checking global model:', error);
    } finally {
      setIsCheckingGlobal(false);
    }
  };

  // Download global model
  const handleDownloadGlobal = async () => {
    if (!user) return;
    
    setIsDownloading(true);
    setDownloadError(null);
    
    try {
      // Normalize lab label: 'Lab A' -> 'lab_A', 'Lab B' -> 'lab_B'
      const rawLabLabel = user?.labName || user?.email || 'lab_sim';
      const labLabel = rawLabLabel
        .replace(/^Lab\s+/i, 'lab_')  // 'Lab A' -> 'lab_A'
        .replace(/\s+/g, '_')         // Replace remaining spaces with underscores
        .replace(/[^a-zA-Z0-9_]/g, ''); // Remove special chars
      
      const response = await fetch(`${serverUrl}/lab/download_global_model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lab_label: labLabel }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to download global model');
      }
      
      const data = await response.json();
      
      // Store improvement metrics if available
      if (data.improvement_metrics) {
        setImprovementMetrics(data.improvement_metrics);
      }
      
      setDownloadSuccess(true);
      setPendingUpdate(null); // Clear pending update after download
      setShowUpdateNotification(false);
      
      setTimeout(() => {
        setDownloadSuccess(false);
        setImprovementMetrics(null); // Clear after showing
      }, 5000);
      
      // Acknowledge download to server if there was a broadcast
      if (pendingUpdate?.broadcast_id) {
        const labLabel = user?.labName || user?.email || 'lab_sim';
        await fetch(`${serverUrl}/lab/acknowledge_download`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            lab_label: labLabel, 
            broadcast_id: pendingUpdate.broadcast_id 
          }),
        });
      }
      
      // Refresh global model info
      checkGlobalModel();
    } catch (error) {
      setDownloadError(error instanceof Error ? error.message : 'Failed to download global model');
      setTimeout(() => setDownloadError(null), 5000);
    } finally {
      setIsDownloading(false);
    }
  };

  // Toggle auto-sync preference
  const handleToggleAutoSync = async () => {
    if (!user) return;
    
    setIsTogglingAutoSync(true);
    try {
      const labLabel = user?.labName || user?.email || 'lab_sim';
      const newValue = !autoSyncEnabled;
      
      const response = await fetch(`${serverUrl}/lab/enable_auto_sync`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lab_label: labLabel, enabled: newValue }),
      });
      
      if (response.ok) {
        setAutoSyncEnabled(newValue);
      }
    } catch (error) {
      console.error('Error toggling auto-sync:', error);
    } finally {
      setIsTogglingAutoSync(false);
    }
  };

  // Check for pushed updates
  const checkForPushedUpdates = async () => {
    if (!user) return;
    
    try {
      const labLabel = user?.labName || user?.email || 'lab_sim';
      const response = await fetch(`${serverUrl}/lab/check_for_updates?lab_label=${encodeURIComponent(labLabel)}`);
      
      if (response.ok) {
        const data = await response.json();
        if (data.new_model_available) {
          setPendingUpdate({ version: data.version, broadcast_id: data.broadcast_id });
          setShowUpdateNotification(true);
          
          // Auto-download if auto-sync is enabled
          if (autoSyncEnabled) {
            handleDownloadGlobal();
          }
        }
      }
    } catch (error) {
      console.error('Error checking for updates:', error);
    }
  };

  // Load auto-sync status on mount
  const loadAutoSyncStatus = async () => {
    if (!user) return;
    
    try {
      const labLabel = user?.labName || user?.email || 'lab_sim';
      const response = await fetch(`${serverUrl}/lab/get_auto_sync_status?lab_label=${encodeURIComponent(labLabel)}`);
      
      if (response.ok) {
        const data = await response.json();
        setAutoSyncEnabled(data.auto_sync_enabled || false);
      }
    } catch (error) {
      console.error('Error loading auto-sync status:', error);
    }
  };

  // Check for global model on mount and periodically
  useEffect(() => {
    checkGlobalModel();
    loadAutoSyncStatus();
    checkForPushedUpdates();
    
    const interval = setInterval(() => {
      checkGlobalModel();
      checkForPushedUpdates();
    }, 30000); // Check every 30 seconds
    
    return () => clearInterval(interval);
  }, [user]);

  const handleSendUpdate = async () => {
    if (!user) return;

    setIsSending(true);
    setSendError(null);

    try {
      // Normalize lab label: 'Lab A' -> 'lab_A', 'Lab B' -> 'lab_B'
      const rawLabLabel = user?.labName || user?.email || 'lab_sim';
      const labLabel = rawLabLabel
        .replace(/^Lab\s+/i, 'lab_')  // 'Lab A' -> 'lab_A'
        .replace(/\s+/g, '_')         // Replace remaining spaces with underscores
        .replace(/[^a-zA-Z0-9_]/g, ''); // Remove special chars

      const response = await fetch('http://127.0.0.1:5001/lab/send_model_update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lab_label: labLabel,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to send model update');
      }

      const data = await response.json();
      
      // Add to mock data for UI display
      const version = `v1.2.${labUpdates.length + 1}`;
      addModelUpdate({
        labId: user.id,
        labName: user.labName || 'Lab',
        modelVersion: version,
        accuracy: data.local_accuracy || 0.92,
        isAggregated: false,
      });

      setSendSuccess(true);
      setTimeout(() => setSendSuccess(false), 3000);
    } catch (error) {
      setSendError(error instanceof Error ? error.message : 'Failed to send model update');
      setTimeout(() => setSendError(null), 5000);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-gray-600">Local Model Updates</p>
              <p className="text-2xl font-bold text-gray-900">{labUpdates.length}</p>
            </div>
            <Brain className="w-10 h-10 text-blue-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-gray-600">Latest Local Accuracy</p>
              <p className="text-2xl font-bold text-green-600">
                {labUpdates.length > 0
                  ? `${(labUpdates[labUpdates.length - 1].accuracy * 100).toFixed(1)}%`
                  : 'N/A'}
              </p>
            </div>
            <TrendingUp className="w-10 h-10 text-green-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-gray-600">Global Model Accuracy</p>
              <p className="text-2xl font-bold text-blue-600">
                {latestGlobalAccuracy > 0
                  ? `${(latestGlobalAccuracy * 100).toFixed(1)}%`
                  : 'N/A'}
              </p>
            </div>
            <CheckCircle className="w-10 h-10 text-blue-600" />
          </div>
        </div>
      </div>

      {/* Update Notification Toast */}
      {showUpdateNotification && pendingUpdate && !autoSyncEnabled && (
        <div className="fixed top-4 right-4 z-50 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg shadow-2xl p-4 max-w-sm animate-bounce">
          <div className="flex items-start justify-between">
            <div className="flex items-center">
              <Bell className="w-6 h-6 mr-3 animate-pulse" />
              <div>
                <p className="font-semibold">New Global Model Available!</p>
                <p className="text-sm text-blue-100">Version {pendingUpdate.version} is ready to download</p>
              </div>
            </div>
            <button 
              onClick={() => setShowUpdateNotification(false)}
              className="text-white/80 hover:text-white ml-2"
            >
              ×
            </button>
          </div>
          <div className="mt-3 flex gap-2">
            <button
              onClick={handleDownloadGlobal}
              disabled={isDownloading}
              className="flex-1 bg-white text-indigo-600 py-2 px-4 rounded font-medium text-sm hover:bg-blue-50"
            >
              Download Now
            </button>
            <button
              onClick={() => setShowUpdateNotification(false)}
              className="px-3 py-2 text-white/80 hover:text-white text-sm"
            >
              Later
            </button>
          </div>
        </div>
      )}

      {/* Auto-Sync Toggle Card */}
      <div className="bg-white rounded-lg shadow p-4 border-l-4 border-indigo-500">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            {autoSyncEnabled ? (
              <Bell className="w-6 h-6 text-indigo-600 mr-3" />
            ) : (
              <BellOff className="w-6 h-6 text-gray-400 mr-3" />
            )}
            <div>
              <h4 className="font-semibold text-gray-900">Auto-Sync Global Models</h4>
              <p className="text-sm text-gray-600">
                {autoSyncEnabled 
                  ? 'New global models will be downloaded automatically when pushed by admin'
                  : 'You will receive a notification when new models are available'}
              </p>
            </div>
          </div>
          <button
            onClick={handleToggleAutoSync}
            disabled={isTogglingAutoSync}
            className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 ${
              autoSyncEnabled ? 'bg-indigo-600' : 'bg-gray-200'
            } ${isTogglingAutoSync ? 'opacity-50' : ''}`}
          >
            <span
              className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                autoSyncEnabled ? 'translate-x-5' : 'translate-x-0'
              }`}
            />
          </button>
        </div>
        {autoSyncEnabled && (
          <div className="mt-2 flex items-center text-xs text-green-600">
            <CheckCircle className="w-3 h-3 mr-1" />
            Auto-sync is enabled - you're always up to date
          </div>
        )}
      </div>

      {/* Global Model Download Section */}
      {globalModelInfo?.available && (
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border-2 border-indigo-200 rounded-lg shadow-lg p-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-1 flex items-center">
                <Brain className="w-5 h-5 mr-2 text-indigo-600" />
                Global Model Available
              </h3>
              <p className="text-sm text-gray-600">
                Download the latest aggregated model trained on data from all participating labs
              </p>
            </div>
            {globalModelInfo.needs_update && (
              <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-orange-100 text-orange-800 animate-pulse">
                New Update
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white rounded-lg p-4 shadow">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">Global Model</h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Version:</span>
                  <span className="font-bold text-indigo-600">v{globalModelInfo.global_model?.version}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Model Type:</span>
                  <span className="font-medium">{globalModelInfo.global_model?.model_type}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Created:</span>
                  <span className="text-xs">{new Date(globalModelInfo.global_model?.created_at || '').toLocaleDateString()}</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 shadow">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">Your Local Model</h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Version:</span>
                  <span className="font-medium">{globalModelInfo.local_model?.version ? `v${globalModelInfo.local_model.version}` : 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Accuracy:</span>
                  <span className="font-medium text-green-600">
                    {globalModelInfo.local_model?.accuracy 
                      ? `${(globalModelInfo.local_model.accuracy * 100).toFixed(1)}%`
                      : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Status:</span>
                  <span className={`text-xs font-medium ${globalModelInfo.needs_update ? 'text-orange-600' : 'text-green-600'}`}>
                    {globalModelInfo.needs_update ? 'Update Available' : 'Up to date'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {downloadSuccess && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
              <div className="flex items-center text-green-800 font-medium mb-2">
                <CheckCircle className="w-5 h-5 mr-2" />
                Global model downloaded successfully!
              </div>
              
              {improvementMetrics && (
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div className="bg-white rounded-lg p-3 shadow-sm">
                    <p className="text-xs text-gray-600 mb-1">Previous Accuracy</p>
                    <p className="text-lg font-bold text-gray-900">
                      {(improvementMetrics.accuracy_before * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className="bg-white rounded-lg p-3 shadow-sm">
                    <p className="text-xs text-gray-600 mb-1">New Accuracy</p>
                    <p className="text-lg font-bold text-green-600">
                      {(improvementMetrics.accuracy_after * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className="col-span-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg p-3 text-white shadow-sm">
                    <p className="text-xs mb-1 opacity-90">Performance Improvement</p>
                    <p className="text-2xl font-bold">
                      {improvementMetrics.improvement_percentage > 0 ? '+' : ''}
                      {improvementMetrics.improvement_percentage.toFixed(2)}%
                    </p>
                    <p className="text-xs opacity-90 mt-1">
                      ({improvementMetrics.absolute_improvement > 0 ? '+' : ''}
                      {(improvementMetrics.absolute_improvement * 100).toFixed(2)} points)
                    </p>
                  </div>
                </div>
              )}
              
              {!improvementMetrics && (
                <p className="text-sm text-green-700 mt-1">
                  You can now use this model for predictions.
                </p>
              )}
            </div>
          )}

          {downloadError && (
            <div className="flex items-center p-3 bg-red-50 border border-red-200 rounded-lg text-red-800 text-sm mb-4">
              <AlertCircle className="w-4 h-4 mr-2" />
              {downloadError}
            </div>
          )}

          <div className="flex gap-3">
            <button
              onClick={handleDownloadGlobal}
              disabled={isDownloading}
              className="flex-1 bg-indigo-600 text-white py-3 px-4 rounded-lg hover:bg-indigo-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed font-medium flex items-center justify-center"
            >
              <Download className="w-5 h-5 mr-2" />
              {isDownloading ? 'Downloading...' : 'Download Global Model'}
            </button>
            
            <button
              onClick={checkGlobalModel}
              disabled={isCheckingGlobal}
              className="bg-white border-2 border-indigo-200 text-indigo-600 py-3 px-4 rounded-lg hover:bg-indigo-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium flex items-center justify-center"
            >
              <RefreshCw className={`w-5 h-5 ${isCheckingGlobal ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
      )}

      <div className="max-w-md mx-auto">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Send Model Update</h3>
          <p className="text-sm text-gray-600 mb-4">
            Send encrypted model gradients/weights to the central server for federated aggregation. Your local data never leaves the lab.
          </p>

          {sendSuccess && (
            <div className="flex items-center p-3 bg-green-50 border border-green-200 rounded-lg text-green-800 text-sm mb-4">
              <CheckCircle className="w-4 h-4 mr-2" />
              Model update sent successfully!
            </div>
          )}

          {sendError && (
            <div className="flex items-center p-3 bg-red-50 border border-red-200 rounded-lg text-red-800 text-sm mb-4">
              <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              {sendError}
            </div>
          )}

          <button
            onClick={handleSendUpdate}
            disabled={isSending}
            className="w-full bg-green-600 text-white py-3 px-4 rounded-lg hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed font-medium flex items-center justify-center"
          >
            <Upload className="w-5 h-5 mr-2" />
            {isSending ? 'Sending Update...' : 'Send Model Update'}
          </button>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Model Update History</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Version
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Accuracy
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {labUpdates.length === 0 ? (
                <tr>
                  <td colSpan={4} className="px-6 py-12 text-center text-gray-500">
                    <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    No model updates yet
                  </td>
                </tr>
              ) : (
                labUpdates.map((update) => (
                  <tr key={update.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {update.modelVersion}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {(update.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                        update.isAggregated
                          ? 'bg-green-100 text-green-800'
                          : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {update.isAggregated ? 'Aggregated' : 'Pending'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(update.createdAt).toLocaleDateString()}
                    </td>
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
