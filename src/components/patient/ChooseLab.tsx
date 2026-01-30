import { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { Building2, CheckCircle, TrendingUp, Clock, AlertCircle } from 'lucide-react';

interface Lab {
  lab_id: string;
  display_name: string;
  accuracy: number;
  accuracy_percent: string;
  last_updated: string | null;
  num_patients: number;
}

interface SelectedLab {
  lab_id: string;
  display_name: string;
  accuracy: number | null;
  accuracy_percent: string;
  selected_at?: string;
}

export default function ChooseLab() {
  const { user } = useAuth();
  const [labs, setLabs] = useState<Lab[]>([]);
  const [selectedLab, setSelectedLab] = useState<SelectedLab | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSelecting, setIsSelecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const serverUrl = 'http://127.0.0.1:5001';

  // Fetch available labs and current selection on mount
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        // Fetch available labs
        const labsResponse = await fetch(`${serverUrl}/patient/labs`);
        if (!labsResponse.ok) {
          throw new Error('Failed to fetch labs');
        }
        const labsData = await labsResponse.json();
        setLabs(labsData.labs || []);
        
        // Fetch current selection
        const patientId = user?.id || user?.email || 'anonymous';
        const selectionResponse = await fetch(`${serverUrl}/patient/selected_lab?patient_id=${encodeURIComponent(patientId)}`);
        if (selectionResponse.ok) {
          const selectionData = await selectionResponse.json();
          if (selectionData.has_selection) {
            setSelectedLab(selectionData.selected_lab);
          } else {
            // Check localStorage as fallback
            const stored = localStorage.getItem('privmed_selected_lab');
            if (stored) {
              try {
                setSelectedLab(JSON.parse(stored));
              } catch {
                // Invalid stored data
              }
            }
          }
        }
      } catch (err) {
        console.error('Error fetching labs:', err);
        setError('Unable to load labs. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchData();
  }, [user, serverUrl]);

  const handleSelectLab = async (lab: Lab) => {
    setIsSelecting(true);
    setError(null);
    setSuccessMessage(null);
    
    try {
      const patientId = user?.id || user?.email || 'anonymous';
      
      const response = await fetch(`${serverUrl}/patient/select_lab`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: patientId,
          lab_id: lab.lab_id
        })
      });
      
      const data = await response.json();
      
      if (response.ok && data.success) {
        const selected: SelectedLab = {
          lab_id: lab.lab_id,
          display_name: lab.display_name,
          accuracy: lab.accuracy,
          accuracy_percent: lab.accuracy_percent
        };
        setSelectedLab(selected);
        
        // Store in localStorage as backup
        localStorage.setItem('privmed_selected_lab', JSON.stringify(selected));
        
        setSuccessMessage(`Successfully selected ${lab.display_name}!`);
        setTimeout(() => setSuccessMessage(null), 3000);
      } else {
        throw new Error(data.error || 'Failed to select lab');
      }
    } catch (err) {
      console.error('Error selecting lab:', err);
      setError(err instanceof Error ? err.message : 'Failed to select lab');
    } finally {
      setIsSelecting(false);
    }
  };

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return 'Unknown';
    try {
      return new Date(dateStr).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      });
    } catch {
      return 'Unknown';
    }
  };

  if (isLoading) {
    return (
      <div className="max-w-4xl">
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Available Labs</h2>
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-3 text-gray-600">Loading available labs...</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Choose Your Lab</h2>
            <p className="text-sm text-gray-500 mt-1">
              Select a lab for your medical reports and care. Labs are ranked by AI model accuracy.
            </p>
          </div>
          {selectedLab && (
            <div className="bg-green-50 border border-green-200 rounded-lg px-4 py-2">
              <div className="flex items-center text-green-700">
                <CheckCircle className="w-4 h-4 mr-2" />
                <span className="text-sm font-medium">Current: {selectedLab.display_name}</span>
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 flex items-center">
            <AlertCircle className="w-5 h-5 mr-2" />
            {error}
          </div>
        )}

        {successMessage && (
          <div className="mb-4 p-4 bg-green-50 border border-green-200 rounded-lg text-green-700 flex items-center">
            <CheckCircle className="w-5 h-5 mr-2" />
            {successMessage}
          </div>
        )}

        {/* Info box */}
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-700">
            <strong>About Lab Accuracy:</strong> Accuracy reflects each lab's current AI model performance 
            on a shared evaluation dataset. Higher accuracy may indicate better diagnostic predictions.
          </p>
        </div>

        {labs.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <Building2 className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <p>No labs are registered yet. Check back later.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {labs.map((lab, index) => {
              const isSelected = selectedLab?.lab_id === lab.lab_id;
              const isTop = index === 0;
              
              return (
                <div
                  key={lab.lab_id}
                  className={`relative border rounded-lg p-4 transition-all ${
                    isSelected 
                      ? 'border-green-500 bg-green-50 ring-2 ring-green-200' 
                      : 'border-gray-200 hover:border-blue-300 hover:shadow-md'
                  }`}
                >
                  {isTop && (
                    <div className="absolute -top-3 left-4 bg-yellow-400 text-yellow-900 text-xs font-bold px-2 py-1 rounded">
                      🏆 Highest Accuracy
                    </div>
                  )}
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                        isSelected ? 'bg-green-500' : 'bg-blue-500'
                      } text-white`}>
                        <Building2 className="w-6 h-6" />
                      </div>
                      <div className="ml-4">
                        <h3 className="text-lg font-semibold text-gray-900">
                          {lab.display_name}
                          {isSelected && (
                            <span className="ml-2 text-green-600 text-sm font-normal">
                              (Your Lab)
                            </span>
                          )}
                        </h3>
                        <div className="flex items-center text-sm text-gray-500 mt-1">
                          <Clock className="w-4 h-4 mr-1" />
                          <span>Updated: {formatDate(lab.last_updated)}</span>
                          {lab.num_patients > 0 && (
                            <span className="ml-3">• {lab.num_patients} patients trained</span>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-6">
                      {/* Accuracy Display */}
                      <div className="text-center">
                        <div className="flex items-center text-2xl font-bold text-gray-900">
                          <TrendingUp className={`w-5 h-5 mr-1 ${
                            lab.accuracy >= 0.7 ? 'text-green-500' :
                            lab.accuracy >= 0.5 ? 'text-yellow-500' : 'text-red-500'
                          }`} />
                          {lab.accuracy_percent}
                        </div>
                        <div className="text-xs text-gray-500 uppercase tracking-wide">
                          Accuracy
                        </div>
                      </div>
                      
                      {/* Select Button */}
                      <button
                        onClick={() => handleSelectLab(lab)}
                        disabled={isSelecting || isSelected}
                        className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                          isSelected
                            ? 'bg-green-100 text-green-700 cursor-default'
                            : 'bg-blue-600 text-white hover:bg-blue-700 disabled:bg-gray-300'
                        }`}
                      >
                        {isSelecting ? 'Selecting...' : isSelected ? 'Selected' : 'Select Lab'}
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
