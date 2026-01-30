import { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { User, Phone, Mail, Building2, TrendingUp, ExternalLink } from 'lucide-react';

interface SelectedLab {
  lab_id: string;
  display_name: string;
  accuracy: number | null;
  accuracy_percent: string;
  selected_at?: string;
}

export default function Profile() {
  const { user } = useAuth();
  const [fullName, setFullName] = useState(user?.fullName || '');
  const [contactPhone, setContactPhone] = useState(user?.contactPhone || '');
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [selectedLab, setSelectedLab] = useState<SelectedLab | null>(null);
  const [isLoadingLab, setIsLoadingLab] = useState(true);
  const serverUrl = 'http://127.0.0.1:5001';

  // Fetch selected lab on mount
  useEffect(() => {
    const fetchSelectedLab = async () => {
      setIsLoadingLab(true);
      try {
        const patientId = user?.id || user?.email || 'anonymous';
        const response = await fetch(`${serverUrl}/patient/selected_lab?patient_id=${encodeURIComponent(patientId)}`);
        if (response.ok) {
          const data = await response.json();
          if (data.has_selection) {
            setSelectedLab(data.selected_lab);
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
        console.error('Error fetching selected lab:', err);
        // Try localStorage as fallback
        const stored = localStorage.getItem('privmed_selected_lab');
        if (stored) {
          try {
            setSelectedLab(JSON.parse(stored));
          } catch {
            // Invalid stored data
          }
        }
      } finally {
        setIsLoadingLab(false);
      }
    };
    
    fetchSelectedLab();
  }, [user, serverUrl]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSaving(true);

    await new Promise(resolve => setTimeout(resolve, 1000));

    setIsSaving(false);
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 3000);
  };

  return (
    <div className="max-w-2xl">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Profile Information</h2>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <div className="flex items-center">
                <User className="w-4 h-4 mr-2" />
                Full Name
              </div>
            </label>
            <input
              type="text"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <div className="flex items-center">
                <Mail className="w-4 h-4 mr-2" />
                Email
              </div>
            </label>
            <input
              type="email"
              value={user?.email}
              disabled
              className="w-full px-4 py-2 border border-gray-300 rounded-lg bg-gray-50 text-gray-500 cursor-not-allowed"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <div className="flex items-center">
                <Phone className="w-4 h-4 mr-2" />
                Contact Phone
              </div>
            </label>
            <input
              type="tel"
              value={contactPhone}
              onChange={(e) => setContactPhone(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <div className="flex items-center">
                <Building2 className="w-4 h-4 mr-2" />
                Selected Lab
              </div>
            </label>
            {isLoadingLab ? (
              <div className="w-full px-4 py-3 border border-gray-300 rounded-lg bg-gray-50 flex items-center">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                <span className="text-gray-500">Loading...</span>
              </div>
            ) : selectedLab ? (
              <div className="w-full px-4 py-3 border border-green-300 rounded-lg bg-green-50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Building2 className="w-5 h-5 text-green-600 mr-2" />
                    <span className="font-medium text-green-800">{selectedLab.display_name}</span>
                  </div>
                  <div className="flex items-center text-green-700">
                    <TrendingUp className="w-4 h-4 mr-1" />
                    <span className="font-semibold">{selectedLab.accuracy_percent}</span>
                    <span className="text-sm text-green-600 ml-1">accuracy</span>
                  </div>
                </div>
                <a 
                  href="#" 
                  onClick={(e) => {
                    e.preventDefault();
                    // Trigger navigation to labs tab - in a real app, use router
                    window.location.hash = 'labs';
                    window.location.reload();
                  }}
                  className="text-sm text-blue-600 hover:text-blue-800 mt-2 inline-flex items-center"
                >
                  Change lab
                  <ExternalLink className="w-3 h-3 ml-1" />
                </a>
              </div>
            ) : (
              <div className="w-full px-4 py-3 border border-gray-300 rounded-lg bg-gray-50">
                <div className="flex items-center justify-between">
                  <span className="text-gray-500">No lab selected</span>
                  <a 
                    href="#" 
                    onClick={(e) => {
                      e.preventDefault();
                      window.location.hash = 'labs';
                      window.location.reload();
                    }}
                    className="text-sm text-blue-600 hover:text-blue-800 inline-flex items-center"
                  >
                    Choose a lab
                    <ExternalLink className="w-3 h-3 ml-1" />
                  </a>
                </div>
              </div>
            )}
          </div>

          {saveSuccess && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg text-green-800 text-sm">
              Profile updated successfully!
            </div>
          )}

          <button
            type="submit"
            disabled={isSaving}
            className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed font-medium"
          >
            {isSaving ? 'Saving...' : 'Save Changes'}
          </button>
        </form>
      </div>
    </div>
  );
}
