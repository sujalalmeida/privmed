import { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { addReport } from '../../utils/mockData';
import { Upload, CheckCircle } from 'lucide-react';

export default function SendReport() {
  const { user } = useAuth();
  const [patientId, setPatientId] = useState('');
  const [testType, setTestType] = useState('Blood Test');
  const [file, setFile] = useState<File | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [sendSuccess, setSendSuccess] = useState(false);

  const testTypes = ['Blood Test', 'X-Ray', 'MRI Scan', 'CT Scan', 'Ultrasound', 'ECG'];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setSendSuccess(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !user) return;

    setIsSending(true);

    await new Promise(resolve => setTimeout(resolve, 1500));

    addReport({
      patientId: patientId || 'patient-1',
      patientName: `Anonymous-${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`,
      labId: user.id,
      labName: user.labName || 'Lab',
      testType,
      fileName: file.name,
      status: 'pending',
      encryptedData: 'encrypted_' + Date.now(),
    });

    setIsSending(false);
    setSendSuccess(true);
    setFile(null);
    setPatientId('');

    setTimeout(() => setSendSuccess(false), 3000);
  };

  return (
    <div className="max-w-2xl">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Encrypt & Send Report to Central Lab</h2>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Patient ID (Optional)
            </label>
            <input
              type="text"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              placeholder="Leave blank for anonymous ID"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Test Type
            </label>
            <select
              value={testType}
              onChange={(e) => setTestType(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {testTypes.map((type) => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Test Data File
            </label>
            <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-lg hover:border-blue-400 transition-colors">
              <div className="space-y-1 text-center">
                <Upload className="mx-auto h-12 w-12 text-gray-400" />
                <div className="flex text-sm text-gray-600">
                  <label className="relative cursor-pointer bg-white rounded-md font-medium text-blue-600 hover:text-blue-500">
                    <span>Upload a file</span>
                    <input
                      type="file"
                      className="sr-only"
                      accept=".pdf,.jpg,.jpeg,.png,.csv,.xlsx"
                      onChange={handleFileChange}
                    />
                  </label>
                  <p className="pl-1">or drag and drop</p>
                </div>
                <p className="text-xs text-gray-500">PDF, CSV, Excel, or Image files</p>
                {file && (
                  <p className="text-sm text-blue-600 font-medium mt-2">
                    Selected: {file.name}
                  </p>
                )}
              </div>
            </div>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-sm text-blue-800">
              Report data will be encrypted using homomorphic encryption before transmission to the central lab, preserving patient privacy.
            </p>
          </div>

          {sendSuccess && (
            <div className="flex items-center p-4 bg-green-50 border border-green-200 rounded-lg text-green-800">
              <CheckCircle className="w-5 h-5 mr-2" />
              <span className="text-sm font-medium">Report encrypted and sent to central lab successfully!</span>
            </div>
          )}

          <button
            type="submit"
            disabled={!file || isSending}
            className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed font-medium"
          >
            {isSending ? 'Encrypting & Sending...' : 'Encrypt & Send to Central Lab'}
          </button>
        </form>
      </div>
    </div>
  );
}
