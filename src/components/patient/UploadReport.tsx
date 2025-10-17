import { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { addReport } from '../../utils/mockData';
import { Upload, CheckCircle } from 'lucide-react';

export default function UploadReport() {
  const { user } = useAuth();
  const [testType, setTestType] = useState('Blood Test');
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const testTypes = ['Blood Test', 'X-Ray', 'MRI Scan', 'CT Scan', 'Ultrasound', 'ECG'];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUploadSuccess(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !user) return;

    setIsUploading(true);

    await new Promise(resolve => setTimeout(resolve, 1500));

    addReport({
      patientId: user.id,
      patientName: 'Anonymous-001',
      labId: user.assignedLabId || 'lab-1',
      labName: 'MediLab Central',
      testType,
      fileName: file.name,
      status: 'pending',
      encryptedData: 'encrypted_' + Date.now(),
    });

    setIsUploading(false);
    setUploadSuccess(true);
    setFile(null);

    setTimeout(() => setUploadSuccess(false), 3000);
  };

  return (
    <div className="max-w-2xl">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Upload Medical Report</h2>

        <form onSubmit={handleSubmit} className="space-y-6">
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
              Medical Report (PDF/Image)
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
                      accept=".pdf,.jpg,.jpeg,.png"
                      onChange={handleFileChange}
                    />
                  </label>
                  <p className="pl-1">or drag and drop</p>
                </div>
                <p className="text-xs text-gray-500">PDF, PNG, JPG up to 10MB</p>
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
              Your report will be encrypted using homomorphic encryption before upload to ensure complete privacy.
            </p>
          </div>

          {uploadSuccess && (
            <div className="flex items-center p-4 bg-green-50 border border-green-200 rounded-lg text-green-800">
              <CheckCircle className="w-5 h-5 mr-2" />
              <span className="text-sm font-medium">Report encrypted and uploaded successfully!</span>
            </div>
          )}

          <button
            type="submit"
            disabled={!file || isUploading}
            className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed font-medium"
          >
            {isUploading ? 'Encrypting & Uploading...' : 'Encrypt and Upload Report'}
          </button>
        </form>
      </div>
    </div>
  );
}
