import { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { supabase } from '../../lib/supabase';

interface ModelUpdate {
  id: string;
  created_at: string;
  local_accuracy: number;
  grad_norm: number;
  num_examples: number;
  storage_path: string;
}

export default function PatientDataCollection() {
  const { user } = useAuth();
  const [serverUrl, setServerUrl] = useState('http://127.0.0.1:5001');
  
  // Basic patient info
  const [age, setAge] = useState<number>(30);
  const [gender, setGender] = useState<string>('male');
  const [bloodType, setBloodType] = useState<string>('O+');
  const [discomfort, setDiscomfort] = useState<number>(5);
  const [duration, setDuration] = useState<number>(7);
  const [prior, setPrior] = useState<string>('');
  
  // New fields
  const [bmi, setBmi] = useState<number>(25.0);
  const [smokerStatus, setSmokerStatus] = useState<string>('no');
  const [heartRate, setHeartRate] = useState<number>(70);
  const [bpSys, setBpSys] = useState<number>(120);
  const [bpDia, setBpDia] = useState<number>(80);
  const [cholesterol, setCholesterol] = useState<number>(200);
  const [glucose, setGlucose] = useState<number>(100);
  const [familyHistory, setFamilyHistory] = useState<string>('none');
  const [medicationUse, setMedicationUse] = useState<string>('none');
  
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<{
    risk_score: number;
    disease_type: string;
    local_accuracy?: number;
    num_examples?: number;
    insights?: {
      risk_factors?: Array<{type: string; severity: string; description: string}>;
      critical_values?: Array<{metric: string; value: number; status: string; message: string}>;
      recommendations?: Array<{priority: string; action: string}>;
      confidence?: string;
      confidence_message?: string;
    };
  } | null>(null);
  const [success, setSuccess] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [modelUpdates, setModelUpdates] = useState<ModelUpdate[]>([]);

  // Load model update history
  useEffect(() => {
    const loadModelUpdates = async () => {
      try {
        const labLabel = user?.labName || user?.email || 'lab_sim';
        const { data, error } = await supabase
          .from('fl_client_updates')
          .select('*')
          .eq('client_label', labLabel)
          .order('created_at', { ascending: false })
          .limit(10);
        
        if (error) throw error;
        setModelUpdates(data || []);
      } catch (err) {
        console.error('Error loading model updates:', err);
      }
    };
    
    loadModelUpdates();
  }, [user]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError('');
    setResult(null);
    setSuccess('');

    try {
      const labLabel = user?.labName || user?.email || 'lab_sim';
      const resp = await fetch(`${serverUrl}/lab/add_patient_data`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lab_label: labLabel,
          age,
          gender,
          blood_type: bloodType,
          discomfort_level: discomfort,
          symptom_duration: duration,
          prior_conditions: prior,
          bmi,
          smoker_status: smokerStatus,
          heart_rate: heartRate,
          bp_sys: bpSys,
          bp_dia: bpDia,
          cholesterol,
          glucose,
          family_history: familyHistory,
          medication_use: medicationUse,
        }),
      });
      if (!resp.ok) {
        const t = await resp.text();
        throw new Error(t || 'Request failed');
      }
      const data = await resp.json();
      setResult({
        risk_score: data.risk_score,
        disease_type: data.disease_type,
        local_accuracy: data.local_accuracy,
        num_examples: data.num_examples,
        insights: data.insights,
      });
      setSuccess('Patient data analyzed and model updated via federated learning');
      
      // Refresh model updates
      const { data: updates, error } = await supabase
        .from('fl_client_updates')
        .select('*')
        .eq('client_label', labLabel)
        .order('created_at', { ascending: false })
        .limit(10);
      
      if (!error) {
        setModelUpdates(updates || []);
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to submit data');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Patient Data Collection</h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm text-gray-700 mb-1">Backend URL</label>
            <input value={serverUrl} onChange={(e) => setServerUrl(e.target.value)} className="w-full px-3 py-2 border rounded" />
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Basic Information */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label htmlFor="age" className="block text-sm text-gray-700 mb-1">Age</label>
              <input id="age" type="number" min={0} value={age} onChange={(e) => setAge(parseInt(e.target.value || '0', 10))} className="w-full px-3 py-2 border rounded" required />
            </div>

            <div>
              <label htmlFor="gender" className="block text-sm text-gray-700 mb-1">Gender</label>
              <select id="gender" value={gender} onChange={(e) => setGender(e.target.value)} className="w-full px-3 py-2 border rounded">
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>

            <div>
              <label htmlFor="bloodType" className="block text-sm text-gray-700 mb-1">Blood Type</label>
              <select id="bloodType" value={bloodType} onChange={(e) => setBloodType(e.target.value)} className="w-full px-3 py-2 border rounded">
                {['O+','O-','A+','A-','B+','B-','AB+','AB-'].map(bt => (
                  <option key={bt} value={bt}>{bt}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Symptoms */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="discomfort" className="block text-sm text-gray-700 mb-1">Discomfort Level (1-10)</label>
              <input id="discomfort" type="number" min={1} max={10} value={discomfort} onChange={(e) => setDiscomfort(parseInt(e.target.value || '1', 10))} className="w-full px-3 py-2 border rounded" required />
            </div>

            <div>
              <label htmlFor="duration" className="block text-sm text-gray-700 mb-1">Symptom Duration (days)</label>
              <input id="duration" type="number" min={0} value={duration} onChange={(e) => setDuration(parseInt(e.target.value || '0', 10))} className="w-full px-3 py-2 border rounded" required />
            </div>
          </div>

          {/* Health Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label htmlFor="bmi" className="block text-sm text-gray-700 mb-1">BMI</label>
              <input id="bmi" type="number" step="0.1" min={10} max={60} value={bmi} onChange={(e) => setBmi(parseFloat(e.target.value || '25.0'))} className="w-full px-3 py-2 border rounded" required />
            </div>

            <div>
              <label htmlFor="heartRate" className="block text-sm text-gray-700 mb-1">Heart Rate (bpm)</label>
              <input id="heartRate" type="number" min={40} max={200} value={heartRate} onChange={(e) => setHeartRate(parseInt(e.target.value || '70', 10))} className="w-full px-3 py-2 border rounded" required />
            </div>

            <div>
              <label htmlFor="smokerStatus" className="block text-sm text-gray-700 mb-1">Smoker Status</label>
              <select id="smokerStatus" value={smokerStatus} onChange={(e) => setSmokerStatus(e.target.value)} className="w-full px-3 py-2 border rounded">
                <option value="no">No</option>
                <option value="yes">Yes</option>
              </select>
            </div>
          </div>

          {/* Blood Pressure */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="bpSys" className="block text-sm text-gray-700 mb-1">Systolic BP (mmHg)</label>
              <input id="bpSys" type="number" min={80} max={250} value={bpSys} onChange={(e) => setBpSys(parseInt(e.target.value || '120', 10))} className="w-full px-3 py-2 border rounded" required />
            </div>

            <div>
              <label htmlFor="bpDia" className="block text-sm text-gray-700 mb-1">Diastolic BP (mmHg)</label>
              <input id="bpDia" type="number" min={50} max={150} value={bpDia} onChange={(e) => setBpDia(parseInt(e.target.value || '80', 10))} className="w-full px-3 py-2 border rounded" required />
            </div>
          </div>

          {/* Lab Values */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="cholesterol" className="block text-sm text-gray-700 mb-1">Cholesterol (mg/dL)</label>
              <input id="cholesterol" type="number" min={100} max={400} value={cholesterol} onChange={(e) => setCholesterol(parseInt(e.target.value || '200', 10))} className="w-full px-3 py-2 border rounded" required />
            </div>

            <div>
              <label htmlFor="glucose" className="block text-sm text-gray-700 mb-1">Glucose (mg/dL)</label>
              <input id="glucose" type="number" min={70} max={300} value={glucose} onChange={(e) => setGlucose(parseInt(e.target.value || '100', 10))} className="w-full px-3 py-2 border rounded" required />
            </div>
          </div>

          {/* Medical History */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="familyHistory" className="block text-sm text-gray-700 mb-1">Family History</label>
              <select id="familyHistory" value={familyHistory} onChange={(e) => setFamilyHistory(e.target.value)} className="w-full px-3 py-2 border rounded">
                <option value="none">None</option>
                <option value="diabetes">Diabetes</option>
                <option value="hypertension">Hypertension</option>
                <option value="heart_disease">Heart Disease</option>
              </select>
            </div>

            <div>
              <label htmlFor="medicationUse" className="block text-sm text-gray-700 mb-1">Current Medications</label>
              <select id="medicationUse" value={medicationUse} onChange={(e) => setMedicationUse(e.target.value)} className="w-full px-3 py-2 border rounded">
                <option value="none">None</option>
                <option value="metformin">Metformin</option>
                <option value="insulin">Insulin</option>
                <option value="lisinopril">Lisinopril</option>
                <option value="amlodipine">Amlodipine</option>
                <option value="atorvastatin">Atorvastatin</option>
              </select>
            </div>
          </div>

          <div>
            <label htmlFor="prior" className="block text-sm text-gray-700 mb-1">Prior Conditions</label>
            <input id="prior" type="text" value={prior} onChange={(e) => setPrior(e.target.value)} className="w-full px-3 py-2 border rounded" placeholder="e.g., diabetes;hypertension" />
          </div>

          <div className="mt-4">
            <button type="submit" disabled={isSubmitting} className="bg-blue-600 text-white py-2 px-6 rounded disabled:bg-gray-400 hover:bg-blue-700">
              {isSubmitting ? 'Submitting...' : 'Submit Patient Data'}
            </button>
          </div>
        </form>

        {/* Prediction Results - Comprehensive Medical Report */}
        {result && (
          <div className="mt-6 space-y-4">
            {/* Header Banner */}
            <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-6 rounded-lg shadow-lg">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-2xl font-bold mb-1">Medical Analysis Report</h3>
                  <p className="text-blue-100 text-sm">Powered by Federated Learning AI</p>
                </div>
                <div className="text-right">
                  <div className="text-xs text-blue-200 mb-1">Report ID</div>
                  <div className="text-sm font-mono">{Date.now().toString(36).toUpperCase()}</div>
                  <div className="text-xs text-blue-200 mt-2">{new Date().toLocaleString()}</div>
                </div>
              </div>
            </div>

            {/* Main Diagnosis Card */}
            <div className="bg-white border-2 border-gray-200 rounded-lg shadow-md overflow-hidden">
              <div className={`p-4 ${
                result.disease_type === 'healthy' ? 'bg-green-50 border-b-4 border-green-500' :
                result.disease_type === 'diabetes' ? 'bg-yellow-50 border-b-4 border-yellow-500' :
                result.disease_type === 'hypertension' ? 'bg-orange-50 border-b-4 border-orange-500' :
                'bg-red-50 border-b-4 border-red-500'
              }`}>
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-sm font-medium text-gray-600 mb-1">Primary Diagnosis</h4>
                    <div className="text-3xl font-bold text-gray-900">
                      {result.disease_type === 'healthy' ? '✓ Healthy' :
                       result.disease_type === 'diabetes' ? '⚠ Diabetes Detected' :
                       result.disease_type === 'hypertension' ? '⚠ Hypertension Detected' :
                       '⚠ Heart Disease Detected'}
                    </div>
                  </div>
                  <div className={`text-6xl ${
                    result.disease_type === 'healthy' ? 'text-green-500' :
                    result.disease_type === 'diabetes' ? 'text-yellow-500' :
                    result.disease_type === 'hypertension' ? 'text-orange-500' :
                    'text-red-500'
                  }`}>
                    {result.disease_type === 'healthy' ? '💚' :
                     result.disease_type === 'diabetes' ? '🩺' :
                     result.disease_type === 'hypertension' ? '❤️' :
                     '🫀'}
                  </div>
                </div>
              </div>

              <div className="p-6 space-y-6">
                {/* Risk Assessment */}
                <div>
                  <h5 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                    <span className="w-2 h-2 bg-blue-600 rounded-full mr-2"></span>
                    Risk Assessment
                  </h5>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-600">Confidence Score</span>
                      <span className="text-lg font-bold text-gray-900">
                        {(result.risk_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="relative w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                      <div 
                        className={`h-4 rounded-full transition-all duration-500 ${
                          result.risk_score < 0.3 ? 'bg-gradient-to-r from-green-400 to-green-600' :
                          result.risk_score < 0.6 ? 'bg-gradient-to-r from-yellow-400 to-yellow-600' :
                          result.risk_score < 0.8 ? 'bg-gradient-to-r from-orange-400 to-orange-600' :
                          'bg-gradient-to-r from-red-400 to-red-600'
                        }`}
                        style={{ width: `${result.risk_score * 100}%` }}
                      ></div>
                    </div>
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>Low Risk</span>
                      <span>Moderate</span>
                      <span>High Risk</span>
                    </div>
                  </div>
                </div>

                {/* Patient Summary */}
                <div>
                  <h5 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                    <span className="w-2 h-2 bg-blue-600 rounded-full mr-2"></span>
                    Patient Summary
                  </h5>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-xs text-gray-500 mb-1">Age</div>
                      <div className="text-lg font-semibold text-gray-900">{age} years</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-xs text-gray-500 mb-1">BMI</div>
                      <div className="text-lg font-semibold text-gray-900">{bmi.toFixed(1)}</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-xs text-gray-500 mb-1">Blood Pressure</div>
                      <div className="text-lg font-semibold text-gray-900">{bpSys}/{bpDia}</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-xs text-gray-500 mb-1">Heart Rate</div>
                      <div className="text-lg font-semibold text-gray-900">{heartRate} bpm</div>
                    </div>
                  </div>
                </div>

                {/* Clinical Indicators */}
                <div>
                  <h5 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                    <span className="w-2 h-2 bg-blue-600 rounded-full mr-2"></span>
                    Clinical Indicators
                  </h5>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="border border-gray-200 p-3 rounded-lg">
                      <div className="text-xs text-gray-500 mb-1">Glucose Level</div>
                      <div className="text-base font-semibold text-gray-900">{glucose} mg/dL</div>
                      <div className={`text-xs mt-1 ${
                        glucose < 100 ? 'text-green-600' : glucose < 126 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {glucose < 100 ? 'Normal' : glucose < 126 ? 'Prediabetic' : 'Diabetic Range'}
                      </div>
                    </div>
                    <div className="border border-gray-200 p-3 rounded-lg">
                      <div className="text-xs text-gray-500 mb-1">Cholesterol</div>
                      <div className="text-base font-semibold text-gray-900">{cholesterol} mg/dL</div>
                      <div className={`text-xs mt-1 ${
                        cholesterol < 200 ? 'text-green-600' : cholesterol < 240 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {cholesterol < 200 ? 'Desirable' : cholesterol < 240 ? 'Borderline' : 'High'}
                      </div>
                    </div>
                    <div className="border border-gray-200 p-3 rounded-lg">
                      <div className="text-xs text-gray-500 mb-1">Symptom Duration</div>
                      <div className="text-base font-semibold text-gray-900">{duration} days</div>
                      <div className={`text-xs mt-1 ${
                        duration < 7 ? 'text-green-600' : duration < 14 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {duration < 7 ? 'Recent' : duration < 14 ? 'Moderate' : 'Chronic'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Risk Factors - Dynamic from AI insights */}
                <div>
                  <h5 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                    <span className="w-2 h-2 bg-blue-600 rounded-full mr-2"></span>
                    Risk Factors Identified
                  </h5>
                  <div className="flex flex-wrap gap-2">
                    {result.insights?.risk_factors && result.insights.risk_factors.length > 0 ? (
                      result.insights.risk_factors.map((factor, idx) => (
                        <span key={idx} className={`px-3 py-1 text-xs font-medium rounded-full border ${
                          factor.severity === 'high' ? 'bg-red-100 text-red-800 border-red-300' :
                          factor.severity === 'moderate' ? 'bg-orange-100 text-orange-800 border-orange-300' :
                          'bg-yellow-100 text-yellow-800 border-yellow-300'
                        }`}>
                          {factor.description}
                        </span>
                      ))
                    ) : (
                      <span className="px-3 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full border border-green-300">
                        ✓ No significant risk factors identified
                      </span>
                    )}
                  </div>
                </div>

                {/* Federated Learning Info */}
                <div className="bg-gradient-to-r from-indigo-50 to-blue-50 border-2 border-indigo-200 p-4 rounded-lg">
                  <h5 className="text-sm font-semibold text-indigo-900 mb-3 flex items-center">
                    <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M13 7H7v6h6V7z"/>
                      <path fillRule="evenodd" d="M7 2a1 1 0 012 0v1h2V2a1 1 0 112 0v1h2a2 2 0 012 2v2h1a1 1 0 110 2h-1v2h1a1 1 0 110 2h-1v2a2 2 0 01-2 2h-2v1a1 1 0 11-2 0v-1H9v1a1 1 0 11-2 0v-1H5a2 2 0 01-2-2v-2H2a1 1 0 110-2h1V9H2a1 1 0 010-2h1V5a2 2 0 012-2h2V2zM5 5h10v10H5V5z"/>
                    </svg>
                    Federated Learning Model Details
                  </h5>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <div className="text-xs text-indigo-600 mb-1">Lab Network</div>
                      <div className="font-semibold text-indigo-900">{user?.labName || 'Lab Network'}</div>
                    </div>
                    {result.local_accuracy && (
                      <div>
                        <div className="text-xs text-indigo-600 mb-1">Model Accuracy</div>
                        <div className="font-semibold text-indigo-900">
                          {(result.local_accuracy * 100).toFixed(1)}%
                        </div>
                      </div>
                    )}
                    {result.num_examples && (
                      <div>
                        <div className="text-xs text-indigo-600 mb-1">Training Samples</div>
                        <div className="font-semibold text-indigo-900">{result.num_examples} records</div>
                      </div>
                    )}
                    <div>
                      <div className="text-xs text-indigo-600 mb-1">Privacy Status</div>
                      <div className="font-semibold text-green-700 flex items-center">
                        <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z"/>
                        </svg>
                        Protected
                      </div>
                    </div>
                  </div>
                  <div className="mt-3 p-3 bg-white bg-opacity-60 rounded border border-indigo-200">
                    <p className="text-xs text-indigo-800 leading-relaxed">
                      <strong>🔒 Privacy-Preserving Analysis:</strong> This diagnosis was generated using federated learning technology. 
                      Your patient data never leaves this lab. Only encrypted model parameters are shared with the global network, 
                      ensuring complete data privacy while benefiting from collaborative AI training across multiple medical institutions.
                    </p>
                  </div>
                </div>

                {/* Recommendations - Dynamic from AI */}
                <div>
                  <h5 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                    <span className="w-2 h-2 bg-blue-600 rounded-full mr-2"></span>
                    Clinical Recommendations
                  </h5>
                  {result.insights?.confidence_message && (
                    <div className="mb-3 p-3 bg-indigo-50 border border-indigo-200 rounded-lg">
                      <p className="text-sm text-indigo-800">
                        <strong>AI Confidence:</strong> {result.insights.confidence_message}
                      </p>
                    </div>
                  )}
                  <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                    <ul className="space-y-2 text-sm text-gray-700">
                      {result.insights?.recommendations && result.insights.recommendations.length > 0 ? (
                        result.insights.recommendations.map((rec, idx) => (
                          <li key={idx} className="flex items-start">
                            <span className={`mr-2 ${
                              rec.priority === 'critical' ? 'text-red-600 font-bold' :
                              rec.priority === 'urgent' ? 'text-red-600' :
                              rec.priority === 'high' ? 'text-orange-600' :
                              rec.priority === 'moderate' ? 'text-yellow-600' :
                              'text-green-600'
                            }`}>
                              {rec.priority === 'critical' || rec.priority === 'urgent' ? '⚠' : 
                               rec.priority === 'routine' ? '✓' : '•'}
                            </span>
                            <span className={rec.priority === 'critical' ? 'font-semibold' : ''}>
                              {rec.action}
                            </span>
                          </li>
                        ))
                      ) : (
                        <li className="flex items-start">
                          <span className="text-green-600 mr-2">✓</span>
                          <span>Continue regular health monitoring and maintain healthy lifestyle</span>
                        </li>
                      )}
                    </ul>
                  </div>
                </div>

                {/* Disclaimer */}
                <div className="border-t pt-4 mt-4">
                  <p className="text-xs text-gray-500 leading-relaxed">
                    <strong>Medical Disclaimer:</strong> This AI-generated report is intended to assist healthcare professionals 
                    and should not replace clinical judgment. Please correlate with clinical findings, patient history, and 
                    additional diagnostic tests. The federated learning model is continuously improving through collaborative 
                    training across multiple healthcare institutions while maintaining patient privacy.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {success && (
          <div className="mt-4 p-3 border rounded text-sm bg-green-50 border-green-200 text-green-700">{success}</div>
        )}
        {error && (
          <div className="mt-4 p-3 border rounded text-sm bg-red-50 border-red-200 text-red-700">{error}</div>
        )}
      </div>

      {/* Model Update History */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Update History</h3>
        {modelUpdates.length === 0 ? (
          <p className="text-gray-500 text-sm">No model updates yet. Submit patient data to see updates.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Local Accuracy</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Gradient Norm</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Examples</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model File</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {modelUpdates.map((update) => (
                  <tr key={update.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {new Date(update.created_at).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {update.local_accuracy ? `${(update.local_accuracy * 100).toFixed(1)}%` : 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {update.grad_norm ? update.grad_norm.toFixed(4) : 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {update.num_examples}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {update.storage_path ? (
                        <span className="text-blue-600 hover:text-blue-800 cursor-pointer">
                          {update.storage_path.split('/').pop()}
                        </span>
                      ) : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
