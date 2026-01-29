import { useState, useEffect, useMemo } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import PatientRegistration from './PatientRegistration';
import VitalSigns from './VitalSigns';
import BloodChemistry from './BloodChemistry';
import CardiacMarkers from './CardiacMarkers';
import MedicalHistory from './MedicalHistory';
import CurrentMedications from './CurrentMedications';
import DiagnosisResult from './DiagnosisResult';
import { 
  calculateBmi, 
  getBmiStatus, 
  getBpStatus, 
  getGlucoseStatus,
  getHba1cStatus,
  getCholesterolStatus,
  getHeartRateStatus 
} from './ReferenceRanges';
import '../../styles/clinical.css';

interface PredictionResult {
  diagnosis: number;
  diagnosis_label: string;
  confidence: number;
  probabilities: {
    healthy: number;
    diabetes: number;
    hypertension: number;
    heart_disease: number;
  };
}

export default function ClinicalDataEntry() {
  const { user } = useAuth();
  const [serverUrl, setServerUrl] = useState('http://127.0.0.1:5001');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  
  // Generate unique patient ID
  const generatePatientId = () => {
    return 'P-' + Date.now().toString(36) + '-' + Math.random().toString(36).substr(2, 5).toUpperCase();
  };
  
  // Patient Registration
  const [patientId] = useState(generatePatientId());
  const [age, setAge] = useState(50);
  const [sex, setSex] = useState('M');
  const [heightCm, setHeightCm] = useState(0);
  const [weightKg, setWeightKg] = useState(0);
  
  // Vital Signs
  const [systolicBp, setSystolicBp] = useState(120);
  const [diastolicBp, setDiastolicBp] = useState(80);
  const [heartRate, setHeartRate] = useState(72);
  
  // Blood Chemistry - Glucose
  const [fastingGlucose, setFastingGlucose] = useState(100);
  const [hba1c, setHba1c] = useState(5.5);
  const [insulin, setInsulin] = useState(0);
  
  // Blood Chemistry - Lipid Panel
  const [totalCholesterol, setTotalCholesterol] = useState(200);
  const [ldlCholesterol, setLdlCholesterol] = useState(100);
  const [hdlCholesterol, setHdlCholesterol] = useState(50);
  const [triglycerides, setTriglycerides] = useState(150);
  
  // Cardiac Markers
  const [chestPainType, setChestPainType] = useState(4);
  const [restingEcg, setRestingEcg] = useState(0);
  const [maxHeartRate, setMaxHeartRate] = useState(170);
  const [exerciseAngina, setExerciseAngina] = useState(0);
  const [stDepression, setStDepression] = useState(0);
  const [stSlope, setStSlope] = useState(2);
  
  // Medical History
  const [smokingStatus, setSmokingStatus] = useState(0);
  const [familyHistoryCvd, setFamilyHistoryCvd] = useState(false);
  const [familyHistoryDiabetes, setFamilyHistoryDiabetes] = useState(false);
  const [familyHistoryHypertension, setFamilyHistoryHypertension] = useState(false);
  const [priorDiabetes, setPriorDiabetes] = useState(false);
  const [priorHypertension, setPriorHypertension] = useState(false);
  const [priorHeartDisease, setPriorHeartDisease] = useState(false);
  const [priorStroke, setPriorStroke] = useState(false);
  
  // Medications
  const [onMetformin, setOnMetformin] = useState(false);
  const [onInsulin, setOnInsulin] = useState(false);
  const [onSulfonylureas, setOnSulfonylureas] = useState(false);
  const [onGlp1Agonists, setOnGlp1Agonists] = useState(false);
  const [onAceInhibitor, setOnAceInhibitor] = useState(false);
  const [onBetaBlocker, setOnBetaBlocker] = useState(false);
  const [onCalciumChannelBlocker, setOnCalciumChannelBlocker] = useState(false);
  const [onDiuretic, setOnDiuretic] = useState(false);
  const [onStatin, setOnStatin] = useState(false);
  const [onAspirin, setOnAspirin] = useState(false);
  
  // Computed values
  const bmi = useMemo(() => calculateBmi(heightCm, weightKg), [heightCm, weightKg]);
  const bmiStatus = useMemo(() => getBmiStatus(bmi), [bmi]);
  const bpStatus = useMemo(() => getBpStatus(systolicBp, diastolicBp), [systolicBp, diastolicBp]);
  const hrStatus = useMemo(() => getHeartRateStatus(heartRate), [heartRate]);
  const glucoseStatus = useMemo(() => getGlucoseStatus(fastingGlucose), [fastingGlucose]);
  const hba1cStatus = useMemo(() => getHba1cStatus(hba1c), [hba1c]);
  const cholesterolStatus = useMemo(() => getCholesterolStatus(totalCholesterol), [totalCholesterol]);
  
  // Update max heart rate when age changes
  useEffect(() => {
    setMaxHeartRate(220 - age);
  }, [age]);
  
  // Build patient data payload
  const buildPatientData = () => {
    return {
      // Demographics
      patient_id: patientId,
      age,
      sex,
      height_cm: heightCm || null,
      weight_kg: weightKg || null,
      bmi: bmi > 0 ? bmi : null,
      
      // Vital Signs
      systolic_bp: systolicBp,
      diastolic_bp: diastolicBp,
      heart_rate: heartRate,
      
      // Blood Chemistry
      fasting_glucose: fastingGlucose,
      hba1c,
      insulin: insulin || null,
      total_cholesterol: totalCholesterol,
      ldl_cholesterol: ldlCholesterol,
      hdl_cholesterol: hdlCholesterol,
      triglycerides,
      
      // Cardiac Markers
      chest_pain_type: chestPainType,
      resting_ecg: restingEcg,
      max_heart_rate: maxHeartRate,
      exercise_angina: exerciseAngina,
      st_depression: stDepression,
      st_slope: stSlope,
      
      // Medical History
      smoking_status: smokingStatus,
      family_history_cvd: familyHistoryCvd ? 1 : 0,
      family_history_diabetes: familyHistoryDiabetes ? 1 : 0,
      prior_hypertension: priorHypertension ? 1 : 0,
      prior_diabetes: priorDiabetes ? 1 : 0,
      prior_heart_disease: priorHeartDisease ? 1 : 0,
      
      // Medications (aggregated)
      on_bp_medication: (onAceInhibitor || onBetaBlocker || onCalciumChannelBlocker || onDiuretic) ? 1 : 0,
      on_diabetes_medication: (onMetformin || onInsulin || onSulfonylureas || onGlp1Agonists) ? 1 : 0,
      on_cholesterol_medication: onStatin ? 1 : 0,
      
      // Lab info
      lab_label: (user as any)?.user_metadata?.lab_label || 'lab_A',
    };
  };
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    setSuccessMessage(null);
    
    const patientData = buildPatientData();
    
    try {
      // Send to backend for prediction
      const response = await fetch(`${serverUrl}/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patientData),
      });
      
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Transform response to match expected format
      const predictionResult: PredictionResult = {
        diagnosis: data.diagnosis ?? 0,
        diagnosis_label: data.diagnosis_label ?? data.disease_type ?? 'unknown',
        confidence: data.confidence ?? data.risk_score ?? 0.5,
        probabilities: data.probabilities ?? {
          healthy: data.disease_type === 'healthy' ? data.risk_score : 0.25,
          diabetes: data.disease_type === 'diabetes' ? data.risk_score : 0.25,
          hypertension: data.disease_type === 'hypertension' ? data.risk_score : 0.25,
          heart_disease: data.disease_type === 'heart_disease' ? data.risk_score : 0.25,
        },
      };
      
      setResult(predictionResult);

      // Backend /submit now persists to patient_records (Option B); no direct frontend insert.
      setSuccessMessage('Prediction complete. Patient record saved by server.');
      
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err instanceof Error ? err.message : 'Failed to get prediction');
    } finally {
      setIsSubmitting(false);
    }
  };
  
  const resetForm = () => {
    setAge(50);
    setSex('M');
    setHeightCm(0);
    setWeightKg(0);
    setSystolicBp(120);
    setDiastolicBp(80);
    setHeartRate(72);
    setFastingGlucose(100);
    setHba1c(5.5);
    setInsulin(0);
    setTotalCholesterol(200);
    setLdlCholesterol(100);
    setHdlCholesterol(50);
    setTriglycerides(150);
    setChestPainType(4);
    setRestingEcg(0);
    setMaxHeartRate(170);
    setExerciseAngina(0);
    setStDepression(0);
    setStSlope(2);
    setSmokingStatus(0);
    setFamilyHistoryCvd(false);
    setFamilyHistoryDiabetes(false);
    setFamilyHistoryHypertension(false);
    setPriorDiabetes(false);
    setPriorHypertension(false);
    setPriorHeartDisease(false);
    setPriorStroke(false);
    setOnMetformin(false);
    setOnInsulin(false);
    setOnSulfonylureas(false);
    setOnGlp1Agonists(false);
    setOnAceInhibitor(false);
    setOnBetaBlocker(false);
    setOnCalciumChannelBlocker(false);
    setOnDiuretic(false);
    setOnStatin(false);
    setOnAspirin(false);
    setResult(null);
    setError(null);
    setSuccessMessage(null);
  };
  
  return (
    <div className="clinical-data-entry">
      <div className="mb-6 flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">Clinical Data Entry</h2>
          <p className="text-gray-600">Enter patient clinical data for AI-powered diagnosis prediction</p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-gray-500">Server:</label>
          <input
            type="text"
            value={serverUrl}
            onChange={(e) => setServerUrl(e.target.value)}
            className="text-sm border rounded px-2 py-1 w-48"
          />
        </div>
      </div>
      
      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
          <strong>Error:</strong> {error}
        </div>
      )}
      
      {successMessage && (
        <div className="mb-4 p-4 bg-green-50 border border-green-200 rounded-lg text-green-700">
          {successMessage}
        </div>
      )}
      
      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-6">
            <PatientRegistration
              patientId={patientId}
              age={age}
              setAge={setAge}
              sex={sex}
              setSex={setSex}
              heightCm={heightCm}
              setHeightCm={setHeightCm}
              weightKg={weightKg}
              setWeightKg={setWeightKg}
              bmi={bmi}
              bmiStatus={bmiStatus}
            />
            
            <VitalSigns
              systolicBp={systolicBp}
              setSystolicBp={setSystolicBp}
              diastolicBp={diastolicBp}
              setDiastolicBp={setDiastolicBp}
              heartRate={heartRate}
              setHeartRate={setHeartRate}
              bpStatus={bpStatus}
              hrStatus={hrStatus}
            />
            
            <BloodChemistry
              fastingGlucose={fastingGlucose}
              setFastingGlucose={setFastingGlucose}
              hba1c={hba1c}
              setHba1c={setHba1c}
              insulin={insulin}
              setInsulin={setInsulin}
              totalCholesterol={totalCholesterol}
              setTotalCholesterol={setTotalCholesterol}
              ldlCholesterol={ldlCholesterol}
              setLdlCholesterol={setLdlCholesterol}
              hdlCholesterol={hdlCholesterol}
              setHdlCholesterol={setHdlCholesterol}
              triglycerides={triglycerides}
              setTriglycerides={setTriglycerides}
              glucoseStatus={glucoseStatus}
              hba1cStatus={hba1cStatus}
              cholesterolStatus={cholesterolStatus}
              sex={sex}
            />
          </div>
          
          {/* Right Column */}
          <div className="space-y-6">
            <CardiacMarkers
              age={age}
              chestPainType={chestPainType}
              setChestPainType={setChestPainType}
              restingEcg={restingEcg}
              setRestingEcg={setRestingEcg}
              maxHeartRate={maxHeartRate}
              setMaxHeartRate={setMaxHeartRate}
              exerciseAngina={exerciseAngina}
              setExerciseAngina={setExerciseAngina}
              stDepression={stDepression}
              setStDepression={setStDepression}
              stSlope={stSlope}
              setStSlope={setStSlope}
            />
            
            <MedicalHistory
              smokingStatus={smokingStatus}
              setSmokingStatus={setSmokingStatus}
              familyHistoryCvd={familyHistoryCvd}
              setFamilyHistoryCvd={setFamilyHistoryCvd}
              familyHistoryDiabetes={familyHistoryDiabetes}
              setFamilyHistoryDiabetes={setFamilyHistoryDiabetes}
              familyHistoryHypertension={familyHistoryHypertension}
              setFamilyHistoryHypertension={setFamilyHistoryHypertension}
              priorDiabetes={priorDiabetes}
              setPriorDiabetes={setPriorDiabetes}
              priorHypertension={priorHypertension}
              setPriorHypertension={setPriorHypertension}
              priorHeartDisease={priorHeartDisease}
              setPriorHeartDisease={setPriorHeartDisease}
              priorStroke={priorStroke}
              setPriorStroke={setPriorStroke}
            />
            
            <CurrentMedications
              onMetformin={onMetformin}
              setOnMetformin={setOnMetformin}
              onInsulin={onInsulin}
              setOnInsulin={setOnInsulin}
              onSulfonylureas={onSulfonylureas}
              setOnSulfonylureas={setOnSulfonylureas}
              onGlp1Agonists={onGlp1Agonists}
              setOnGlp1Agonists={setOnGlp1Agonists}
              onAceInhibitor={onAceInhibitor}
              setOnAceInhibitor={setOnAceInhibitor}
              onBetaBlocker={onBetaBlocker}
              setOnBetaBlocker={setOnBetaBlocker}
              onCalciumChannelBlocker={onCalciumChannelBlocker}
              setOnCalciumChannelBlocker={setOnCalciumChannelBlocker}
              onDiuretic={onDiuretic}
              setOnDiuretic={setOnDiuretic}
              onStatin={onStatin}
              setOnStatin={setOnStatin}
              onAspirin={onAspirin}
              setOnAspirin={setOnAspirin}
            />
          </div>
        </div>
        
        {/* Submit buttons */}
        <div className="mt-6 flex gap-4">
          <button
            type="submit"
            disabled={isSubmitting}
            className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed transition-colors"
          >
            {isSubmitting ? 'Analyzing...' : 'Submit for AI Prediction'}
          </button>
          <button
            type="button"
            onClick={resetForm}
            className="px-6 py-3 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
          >
            Reset Form
          </button>
        </div>
      </form>
      
      {/* Results Section */}
      <div className="mt-8">
        <DiagnosisResult result={result} isLoading={isSubmitting} />
      </div>
    </div>
  );
}
