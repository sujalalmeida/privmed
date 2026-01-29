

interface MedicalHistoryProps {
  smokingStatus: number;
  setSmokingStatus: (value: number) => void;
  familyHistoryCvd: boolean;
  setFamilyHistoryCvd: (value: boolean) => void;
  familyHistoryDiabetes: boolean;
  setFamilyHistoryDiabetes: (value: boolean) => void;
  familyHistoryHypertension: boolean;
  setFamilyHistoryHypertension: (value: boolean) => void;
  priorDiabetes: boolean;
  setPriorDiabetes: (value: boolean) => void;
  priorHypertension: boolean;
  setPriorHypertension: (value: boolean) => void;
  priorHeartDisease: boolean;
  setPriorHeartDisease: (value: boolean) => void;
  priorStroke: boolean;
  setPriorStroke: (value: boolean) => void;
}

export default function MedicalHistory({
  smokingStatus,
  setSmokingStatus,
  familyHistoryCvd,
  setFamilyHistoryCvd,
  familyHistoryDiabetes,
  setFamilyHistoryDiabetes,
  familyHistoryHypertension,
  setFamilyHistoryHypertension,
  priorDiabetes,
  setPriorDiabetes,
  priorHypertension,
  setPriorHypertension,
  priorHeartDisease,
  setPriorHeartDisease,
  priorStroke,
  setPriorStroke
}: MedicalHistoryProps) {
  return (
    <div className="clinical-section medical-history">
      <h3>Medical History</h3>
      
      <div className="checkbox-group">
        <h4 className="subsection-title">Personal Medical History</h4>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={priorDiabetes}
            onChange={(e) => setPriorDiabetes(e.target.checked)}
          />
          <span>Prior Diabetes Diagnosis</span>
        </label>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={priorHypertension}
            onChange={(e) => setPriorHypertension(e.target.checked)}
          />
          <span>Prior Hypertension Diagnosis</span>
        </label>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={priorHeartDisease}
            onChange={(e) => setPriorHeartDisease(e.target.checked)}
          />
          <span>Prior Heart Disease / MI</span>
        </label>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={priorStroke}
            onChange={(e) => setPriorStroke(e.target.checked)}
          />
          <span>Prior Stroke / TIA</span>
        </label>
      </div>
      
      <div className="checkbox-group mt-4">
        <h4 className="subsection-title">Family History</h4>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={familyHistoryDiabetes}
            onChange={(e) => setFamilyHistoryDiabetes(e.target.checked)}
          />
          <span>Family History of Diabetes</span>
        </label>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={familyHistoryCvd}
            onChange={(e) => setFamilyHistoryCvd(e.target.checked)}
          />
          <span>Family History of Heart Disease</span>
        </label>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={familyHistoryHypertension}
            onChange={(e) => setFamilyHistoryHypertension(e.target.checked)}
          />
          <span>Family History of Hypertension</span>
        </label>
      </div>
      
      <div className="form-row mt-4">
        <label>Smoking Status</label>
        <select 
          value={smokingStatus}
          onChange={(e) => setSmokingStatus(parseInt(e.target.value))}
        >
          <option value={0}>Never Smoked</option>
          <option value={1}>Former Smoker</option>
          <option value={2}>Current Smoker</option>
        </select>
        <span className={smokingStatus === 2 ? 'status-critical' : smokingStatus === 1 ? 'status-warning' : 'status-normal'}>
          {smokingStatus === 0 ? 'No risk' : smokingStatus === 1 ? 'Reduced risk' : '⚠️ High CV risk'}
        </span>
      </div>
    </div>
  );
}
