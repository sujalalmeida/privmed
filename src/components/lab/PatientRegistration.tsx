

interface PatientRegistrationProps {
  patientId: string;
  age: number;
  setAge: (value: number) => void;
  sex: string;
  setSex: (value: string) => void;
  heightCm: number;
  setHeightCm: (value: number) => void;
  weightKg: number;
  setWeightKg: (value: number) => void;
  bmi: number;
  bmiStatus: { label: string; severity: string };
}

export default function PatientRegistration({
  patientId,
  age,
  setAge,
  sex,
  setSex,
  heightCm,
  setHeightCm,
  weightKg,
  setWeightKg,
  bmi,
  bmiStatus
}: PatientRegistrationProps) {
  return (
    <div className="clinical-section">
      <h3>Patient Registration</h3>
      
      <div className="form-row">
        <label>Patient ID</label>
        <input 
          type="text" 
          disabled 
          value={patientId} 
          className="bg-gray-100"
        />
      </div>
      
      <div className="form-row">
        <label>Age</label>
        <input 
          type="number" 
          min={18} 
          max={100}
          value={age}
          onChange={(e) => setAge(parseInt(e.target.value) || 18)}
        />
        <span className="unit">years</span>
        <span className="reference">Valid: 18-100</span>
      </div>
      
      <div className="form-row">
        <label>Sex</label>
        <select value={sex} onChange={(e) => setSex(e.target.value)}>
          <option value="M">Male</option>
          <option value="F">Female</option>
        </select>
      </div>
      
      <div className="form-row">
        <label>Height</label>
        <input 
          type="number" 
          min={100} 
          max={250}
          value={heightCm || ''}
          onChange={(e) => setHeightCm(parseFloat(e.target.value) || 0)}
          placeholder="Optional"
        />
        <span className="unit">cm</span>
        <span className="reference">100-250 cm</span>
      </div>
      
      <div className="form-row">
        <label>Weight</label>
        <input 
          type="number" 
          min={30} 
          max={300}
          value={weightKg || ''}
          onChange={(e) => setWeightKg(parseFloat(e.target.value) || 0)}
          placeholder="Optional"
        />
        <span className="unit">kg</span>
        <span className="reference">30-300 kg</span>
      </div>
      
      <div className="form-row calculated">
        <label>BMI</label>
        <input 
          type="text" 
          disabled 
          value={bmi > 0 ? bmi.toFixed(1) : 'Enter height/weight'}
          className="bg-gray-100"
        />
        <span className="unit">kg/m²</span>
        {bmi > 0 && (
          <span className={`status-${bmiStatus.severity}`}>
            {bmiStatus.label}
          </span>
        )}
        <span className="reference">Normal: 18.5-25</span>
      </div>
    </div>
  );
}
