

interface BloodChemistryProps {
  fastingGlucose: number;
  setFastingGlucose: (value: number) => void;
  hba1c: number;
  setHba1c: (value: number) => void;
  insulin: number;
  setInsulin: (value: number) => void;
  totalCholesterol: number;
  setTotalCholesterol: (value: number) => void;
  ldlCholesterol: number;
  setLdlCholesterol: (value: number) => void;
  hdlCholesterol: number;
  setHdlCholesterol: (value: number) => void;
  triglycerides: number;
  setTriglycerides: (value: number) => void;
  glucoseStatus: { label: string; severity: string };
  hba1cStatus: { label: string; severity: string };
  cholesterolStatus: { label: string; severity: string };
  sex: string;
}

export default function BloodChemistry({
  fastingGlucose,
  setFastingGlucose,
  hba1c,
  setHba1c,
  insulin,
  setInsulin,
  totalCholesterol,
  setTotalCholesterol,
  ldlCholesterol,
  setLdlCholesterol,
  hdlCholesterol,
  setHdlCholesterol,
  triglycerides,
  setTriglycerides,
  glucoseStatus,
  hba1cStatus,
  cholesterolStatus,
  sex
}: BloodChemistryProps) {
  return (
    <div className="clinical-section blood-chemistry">
      <h3>Blood Chemistry</h3>
      
      <h4 className="subsection-title">Glucose Metabolism</h4>
      
      <div className="form-row">
        <label>Fasting Glucose</label>
        <input 
          type="number" 
          min={50} 
          max={400}
          value={fastingGlucose}
          onChange={(e) => setFastingGlucose(parseInt(e.target.value) || 100)}
        />
        <span className="unit">mg/dL</span>
        <span className={`status-${glucoseStatus.severity}`}>
          {glucoseStatus.label}
        </span>
        <span className="reference">Normal: 70-100</span>
      </div>
      
      <div className="form-row">
        <label>HbA1c</label>
        <input 
          type="number" 
          step="0.1" 
          min={4} 
          max={14}
          value={hba1c}
          onChange={(e) => setHba1c(parseFloat(e.target.value) || 5.5)}
        />
        <span className="unit">%</span>
        <span className={`status-${hba1cStatus.severity}`}>
          {hba1cStatus.label}
        </span>
        <span className="reference">Normal: &lt;5.7%</span>
      </div>
      
      <div className="form-row">
        <label>Insulin (Optional)</label>
        <input 
          type="number" 
          step="0.1" 
          min={0} 
          max={500}
          value={insulin || ''}
          onChange={(e) => setInsulin(parseFloat(e.target.value) || 0)}
          placeholder="Optional"
        />
        <span className="unit">μU/mL</span>
        <span className="reference">Fasting: 2-25</span>
      </div>
      
      <h4 className="subsection-title mt-4">Lipid Panel</h4>
      
      <div className="form-row">
        <label>Total Cholesterol</label>
        <input 
          type="number" 
          min={100} 
          max={400}
          value={totalCholesterol}
          onChange={(e) => setTotalCholesterol(parseInt(e.target.value) || 200)}
        />
        <span className="unit">mg/dL</span>
        <span className={`status-${cholesterolStatus.severity}`}>
          {cholesterolStatus.label}
        </span>
        <span className="reference">Desirable: &lt;200</span>
      </div>
      
      <div className="form-row">
        <label>LDL Cholesterol</label>
        <input 
          type="number" 
          min={50} 
          max={250}
          value={ldlCholesterol}
          onChange={(e) => setLdlCholesterol(parseInt(e.target.value) || 100)}
        />
        <span className="unit">mg/dL</span>
        <span className="reference">Optimal: &lt;100</span>
      </div>
      
      <div className="form-row">
        <label>HDL Cholesterol</label>
        <input 
          type="number" 
          min={20} 
          max={100}
          value={hdlCholesterol}
          onChange={(e) => setHdlCholesterol(parseInt(e.target.value) || 50)}
        />
        <span className="unit">mg/dL</span>
        <span className="reference">Good: &gt;{sex === 'M' ? '40' : '50'}</span>
      </div>
      
      <div className="form-row">
        <label>Triglycerides</label>
        <input 
          type="number" 
          min={50} 
          max={500}
          value={triglycerides}
          onChange={(e) => setTriglycerides(parseInt(e.target.value) || 150)}
        />
        <span className="unit">mg/dL</span>
        <span className="reference">Normal: &lt;150</span>
      </div>
    </div>
  );
}
