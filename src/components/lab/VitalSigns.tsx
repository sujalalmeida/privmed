

interface VitalSignsProps {
  systolicBp: number;
  setSystolicBp: (value: number) => void;
  diastolicBp: number;
  setDiastolicBp: (value: number) => void;
  heartRate: number;
  setHeartRate: (value: number) => void;
  bpStatus: { label: string; severity: string };
  hrStatus: { label: string; severity: string };
}

export default function VitalSigns({
  systolicBp,
  setSystolicBp,
  diastolicBp,
  setDiastolicBp,
  heartRate,
  setHeartRate,
  bpStatus,
  hrStatus
}: VitalSignsProps) {
  return (
    <div className="clinical-section vital-signs">
      <h3>Vital Signs</h3>
      
      <div className="form-row">
        <label>Blood Pressure</label>
        <div className="bp-inputs flex items-center gap-1">
          <input 
            type="number" 
            placeholder="Systolic" 
            min={80} 
            max={250}
            value={systolicBp}
            onChange={(e) => setSystolicBp(parseInt(e.target.value) || 120)}
            className="w-20"
          />
          <span className="text-gray-500">/</span>
          <input 
            type="number" 
            placeholder="Diastolic" 
            min={40} 
            max={150}
            value={diastolicBp}
            onChange={(e) => setDiastolicBp(parseInt(e.target.value) || 80)}
            className="w-20"
          />
        </div>
        <span className="unit">mmHg</span>
        <span className={`status-${bpStatus.severity}`}>
          {bpStatus.label}
        </span>
        <span className="reference">Normal: &lt;120/80</span>
      </div>
      
      <div className="form-row">
        <label>Heart Rate</label>
        <input 
          type="number" 
          min={40} 
          max={200}
          value={heartRate}
          onChange={(e) => setHeartRate(parseInt(e.target.value) || 72)}
        />
        <span className="unit">bpm</span>
        <span className={`status-${hrStatus.severity}`}>
          {hrStatus.label}
        </span>
        <span className="reference">Normal: 60-100</span>
      </div>
    </div>
  );
}
