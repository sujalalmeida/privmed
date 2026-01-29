

interface CardiacMarkersProps {
  age: number;
  chestPainType: number;
  setChestPainType: (value: number) => void;
  restingEcg: number;
  setRestingEcg: (value: number) => void;
  maxHeartRate: number;
  setMaxHeartRate: (value: number) => void;
  exerciseAngina: number;
  setExerciseAngina: (value: number) => void;
  stDepression: number;
  setStDepression: (value: number) => void;
  stSlope: number;
  setStSlope: (value: number) => void;
}

export default function CardiacMarkers({
  age,
  chestPainType,
  setChestPainType,
  restingEcg,
  setRestingEcg,
  maxHeartRate,
  setMaxHeartRate,
  exerciseAngina,
  setExerciseAngina,
  stDepression,
  setStDepression,
  stSlope,
  setStSlope
}: CardiacMarkersProps) {
  const targetMaxHr = 220 - age;
  
  return (
    <div className="clinical-section cardiac">
      <h3>Cardiac Assessment</h3>
      
      <div className="form-row">
        <label>Chest Pain Type</label>
        <select 
          value={chestPainType} 
          onChange={(e) => setChestPainType(parseInt(e.target.value))}
        >
          <option value={1}>Typical Angina</option>
          <option value={2}>Atypical Angina</option>
          <option value={3}>Non-Anginal Pain</option>
          <option value={4}>Asymptomatic</option>
        </select>
        <span className="reference">
          {chestPainType === 1 && '⚠️ High cardiac risk indicator'}
          {chestPainType === 2 && '⚠️ Moderate concern'}
          {chestPainType === 3 && 'Less specific'}
          {chestPainType === 4 && 'No chest pain reported'}
        </span>
      </div>
      
      <div className="form-row">
        <label>Resting ECG</label>
        <select 
          value={restingEcg} 
          onChange={(e) => setRestingEcg(parseInt(e.target.value))}
        >
          <option value={0}>Normal</option>
          <option value={1}>ST-T Wave Abnormality</option>
          <option value={2}>Left Ventricular Hypertrophy</option>
        </select>
        <span className={restingEcg === 0 ? 'status-normal' : 'status-warning'}>
          {restingEcg === 0 ? 'Normal' : 'Abnormal'}
        </span>
      </div>
      
      <div className="form-row">
        <label>Max Heart Rate (Exercise)</label>
        <input 
          type="number" 
          min={60} 
          max={220}
          value={maxHeartRate}
          onChange={(e) => setMaxHeartRate(parseInt(e.target.value) || targetMaxHr)}
        />
        <span className="unit">bpm</span>
        <span className={maxHeartRate < targetMaxHr * 0.85 ? 'status-warning' : 'status-normal'}>
          {Math.round((maxHeartRate / targetMaxHr) * 100)}% of max
        </span>
        <span className="reference">Target: {targetMaxHr} (age-adjusted max)</span>
      </div>
      
      <div className="form-row">
        <label>Exercise-Induced Angina</label>
        <div className="radio-group flex gap-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input 
              type="radio" 
              name="exang" 
              value={0}
              checked={exerciseAngina === 0}
              onChange={() => setExerciseAngina(0)}
            />
            <span>No</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input 
              type="radio" 
              name="exang" 
              value={1}
              checked={exerciseAngina === 1}
              onChange={() => setExerciseAngina(1)}
            />
            <span>Yes</span>
          </label>
        </div>
        {exerciseAngina === 1 && (
          <span className="status-critical">⚠️ Positive finding</span>
        )}
      </div>
      
      <div className="form-row">
        <label>ST Depression</label>
        <input 
          type="number" 
          step="0.1" 
          min={0} 
          max={6}
          value={stDepression}
          onChange={(e) => setStDepression(parseFloat(e.target.value) || 0)}
        />
        <span className="unit">mm</span>
        <span className={stDepression > 1 ? 'status-critical' : stDepression > 0 ? 'status-warning' : 'status-normal'}>
          {stDepression > 2 ? 'Significant' : stDepression > 1 ? 'Abnormal' : stDepression > 0 ? 'Borderline' : 'Normal'}
        </span>
        <span className="reference">Normal: 0 mm</span>
      </div>
      
      <div className="form-row">
        <label>ST Slope</label>
        <select 
          value={stSlope} 
          onChange={(e) => setStSlope(parseInt(e.target.value))}
        >
          <option value={1}>Upsloping</option>
          <option value={2}>Flat</option>
          <option value={3}>Downsloping</option>
        </select>
        <span className={stSlope === 3 ? 'status-critical' : stSlope === 2 ? 'status-warning' : 'status-normal'}>
          {stSlope === 1 ? 'Normal response' : stSlope === 2 ? 'Concerning' : 'Abnormal'}
        </span>
      </div>
    </div>
  );
}
