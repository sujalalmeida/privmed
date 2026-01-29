

interface CurrentMedicationsProps {
  // Diabetes medications
  onMetformin: boolean;
  setOnMetformin: (value: boolean) => void;
  onInsulin: boolean;
  setOnInsulin: (value: boolean) => void;
  onSulfonylureas: boolean;
  setOnSulfonylureas: (value: boolean) => void;
  onGlp1Agonists: boolean;
  setOnGlp1Agonists: (value: boolean) => void;
  
  // Cardiovascular medications
  onAceInhibitor: boolean;
  setOnAceInhibitor: (value: boolean) => void;
  onBetaBlocker: boolean;
  setOnBetaBlocker: (value: boolean) => void;
  onCalciumChannelBlocker: boolean;
  setOnCalciumChannelBlocker: (value: boolean) => void;
  onDiuretic: boolean;
  setOnDiuretic: (value: boolean) => void;
  onStatin: boolean;
  setOnStatin: (value: boolean) => void;
  onAspirin: boolean;
  setOnAspirin: (value: boolean) => void;
}

export default function CurrentMedications({
  onMetformin,
  setOnMetformin,
  onInsulin,
  setOnInsulin,
  onSulfonylureas,
  setOnSulfonylureas,
  onGlp1Agonists,
  setOnGlp1Agonists,
  onAceInhibitor,
  setOnAceInhibitor,
  onBetaBlocker,
  setOnBetaBlocker,
  onCalciumChannelBlocker,
  setOnCalciumChannelBlocker,
  onDiuretic,
  setOnDiuretic,
  onStatin,
  setOnStatin,
  onAspirin,
  setOnAspirin
}: CurrentMedicationsProps) {
  return (
    <div className="clinical-section medications">
      <h3>Current Medications</h3>
      
      <div className="checkbox-group">
        <h4 className="subsection-title">Diabetes Medications</h4>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={onMetformin}
            onChange={(e) => setOnMetformin(e.target.checked)}
          />
          <span>Metformin</span>
        </label>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={onInsulin}
            onChange={(e) => setOnInsulin(e.target.checked)}
          />
          <span>Insulin</span>
        </label>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={onSulfonylureas}
            onChange={(e) => setOnSulfonylureas(e.target.checked)}
          />
          <span>Sulfonylureas</span>
        </label>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={onGlp1Agonists}
            onChange={(e) => setOnGlp1Agonists(e.target.checked)}
          />
          <span>GLP-1 Agonists</span>
        </label>
      </div>
      
      <div className="checkbox-group mt-4">
        <h4 className="subsection-title">Cardiovascular Medications</h4>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={onAceInhibitor}
            onChange={(e) => setOnAceInhibitor(e.target.checked)}
          />
          <span>ACE Inhibitor / ARB</span>
        </label>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={onBetaBlocker}
            onChange={(e) => setOnBetaBlocker(e.target.checked)}
          />
          <span>Beta Blocker</span>
        </label>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={onCalciumChannelBlocker}
            onChange={(e) => setOnCalciumChannelBlocker(e.target.checked)}
          />
          <span>Calcium Channel Blocker</span>
        </label>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={onDiuretic}
            onChange={(e) => setOnDiuretic(e.target.checked)}
          />
          <span>Diuretic</span>
        </label>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={onStatin}
            onChange={(e) => setOnStatin(e.target.checked)}
          />
          <span>Statin</span>
        </label>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={onAspirin}
            onChange={(e) => setOnAspirin(e.target.checked)}
          />
          <span>Aspirin</span>
        </label>
      </div>
    </div>
  );
}
