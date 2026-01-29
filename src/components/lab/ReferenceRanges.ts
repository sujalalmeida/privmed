/**
 * Reference Ranges for Clinical Measurements
 * 
 * Provides clinical reference ranges and status indicators for various health metrics.
 * Based on standard medical guidelines (ADA, AHA, ACC).
 */

export interface Range {
  min: number;
  max: number;
  label: string;
}

export interface ReferenceRanges {
  [key: string]: {
    [status: string]: Range;
  };
}

export const REFERENCE_RANGES: ReferenceRanges = {
  fasting_glucose: {
    normal: { min: 70, max: 100, label: 'Normal' },
    prediabetes: { min: 100, max: 126, label: 'Prediabetes' },
    diabetes: { min: 126, max: 400, label: 'Diabetes Range' },
  },
  systolic_bp: {
    normal: { min: 90, max: 120, label: 'Normal' },
    elevated: { min: 120, max: 130, label: 'Elevated' },
    high_stage1: { min: 130, max: 140, label: 'High (Stage 1)' },
    high_stage2: { min: 140, max: 180, label: 'High (Stage 2)' },
    crisis: { min: 180, max: 300, label: 'Hypertensive Crisis' },
  },
  diastolic_bp: {
    normal: { min: 60, max: 80, label: 'Normal' },
    high_stage1: { min: 80, max: 90, label: 'High (Stage 1)' },
    high_stage2: { min: 90, max: 120, label: 'High (Stage 2)' },
    crisis: { min: 120, max: 180, label: 'Hypertensive Crisis' },
  },
  hba1c: {
    normal: { min: 4.0, max: 5.7, label: 'Normal' },
    prediabetes: { min: 5.7, max: 6.5, label: 'Prediabetes' },
    diabetes: { min: 6.5, max: 14.0, label: 'Diabetes' },
  },
  total_cholesterol: {
    desirable: { min: 0, max: 200, label: 'Desirable' },
    borderline: { min: 200, max: 240, label: 'Borderline High' },
    high: { min: 240, max: 400, label: 'High' },
  },
  ldl_cholesterol: {
    optimal: { min: 0, max: 100, label: 'Optimal' },
    near_optimal: { min: 100, max: 130, label: 'Near Optimal' },
    borderline: { min: 130, max: 160, label: 'Borderline High' },
    high: { min: 160, max: 190, label: 'High' },
    very_high: { min: 190, max: 300, label: 'Very High' },
  },
  hdl_cholesterol: {
    low: { min: 0, max: 40, label: 'Low (Risk)' },
    borderline: { min: 40, max: 60, label: 'Borderline' },
    good: { min: 60, max: 100, label: 'Good (Protective)' },
  },
  triglycerides: {
    normal: { min: 0, max: 150, label: 'Normal' },
    borderline: { min: 150, max: 200, label: 'Borderline High' },
    high: { min: 200, max: 500, label: 'High' },
    very_high: { min: 500, max: 1000, label: 'Very High' },
  },
  bmi: {
    underweight: { min: 0, max: 18.5, label: 'Underweight' },
    normal: { min: 18.5, max: 25, label: 'Normal' },
    overweight: { min: 25, max: 30, label: 'Overweight' },
    obese_1: { min: 30, max: 35, label: 'Obese Class I' },
    obese_2: { min: 35, max: 40, label: 'Obese Class II' },
    obese_3: { min: 40, max: 100, label: 'Obese Class III' },
  },
  heart_rate: {
    bradycardia: { min: 0, max: 60, label: 'Bradycardia' },
    normal: { min: 60, max: 100, label: 'Normal' },
    tachycardia: { min: 100, max: 200, label: 'Tachycardia' },
  },
};

export interface StatusResult {
  status: string;
  label: string;
  severity: 'normal' | 'warning' | 'critical';
}

/**
 * Get the status for a given value based on reference ranges
 */
export function getStatus(value: number, metricKey: keyof typeof REFERENCE_RANGES): StatusResult {
  const ranges = REFERENCE_RANGES[metricKey];
  
  if (!ranges) {
    return { status: 'unknown', label: 'Unknown', severity: 'normal' };
  }
  
  for (const [key, range] of Object.entries(ranges)) {
    if (value >= range.min && value < range.max) {
      // Determine severity based on status key
      let severity: 'normal' | 'warning' | 'critical' = 'normal';
      
      if (key.includes('high') || key.includes('diabetes') || key.includes('crisis') || 
          key.includes('very_high') || key.includes('obese') || key.includes('low') ||
          key === 'tachycardia' || key === 'bradycardia') {
        severity = 'critical';
      } else if (key.includes('borderline') || key.includes('elevated') || 
                 key.includes('prediabetes') || key.includes('overweight') ||
                 key.includes('underweight') || key.includes('near_optimal')) {
        severity = 'warning';
      }
      
      return { status: key, label: range.label, severity };
    }
  }
  
  return { status: 'unknown', label: 'Out of range', severity: 'critical' };
}

/**
 * Get BMI status with color class
 */
export function getBmiStatus(bmi: number): StatusResult {
  return getStatus(bmi, 'bmi');
}

/**
 * Get blood pressure status
 */
export function getBpStatus(systolic: number, diastolic: number): StatusResult {
  const sysStatus = getStatus(systolic, 'systolic_bp');
  const diaStatus = getStatus(diastolic, 'diastolic_bp');
  
  // Return the worse of the two
  if (sysStatus.severity === 'critical' || diaStatus.severity === 'critical') {
    return sysStatus.severity === 'critical' ? sysStatus : diaStatus;
  }
  if (sysStatus.severity === 'warning' || diaStatus.severity === 'warning') {
    return sysStatus.severity === 'warning' ? sysStatus : diaStatus;
  }
  return sysStatus;
}

/**
 * Get glucose status
 */
export function getGlucoseStatus(glucose: number): StatusResult {
  return getStatus(glucose, 'fasting_glucose');
}

/**
 * Get HbA1c status
 */
export function getHba1cStatus(hba1c: number): StatusResult {
  return getStatus(hba1c, 'hba1c');
}

/**
 * Get cholesterol status
 */
export function getCholesterolStatus(cholesterol: number): StatusResult {
  return getStatus(cholesterol, 'total_cholesterol');
}

/**
 * Get heart rate status
 */
export function getHeartRateStatus(heartRate: number): StatusResult {
  return getStatus(heartRate, 'heart_rate');
}

/**
 * Calculate BMI from height (cm) and weight (kg)
 */
export function calculateBmi(heightCm: number, weightKg: number): number {
  if (heightCm <= 0 || weightKg <= 0) return 0;
  const heightM = heightCm / 100;
  return weightKg / (heightM * heightM);
}

/**
 * Get CSS class for status severity
 */
export function getStatusClass(severity: 'normal' | 'warning' | 'critical'): string {
  switch (severity) {
    case 'normal':
      return 'status-normal';
    case 'warning':
      return 'status-warning';
    case 'critical':
      return 'status-critical';
    default:
      return '';
  }
}

/**
 * Format reference range for display
 */
export function formatReferenceRange(metricKey: keyof typeof REFERENCE_RANGES): string {
  const ranges = REFERENCE_RANGES[metricKey];
  if (!ranges) return '';
  
  const normalRange = ranges['normal'] || ranges['desirable'] || ranges['optimal'];
  if (normalRange) {
    if (normalRange.min === 0) {
      return `Normal: <${normalRange.max}`;
    }
    return `Normal: ${normalRange.min}-${normalRange.max}`;
  }
  return '';
}
