import { User, MedicalReport, Diagnosis, ModelUpdate } from '../types';

export const mockUsers: User[] = [
  {
    id: 'patient-1',
    email: 'patient@demo.com',
    fullName: 'John Smith',
    role: 'patient',
    contactPhone: '+1-555-0101',
    assignedLabId: 'lab-1',
  },
  {
    id: 'lab-1',
    email: 'lab@demo.com',
    fullName: 'Dr. Sarah Johnson',
    role: 'lab',
    labName: 'MediLab Central',
    contactPhone: '+1-555-0202',
  },
  {
    id: 'admin-1',
    email: 'admin@demo.com',
    fullName: 'Dr. Michael Chen',
    role: 'central_admin',
    contactPhone: '+1-555-0303',
  },
];

export const mockReports: MedicalReport[] = [
  {
    id: 'report-1',
    patientId: 'patient-1',
    patientName: 'Anonymous-001',
    labId: 'lab-1',
    labName: 'MediLab Central',
    testType: 'Blood Test',
    fileName: 'blood_test_2024_001.pdf',
    status: 'diagnosed',
    aiPrediction: [
      { disease: 'Type 2 Diabetes', confidence: 0.87 },
      { disease: 'Prediabetes', confidence: 0.13 },
    ],
    createdAt: '2024-10-01T10:30:00Z',
    updatedAt: '2024-10-02T14:20:00Z',
  },
  {
    id: 'report-2',
    patientId: 'patient-1',
    patientName: 'Anonymous-001',
    labId: 'lab-1',
    labName: 'MediLab Central',
    testType: 'X-Ray',
    fileName: 'chest_xray_2024_002.pdf',
    status: 'under_review',
    aiPrediction: [
      { disease: 'Normal', confidence: 0.92 },
      { disease: 'Pneumonia', confidence: 0.08 },
    ],
    createdAt: '2024-10-03T09:15:00Z',
    updatedAt: '2024-10-03T09:15:00Z',
  },
  {
    id: 'report-3',
    patientId: 'patient-1',
    patientName: 'Anonymous-001',
    labId: 'lab-1',
    labName: 'MediLab Central',
    testType: 'MRI Scan',
    fileName: 'brain_mri_2024_003.pdf',
    status: 'pending',
    createdAt: '2024-10-04T11:00:00Z',
    updatedAt: '2024-10-04T11:00:00Z',
  },
];

export const mockDiagnoses: Diagnosis[] = [
  {
    id: 'diag-1',
    reportId: 'report-1',
    doctorId: 'admin-1',
    doctorName: 'Dr. Michael Chen',
    encryptedFeedback: 'encrypted_feedback_data',
    diagnosisResult: 'Early stage Type 2 Diabetes detected. Recommend lifestyle modifications and follow-up in 3 months.',
    confidenceScore: 0.87,
    createdAt: '2024-10-02T14:20:00Z',
  },
];

export const mockModelUpdates: ModelUpdate[] = [
  {
    id: 'model-1',
    labId: 'lab-1',
    labName: 'MediLab Central',
    modelVersion: 'v1.2.3',
    accuracy: 0.94,
    isAggregated: true,
    createdAt: '2024-10-01T08:00:00Z',
  },
  {
    id: 'model-2',
    labId: 'lab-1',
    labName: 'MediLab Central',
    modelVersion: 'v1.2.4',
    accuracy: 0.96,
    isAggregated: false,
    createdAt: '2024-10-04T16:30:00Z',
  },
];

let reportsStore = [...mockReports];
let diagnosesStore = [...mockDiagnoses];
let modelUpdatesStore = [...mockModelUpdates];

export function getReportsByPatient(patientId: string): MedicalReport[] {
  return reportsStore.filter(r => r.patientId === patientId);
}

export function getReportsByLab(labId: string): MedicalReport[] {
  return reportsStore.filter(r => r.labId === labId);
}

export function getAllReports(): MedicalReport[] {
  return reportsStore;
}

export function addReport(report: Omit<MedicalReport, 'id' | 'createdAt' | 'updatedAt'>): MedicalReport {
  const newReport: MedicalReport = {
    ...report,
    id: `report-${Date.now()}`,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  reportsStore.push(newReport);
  return newReport;
}

export function updateReportStatus(reportId: string, status: MedicalReport['status']): void {
  const report = reportsStore.find(r => r.id === reportId);
  if (report) {
    report.status = status;
    report.updatedAt = new Date().toISOString();
  }
}

export function getDiagnosisByReportId(reportId: string): Diagnosis | undefined {
  return diagnosesStore.find(d => d.reportId === reportId);
}

export function addDiagnosis(diagnosis: Omit<Diagnosis, 'id' | 'createdAt'>): Diagnosis {
  const newDiagnosis: Diagnosis = {
    ...diagnosis,
    id: `diag-${Date.now()}`,
    createdAt: new Date().toISOString(),
  };
  diagnosesStore.push(newDiagnosis);
  return newDiagnosis;
}

export function getModelUpdatesByLab(labId: string): ModelUpdate[] {
  return modelUpdatesStore.filter(m => m.labId === labId);
}

export function getAllModelUpdates(): ModelUpdate[] {
  return modelUpdatesStore;
}

export function addModelUpdate(update: Omit<ModelUpdate, 'id' | 'createdAt'>): ModelUpdate {
  const newUpdate: ModelUpdate = {
    ...update,
    id: `model-${Date.now()}`,
    createdAt: new Date().toISOString(),
  };
  modelUpdatesStore.push(newUpdate);
  return newUpdate;
}

export function aggregateModelUpdate(updateId: string): void {
  const update = modelUpdatesStore.find(m => m.id === updateId);
  if (update) {
    update.isAggregated = true;
  }
}
