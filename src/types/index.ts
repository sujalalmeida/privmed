export type UserRole = 'patient' | 'lab' | 'central_admin';

export interface User {
  id: string;
  email: string;
  fullName: string;
  role: UserRole;
  contactPhone?: string;
  assignedLabId?: string;
  labName?: string;
}

export interface MedicalReport {
  id: string;
  patientId: string;
  patientName?: string;
  labId: string;
  labName?: string;
  testType: string;
  encryptedData?: string;
  fileName?: string;
  status: 'pending' | 'under_review' | 'diagnosed';
  aiPrediction?: {
    disease: string;
    confidence: number;
  }[];
  createdAt: string;
  updatedAt: string;
}

export interface Diagnosis {
  id: string;
  reportId: string;
  doctorId: string;
  doctorName: string;
  encryptedFeedback: string;
  diagnosisResult: string;
  confidenceScore: number;
  createdAt: string;
}

export interface ModelUpdate {
  id: string;
  labId: string;
  labName: string;
  modelVersion: string;
  accuracy: number;
  isAggregated: boolean;
  createdAt: string;
}
