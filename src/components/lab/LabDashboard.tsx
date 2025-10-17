import { useState } from 'react';
import Layout from '../Layout';
import IncomingPatients from './IncomingPatients';
import SendReport from './SendReport';
import DoctorFeedback from './DoctorFeedback';
import FederatedLearning from './FederatedLearning';
import PatientDataCollection from './PatientDataCollection';
import { Users, Upload, MessageSquare, Brain, ClipboardList } from 'lucide-react';

type Tab = 'patients' | 'send' | 'feedback' | 'federated' | 'collect';

export default function LabDashboard() {
  const [activeTab, setActiveTab] = useState<Tab>('patients');

  const tabs = [
    { id: 'patients' as Tab, label: 'Incoming Patients', icon: Users },
    { id: 'send' as Tab, label: 'Send Report', icon: Upload },
    { id: 'feedback' as Tab, label: 'Doctor Feedback', icon: MessageSquare },
    { id: 'collect' as Tab, label: 'Patient Data Collection', icon: ClipboardList },
    { id: 'federated' as Tab, label: 'Model Training', icon: Brain },
  ];

  return (
    <Layout title="Lab Dashboard">
      <div className="mb-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    flex items-center py-4 px-1 border-b-2 font-medium text-sm transition-colors
                    ${activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }
                  `}
                >
                  <Icon className="w-5 h-5 mr-2" />
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      <div>
        {activeTab === 'patients' && <IncomingPatients />}
        {activeTab === 'send' && <SendReport />}
        {activeTab === 'feedback' && <DoctorFeedback />}
        {activeTab === 'collect' && <PatientDataCollection />}
        {activeTab === 'federated' && <FederatedLearning />}
      </div>
    </Layout>
  );
}
