import { useState } from 'react';
import Layout from '../Layout';
import IncomingPatients from './IncomingPatients';
import SendReport from './SendReport';
import DoctorFeedback from './DoctorFeedback';
import FederatedLearning from './FederatedLearning';
import ClinicalDataEntry from './ClinicalDataEntry';
import { Users, Upload, MessageSquare, Brain, ClipboardList } from 'lucide-react';

type Tab = 'patients' | 'send' | 'feedback' | 'federated' | 'collect';

export default function LabDashboard() {
  const [activeTab, setActiveTab] = useState<Tab>('collect');

  const tabs = [
    { id: 'collect' as Tab, label: 'Clinical Data Entry', icon: ClipboardList },
    { id: 'patients' as Tab, label: 'Incoming Patients', icon: Users },
    { id: 'send' as Tab, label: 'Send Report', icon: Upload },
    { id: 'feedback' as Tab, label: 'Doctor Feedback', icon: MessageSquare },
    { id: 'federated' as Tab, label: 'Model Training', icon: Brain },
  ];

  return (
    <Layout title="Lab Dashboard">
      <div className="mb-6">
        <nav className="tab-nav overflow-x-auto scrollbar-hide">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`tab-item whitespace-nowrap ${activeTab === tab.id ? 'tab-item-active' : ''}`}
              >
                <Icon className="w-4 h-4 mr-2" />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>

      <div>
        {activeTab === 'collect' && <ClinicalDataEntry />}
        {activeTab === 'patients' && <IncomingPatients />}
        {activeTab === 'send' && <SendReport />}
        {activeTab === 'feedback' && <DoctorFeedback />}
        {activeTab === 'federated' && <FederatedLearning />}
      </div>
    </Layout>
  );
}
