import { useState } from 'react';
import Layout from '../Layout';
import UploadReport from './UploadReport';
import ViewReports from './ViewReports';
import Profile from './Profile';
import ChooseLab from './ChooseLab';
import { Upload, FileText, User, Building2 } from 'lucide-react';

type Tab = 'upload' | 'reports' | 'labs' | 'profile';

export default function PatientDashboard() {
  const [activeTab, setActiveTab] = useState<Tab>('upload');

  const tabs = [
    { id: 'upload' as Tab, label: 'Upload Report', icon: Upload },
    { id: 'reports' as Tab, label: 'My Reports', icon: FileText },
    { id: 'labs' as Tab, label: 'Choose Lab', icon: Building2 },
    { id: 'profile' as Tab, label: 'Profile', icon: User },
  ];

  return (
    <Layout title="Patient Dashboard">
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
        {activeTab === 'upload' && <UploadReport />}
        {activeTab === 'reports' && <ViewReports />}
        {activeTab === 'labs' && <ChooseLab />}
        {activeTab === 'profile' && <Profile />}
      </div>
    </Layout>
  );
}
