import { useState } from 'react';
import Layout from '../Layout';
import UploadReport from './UploadReport';
import ViewReports from './ViewReports';
import Profile from './Profile';
import { Upload, FileText, User } from 'lucide-react';

type Tab = 'upload' | 'reports' | 'profile';

export default function PatientDashboard() {
  const [activeTab, setActiveTab] = useState<Tab>('upload');

  const tabs = [
    { id: 'upload' as Tab, label: 'Upload Report', icon: Upload },
    { id: 'reports' as Tab, label: 'My Reports', icon: FileText },
    { id: 'profile' as Tab, label: 'Profile', icon: User },
  ];

  return (
    <Layout title="Patient Dashboard">
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
        {activeTab === 'upload' && <UploadReport />}
        {activeTab === 'reports' && <ViewReports />}
        {activeTab === 'profile' && <Profile />}
      </div>
    </Layout>
  );
}
