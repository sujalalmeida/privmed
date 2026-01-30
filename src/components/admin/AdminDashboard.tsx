import { useState } from 'react';
import Layout from '../Layout';
import IncomingReports from './IncomingReports';
import ModelAggregation from './ModelAggregation';
import RoundHistory from './RoundHistory';
import ABTestDashboard from './ABTestDashboard';
import FeedbackStats from './FeedbackStats';
import { FileText, Brain, Activity, FlaskConical, BarChart3 } from 'lucide-react';

type Tab = 'reports' | 'models' | 'rounds' | 'abtest' | 'feedback';

export default function AdminDashboard() {
  const [activeTab, setActiveTab] = useState<Tab>('reports');

  const tabs = [
    { id: 'reports' as Tab, label: 'Incoming Reports', icon: FileText },
    { id: 'feedback' as Tab, label: 'Feedback Stats', icon: BarChart3 },
    { id: 'models' as Tab, label: 'Model Aggregation', icon: Brain },
    { id: 'rounds' as Tab, label: 'FL Performance', icon: Activity },
    { id: 'abtest' as Tab, label: 'A/B Testing', icon: FlaskConical },
  ];

  return (
    <Layout title="Central Admin Dashboard">
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
        {activeTab === 'reports' && <IncomingReports />}
        {activeTab === 'feedback' && <FeedbackStats />}
        {activeTab === 'models' && <ModelAggregation />}
        {activeTab === 'rounds' && <RoundHistory />}
        {activeTab === 'abtest' && <ABTestDashboard />}
      </div>
    </Layout>
  );
}
