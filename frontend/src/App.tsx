import React, { useState } from 'react';
import { NavTab } from './types';
import { ShadowBanner } from './components/ShadowBanner';
import { Sidebar } from './components/Sidebar';
import { TopNav } from './components/TopNav';
import { OverviewPage } from './pages/OverviewPage';
import { StudentExplorerPage } from './pages/StudentExplorerPage';
import { ModuleAnalyticsPage } from './pages/ModuleAnalyticsPage';
import { AssessmentAnalyticsPage } from './pages/AssessmentAnalyticsPage';
import { ExplainabilityPage } from './pages/ExplainabilityPage';
import { GovernanceAuditPage } from './pages/GovernanceAuditPage';
import { ModelMonitoringPage } from './pages/ModelMonitoringPage';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>('overview');

  const renderActivePage = () => {
    switch (currentTab) {
      case 'overview':
        return <OverviewPage />;
      case 'student-explorer':
        return <StudentExplorerPage />;
      case 'modules':
        return <ModuleAnalyticsPage />;
      case 'assessments':
        return <AssessmentAnalyticsPage />;
      case 'explainability':
        return <ExplainabilityPage />;
      case 'governance':
        return <GovernanceAuditPage />;
      case 'monitoring':
        return <ModelMonitoringPage />;
      default:
        return <OverviewPage />;
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-canvas-bg text-text-primary antialiased font-sans">
      {/* Persistent Shadow Mode Safety Banner */}
      <ShadowBanner />

      {/* Main Layout: Fixed 260px Sidebar + Fluid Content Area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar currentTab={currentTab} onSelectTab={setCurrentTab} />

        {/* Right Content Column */}
        <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
          {/* Top Context Navigation */}
          <TopNav currentTab={currentTab} />

          {/* Page Content Container */}
          <main className="flex-1 min-w-0">
            {renderActivePage()}
          </main>
        </div>
      </div>
    </div>
  );
};
export default App;

