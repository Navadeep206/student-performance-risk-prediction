import React from 'react';
import { ChevronRight, Calendar, UserCheck } from 'lucide-react';
import { NavTab } from '../types';

interface TopNavProps {
  currentTab: NavTab;
}

const TAB_TITLES: Record<NavTab, { title: string; subtitle: string }> = {
  overview: {
    title: 'Executive Cohort Overview',
    subtitle: 'Prospective 2014J intake risk distribution and workload modeling',
  },
  'student-explorer': {
    title: 'Student Risk Explorer',
    subtitle: 'Pseudonymized assessment snapshot ledger with calibrated triage thresholds',
  },
  modules: {
    title: 'Course Module Analytics',
    subtitle: '7-Module comparative failure prevalence, alert density, and GGG governance audit',
  },
  assessments: {
    title: 'Assessment & Milestone Dynamics',
    subtitle: 'TMA coursework tracking, CMA outreach suppression, and cold-start lateness analysis',
  },
  explainability: {
    title: 'Model Explainability (TreeSHAP)',
    subtitle: 'Certified global feature attribution and advisory interpretation guidance',
  },
  governance: {
    title: 'Governance & Compliance Ledger',
    subtitle: 'Audit trail of model inferences, threshold enforcement, and privacy safeguards',
  },
  monitoring: {
    title: 'Model Monitoring & Generalization',
    subtitle: 'Evaluation gap benchmarking, population stability index (PSI), and calibration drift',
  },
};

export const TopNav: React.FC<TopNavProps> = ({ currentTab }) => {
  const current = TAB_TITLES[currentTab] || { title: 'Analytics', subtitle: '' };

  return (
    <header className="h-16 bg-white border-b border-border-subtle px-6 flex items-center justify-between z-10 shrink-0">
      {/* Breadcrumb & Title */}
      <div className="flex flex-col justify-center">
        <div className="flex items-center space-x-1.5 text-xs text-text-secondary">
          <span>Institutional Portal</span>
          <ChevronRight size={12} className="text-slate-400" />
          <span className="font-medium text-text-primary capitalize">{currentTab.replace('-', ' ')}</span>
        </div>
        <h1 className="text-base font-semibold text-text-primary leading-tight mt-0.5">
          {current.title}
        </h1>
      </div>

      {/* Action Controls & Session Context */}
      <div className="flex items-center space-x-4">
        {/* Intake Selector */}
        <div className="flex items-center space-x-2 bg-slate-50 border border-border-subtle rounded-md px-3 py-1.5 text-xs">
          <Calendar size={14} className="text-slate-500" />
          <span className="text-slate-600 font-medium">Cohort Intake:</span>
          <span className="font-semibold text-primary-blue">2014J (Holdout)</span>
        </div>

        {/* User Session Indicator */}
        <div className="flex items-center space-x-2 bg-slate-50 border border-border-subtle rounded-md px-3 py-1.5 text-xs">
          <UserCheck size={14} className="text-emerald-600" />
          <span className="text-slate-600">Role:</span>
          <span className="font-medium text-slate-800">Academic Advisor</span>
          <span className="text-[10px] bg-slate-200 text-slate-700 px-1.5 py-0.2 rounded font-semibold">READ-ONLY</span>
        </div>
      </div>
    </header>
  );
};

