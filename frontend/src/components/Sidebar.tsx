import React from 'react';
import { 
  LayoutDashboard, 
  Users, 
  BookOpen, 
  Cpu, 
  ShieldCheck, 
  Activity, 
  GraduationCap,
  Lock
} from 'lucide-react';
import { NavTab } from '../types';

interface SidebarProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentTab, onSelectTab }) => {
  const navItems: { id: NavTab; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview', icon: <LayoutDashboard size={18} /> },
    { id: 'student-explorer', label: 'Student Risk Explorer', icon: <Users size={18} /> },
    { id: 'modules', label: 'Module Analytics', icon: <BookOpen size={18} /> },
    { id: 'explainability', label: 'Model Explainability', icon: <Cpu size={18} /> },
    { id: 'governance', label: 'Governance & Audit', icon: <ShieldCheck size={18} /> },
    { id: 'monitoring', label: 'Model Monitoring', icon: <Activity size={18} /> },
  ];

  return (
    <aside className="w-[260px] bg-sidebar-bg min-h-screen flex flex-col text-slate-200 border-r border-[#1C3250] select-none shrink-0">
      {/* Brand Header */}
      <div className="h-16 flex items-center px-5 border-b border-[#1C3250] space-x-3">
        <div className="w-8 h-8 rounded bg-primary-blue flex items-center justify-center text-white font-bold">
          <GraduationCap size={20} />
        </div>
        <div className="flex flex-col">
          <span className="text-sm font-semibold text-white tracking-tight">Academic Analytics</span>
          <span className="text-[11px] text-slate-400">Student Early Warning Portal</span>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 py-4 px-3 space-y-1">
        <div className="px-3 pb-2 text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
          Decision Intelligence
        </div>
        {navItems.map((item) => {
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-md text-xs font-medium transition-colors text-left ${
                isActive
                  ? 'bg-sidebar-active text-white border-l-[3px] border-primary-blue font-semibold pl-[9px]'
                  : 'text-slate-300 hover:bg-sidebar-hover hover:text-white'
              }`}
            >
              <span className={isActive ? 'text-blue-400' : 'text-slate-400'}>
                {item.icon}
              </span>
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Sidebar Footer: Shadow Mode & Status Indicator */}
      <div className="p-4 border-t border-[#1C3250] bg-[#0E1A2B] text-[11px] space-y-2">
        <div className="flex items-center justify-between text-slate-400">
          <span className="flex items-center space-x-1.5">
            <Lock size={12} className="text-amber-400" />
            <span>Mode:</span>
          </span>
          <span className="font-semibold text-slate-200">Shadow (Read-Only)</span>
        </div>

        <div className="flex items-center justify-between text-slate-400">
          <span>Model:</span>
          <span className="font-mono text-slate-200">HGB v1.0.0</span>
        </div>

        <div className="flex items-center justify-between text-slate-400 pt-1 border-t border-[#1C3250]/60">
          <span>System Health:</span>
          <span className="flex items-center space-x-1.5 text-emerald-400 font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            <span>Operational</span>
          </span>
        </div>
      </div>
    </aside>
  );
};

