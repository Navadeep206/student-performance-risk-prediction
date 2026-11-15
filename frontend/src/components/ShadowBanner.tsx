import React from 'react';
import { ShieldAlert } from 'lucide-react';

export const ShadowBanner: React.FC = () => {
  return (
    <div className="w-full bg-[#1A365D] border-b border-[#2A4365] px-4 py-2 flex items-center justify-between text-white text-xs font-medium z-50">
      <div className="flex items-center space-x-2">
        <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-[#2B6CB0] text-white uppercase tracking-wider">
          Passive Advisory
        </span>
        <span className="font-semibold tracking-wide">
          SHADOW MODE ACTIVE — PASSIVE ACADEMIC ADVISORY VIEW ONLY. STUDENT COMMUNICATION AND AUTOMATED OUTREACH ARE PROHIBITED.
        </span>
      </div>
      <div className="flex items-center space-x-3 text-slate-300">
        <span className="hidden sm:inline">Governance Policy: v1.0 (Strict Lockdown)</span>
        <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
      </div>
    </div>
  );
};

