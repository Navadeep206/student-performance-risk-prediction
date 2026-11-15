import React, { useState } from 'react';
import { X, Copy, Check, ShieldAlert, AlertTriangle, Info, BookOpen } from 'lucide-react';
import { StudentExplorerItem } from '../types';

interface DetailDrawerProps {
  student: StudentExplorerItem | null;
  onClose: () => void;
}

export const DetailDrawer: React.FC<DetailDrawerProps> = ({ student, onClose }) => {
  const [copied, setCopied] = useState(false);

  if (!student) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(student.pseudonymized_student_id);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isHighRisk = student.risk_tier === 'HIGH_RISK';
  const isMediumRisk = student.risk_tier === 'MEDIUM_RISK';
  const isSuppressed = student.risk_tier === 'SUPPRESSED';

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-slate-900/40 backdrop-blur-[1px] flex justify-end">
      <div className="w-full max-w-lg bg-white h-full shadow-2xl flex flex-col border-l border-border-medium animate-in slide-in-from-right duration-200">
        {/* Drawer Header */}
        <div className="p-5 border-b border-border-subtle flex items-start justify-between bg-slate-50">
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Student Assessment Snapshot
              </span>
              <span className="text-[10px] bg-slate-200 text-slate-700 px-1.5 py-0.5 rounded font-mono font-medium">
                {student.code_presentation}
              </span>
            </div>
            <div className="flex items-center space-x-2 mt-1">
              <span className="text-base font-bold font-mono text-text-primary">
                {student.pseudonymized_student_id}
              </span>
              <button
                onClick={handleCopy}
                title="Copy Pseudonymized ID"
                className="text-slate-400 hover:text-slate-600 p-1 rounded hover:bg-slate-200 transition-colors"
              >
                {copied ? <Check size={14} className="text-emerald-600" /> : <Copy size={14} />}
              </button>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 p-1.5 rounded-md hover:bg-slate-200 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Drawer Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {/* Risk Prediction Summary Banner */}
          <div
            className={`p-4 rounded-lg border ${
              isHighRisk
                ? 'bg-risk-high-surface border-risk-high-border text-risk-high-text'
                : isMediumRisk
                ? 'bg-risk-mod-surface border-risk-mod-border text-risk-mod-text'
                : isSuppressed
                ? 'bg-risk-suppressed-surface border-risk-suppressed-border text-risk-suppressed-text'
                : 'bg-risk-low-surface border-risk-low-border text-risk-low-text'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <AlertTriangle size={18} />
                <span className="font-semibold text-sm">
                  {student.risk_tier.replace('_', ' ')}
                </span>
              </div>
              <span className="text-xl font-bold font-mono">
                {(student.predicted_risk_score * 100).toFixed(1)}%
              </span>
            </div>
            <div className="mt-2 text-xs leading-relaxed opacity-90">
              {student.advising_recommendation}
            </div>
          </div>

          {/* Assessment Context Card */}
          <div className="enterprise-card p-4 space-y-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-text-secondary flex items-center space-x-1.5">
              <BookOpen size={14} className="text-primary-blue" />
              <span>Assessment Context</span>
            </h3>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-slate-50 p-2.5 rounded border border-border-subtle">
                <span className="text-text-muted block text-[11px]">Module & Code</span>
                <span className="font-semibold text-text-primary text-sm">
                  Module {student.code_module}
                </span>
              </div>
              <div className="bg-slate-50 p-2.5 rounded border border-border-subtle">
                <span className="text-text-muted block text-[11px]">Assessment Type</span>
                <span className="font-semibold text-text-primary text-sm">
                  {student.assessment_type} (ID #{student.id_assessment})
                </span>
              </div>
              <div className="bg-slate-50 p-2.5 rounded border border-border-subtle">
                <span className="text-text-muted block text-[11px]">Scheduled Day</span>
                <span className="font-semibold text-text-primary text-sm">
                  Day {student.scheduled_date}
                </span>
              </div>
              <div className="bg-slate-50 p-2.5 rounded border border-border-subtle">
                <span className="text-text-muted block text-[11px]">Coursework Weight</span>
                <span className="font-semibold text-text-primary text-sm">
                  {student.assessment_weight}%
                </span>
              </div>
            </div>
          </div>

          {/* Threshold & Governance Policy Card */}
          <div className="enterprise-card p-4 space-y-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-text-secondary flex items-center space-x-1.5">
              <ShieldAlert size={14} className="text-primary-blue" />
              <span>Operational Governance Policy</span>
            </h3>
            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between py-1.5 border-b border-border-subtle">
                <span className="text-text-secondary">Operational Threshold Applied:</span>
                <span className="font-mono font-semibold text-text-primary">
                  {student.operational_threshold_applied !== null
                    ? `${(student.operational_threshold_applied * 100).toFixed(0)}% (${
                        student.code_module === 'GGG' ? 'Course Calibrated' : 'University Standard'
                      })`
                    : 'N/A (Suppressed)'}
                </span>
              </div>
              <div className="flex items-center justify-between py-1.5 border-b border-border-subtle">
                <span className="text-text-secondary">Governance Action:</span>
                <span className="font-medium text-slate-800">
                  {student.governance_action.replace(/_/g, ' ')}
                </span>
              </div>
              <div className="flex items-center justify-between py-1.5 border-b border-border-subtle">
                <span className="text-text-secondary">Automated Outreach:</span>
                <span className="text-emerald-700 font-semibold bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                  Strictly Prohibited
                </span>
              </div>
              <div className="flex items-center justify-between py-1.5">
                <span className="text-text-secondary">Student Communication Permission:</span>
                <span className="text-slate-600 font-mono">student_outreach_enabled = false</span>
              </div>
            </div>
          </div>

          {/* Contextual Behavior Attributes */}
          <div className="enterprise-card p-4 space-y-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-text-secondary flex items-center space-x-1.5">
              <Info size={14} className="text-primary-blue" />
              <span>Curricular Trajectory Signals</span>
            </h3>
            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between py-1 border-b border-border-subtle/60">
                <span className="text-text-secondary">Academic Milestone:</span>
                <span className="font-medium text-text-primary">
                  {student.influential_features.course_milestone || 'In-Course Milestone'}
                </span>
              </div>
              <div className="flex items-center justify-between py-1 border-b border-border-subtle/60">
                <span className="text-text-secondary">Cohort Experience:</span>
                <span className="font-medium text-text-primary">
                  {student.is_novel_student ? 'First-Time Enrolled Learner' : 'Continuing Student'}
                </span>
              </div>
              <div className="flex items-center justify-between py-1">
                <span className="text-text-secondary">Prediction Evaluation Horizon:</span>
                <span className="font-medium text-text-primary">
                  7-Day Advance Warning (14-Day Grading Delay)
                </span>
              </div>
            </div>
          </div>

          {/* Non-Punitive Advisory Disclaimer */}
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3.5 text-xs text-amber-900 space-y-1">
            <span className="font-semibold flex items-center space-x-1.5">
              <ShieldAlert size={14} className="text-amber-700" />
              <span>Mandatory Advisory Notice</span>
            </span>
            <p className="leading-relaxed text-[11px] text-amber-800">
              Risk predictions are generated in read-only shadow mode to assist academic advisors with proactive,
              non-punitive study support. Automated messaging is disabled. Predictions reflect statistical patterns
              and must be evaluated in human context by qualified advisors.
            </p>
          </div>
        </div>

        {/* Drawer Footer */}
        <div className="p-4 border-t border-border-subtle bg-slate-50 flex items-center justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-white border border-border-subtle hover:bg-slate-100 rounded-md text-xs font-medium text-text-secondary transition-colors"
          >
            Close Snapshot
          </button>
        </div>
      </div>
    </div>
  );
};

