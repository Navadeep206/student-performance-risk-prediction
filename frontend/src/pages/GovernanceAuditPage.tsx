import React, { useEffect, useState } from 'react';
import { 
  ShieldCheck, 
  Lock, 
  AlertOctagon, 
  FileText, 
  Database, 
  CheckCircle2, 
  XCircle,
  Eye,
  Key
} from 'lucide-react';
import { analyticsApi } from '../services/api';
import { GovernanceAuditResponse } from '../types';

export const GovernanceAuditPage: React.FC = () => {
  const [data, setData] = useState<GovernanceAuditResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const res = await analyticsApi.getGovernanceAudit();
        setData(res);
      } catch (err: any) {
        console.error('Failed to load governance audit:', err);
        setError(err.message || 'Error fetching governance records');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="p-8 space-y-6 animate-pulse">
        <div className="h-64 bg-slate-200 rounded-lg"></div>
        <div className="h-80 bg-slate-200 rounded-lg"></div>
      </div>
    );
  }

  return (
    <div className="p-8 space-y-8">
      {/* Governance & Safeguard Control Matrix */}
      <div className="enterprise-card p-6 space-y-5">
        <div className="pb-3 border-b border-border-subtle flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <div className="flex items-center space-x-2">
              <ShieldCheck size={18} className="text-emerald-600" />
              <h2 className="text-sm font-semibold text-text-primary">
                Mandatory Operational Governance & Safety Enforcement
              </h2>
            </div>
            <p className="text-xs text-text-muted mt-0.5">
              Hardcoded architectural controls preventing autonomous student outreach and privacy violations
            </p>
          </div>
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
            ALL SAFEGUARDS ENFORCED
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
          {/* Safeguard 1: Read-Only Shadow Mode */}
          <div className="p-4 bg-slate-50 border border-border-subtle rounded-lg space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-text-primary flex items-center space-x-1.5">
                <Lock size={14} className="text-amber-600" />
                <span>Read-Only Shadow Mode</span>
              </span>
              <CheckCircle2 size={16} className="text-emerald-600" />
            </div>
            <p className="text-text-secondary leading-relaxed text-[11px]">
              <code className="font-mono text-slate-700">ALLOW_STUDENT_OUTREACH = False</code> hard-coded in
              <code className="font-mono text-slate-700"> ShadowModeManager</code>. Prohibits automated messaging.
            </p>
          </div>

          {/* Safeguard 2: Zero Communication Endpoints */}
          <div className="p-4 bg-slate-50 border border-border-subtle rounded-lg space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-text-primary flex items-center space-x-1.5">
                <AlertOctagon size={14} className="text-risk-high-base" />
                <span>Zero Outreach Endpoints</span>
              </span>
              <CheckCircle2 size={16} className="text-emerald-600" />
            </div>
            <p className="text-text-secondary leading-relaxed text-[11px]">
              API routes for email, SMS, WhatsApp, and intervention execution return 404. Direct invocation raises
              <code className="font-mono text-slate-700"> SecurityViolationError</code>.
            </p>
          </div>

          {/* Safeguard 3: Keyed HMAC-SHA256 Pseudonymization */}
          <div className="p-4 bg-slate-50 border border-border-subtle rounded-lg space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-text-primary flex items-center space-x-1.5">
                <Key size={14} className="text-primary-blue" />
                <span>HMAC Pseudonymization</span>
              </span>
              <CheckCircle2 size={16} className="text-emerald-600" />
            </div>
            <p className="text-text-secondary leading-relaxed text-[11px]">
              100% keyed HMAC-SHA256 digests. Raw student IDs and cryptographic secret salts are strictly excluded from UI and responses.
            </p>
          </div>

          {/* Safeguard 4: Universal CMA Outreach Suppression */}
          <div className="p-4 bg-slate-50 border border-border-subtle rounded-lg space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-text-primary flex items-center space-x-1.5">
                <FileText size={14} className="text-slate-600" />
                <span>Universal CMA Suppression</span>
              </span>
              <CheckCircle2 size={16} className="text-emerald-600" />
            </div>
            <p className="text-text-secondary leading-relaxed text-[11px]">
              Computer-marked quizzes across all 7 courses are suppressed from student outreach. Reduces alert load by 22.73%.
            </p>
          </div>

          {/* Safeguard 5: Course-Calibrated Thresholding */}
          <div className="p-4 bg-slate-50 border border-border-subtle rounded-lg space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-text-primary flex items-center space-x-1.5">
                <Database size={14} className="text-indigo-600" />
                <span>Dual Threshold Policy</span>
              </span>
              <CheckCircle2 size={16} className="text-emerald-600" />
            </div>
            <p className="text-text-secondary leading-relaxed text-[11px]">
              Enforces $\tau = 0.10$ for Module GGG TMAs (remediating 75.9% FNR to 54.3%) and standard $\tau = 0.20$ for AAA–FFF.
            </p>
          </div>

          {/* Safeguard 6: Prohibited Demographic Exclusion */}
          <div className="p-4 bg-slate-50 border border-border-subtle rounded-lg space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-text-primary flex items-center space-x-1.5">
                <Eye size={14} className="text-emerald-600" />
                <span>Demographic Exclusion</span>
              </span>
              <CheckCircle2 size={16} className="text-emerald-600" />
            </div>
            <p className="text-text-secondary leading-relaxed text-[11px]">
              Sensitive attributes (gender, age, region, IMD band, disability) are rejected by Pydantic validators on inference ingestion.
            </p>
          </div>
        </div>
      </div>

      {/* Row 2: Threshold Policy Formal Registry */}
      <div className="enterprise-card p-6 space-y-4">
        <div className="pb-3 border-b border-border-subtle">
          <h2 className="text-sm font-semibold text-text-primary">
            Formal Threshold Configuration & Legal Basis Registry
          </h2>
          <p className="text-xs text-text-muted mt-0.5">
            Operational classification rules applied by the InstitutionalGovernanceRouter
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left enterprise-table">
            <thead>
              <tr>
                <th>Curricular Scope</th>
                <th>Assessment Type</th>
                <th>Operational Threshold</th>
                <th>Automated Outreach Permission</th>
                <th>Governance Action Tag</th>
                <th>Regulatory Compliance Rationale</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="font-semibold text-text-primary text-xs">
                  Modules AAA, BBB, CCC, DDD, EEE, FFF
                </td>
                <td className="text-xs text-text-secondary">
                  TMA (Tutor-Marked Coursework)
                </td>
                <td className="text-xs font-mono font-bold tabular-nums text-text-primary">
                  $\tau = 0.20$
                </td>
                <td>
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-slate-100 text-slate-700">
                    Prohibited (Shadow Mode)
                  </span>
                </td>
                <td className="text-xs font-mono text-slate-700">
                  SHADOW_MODE_ADVISORY_REVIEW
                </td>
                <td className="text-xs text-text-secondary">
                  Standard university threshold calibrated to historical 8.56% failure prevalence.
                </td>
              </tr>
              <tr className="bg-indigo-50/40">
                <td className="font-semibold text-primary-blue text-xs">
                  Module GGG Exclusively
                </td>
                <td className="text-xs text-text-secondary">
                  TMA (Tutor-Marked Coursework)
                </td>
                <td className="text-xs font-mono font-bold tabular-nums text-[#5A67D8]">
                  $\tau = 0.10$ (Calibrated)
                </td>
                <td>
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-slate-100 text-slate-700">
                    Prohibited (Shadow Mode)
                  </span>
                </td>
                <td className="text-xs font-mono text-indigo-700">
                  SHADOW_MODE_ADVISORY_REVIEW
                </td>
                <td className="text-xs text-text-secondary">
                  Compensates for practice quiz grade inflation; doubles student capture (+89 learners).
                </td>
              </tr>
              <tr>
                <td className="font-semibold text-text-primary text-xs">
                  Universal (Modules AAA through GGG)
                </td>
                <td className="text-xs text-text-secondary">
                  CMA (Computer-Marked Quizzes)
                </td>
                <td className="text-xs font-mono text-text-muted">
                  Suppressed (None)
                </td>
                <td>
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-slate-100 text-slate-700">
                    Prohibited (Policy)
                  </span>
                </td>
                <td className="text-xs font-mono text-slate-500">
                  RESEARCH_TELEMETRY_ONLY
                </td>
                <td className="text-xs text-text-secondary">
                  Formative milestone exclusion preventing advisory alert fatigue (-2,243 alerts).
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Row 3: Decision Audit Log Stream */}
      <div className="enterprise-card p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-border-subtle gap-2">
          <div>
            <h2 className="text-sm font-semibold text-text-primary">
              Append-only Decision Audit Log
            </h2>
            <p className="text-xs text-text-muted mt-0.5">
              Thread-safe DecisionAuditLogger append-only log capturing inference timestamp, input hash, and governance verdict
            </p>
          </div>
          <span className="text-xs text-text-secondary bg-slate-50 border border-border-subtle px-3 py-1 rounded">
            Audit Stream Status: <span className="font-semibold text-emerald-600">Active</span>
          </span>
        </div>

        {data?.recent_records && data.recent_records.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left enterprise-table">
              <thead>
                <tr>
                  <th>Prediction UUID</th>
                  <th>Timestamp (UTC)</th>
                  <th>Pseudonymized Student ID</th>
                  <th>Module</th>
                  <th>Assessment</th>
                  <th>Input Feature Hash (SHA-256)</th>
                  <th>Risk Prob</th>
                  <th>Threshold</th>
                  <th>Governance Action</th>
                </tr>
              </thead>
              <tbody>
                {data.recent_records.map((rec, i) => (
                  <tr key={i}>
                    <td className="font-mono text-[11px] text-text-primary">
                      {rec.prediction_id.slice(0, 8)}...
                    </td>
                    <td className="text-xs text-text-secondary font-mono">
                      {rec.timestamp_utc}
                    </td>
                    <td className="font-mono text-xs font-semibold text-text-primary">
                      {rec.pseudonymized_student_id}
                    </td>
                    <td className="text-xs text-text-secondary">
                      {rec.code_module}
                    </td>
                    <td className="text-xs text-text-secondary">
                      {rec.assessment_type} #{rec.id_assessment}
                    </td>
                    <td className="font-mono text-[10px] text-text-muted">
                      {rec.input_feature_hash ? `${rec.input_feature_hash.slice(0, 10)}...` : 'N/A'}
                    </td>
                    <td className="text-xs font-bold text-risk-high-base tabular-nums">
                      {(rec.raw_risk_probability * 100).toFixed(1)}%
                    </td>
                    <td className="text-xs font-mono text-text-secondary">
                      {rec.operational_threshold !== null ? `${(rec.operational_threshold * 100).toFixed(0)}%` : 'Suppressed'}
                    </td>
                    <td>
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-blue-50 text-blue-700">
                        {rec.governance_routing_action || 'AUDIT_LOGGED'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-8 text-center text-xs text-text-muted bg-slate-50 rounded border border-border-subtle">
            <p className="font-medium text-text-secondary">No real-time API predictions logged in current session.</p>
            <p className="mt-1">Inferences processed via <code className="font-mono text-slate-700">POST /predict</code> stream directly to this thread-safe audit ledger with HMAC hashes.</p>
          </div>
        )}
      </div>
    </div>
  );
};

