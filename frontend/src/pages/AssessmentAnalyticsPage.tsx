import React from 'react';
import { 
  FileSpreadsheet, 
  AlertCircle, 
  Clock, 
  Layers, 
  ShieldAlert, 
  CheckCircle,
  HelpCircle
} from 'lucide-react';
import { KpiCard } from '../components/KpiCard';

export const AssessmentAnalyticsPage: React.FC = () => {
  return (
    <div className="p-8 space-y-8">
      {/* 3 Overview KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <KpiCard
          title="TMA Coursework Snapshots"
          value="32,506"
          subtitle="65.0% of 2014J prospective snapshots"
          annotation="Human-graded coursework assessments"
          badge={{ text: "Actionable Outreach", variant: "warning" }}
          icon={<FileSpreadsheet size={20} />}
        />
        <KpiCard
          title="CMA Formative Quizzes"
          value="17,508"
          subtitle="35.0% of 2014J prospective snapshots"
          annotation="Universal automated outreach suppression"
          badge={{ text: "Telemetry Only", variant: "neutral" }}
          icon={<ShieldAlert size={20} />}
        />
        <KpiCard
          title="Final Examinations"
          value="Unobserved"
          subtitle="Administered post-cutoff at course end"
          annotation="Graded externally; excluded from in-course alerting"
          badge={{ text: "Post-Course", variant: "info" }}
          icon={<Clock size={20} />}
        />
      </div>

      {/* Row 2: Cold-Start vs Non-Cold-Start Performance Investigation */}
      <div className="enterprise-card p-6 space-y-5">
        <div className="pb-3 border-b border-border-subtle flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold text-text-primary">
              Cold-Start Milestone 1 vs Non-Cold-Start Predictive Generalization
            </h2>
            <p className="text-xs text-text-muted mt-0.5">
              Evaluating model discrimination when prior assessment history is strictly unobserved
            </p>
          </div>
          <span className="text-xs text-text-secondary bg-slate-50 border border-border-subtle px-2.5 py-1 rounded">
            Phase 9 Task 2 Certified
          </span>
        </div>

        {/* Cold-Start Metrics Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left enterprise-table">
            <thead>
              <tr>
                <th>Evaluation Population</th>
                <th>Prior Assessment History</th>
                <th className="text-right">Grouped 5-Fold CV PR-AUC</th>
                <th className="text-right">Chronological 2014J PR-AUC</th>
                <th className="text-right">Generalization Degradation</th>
                <th>Diagnostic Findings & Governance</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="font-semibold text-text-primary text-xs">
                  Cold-Start (Assessment 1 Milestones)
                </td>
                <td className="text-xs text-text-secondary">
                  Strictly 0 prior scores (unobserved)
                </td>
                <td className="text-xs font-mono text-right tabular-nums">
                  0.3203
                </td>
                <td className="text-xs font-mono font-bold text-right tabular-nums text-risk-high-base">
                  0.1687
                </td>
                <td className="text-xs font-mono text-right tabular-nums text-risk-high-base">
                  -47.3%
                </td>
                <td className="text-xs text-text-secondary">
                  High reliance on VLE onboarding clicks and registration timing. Nudges strictly advisory.
                </td>
              </tr>
              <tr>
                <td className="font-semibold text-text-primary text-xs">
                  Non-Cold-Start (Milestones 2 & 3)
                </td>
                <td className="text-xs text-text-secondary">
                  1 to 9 observed prior scores available
                </td>
                <td className="text-xs font-mono text-right tabular-nums">
                  0.4882
                </td>
                <td className="text-xs font-mono font-bold text-right tabular-nums text-primary-blue">
                  0.3478
                </td>
                <td className="text-xs font-mono text-right tabular-nums text-amber-700">
                  -28.7%
                </td>
                <td className="text-xs text-text-secondary">
                  Strong discrimination driven by recent prior score and cumulative grade momentum.
                </td>
              </tr>
              <tr className="bg-slate-50 font-medium">
                <td className="text-text-primary text-xs font-bold">
                  All Milestones Combined (2014J)
                </td>
                <td className="text-xs text-text-secondary">
                  Full prospective cohort ($N=50,014$)
                </td>
                <td className="text-xs font-mono text-right tabular-nums">
                  0.4606
                </td>
                <td className="text-xs font-mono font-bold text-right tabular-nums text-primary-blue">
                  0.3005
                </td>
                <td className="text-xs font-mono text-right tabular-nums text-slate-700">
                  -34.8%
                </td>
                <td className="text-xs text-text-secondary">
                  Overall holdout benchmark incorporating temporal intake drift and novel learner shift.
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="p-4 bg-slate-50 rounded border border-border-subtle text-xs text-text-secondary leading-relaxed space-y-1">
          <span className="font-semibold text-text-primary block">
            Methodological Separation Note
          </span>
          <p>
            The values <strong>0.3203</strong> and <strong>0.4882</strong> represent Grouped Cross-Validation performance,
            whereas <strong>0.1687</strong> and <strong>0.3478</strong> represent real prospective performance on the 2014J
            chronological holdout. These evaluations must remain strictly separated to avoid reporting inflated retrospective
            cross-validation figures as prospective expectations.
          </p>
        </div>
      </div>

      {/* Row 3: Submission Lateness Behavior & Assessment Type Architecture */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Submission Delay vs Academic Failure Correlation */}
        <div className="enterprise-card p-6 flex flex-col justify-between">
          <div>
            <div className="pb-3 border-b border-border-subtle">
              <h3 className="text-sm font-semibold text-text-primary">
                Submission Delay Dynamics & Failure Trajectory
              </h3>
              <p className="text-xs text-text-muted mt-0.5">
                Empirical relationship between submission lateness and assessment score &lt; 50
              </p>
            </div>

            <div className="mt-5 space-y-3 text-xs">
              {[
                { delay: 'Early Submission (> 2 days before due)', failureRate: '3.4%', volume: '48,210 submissions', bar: '15%' },
                { delay: 'On-Time Submission (Day of deadline)', failureRate: '7.8%', volume: '84,190 submissions', bar: '35%' },
                { delay: 'Slightly Overdue (1–3 days late)', failureRate: '18.9%', volume: '22,415 submissions', bar: '60%' },
                { delay: 'Severely Overdue (> 4 days late)', failureRate: '38.2%', volume: '11,097 submissions', bar: '95%' },
              ].map((row, i) => (
                <div key={i} className="p-3 bg-slate-50 border border-border-subtle rounded space-y-1">
                  <div className="flex justify-between font-medium">
                    <span className="text-text-primary">{row.delay}</span>
                    <span className="font-mono font-bold text-risk-high-base">{row.failureRate} Failure</span>
                  </div>
                  <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                    <div className="bg-primary-blue h-full" style={{ width: row.bar }}></div>
                  </div>
                  <div className="text-[11px] text-text-muted">{row.volume}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-border-subtle text-[11px] text-text-muted">
            Prior submission lateness mean carries mean absolute TreeSHAP = 0.1873.
          </div>
        </div>

        {/* CMA Policy Disaggregation */}
        <div className="enterprise-card p-6 flex flex-col justify-between">
          <div>
            <div className="pb-3 border-b border-border-subtle">
              <h3 className="text-sm font-semibold text-text-primary">
                Universal CMA Telemetry Outreach Suppression Policy
              </h3>
              <p className="text-xs text-text-muted mt-0.5">
                Institutional rationale for suppressing automated quiz messaging across all courses
              </p>
            </div>

            <div className="mt-5 space-y-3 text-xs leading-relaxed">
              <div className="p-3 bg-slate-50 border border-border-subtle rounded space-y-1">
                <span className="font-semibold text-text-primary block">
                  1. Alert Fatigue Prevention
                </span>
                <p className="text-text-secondary">
                  Computer-Marked Assessments are frequent, formative milestones. In 2014J alone, raw thresholding
                  would have triggered <strong>2,243 CMA alerts</strong>. Suppressing CMAs reduces advisor and student
                  alert burden by <strong>22.73%</strong>.
                </p>
              </div>

              <div className="p-3 bg-slate-50 border border-border-subtle rounded space-y-1">
                <span className="font-semibold text-text-primary block">
                  2. Module GGG vs General CMA Structure
                </span>
                <p className="text-text-secondary">
                  <strong>Module GGG:</strong> CMAs carry 0% weight and are 100% decoupled practice quizzes.<br />
                  <strong>Modules BBB, CCC, DDD, FFF:</strong> CMAs carry positive weight (1–5%), but are suppressed
                  strictly as an operational governance measure to focus interventions on high-stakes TMAs.
                </p>
              </div>

              <div className="p-3 bg-slate-50 border border-border-subtle rounded space-y-1">
                <span className="font-semibold text-text-primary block">
                  3. Diagnostic Role as Predictive Features
                </span>
                <p className="text-text-secondary">
                  While CMAs do not trigger student notifications, student activity on CMAs is fully retained
                  as input telemetry (e.g., quiz clicks and practice attempts) to enhance TMA prediction accuracy.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-border-subtle text-[11px] text-text-muted">
            Outreach permission: <span className="font-mono text-slate-700">student_outreach_enabled = false</span> (Universal)
          </div>
        </div>
      </div>
    </div>
  );
};

