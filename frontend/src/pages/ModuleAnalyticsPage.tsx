import React, { useEffect, useState } from 'react';
import { 
  BookOpen, 
  AlertTriangle, 
  TrendingUp, 
  CheckCircle2, 
  HelpCircle, 
  Layers,
  ShieldCheck,
  Info
} from 'lucide-react';
import { analyticsApi } from '../services/api';
import { ModulesComparisonResponse } from '../types';

export const ModuleAnalyticsPage: React.FC = () => {
  const [data, setData] = useState<ModulesComparisonResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const res = await analyticsApi.getModulesComparison();
        setData(res);
      } catch (err: any) {
        console.error('Failed to load module analytics:', err);
        setError(err.message || 'Failed to fetch module comparisons');
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
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="h-80 bg-slate-200 rounded-lg"></div>
          <div className="h-80 bg-slate-200 rounded-lg"></div>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-8">
        <div className="p-6 bg-risk-high-surface border border-risk-high-border text-risk-high-text rounded-lg text-xs">
          Failed to load module analytics: {error}
        </div>
      </div>
    );
  }

  const ggg = data.module_ggg_matched_evaluation;

  return (
    <div className="p-8 space-y-8">
      {/* 7-Module Comparative Health Matrix */}
      <div className="enterprise-card p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-border-subtle gap-2">
          <div>
            <h2 className="text-sm font-semibold text-text-primary">
              Undergraduate Course Module Comparative Matrix (AAA through GGG)
            </h2>
            <p className="text-xs text-text-muted mt-0.5">
              Strictly disaggregated snapshot denominators, benchmark vs prospective prevalence, and governed alert rates
            </p>
          </div>
          <div className="text-xs text-text-secondary bg-slate-50 border border-border-subtle px-3 py-1 rounded">
            <span>Overall 2014J TMA Alert Rate: </span>
            <span className="font-bold text-primary-blue tabular-nums">21.93%</span>
            <span className="text-text-muted"> (7,130 / 32,506)</span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left enterprise-table">
            <thead>
              <tr>
                <th>Module</th>
                <th>Domain</th>
                <th className="text-right">Benchmark Snapshots ($N=170,874$)</th>
                <th className="text-right">Benchmark Failure %</th>
                <th className="text-right">2014J Snapshots ($N=50,014$)</th>
                <th className="text-right">2014J Failure %</th>
                <th className="text-right">2014J Actionable TMAs ($N=32,506$)</th>
                <th className="text-right">Governed Alerts ($N=7,130$)</th>
                <th className="text-right">TMA Alert Rate %</th>
                <th>Dominant Milestone</th>
              </tr>
            </thead>
            <tbody>
              {data.modules.map((m) => {
                const isHighPrevalence = m.holdout_2014j_failure_rate_pct >= 10.0;
                const isGGG = m.code_module === 'GGG';

                return (
                  <tr key={m.code_module} className={isGGG ? 'bg-blue-50/40' : ''}>
                    <td className="font-semibold text-text-primary">
                      <div className="flex items-center space-x-2">
                        <span className="inline-block px-2 py-0.5 rounded text-xs font-bold bg-slate-100 text-slate-800">
                          Module {m.code_module}
                        </span>
                        {isGGG && (
                          <span className="text-[10px] bg-indigo-100 text-indigo-800 px-1.5 py-0.2 rounded font-semibold border border-indigo-200">
                            Calibrated $\tau=0.10$
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="text-xs text-text-secondary">
                      {m.faculty_domain}
                    </td>
                    <td className="text-xs text-text-secondary text-right tabular-nums">
                      {m.benchmark_snapshots.toLocaleString()}
                    </td>
                    <td className="text-xs font-mono text-right tabular-nums text-text-primary">
                      {m.benchmark_failure_rate_pct.toFixed(2)}%
                    </td>
                    <td className="text-xs text-text-secondary text-right tabular-nums">
                      {m.holdout_2014j_snapshots.toLocaleString()}
                    </td>
                    <td className="text-xs font-mono text-right tabular-nums">
                      <span className={isHighPrevalence ? 'text-amber-700 font-bold' : 'text-text-primary'}>
                        {m.holdout_2014j_failure_rate_pct.toFixed(2)}%
                      </span>
                    </td>
                    <td className="text-xs text-text-secondary text-right tabular-nums">
                      {m.actionable_tma_snapshots_2014j.toLocaleString()}
                    </td>
                    <td className="text-xs font-bold text-text-primary text-right tabular-nums">
                      {m.governed_tma_alerts_2014j.toLocaleString()}
                    </td>
                    <td className="text-xs font-mono font-bold text-right tabular-nums">
                      <span className={m.governed_tma_alert_rate_pct > 25 ? 'text-risk-high-base' : 'text-primary-blue'}>
                        {m.governed_tma_alert_rate_pct.toFixed(2)}%
                      </span>
                    </td>
                    <td className="text-xs text-text-muted">
                      {m.dominant_milestone_risk}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Row 2: Visual Chart Comparisons */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Failure Prevalence vs Alert Rate Grouped Comparison */}
        <div className="enterprise-card p-6 flex flex-col justify-between">
          <div>
            <div className="pb-3 border-b border-border-subtle">
              <h3 className="text-sm font-semibold text-text-primary">
                Failure Prevalence vs Governed TMA Alert Rate
              </h3>
              <p className="text-xs text-text-muted mt-0.5">
                Proportionate alert allocation prevents over-alerting in low-failure courses
              </p>
            </div>

            <div className="mt-6 space-y-4">
              {data.modules.map((m) => {
                const maxPct = 40.0;
                const prevWidth = Math.min(100, (m.holdout_2014j_failure_rate_pct / maxPct) * 100);
                const alertWidth = Math.min(100, (m.governed_tma_alert_rate_pct / maxPct) * 100);

                return (
                  <div key={m.code_module} className="space-y-1">
                    <div className="flex justify-between text-xs font-medium">
                      <span>Module {m.code_module} ({m.faculty_domain})</span>
                      <span className="font-mono text-slate-600">
                        Prev: {m.holdout_2014j_failure_rate_pct.toFixed(1)}% | Alerts: {m.governed_tma_alert_rate_pct.toFixed(1)}%
                      </span>
                    </div>
                    <div className="space-y-1">
                      {/* Prevalence bar */}
                      <div className="w-full bg-slate-100 h-2.5 rounded overflow-hidden flex items-center">
                        <div
                          style={{ width: `${prevWidth}%` }}
                          className="bg-slate-400 h-full rounded"
                          title={`Prevalence: ${m.holdout_2014j_failure_rate_pct.toFixed(1)}%`}
                        ></div>
                      </div>
                      {/* Alert rate bar */}
                      <div className="w-full bg-slate-100 h-2.5 rounded overflow-hidden flex items-center">
                        <div
                          style={{ width: `${alertWidth}%` }}
                          className="bg-primary-blue h-full rounded"
                          title={`TMA Alert Rate: ${m.governed_tma_alert_rate_pct.toFixed(1)}%`}
                        ></div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-5 pt-3 border-t border-border-subtle flex items-center space-x-5 text-xs text-text-secondary">
            <span className="flex items-center space-x-1.5">
              <span className="w-3 h-2 bg-slate-400 rounded-sm"></span>
              <span>2014J Failure Prevalence %</span>
            </span>
            <span className="flex items-center space-x-1.5">
              <span className="w-3 h-2 bg-primary-blue rounded-sm"></span>
              <span>Governed TMA Alert Rate %</span>
            </span>
          </div>
        </div>

        {/* Curricular Ingestion & Telemetry Architecture */}
        <div className="enterprise-card p-6 flex flex-col justify-between">
          <div>
            <div className="pb-3 border-b border-border-subtle">
              <h3 className="text-sm font-semibold text-text-primary">
                Curricular Architecture & Assessment Disaggregation
              </h3>
              <p className="text-xs text-text-muted mt-0.5">
                Structural assessment differences across undergraduate faculties
              </p>
            </div>

            <div className="mt-5 space-y-3 text-xs leading-relaxed">
              <div className="p-3 bg-slate-50 border border-border-subtle rounded space-y-1">
                <span className="font-semibold text-text-primary block">
                  1. High-Failure Vulnerability Clusters (BBB, CCC, DDD)
                </span>
                <p className="text-text-secondary">
                  STEM modules CCC (13.49%) and DDD (11.76%) and Social Sciences module BBB (16.71%) account for
                  <strong> 5,785 of the 7,130 alerts (81.1%)</strong>. High alert rates reflect genuine academic
                  difficulty and submission lateness in early coursework assignments.
                </p>
              </div>

              <div className="p-3 bg-slate-50 border border-border-subtle rounded space-y-1">
                <span className="font-semibold text-text-primary block">
                  2. Stable Low-Failure Courses (EEE, FFF)
                </span>
                <p className="text-text-secondary">
                  STEM modules EEE (3.90% holdout failure) and FFF (3.99% holdout failure) exhibit high baseline
                  completion rates. Governed alerts remain appropriately restrained (4.56% and 10.63%),
                  preventing unnecessary advisory intervention.
                </p>
              </div>

              <div className="p-3 bg-slate-50 border border-border-subtle rounded space-y-1">
                <span className="font-semibold text-text-primary block">
                  3. Computer-Marked Assessment (CMA) Outreach Exclusion
                </span>
                <p className="text-text-secondary">
                  Across CCC (5,846 CMAs) and FFF (8,915 CMAs), quizzes carry positive marks but are strictly
                  excluded from automated outreach to prevent student alert fatigue. Quizzes remain available as
                  research telemetry.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-border-subtle text-[11px] text-text-muted">
            Faculty Domain Allocation: 2 Social Sciences courses (AAA, BBB), 5 STEM courses (CCC through GGG).
          </div>
        </div>
      </div>

      {/* Row 3: Module GGG Remediation & Governance Audit Panel */}
      <div className="enterprise-card p-6 border-indigo-200 bg-gradient-to-b from-indigo-50/20 to-white space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-indigo-100 gap-2">
          <div>
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-[#5A67D8]"></span>
              <h2 className="text-sm font-bold text-text-primary">
                Module GGG Special Governance Audit & Matched-Cohort Evaluation
              </h2>
            </div>
            <p className="text-xs text-text-muted mt-0.5">
              Remediation findings across exactly 5,656 matched TMA snapshots across 2,097 unique learners
            </p>
          </div>
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-100 text-indigo-800 border border-indigo-200">
            Phase 9 Task 3 Certified
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <div className="bg-white p-3.5 rounded border border-indigo-100 space-y-1">
            <span className="text-text-muted block text-[11px]">Matched Population</span>
            <span className="text-lg font-bold text-text-primary tabular-nums">5,656 snapshots</span>
            <span className="text-[11px] text-slate-500 block">2,097 unique students</span>
          </div>

          <div className="bg-white p-3.5 rounded border border-indigo-100 space-y-1">
            <span className="text-text-muted block text-[11px]">Observed True Failures</span>
            <span className="text-lg font-bold text-text-primary tabular-nums">411 cases</span>
            <span className="text-[11px] text-slate-500 block">7.27% baseline prevalence</span>
          </div>

          <div className="bg-white p-3.5 rounded border border-indigo-100 space-y-1">
            <span className="text-text-muted block text-[11px]">Discrimination (PR-AUC)</span>
            <span className="text-lg font-bold text-primary-blue tabular-nums">0.2508 vs 0.2353</span>
            <span className="text-[11px] text-slate-500 block">Baseline vs Remediated ($\Delta = -0.0155$)</span>
          </div>

          <div className="bg-white p-3.5 rounded border border-indigo-100 space-y-1">
            <span className="text-text-muted block text-[11px]">Calibrated Recall ($\tau=0.10$)</span>
            <span className="text-lg font-bold text-emerald-700 tabular-nums">45.01% – 45.74%</span>
            <span className="text-[11px] text-emerald-600 block">Baseline: +86 snapshots | Remediated: +107 snapshots</span>
          </div>
        </div>

        {/* Detailed Decision Threshold Tradeoff Matrix */}
        <div className="overflow-x-auto">
          <table className="w-full text-left enterprise-table bg-white">
            <thead>
              <tr>
                <th>Pipeline & Threshold Specification</th>
                <th className="text-right">Precision %</th>
                <th className="text-right">Recall %</th>
                <th className="text-right">F1-Score %</th>
                <th className="text-right">False Negative Rate (FNR)</th>
                <th>Confusion Matrix (TN / FP / FN / TP)</th>
                <th>Operational Impact</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="font-medium text-text-primary text-xs">
                  Baseline Pipeline ($\tau=0.20$ Global)
                </td>
                <td className="text-xs text-right tabular-nums">37.08%</td>
                <td className="text-xs text-right tabular-nums text-risk-high-base font-bold">24.09%</td>
                <td className="text-xs text-right tabular-nums">29.20%</td>
                <td className="text-xs text-right tabular-nums text-risk-high-base">75.91%</td>
                <td className="text-xs font-mono">5077 / 168 / 312 / 99</td>
                <td className="text-xs text-text-muted">Misses 312 failing snapshots</td>
              </tr>
              <tr>
                <td className="font-medium text-text-primary text-xs">
                  Remediated Pipeline ($\tau=0.20$ Global)
                </td>
                <td className="text-xs text-right tabular-nums">36.65%</td>
                <td className="text-xs text-right tabular-nums text-risk-high-base font-bold">19.71%</td>
                <td className="text-xs text-right tabular-nums">25.63%</td>
                <td className="text-xs text-right tabular-nums text-risk-high-base">80.29%</td>
                <td className="text-xs font-mono">5105 / 140 / 330 / 81</td>
                <td className="text-xs text-text-muted">Misses 330 failing snapshots</td>
              </tr>
              <tr className="bg-indigo-50/50 font-medium">
                <td className="text-primary-blue text-xs font-bold">
                  Baseline Pipeline ($\tau=0.10$ Calibrated)
                </td>
                <td className="text-xs text-right tabular-nums">21.24%</td>
                <td className="text-xs text-right tabular-nums text-emerald-700 font-bold">45.01%</td>
                <td className="text-xs text-right tabular-nums">28.86%</td>
                <td className="text-xs text-right tabular-nums text-emerald-700">54.99%</td>
                <td className="text-xs font-mono">4559 / 686 / 226 / 185</td>
                <td className="text-xs text-emerald-700 font-semibold">+86 Snapshots (FN: 312 &rarr; 226)</td>
              </tr>
              <tr className="bg-indigo-50/70 font-medium">
                <td className="text-primary-blue text-xs font-bold">
                  Remediated Pipeline ($\tau=0.10$ Calibrated)
                </td>
                <td className="text-xs text-right tabular-nums">18.63%</td>
                <td className="text-xs text-right tabular-nums text-emerald-700 font-bold">45.74%</td>
                <td className="text-xs text-right tabular-nums">26.48%</td>
                <td className="text-xs text-right tabular-nums text-emerald-700">54.26%</td>
                <td className="text-xs font-mono">4424 / 821 / 223 / 188</td>
                <td className="text-xs text-emerald-700 font-semibold">+107 Snapshots (FN: 330 &rarr; 223)</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Curricular & Governance Findings */}
        <div className="p-4 bg-white rounded-lg border border-indigo-100 space-y-2 text-xs">
          <span className="font-semibold text-text-primary flex items-center space-x-1.5">
            <Info size={14} className="text-indigo-600" />
            <span>Root-Cause Explanation & Policy Reconciliation</span>
          </span>
          <p className="text-text-secondary leading-relaxed">
            <strong>Why feature engineering does not raise PR-AUC:</strong> In Module GGG, all 18 Computer-Marked Assessments
            (6 per intake across 2013J, 2014B, 2014J) are 100% formative practice quizzes scheduled on Day 229/222 with
            <strong> zero course weight (0.0%)</strong>. Students achieve high marks (mean 80–87%) through open-book attempts,
            which inflates prior score features and dilutes diagnostic discrimination for written TMAs.
          </p>
          <p className="text-text-secondary leading-relaxed">
            <strong>The operational solution is within-pipeline threshold calibration:</strong> Lowering the operational threshold to
            $\tau = 0.10$ captures <strong>86 additional failure snapshots detected in Baseline</strong> (raising recall from 24.09% to 45.01%,
            reducing FN from 312 to 226) and <strong>107 additional failure snapshots detected in Remediated</strong> (raising recall from 19.71%
            to 45.74%, reducing FN from 330 to 223), while routing solely to non-punitive supportive reviews.
          </p>
          <p className="text-text-muted leading-relaxed text-[11px] italic border-t border-indigo-50 pt-1.5">
            <strong>Snapshot-Level Evaluation Note:</strong> All confusion-matrix counts (TN, FP, FN, TP) and recall calculations
            are evaluated strictly at the <em>assessment-snapshot level</em> across 5,656 snapshots, not at the unique-student level
            (representing 2,097 unique learners). A single student may contribute multiple assessment snapshots across the course.
          </p>
        </div>
      </div>
    </div>
  );
};

