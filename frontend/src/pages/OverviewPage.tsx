import React, { useEffect, useState } from 'react';
import { 
  Users, 
  AlertTriangle, 
  UserCheck, 
  Clock, 
  TrendingUp, 
  Calendar,
  Layers,
  Shield,
  HelpCircle
} from 'lucide-react';
import { KpiCard } from '../components/KpiCard';
import { analyticsApi } from '../services/api';
import { CohortSummary, RiskDistributionResponse } from '../types';

export const OverviewPage: React.FC = () => {
  const [summary, setSummary] = useState<CohortSummary | null>(null);
  const [riskDist, setRiskDist] = useState<RiskDistributionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [sumData, distData] = await Promise.all([
          analyticsApi.getCohortSummary(),
          analyticsApi.getRiskDistribution(),
        ]);
        setSummary(sumData);
        setRiskDist(distData);
      } catch (err: any) {
        console.error('Failed to load overview analytics:', err);
        setError(err.message || 'Failed to connect to Analytics API');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="p-8 space-y-6 animate-pulse">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-28 bg-slate-200 rounded-lg"></div>
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="h-80 bg-slate-200 rounded-lg"></div>
          <div className="h-80 bg-slate-200 rounded-lg"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8 space-y-8">
      {/* 4 Standardized KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        <KpiCard
          title="2014J Enrolled Learners"
          value="8,607"
          subtitle="Total unique learners enrolled in 2014J"
          annotation="Excludes assumption of 100% retention"
          icon={<Users size={20} />}
          badge={{ text: "Cohort Size", variant: "neutral" }}
        />
        <KpiCard
          title="Governed TMA Alerts"
          value="7,130"
          subtitle="TMA alerts under approved thresholds"
          annotation="21.93% of 32,506 actionable TMAs"
          icon={<AlertTriangle size={20} />}
          badge={{ text: "TMA Only", variant: "warning" }}
        />
        <KpiCard
          title="Unique Learners Alerted"
          value="3,299"
          subtitle="38.33% of 2014J enrolled cohort"
          annotation="Mean 2.16 alerts per alerted learner"
          icon={<UserCheck size={20} />}
          badge={{ text: "Student Coverage", variant: "info" }}
        />
        <KpiCard
          title="Estimated Weekly Alert Load"
          value="~209.7"
          subtitle="Historic average: 7,130 alerts / 34 wks"
          annotation="Historical derived estimate; not institutional capacity"
          icon={<Clock size={20} />}
          badge={{ text: "Historical Model", variant: "neutral" }}
        />
      </div>

      {/* Row 2: Risk Distribution Histogram + Milestone Allocation */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Risk Distribution Histogram (7 Cols) */}
        <div className="lg:col-span-7 enterprise-card p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-border-subtle">
              <div>
                <h2 className="text-sm font-semibold text-text-primary">
                  Prospective Risk Probability Distribution
                </h2>
                <p className="text-xs text-text-muted mt-0.5">
                  Binned probability distribution across 50,014 snapshots (32,506 TMAs vs 17,508 CMAs)
                </p>
              </div>
              <div className="flex items-center space-x-3 text-xs">
                <span className="flex items-center space-x-1">
                  <span className="w-3 h-3 bg-primary-blue rounded-sm"></span>
                  <span className="text-text-secondary text-[11px]">TMA (Actionable)</span>
                </span>
                <span className="flex items-center space-x-1">
                  <span className="w-3 h-3 bg-slate-300 rounded-sm"></span>
                  <span className="text-text-secondary text-[11px]">CMA (Suppressed)</span>
                </span>
              </div>
            </div>

            {/* Visual SVG Histogram Bars */}
            <div className="mt-6 space-y-2.5">
              {riskDist?.bins.map((b, idx) => {
                const maxTotal = 25000;
                const tmaWidth = Math.min(100, (b.tma_count / maxTotal) * 100);
                const cmaWidth = Math.min(100, (b.cma_count / maxTotal) * 100);
                const isOverTmaThresh = b.min_prob >= 0.20;
                const isOverGggThresh = b.min_prob >= 0.10 && b.min_prob < 0.20;

                return (
                  <div key={idx} className="flex items-center text-xs space-x-3">
                    <span className="w-20 font-mono text-[11px] text-text-secondary shrink-0 text-right">
                      {b.bin_label}
                    </span>
                    <div className="flex-1 bg-slate-100 h-5 rounded overflow-hidden flex relative">
                      {/* TMA portion */}
                      <div
                        style={{ width: `${tmaWidth}%` }}
                        className={`h-full ${
                          isOverTmaThresh
                            ? 'bg-[#C53030]'
                            : isOverGggThresh
                            ? 'bg-[#5A67D8]'
                            : 'bg-primary-blue'
                        }`}
                        title={`TMA: ${b.tma_count} (${b.tma_percentage}%)`}
                      ></div>
                      {/* CMA portion */}
                      <div
                        style={{ width: `${cmaWidth}%` }}
                        className="h-full bg-slate-300"
                        title={`CMA: ${b.cma_count} (${b.cma_percentage}%)`}
                      ></div>
                    </div>
                    <span className="w-16 font-mono text-[11px] text-text-primary text-right shrink-0">
                      {(b.total_count).toLocaleString()}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-5 pt-3 border-t border-border-subtle flex items-center justify-between text-[11px] text-text-muted">
            <span className="flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-[#5A67D8]"></span>
              <span>Calibrated &tau; = 0.10 (GGG TMA)</span>
              <span className="w-2 h-2 rounded-full bg-[#C53030] ml-3"></span>
              <span>Standard &tau; = 0.20 (AAA–FFF TMA)</span>
            </span>
            <span className="italic">CMAs 100% research telemetry</span>
          </div>
        </div>

        {/* Assessment Milestone Risk Breakdown (5 Cols) */}
        <div className="lg:col-span-5 enterprise-card p-6 flex flex-col justify-between">
          <div>
            <div className="pb-3 border-b border-border-subtle">
              <h2 className="text-sm font-semibold text-text-primary">
                TMA Alerts by Course Milestone
              </h2>
              <p className="text-xs text-text-muted mt-0.5">
                Curricular progression across the 34-week semester
              </p>
            </div>

            <div className="mt-5 space-y-4">
              {/* Milestone 1: Onboarding */}
              <div className="p-3.5 rounded-lg bg-slate-50 border border-border-subtle space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center space-x-2">
                    <span className="font-semibold text-text-primary">Milestone 1: Onboarding</span>
                    <span className="text-[10px] text-slate-500 font-mono">(Weeks 1–10)</span>
                  </div>
                  <span className="font-bold text-text-primary tabular-nums">2,280 alerts</span>
                </div>
                <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                  <div className="bg-primary-blue h-full" style={{ width: '31.98%' }}></div>
                </div>
                <div className="flex justify-between text-[11px] text-text-muted">
                  <span>31.98% of total alerts</span>
                  <span>Early attrition prevention</span>
                </div>
              </div>

              {/* Milestone 2: Midterm */}
              <div className="p-3.5 rounded-lg bg-slate-50 border border-border-subtle space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center space-x-2">
                    <span className="font-semibold text-text-primary">Milestone 2: Midterm</span>
                    <span className="text-[10px] text-slate-500 font-mono">(Weeks 11–20)</span>
                  </div>
                  <span className="font-bold text-text-primary tabular-nums">1,801 alerts</span>
                </div>
                <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                  <div className="bg-primary-blue h-full" style={{ width: '25.26%' }}></div>
                </div>
                <div className="flex justify-between text-[11px] text-text-muted">
                  <span>25.26% of total alerts</span>
                  <span>Mid-course trajectory intervention</span>
                </div>
              </div>

              {/* Milestone 3: Capstone */}
              <div className="p-3.5 rounded-lg bg-slate-50 border border-border-subtle space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center space-x-2">
                    <span className="font-semibold text-text-primary">Milestone 3: Capstone</span>
                    <span className="text-[10px] text-slate-500 font-mono">(Weeks 21–34)</span>
                  </div>
                  <span className="font-bold text-text-primary tabular-nums">3,049 alerts</span>
                </div>
                <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                  <div className="bg-primary-blue h-full" style={{ width: '42.76%' }}></div>
                </div>
                <div className="flex justify-between text-[11px] text-text-muted">
                  <span>42.76% of total alerts</span>
                  <span>Pre-exam retention support</span>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-border-subtle text-[11px] text-text-muted flex items-center justify-between">
            <span>Total Governed Alerts: 7,130</span>
            <span>CMA Alerts Suppressed: 2,243</span>
          </div>
        </div>
      </div>

      {/* Row 3: Illustrative Demonstration Triage Table with DEMO DATA Badge */}
      <div className="enterprise-card p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-border-subtle gap-2">
          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-sm font-semibold text-text-primary">
                Illustrative Advisor Triage Queue
              </h2>
              <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-amber-100 text-amber-800 border border-amber-300">
                DEMO DATA - NOT LIVE SIS/LMS QUEUE
              </span>
            </div>
            <p className="text-xs text-text-muted mt-0.5">
              Exemplar rendering of high-risk student snapshots for authorized advisor consultation
            </p>
          </div>
          <span className="text-xs text-text-secondary italic">
            Demonstration purposes only • Read-Only Shadow Mode
          </span>
        </div>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left enterprise-table">
            <thead>
              <tr>
                <th>Pseudonymized Student ID</th>
                <th>Module</th>
                <th>Assessment</th>
                <th>Scheduled Day</th>
                <th>Predicted Risk</th>
                <th>Operational Threshold</th>
                <th>Governance Routing</th>
                <th>Outreach Status</th>
              </tr>
            </thead>
            <tbody>
              {[
                { id: '#8a3e7b16c4d8e2f1', mod: 'BBB', assess: 'TMA 2 (#1754)', day: 54, prob: 0.384, thresh: 0.20, action: 'SHADOW_MODE_ADVISORY_REVIEW' },
                { id: '#b2c4e6a8d0f21357', mod: 'GGG', assess: 'TMA 1 (#26100)', day: 61, prob: 0.142, thresh: 0.10, action: 'SHADOW_MODE_ADVISORY_REVIEW' },
                { id: '#f9e8d7c6b5a41234', mod: 'DDD', assess: 'TMA 3 (#25360)', day: 110, prob: 0.428, thresh: 0.20, action: 'SHADOW_MODE_ADVISORY_REVIEW' },
                { id: '#3c4d5e6f7a8b9c0d', mod: 'CCC', assess: 'TMA 1 (#24290)', day: 32, prob: 0.289, thresh: 0.20, action: 'SHADOW_MODE_ADVISORY_REVIEW' },
                { id: '#1a2b3c4d5e6f7a8b', mod: 'FFF', assess: 'TMA 2 (#1760)', day: 88, prob: 0.215, thresh: 0.20, action: 'SHADOW_MODE_ADVISORY_REVIEW' },
              ].map((row, i) => (
                <tr key={i}>
                  <td className="font-mono text-xs font-semibold text-text-primary">
                    {row.id}
                  </td>
                  <td>
                    <span className="inline-block px-1.5 py-0.5 rounded text-[11px] font-semibold bg-slate-100 text-slate-800">
                      Module {row.mod}
                    </span>
                  </td>
                  <td className="text-xs text-text-secondary font-medium">
                    {row.assess}
                  </td>
                  <td className="text-xs text-text-secondary tabular-nums">
                    Day {row.day}
                  </td>
                  <td className="text-xs font-bold text-risk-high-base tabular-nums">
                    {(row.prob * 100).toFixed(1)}%
                  </td>
                  <td className="text-xs font-mono text-text-secondary">
                    {(row.thresh * 100).toFixed(0)}%
                  </td>
                  <td>
                    <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-blue-50 text-blue-700 border border-blue-200">
                      Advisory Review
                    </span>
                  </td>
                  <td>
                    <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-slate-100 text-slate-600 border border-slate-200">
                      Prohibited (Shadow)
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

