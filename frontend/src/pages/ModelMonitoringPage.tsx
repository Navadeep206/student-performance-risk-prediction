import React, { useEffect, useState } from 'react';
import { 
  Activity, 
  TrendingDown, 
  AlertTriangle, 
  Info, 
  CheckCircle2, 
  Layers,
  Database,
  ArrowRight
} from 'lucide-react';
import { KpiCard } from '../components/KpiCard';
import { analyticsApi } from '../services/api';
import { MonitoringResponse } from '../types';

export const ModelMonitoringPage: React.FC = () => {
  const [data, setData] = useState<MonitoringResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const res = await analyticsApi.getMonitoringDrift();
        setData(res);
      } catch (err: any) {
        console.error('Failed to load monitoring analytics:', err);
        setError(err.message || 'Error fetching monitoring metrics');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="p-8 space-y-6 animate-pulse">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-28 bg-slate-200 rounded-lg"></div>
          ))}
        </div>
        <div className="h-72 bg-slate-200 rounded-lg"></div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-8">
        <div className="p-6 bg-risk-high-surface border border-risk-high-border text-risk-high-text rounded-lg text-xs">
          Failed to load model monitoring data: {error}
        </div>
      </div>
    );
  }

  return (
    <div className="p-8 space-y-8">
      {/* 4 Certified Validation Metrics KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        <KpiCard
          title="Chronological PR-AUC (2014J)"
          value="0.3005"
          subtitle="Prospective Holdout (All 50,014 Snapshots)"
          annotation="Grouped 5-Fold CV: 0.4606 (-34.8%)"
          badge={{ text: "Holdout Benchmark", variant: "info" }}
          icon={<Activity size={20} />}
        />
        <KpiCard
          title="Chronological ROC-AUC"
          value="0.7969"
          subtitle="Overall discrimination across 2014J"
          annotation="Grouped 5-Fold CV: 0.8687 (-8.3%)"
          badge={{ text: "High Discrimination", variant: "success" }}
          icon={<CheckCircle2 size={20} />}
        />
        <KpiCard
          title="Chronological Brier Score"
          value="0.0812"
          subtitle="Mean squared error on probabilities"
          annotation="Grouped 5-Fold CV: 0.0592"
          badge={{ text: "Well-Calibrated", variant: "neutral" }}
          icon={<Database size={20} />}
        />
        <KpiCard
          title="Expected Calibration Error (ECE)"
          value="0.0420"
          subtitle="10-bin calibration gap on 2014J"
          annotation="Grouped 5-Fold CV: 0.0163 (+157.7%)"
          badge={{ text: "Calibrated", variant: "neutral" }}
          icon={<TrendingDown size={20} />}
        />
      </div>

      {/* Row 2: Generalization Gap Comparison Table */}
      <div className="enterprise-card p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-border-subtle gap-2">
          <div>
            <h2 className="text-sm font-semibold text-text-primary">
              Performance Generalization Gap: Grouped CV vs Chronological 2014J Holdout
            </h2>
            <p className="text-xs text-text-muted mt-0.5">
              Empirical quantification of prospective model decay under temporal shift and 87.37% novel students
            </p>
          </div>
          <span className="text-xs text-text-secondary bg-slate-50 border border-border-subtle px-3 py-1 rounded">
            Temporal Cohort Shift Analysis
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left enterprise-table">
            <thead>
              <tr>
                <th>Performance Metric Dimension</th>
                <th className="text-right">Grouped 5-Fold CV (N = 170,874)</th>
                <th className="text-right">Chronological 2014J Holdout (N = 50,014)</th>
                <th className="text-right">Observed Drift Delta (&Delta;)</th>
                <th className="text-right">Relative Shift %</th>
                <th>Monitoring Evaluation Status</th>
              </tr>
            </thead>
            <tbody>
              {data.performance_generalization_gap.map((m, i) => (
                <tr key={i}>
                  <td className="font-semibold text-text-primary text-xs">
                    {m.metric_name}
                  </td>
                  <td className="text-xs font-mono text-right tabular-nums text-text-secondary">
                    {m.grouped_cv_value.toFixed(4)}
                  </td>
                  <td className="text-xs font-mono font-bold text-right tabular-nums text-primary-blue">
                    {m.chronological_2014j_value.toFixed(4)}
                  </td>
                  <td className="text-xs font-mono text-right tabular-nums">
                    {m.drift_delta > 0 ? `+${m.drift_delta.toFixed(4)}` : m.drift_delta.toFixed(4)}
                  </td>
                  <td className="text-xs font-mono text-right tabular-nums font-bold">
                    <span className={m.drift_pct < -20 || m.drift_pct > 50 ? 'text-risk-high-base' : 'text-amber-700'}>
                      {m.drift_pct > 0 ? `+${m.drift_pct.toFixed(1)}%` : `${m.drift_pct.toFixed(1)}%`}
                    </span>
                  </td>
                  <td>
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-100 text-slate-800 border border-slate-200">
                      {m.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Row 3: Population Stability Index (PSI) Feature Drift */}
      <div className="enterprise-card p-6 space-y-4">
        <div className="pb-3 border-b border-border-subtle flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold text-text-primary">
              Feature Population Stability Index (PSI) Drift Analysis
            </h2>
            <p className="text-xs text-text-muted mt-0.5">
              Comparing feature distribution shift between Historical Training (N = 120,860) and Prospective 2014J (N = 50,014)
            </p>
          </div>
          <div className="flex items-center space-x-3 text-xs text-text-muted">
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
              <span>Stable (&lt; 0.10)</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
              <span>Moderate (0.10–0.25)</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500"></span>
              <span>Severe (&ge; 0.25)</span>
            </span>
          </div>
        </div>

        <div className="space-y-3">
          {data.top_feature_psi_drift.map((p, i) => {
            const isSevere = p.psi_value >= 0.25;
            const isModerate = p.psi_value >= 0.10 && p.psi_value < 0.25;

            return (
              <div key={i} className="p-4 bg-slate-50 border border-border-subtle rounded-lg flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs">
                <div className="space-y-1 max-w-xl">
                  <div className="flex items-center space-x-2">
                    <span className="font-mono font-bold text-text-primary text-sm">
                      {p.feature_name}
                    </span>
                    <span
                      className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold border ${
                        isSevere
                          ? 'bg-red-50 text-red-700 border-red-200'
                          : isModerate
                          ? 'bg-amber-50 text-amber-800 border-amber-200'
                          : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                      }`}
                    >
                      {p.drift_severity}
                    </span>
                  </div>
                  <p className="text-text-secondary leading-relaxed text-[11px]">
                    {p.commentary}
                  </p>
                </div>

                <div className="flex items-center space-x-4 shrink-0">
                  <div className="text-right">
                    <span className="text-[10px] text-text-muted block">PSI Metric</span>
                    <span className="font-mono font-bold text-sm tabular-nums text-text-primary">
                      {p.psi_value.toFixed(4)}
                    </span>
                  </div>
                  <div className="w-28 bg-slate-200 h-3 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${
                        isSevere
                          ? 'bg-red-500'
                          : isModerate
                          ? 'bg-amber-500'
                          : 'bg-emerald-500'
                      }`}
                      style={{ width: `${Math.min(100, (p.psi_value / 0.50) * 100)}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Row 4: Data Quality & Ingestion Limitations Documentation */}
      <div className="enterprise-card p-6 space-y-4">
        <div className="pb-3 border-b border-border-subtle flex items-center space-x-2">
          <Info size={16} className="text-primary-blue" />
          <h2 className="text-sm font-semibold text-text-primary">
            Data Quality, Ingestion Telemetry & Operational Limitations
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs leading-relaxed">
          {data.ingestion_and_telemetry_limitations.map((lim, i) => (
            <div key={i} className="p-3.5 bg-slate-50 border border-border-subtle rounded-md space-y-1">
              <span className="font-semibold text-text-primary block flex items-center space-x-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-slate-500"></span>
                <span>Limitation #{i + 1}</span>
              </span>
              <p className="text-text-secondary text-[11px]">{lim}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

