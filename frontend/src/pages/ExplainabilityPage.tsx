import React, { useEffect, useState } from 'react';
import { 
  Cpu, 
  HelpCircle, 
  Info, 
  ShieldAlert, 
  BarChart3, 
  Layers,
  ArrowRight,
  TrendingDown,
  TrendingUp
} from 'lucide-react';
import { analyticsApi } from '../services/api';
import { ExplainabilityResponse } from '../types';

export const ExplainabilityPage: React.FC = () => {
  const [data, setData] = useState<ExplainabilityResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const res = await analyticsApi.getGlobalImportance();
        setData(res);
      } catch (err: any) {
        console.error('Failed to load explainability metrics:', err);
        setError(err.message || 'Error fetching TreeSHAP explanations');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="p-8 space-y-6 animate-pulse">
        <div className="h-96 bg-slate-200 rounded-lg"></div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-8">
        <div className="p-6 bg-risk-high-surface border border-risk-high-border text-risk-high-text rounded-lg text-xs flex items-center justify-between">
          <span>Failed to load explainability data: {error}</span>
          <button
            onClick={() => window.location.reload()}
            className="px-3 py-1 bg-white border border-red-300 text-red-800 rounded font-semibold text-xs hover:bg-red-50"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const maxShap = 0.80; // normalized axis scale

  return (
    <div className="p-8 space-y-8">
      {/* Top Header Card */}
      <div className="enterprise-card p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-border-subtle gap-2">
          <div>
            <div className="flex items-center space-x-2">
              <Cpu size={18} className="text-primary-blue" />
              <h2 className="text-sm font-semibold text-text-primary">
                Global TreeSHAP Feature Attribution (Primary HGB Model)
              </h2>
            </div>
            <p className="text-xs text-text-muted mt-0.5">
              Empirical Shapley values computed across 170,874 benchmark snapshot vectors
            </p>
          </div>
          <div className="text-xs text-text-secondary bg-slate-50 border border-border-subtle px-3 py-1 rounded">
            <span>Method: </span>
            <span className="font-semibold text-primary-blue">TreeExplainer (Log-Odds Impact)</span>
          </div>
        </div>

        {/* Horizontal Bar Chart */}
        <div className="mt-6 space-y-4">
          {data.features.map((feat, idx) => {
            const widthPct = Math.min(100, (feat.mean_abs_shap / maxShap) * 100);

            return (
              <div key={idx} className="space-y-1">
                <div className="flex justify-between items-baseline text-xs">
                  <div className="flex items-center space-x-2">
                    <span className="font-semibold text-text-primary">{feat.display_name}</span>
                    <span className="font-mono text-[10px] text-text-muted">({feat.feature_name})</span>
                    <span className="text-[10px] bg-slate-100 text-slate-700 px-1.5 py-0.2 rounded font-medium">
                      {feat.category}
                    </span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className="text-[11px] text-text-muted hidden md:inline">
                      {feat.risk_impact_direction}
                    </span>
                    <span className="font-mono font-bold text-primary-blue tabular-nums text-xs">
                      {feat.mean_abs_shap.toFixed(4)}
                    </span>
                  </div>
                </div>

                {/* Bar */}
                <div className="w-full bg-slate-100 h-4 rounded overflow-hidden flex items-center">
                  <div
                    style={{ width: `${widthPct}%` }}
                    className="bg-primary-blue hover:bg-primary-hover transition-all h-full rounded"
                    title={`${feat.display_name}: mean |SHAP| = ${feat.mean_abs_shap}`}
                  ></div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-6 pt-4 border-t border-border-subtle flex items-center justify-between text-xs text-text-muted">
          <span>Scale: Mean Absolute SHAP Value (E[|&phi;<sub>i</sub>|] on log-odds of risk)</span>
          <span>Evaluated on 23 Permitted Predictive Features</span>
        </div>
      </div>

      {/* Row 2: Methodological Guidance & Log-Odds Mathematical Principles */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Advisory Interpretation Guidance */}
        <div className="enterprise-card p-6 space-y-4">
          <div className="pb-3 border-b border-border-subtle flex items-center space-x-2">
            <Info size={16} className="text-primary-blue" />
            <h3 className="text-sm font-semibold text-text-primary">
              Methodological Interpretation Guidance
            </h3>
          </div>

          <div className="space-y-3 text-xs leading-relaxed">
            {data.methodological_guidance.map((note, i) => (
              <div key={i} className="flex items-start space-x-2.5">
                <span className="inline-block w-4 h-4 rounded-full bg-blue-50 text-primary-blue text-[10px] font-bold text-center shrink-0 mt-0.5">
                  {i + 1}
                </span>
                <p className="text-text-secondary">{note}</p>
              </div>
            ))}
          </div>
        </div>

        {/* SHAP Directional Mechanics Card */}
        <div className="enterprise-card p-6 space-y-4">
          <div className="pb-3 border-b border-border-subtle flex items-center space-x-2">
            <BarChart3 size={16} className="text-primary-blue" />
            <h3 className="text-sm font-semibold text-text-primary">
              Directional Mechanics on Risk Log-Odds
            </h3>
          </div>

          <div className="space-y-3 text-xs">
            <div className="p-3.5 bg-red-50 border border-red-200 rounded-lg space-y-1">
              <div className="flex items-center space-x-2 text-risk-high-text font-semibold">
                <TrendingUp size={16} />
                <span>Positive SHAP Value (&phi;<sub>i</sub> &gt; 0)</span>
              </div>
              <p className="text-risk-high-text leading-relaxed text-[11px]">
                Pushes the model prediction <strong>towards academic vulnerability</strong> (increases the log-odds of scoring &lt; 50).
                Common triggers: low prior scores, late course registration, or sudden drop in VLE click volume.
              </p>
            </div>

            <div className="p-3.5 bg-emerald-50 border border-emerald-200 rounded-lg space-y-1">
              <div className="flex items-center space-x-2 text-risk-low-text font-semibold">
                <TrendingDown size={16} />
                <span>Negative SHAP Value (&phi;<sub>i</sub> &lt; 0)</span>
              </div>
              <p className="text-risk-low-text leading-relaxed text-[11px]">
                Pushes the model prediction <strong>towards passing</strong> (decreases the log-odds of risk).
                Common triggers: high recent assignment scores, consistent weekly VLE logins, and early registration.
              </p>
            </div>
          </div>

          <div className="pt-2 text-[11px] text-text-muted italic">
            Zero Demographic Features: Sensitive demographic variables (gender, region, imd_band, etc.) are strictly excluded.
          </div>
        </div>
      </div>
    </div>
  );
};
