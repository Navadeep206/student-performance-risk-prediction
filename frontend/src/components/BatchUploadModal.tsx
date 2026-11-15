import React, { useState, useRef } from 'react';
import { 
  X, 
  Upload, 
  FileText, 
  AlertTriangle, 
  CheckCircle, 
  Download, 
  RefreshCw, 
  ShieldCheck, 
  Info,
  Copy,
  Check,
  Play
} from 'lucide-react';
import { analyticsApi } from '../services/api';
import { 
  PredictionItemInput, 
  BatchPredictionResponse, 
  BatchItemResult 
} from '../types';

interface BatchUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const SAMPLE_RECORDS: PredictionItemInput[] = [
  {
    id_student: 100234,
    code_module: 'BBB',
    code_presentation: '2014J',
    id_assessment: 34885,
    target_assessment_type: 'TMA',
    scheduled_date: 54,
    target_assessment_weight: 18.0,
    num_previous_scheduled_assessments: 1,
    registration_timing_days: -15,
    vle_total_clicks_to_date: 420.0,
    vle_active_days_to_date: 18.0,
    vle_core_clicks_to_date: 190.0,
    vle_forum_clicks_to_date: 12.0,
    vle_quiz_clicks_to_date: 0.0,
    vle_days_since_last_activity: 8.0,
    vle_clicks_last_7d: 15.0,
    vle_clicks_last_14d: 40.0,
    vle_activity_trend_ratio: 0.35,
    num_prior_submissions: 1,
    prior_submission_lateness_mean: 4.0,
    num_prior_scored_assessments: 1,
    mean_prior_score: 42.0,
    std_prior_score: 0.0,
    prior_at_risk_rate: 1.0,
    recent_prior_score: 42.0
  },
  {
    id_student: 200456,
    code_module: 'GGG',
    code_presentation: '2014J',
    id_assessment: 37435,
    target_assessment_type: 'TMA',
    scheduled_date: 124,
    target_assessment_weight: 33.3,
    num_previous_scheduled_assessments: 2,
    registration_timing_days: -30,
    vle_total_clicks_to_date: 650.0,
    vle_active_days_to_date: 24.0,
    vle_core_clicks_to_date: 290.0,
    vle_forum_clicks_to_date: 35.0,
    vle_quiz_clicks_to_date: 45.0,
    vle_days_since_last_activity: 5.0,
    vle_clicks_last_7d: 30.0,
    vle_clicks_last_14d: 90.0,
    vle_activity_trend_ratio: 0.45,
    num_prior_submissions: 2,
    prior_submission_lateness_mean: 1.0,
    num_prior_scored_assessments: 2,
    mean_prior_score: 48.0,
    std_prior_score: 5.0,
    prior_at_risk_rate: 0.5,
    recent_prior_score: 46.0
  },
  {
    id_student: 300789,
    code_module: 'CCC',
    code_presentation: '2014J',
    id_assessment: 34890,
    target_assessment_type: 'CMA',
    scheduled_date: 35,
    target_assessment_weight: 0.0,
    num_previous_scheduled_assessments: 0,
    registration_timing_days: -45,
    vle_total_clicks_to_date: 310.0,
    vle_active_days_to_date: 12.0,
    vle_core_clicks_to_date: 140.0,
    vle_forum_clicks_to_date: 5.0,
    vle_quiz_clicks_to_date: 120.0,
    vle_days_since_last_activity: 3.0,
    vle_clicks_last_7d: 45.0,
    vle_clicks_last_14d: 110.0,
    vle_activity_trend_ratio: 0.8,
    num_prior_submissions: 0,
    prior_submission_lateness_mean: null,
    num_prior_scored_assessments: 0,
    mean_prior_score: null,
    std_prior_score: null,
    prior_at_risk_rate: null,
    recent_prior_score: null
  },
  {
    id_student: 400112,
    code_module: 'AAA',
    code_presentation: '2013J',
    id_assessment: 1752,
    target_assessment_type: 'TMA',
    scheduled_date: 19,
    target_assessment_weight: 10.0,
    num_previous_scheduled_assessments: 0,
    registration_timing_days: -25,
    vle_total_clicks_to_date: 215.0,
    vle_active_days_to_date: 9.0,
    vle_core_clicks_to_date: 120.0,
    vle_forum_clicks_to_date: 20.0,
    vle_quiz_clicks_to_date: 0.0,
    vle_days_since_last_activity: 2.0,
    vle_clicks_last_7d: 65.0,
    vle_clicks_last_14d: 120.0,
    vle_activity_trend_ratio: 0.9,
    num_prior_submissions: 0,
    prior_submission_lateness_mean: null,
    num_prior_scored_assessments: 0,
    mean_prior_score: null,
    std_prior_score: null,
    prior_at_risk_rate: null,
    recent_prior_score: null
  },
  {
    id_student: 500998,
    code_module: 'BBB',
    code_presentation: '2014J',
    id_assessment: 34885,
    target_assessment_type: 'TMA',
    scheduled_date: 54,
    target_assessment_weight: 18.0,
    num_previous_scheduled_assessments: 1,
    registration_timing_days: -60,
    vle_total_clicks_to_date: 1850.0,
    vle_active_days_to_date: 45.0,
    vle_core_clicks_to_date: 850.0,
    vle_forum_clicks_to_date: 210.0,
    vle_quiz_clicks_to_date: 0.0,
    vle_days_since_last_activity: 1.0,
    vle_clicks_last_7d: 280.0,
    vle_clicks_last_14d: 510.0,
    vle_activity_trend_ratio: 1.15,
    num_prior_submissions: 1,
    prior_submission_lateness_mean: -2.0,
    num_prior_scored_assessments: 1,
    mean_prior_score: 92.0,
    std_prior_score: 0.0,
    prior_at_risk_rate: 0.0,
    recent_prior_score: 92.0
  }
];

const PROHIBITED_FIELDS = [
  'gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability',
  'final_result', 'date_unregistration', 'target_assessment_score', 'next_assessment_score', 'at_risk'
];

export const BatchUploadModal: React.FC<BatchUploadModalProps> = ({ isOpen, onClose }) => {
  const [records, setRecords] = useState<PredictionItemInput[]>([]);
  const [fileName, setFileName] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [prohibitedColumnsFound, setProhibitedColumnsFound] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [batchResults, setBatchResults] = useState<BatchPredictionResponse | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleCopy = (id: string) => {
    navigator.clipboard.writeText(id);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const parseCSV = (csvText: string, name: string) => {
    setValidationError(null);
    setProhibitedColumnsFound([]);
    setBatchResults(null);

    const lines = csvText.split(/\r?\n/).filter(line => line.trim().length > 0);
    if (lines.length < 2) {
      setValidationError("CSV must contain a header row and at least one data row.");
      return;
    }

    const rawHeaders = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
    const normalizedHeaders = rawHeaders.map(h => h.toLowerCase().replace(/[\s-]+/g, '_'));

    // Check for prohibited features
    const foundProhibited = normalizedHeaders.filter(h => PROHIBITED_FIELDS.includes(h));
    if (foundProhibited.length > 0) {
      setProhibitedColumnsFound(foundProhibited);
      setValidationError(`Prohibited columns detected: [${foundProhibited.join(', ')}]. Demographic attributes and target leakage columns are strictly disallowed by institutional governance.`);
      return;
    }

    // Required column validation
    const requiredCols = ['id_student', 'code_module', 'code_presentation', 'id_assessment', 'target_assessment_type', 'scheduled_date'];
    const missing = requiredCols.filter(col => !normalizedHeaders.includes(col));
    if (missing.length > 0) {
      setValidationError(`Missing required column(s): ${missing.join(', ')}. Please ensure headers match the required schema.`);
      return;
    }

    const parsed: PredictionItemInput[] = [];
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      const values = line.split(',').map(v => v.trim().replace(/^["']|["']$/g, ''));
      if (values.length < normalizedHeaders.length) continue;

      const row: any = {};
      normalizedHeaders.forEach((header, index) => {
        const val = values[index];
        if (val === '' || val === 'null' || val === 'NULL' || val === undefined) {
          row[header] = null;
        } else if (['id_assessment', 'scheduled_date', 'num_previous_scheduled_assessments', 'num_prior_submissions', 'num_prior_scored_assessments'].includes(header)) {
          row[header] = parseInt(val, 10) || 0;
        } else if (['target_assessment_weight', 'registration_timing_days', 'vle_total_clicks_to_date', 'vle_active_days_to_date', 'vle_core_clicks_to_date', 'vle_forum_clicks_to_date', 'vle_quiz_clicks_to_date', 'vle_days_since_last_activity', 'vle_clicks_last_7d', 'vle_clicks_last_14d', 'vle_activity_trend_ratio', 'prior_submission_lateness_mean', 'mean_prior_score', 'std_prior_score', 'prior_at_risk_rate', 'recent_prior_score'].includes(header)) {
          const num = parseFloat(val);
          row[header] = isNaN(num) ? null : num;
        } else {
          row[header] = val;
        }
      });

      parsed.push(row as PredictionItemInput);
    }

    if (parsed.length === 0) {
      setValidationError("No valid data rows could be parsed from the CSV file.");
      return;
    }

    setRecords(parsed);
    setFileName(name);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      parseCSV(content, file.name);
    };
    reader.readAsText(file);
  };

  const handleLoadSample = () => {
    setValidationError(null);
    setProhibitedColumnsFound([]);
    setBatchResults(null);
    setRecords(SAMPLE_RECORDS);
    setFileName("sample_oulad_prospective_cohort.csv");
  };

  const handleRunBatchPrediction = async () => {
    if (records.length === 0) return;
    try {
      setIsSubmitting(true);
      setValidationError(null);
      const res = await analyticsApi.predictBatch(records);
      setBatchResults(res);
    } catch (err: any) {
      console.error("Batch prediction failure:", err);
      setValidationError(err.message || "Failed to execute batch prediction.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleExportCSV = () => {
    if (!batchResults || !batchResults.results) return;

    const headers = [
      "item_index",
      "pseudonymized_student_id",
      "code_module",
      "code_presentation",
      "id_assessment",
      "assessment_type",
      "scheduled_date",
      "raw_risk_probability",
      "risk_tier",
      "operational_threshold_applied",
      "predicted_at_risk",
      "governance_action",
      "suppress_automated_outreach",
      "advising_recommendation",
      "model_version"
    ];

    const rows = batchResults.results.map((r: BatchItemResult) => {
      if (!r.success || !r.data) {
        return [r.item_index, "FAILED", "", "", "", "", "", "", "ERROR", "", "", "", "", r.error || "Execution error", ""].join(",");
      }
      const d = r.data;
      return [
        r.item_index,
        `"${d.pseudonymized_student_id}"`,
        d.code_module,
        d.code_presentation,
        d.id_assessment,
        d.assessment_type,
        d.scheduled_date,
        d.raw_risk_probability.toFixed(4),
        d.risk_tier,
        d.operational_threshold_applied ?? "",
        d.predicted_at_risk ? "TRUE" : "FALSE",
        d.governance_action,
        d.suppress_automated_outreach ? "TRUE" : "FALSE",
        `"${d.advising_recommendation.replace(/"/g, '""')}"`,
        d.model_version
      ].join(",");
    });

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `prospective_batch_risk_scoring_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const highRiskCount = batchResults?.results.filter(r => r.data?.risk_tier === 'HIGH_RISK').length || 0;
  const suppressedCount = batchResults?.results.filter(r => r.data?.suppress_automated_outreach).length || 0;
  const gggCalibratedCount = batchResults?.results.filter(r => r.data?.code_module === 'GGG' && r.data.operational_threshold_applied === 0.10).length || 0;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/60 backdrop-blur-[2px] flex items-center justify-center p-4">
      <div className="bg-white rounded-xl shadow-2xl border border-border-medium w-full max-w-5xl max-h-[90vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        
        {/* Modal Header */}
        <div className="px-6 py-5 border-b border-border-subtle bg-slate-50 flex items-start justify-between">
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-bold text-primary-blue uppercase tracking-wider">
                Model Inference & Assessment Telemetry
              </span>
              <span className="text-[10px] bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full font-semibold">
                POST /predict/batch
              </span>
            </div>
            <h2 className="text-lg font-bold text-text-primary mt-1">
              Upload Assessment CSV for Prospective Risk Scoring
            </h2>
            <p className="text-xs text-text-muted mt-0.5">
              Submit student assessment snapshots to evaluate prospective risk, apply calibrated module thresholds, and audit institutional governance routing.
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 p-1.5 rounded-lg hover:bg-slate-200 transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 text-xs">
          
          {/* Institutional Policy Notice */}
          <div className="bg-slate-50 border border-border-subtle rounded-lg p-4 flex items-start space-x-3 text-slate-700">
            <ShieldCheck size={18} className="text-primary-blue shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-semibold text-text-primary">
                Governance & Fair ML Safeguards Active
              </span>
              <p className="text-[11px] text-text-secondary leading-relaxed">
                Raw student IDs are pseudonymized on ingestion via HMAC-SHA256 with institutional secret salt. Demographic features (<code className="font-mono text-slate-800">gender</code>, <code className="font-mono text-slate-800">imd_band</code>, <code className="font-mono text-slate-800">disability</code>, etc.) and target leakage (<code className="font-mono text-slate-800">final_result</code>) are strictly prohibited and validated client-side and server-side. Universal CMA outreach suppression is enforced.
              </p>
            </div>
          </div>

          {/* Action Row: Upload or Load Sample */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            {/* File Dropzone */}
            <div 
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-300 hover:border-primary-blue rounded-lg p-5 flex flex-col items-center justify-center cursor-pointer bg-slate-50/50 hover:bg-blue-50/30 transition-all text-center group"
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileUpload} 
                accept=".csv" 
                className="hidden" 
              />
              <Upload size={24} className="text-slate-400 group-hover:text-primary-blue mb-2 transition-colors" />
              <span className="font-semibold text-text-primary text-xs">
                Click to upload assessment CSV
              </span>
              <span className="text-[11px] text-text-muted mt-1">
                Supports RFC 4180 CSV files with snapshot features
              </span>
              {fileName && (
                <div className="mt-2 text-xs font-mono bg-blue-50 text-blue-800 border border-blue-200 px-2 py-0.5 rounded">
                  {fileName} ({records.length} records parsed)
                </div>
              )}
            </div>

            {/* Load Representative Sample */}
            <div className="border border-border-subtle rounded-lg p-5 flex flex-col justify-between bg-white">
              <div>
                <div className="flex items-center space-x-1.5 text-text-primary font-semibold">
                  <FileText size={16} className="text-primary-blue" />
                  <span>Instant Verification with Sample Dataset</span>
                </div>
                <p className="text-[11px] text-text-muted mt-1 leading-relaxed">
                  Loads 5 realistic, pre-configured assessment snapshots representing diverse operational scenarios: High Risk BBB TMA, Calibrated GGG TMA (&#964;=0.10), CCC CMA Quiz suppression, and Milestone 1 Cold Start.
                </p>
              </div>
              <button
                type="button"
                onClick={handleLoadSample}
                className="mt-3 px-3 py-2 bg-slate-100 hover:bg-slate-200 text-text-primary rounded-md font-medium text-xs flex items-center justify-center space-x-1.5 transition-colors border border-slate-200"
              >
                <RefreshCw size={13} />
                <span>Load Sample Assessment CSV (5 Scenarios)</span>
              </button>
            </div>
          </div>

          {/* Validation Warning Alert */}
          {validationError && (
            <div className="bg-red-50 border border-red-200 text-red-800 rounded-lg p-4 flex items-start space-x-3">
              <AlertTriangle size={18} className="text-red-600 shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold text-xs">Upload Validation Error</div>
                <div className="text-[11px] mt-0.5">{validationError}</div>
                {prohibitedColumnsFound.length > 0 && (
                  <div className="mt-2 text-[11px] bg-red-100/70 p-2 rounded font-mono">
                    Rejected columns: {prohibitedColumnsFound.join(', ')}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Records Preview Table (Before Scoring) */}
          {records.length > 0 && !batchResults && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-text-primary">
                    Parsed Assessment Telemetry Records
                  </span>
                  <span className="bg-slate-200 text-slate-800 text-[10px] px-2 py-0.5 rounded-full font-bold">
                    {records.length} Records
                  </span>
                </div>
                <button
                  type="button"
                  onClick={handleRunBatchPrediction}
                  disabled={isSubmitting}
                  className="px-4 py-2 bg-primary-blue hover:bg-blue-800 text-white rounded-md font-semibold text-xs flex items-center space-x-2 shadow-sm transition-colors disabled:opacity-50"
                >
                  {isSubmitting ? (
                    <>
                      <RefreshCw size={14} className="animate-spin" />
                      <span>Scoring Batch Telemetry...</span>
                    </>
                  ) : (
                    <>
                      <Play size={14} />
                      <span>Run Batch Risk Scoring ({records.length} records)</span>
                    </>
                  )}
                </button>
              </div>

              <div className="border border-border-subtle rounded-lg overflow-x-auto max-h-56">
                <table className="w-full text-left text-xs border-collapse">
                  <thead className="bg-slate-50 border-b border-border-subtle text-slate-600 font-semibold sticky top-0">
                    <tr>
                      <th className="p-2.5">#</th>
                      <th className="p-2.5">Student ID</th>
                      <th className="p-2.5">Module</th>
                      <th className="p-2.5">Cohort</th>
                      <th className="p-2.5">Type</th>
                      <th className="p-2.5">Assessment ID</th>
                      <th className="p-2.5 text-right">Cutoff Day</th>
                      <th className="p-2.5 text-right">Weight</th>
                      <th className="p-2.5 text-right">VLE Clicks</th>
                      <th className="p-2.5 text-right">Prior Score</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border-subtle">
                    {records.map((r, i) => (
                      <tr key={i} className="hover:bg-slate-50 font-mono text-[11px]">
                        <td className="p-2.5 text-slate-400">{i + 1}</td>
                        <td className="p-2.5 font-bold text-slate-700">{r.id_student}</td>
                        <td className="p-2.5">{r.code_module}</td>
                        <td className="p-2.5">{r.code_presentation}</td>
                        <td className="p-2.5">
                          <span className={`px-1.5 py-0.5 rounded text-[10px] font-sans font-semibold ${
                            r.target_assessment_type === 'TMA' ? 'bg-indigo-50 text-indigo-700 border border-indigo-200' : 'bg-slate-100 text-slate-700'
                          }`}>
                            {r.target_assessment_type}
                          </span>
                        </td>
                        <td className="p-2.5">{r.id_assessment}</td>
                        <td className="p-2.5 text-right">Day {r.scheduled_date}</td>
                        <td className="p-2.5 text-right">{r.target_assessment_weight ?? 0}%</td>
                        <td className="p-2.5 text-right">{r.vle_total_clicks_to_date?.toLocaleString() ?? 0}</td>
                        <td className="p-2.5 text-right">
                          {r.recent_prior_score !== null && r.recent_prior_score !== undefined
                            ? `${r.recent_prior_score}%`
                            : <span className="text-slate-400 italic font-sans">Cold-Start</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Results View (After Scoring) */}
          {batchResults && (
            <div className="space-y-4">
              
              {/* Summary KPIs */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
                  <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Processed</div>
                  <div className="text-xl font-bold text-slate-800 mt-1">{batchResults.successful_count} / {batchResults.total_requested}</div>
                  <div className="text-[10px] text-emerald-700 font-medium mt-0.5">100% Ingestion Integrity</div>
                </div>

                <div className="p-3.5 bg-red-50 border border-red-200 rounded-lg">
                  <div className="text-[10px] font-bold text-red-700 uppercase tracking-wider">High Risk Alerts</div>
                  <div className="text-xl font-bold text-red-900 mt-1">{highRiskCount}</div>
                  <div className="text-[10px] text-red-600 font-medium mt-0.5">Prioritized for Academic Review</div>
                </div>

                <div className="p-3.5 bg-indigo-50 border border-indigo-200 rounded-lg">
                  <div className="text-[10px] font-bold text-indigo-700 uppercase tracking-wider">Calibrated GGG</div>
                  <div className="text-xl font-bold text-indigo-900 mt-1">{gggCalibratedCount}</div>
                  <div className="text-[10px] text-indigo-600 font-medium mt-0.5">&#964; = 0.10 Threshold Applied</div>
                </div>

                <div className="p-3.5 bg-slate-100 border border-slate-300 rounded-lg">
                  <div className="text-[10px] font-bold text-slate-600 uppercase tracking-wider">Suppressed (CMA)</div>
                  <div className="text-xl font-bold text-slate-800 mt-1">{suppressedCount}</div>
                  <div className="text-[10px] text-slate-500 font-medium mt-0.5">Zero Outreach Telemetry</div>
                </div>
              </div>

              {/* Action Bar for Results */}
              <div className="flex items-center justify-between pt-2">
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-text-primary text-xs">
                    Prospective Inference & Governance Decision Audit Log
                  </span>
                  <span className="text-[10px] bg-slate-100 text-slate-700 px-2 py-0.5 rounded border border-slate-200 font-mono">
                    Model: HistogramGradientBoosting (v1.0.0)
                  </span>
                </div>

                <div className="flex items-center space-x-2">
                  <button
                    type="button"
                    onClick={handleExportCSV}
                    className="px-3 py-1.5 bg-white border border-border-medium hover:bg-slate-50 text-text-primary rounded-md font-medium text-xs flex items-center space-x-1.5 shadow-sm transition-colors"
                  >
                    <Download size={13} className="text-primary-blue" />
                    <span>Export Results CSV</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      setBatchResults(null);
                      setRecords([]);
                      setFileName(null);
                    }}
                    className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-text-secondary rounded-md font-medium text-xs flex items-center space-x-1.5 transition-colors"
                  >
                    <RefreshCw size={13} />
                    <span>Score Another Batch</span>
                  </button>
                </div>
              </div>

              {/* Predictions Table */}
              <div className="border border-border-subtle rounded-lg overflow-x-auto max-h-72">
                <table className="w-full text-left text-xs border-collapse">
                  <thead className="bg-slate-50 border-b border-border-subtle text-slate-600 font-semibold sticky top-0">
                    <tr>
                      <th className="p-2.5">Pseudonymized ID</th>
                      <th className="p-2.5">Module</th>
                      <th className="p-2.5">Assessment</th>
                      <th className="p-2.5 text-right">Risk Prob</th>
                      <th className="p-2.5">Operational Tier</th>
                      <th className="p-2.5 text-center">&#964; Applied</th>
                      <th className="p-2.5">Governance Action</th>
                      <th className="p-2.5">Advising Recommendation</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border-subtle font-mono text-[11px]">
                    {batchResults.results.map((res: BatchItemResult) => {
                      if (!res.success || !res.data) {
                        return (
                          <tr key={res.item_index} className="bg-red-50/50">
                            <td colSpan={8} className="p-2.5 text-red-700">
                              Item #{res.item_index}: {res.error}
                            </td>
                          </tr>
                        );
                      }
                      const d = res.data;
                      const isHigh = d.risk_tier === 'HIGH_RISK';
                      const isSuppressed = d.suppress_automated_outreach;

                      return (
                        <tr key={res.item_index} className="hover:bg-slate-50 transition-colors">
                          <td className="p-2.5 font-mono text-slate-700 flex items-center space-x-1">
                            <span title={d.pseudonymized_student_id}>
                              {d.pseudonymized_student_id.slice(0, 10)}...
                            </span>
                            <button
                              onClick={() => handleCopy(d.pseudonymized_student_id)}
                              className="text-slate-400 hover:text-slate-600 p-0.5 rounded"
                              title="Copy Full Pseudonym"
                            >
                              {copiedId === d.pseudonymized_student_id ? <Check size={11} className="text-emerald-600" /> : <Copy size={11} />}
                            </button>
                          </td>
                          <td className="p-2.5">{d.code_module} ({d.code_presentation})</td>
                          <td className="p-2.5">
                            <span className="font-sans font-medium text-slate-800">{d.assessment_type} #{d.id_assessment}</span>
                          </td>
                          <td className="p-2.5 text-right font-bold text-slate-800">
                            {(d.raw_risk_probability * 100).toFixed(1)}%
                          </td>
                          <td className="p-2.5 font-sans">
                            {isHigh ? (
                              <span className="bg-red-50 text-red-700 border border-red-200 px-2 py-0.5 rounded-full text-[10px] font-bold">
                                High Risk Alert
                              </span>
                            ) : (
                              <span className="bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5 rounded-full text-[10px] font-medium">
                                Low Risk Monitor
                              </span>
                            )}
                          </td>
                          <td className="p-2.5 text-center font-sans">
                            {d.operational_threshold_applied !== null ? (
                              <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold ${
                                d.code_module === 'GGG' ? 'bg-indigo-50 text-indigo-700 border border-indigo-200' : 'bg-slate-100 text-slate-700'
                              }`}>
                                {d.operational_threshold_applied.toFixed(2)}
                              </span>
                            ) : (
                              <span className="text-slate-400">-</span>
                            )}
                          </td>
                          <td className="p-2.5 font-sans">
                            {isSuppressed ? (
                              <span className="bg-slate-100 text-slate-700 border border-slate-200 px-1.5 py-0.5 rounded text-[10px] font-medium">
                                Suppressed (Telemtry Only)
                              </span>
                            ) : (
                              <span className="bg-blue-50 text-blue-700 border border-blue-200 px-1.5 py-0.5 rounded text-[10px] font-medium">
                                Actionable Advisor Outreach
                              </span>
                            )}
                          </td>
                          <td className="p-2.5 font-sans text-slate-600 text-[11px] max-w-xs truncate" title={d.advising_recommendation}>
                            {d.advising_recommendation}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 border-t border-border-subtle bg-slate-50 flex items-center justify-between text-xs text-text-muted">
          <div className="flex items-center space-x-1.5">
            <Info size={14} className="text-slate-400" />
            <span>Read-Only Shadow Mode Enforced &bull; Automated Student Outreach Disabled</span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 border border-border-medium bg-white hover:bg-slate-100 text-text-primary rounded-md font-medium transition-colors"
          >
            Close
          </button>
        </div>

      </div>
    </div>
  );
};

