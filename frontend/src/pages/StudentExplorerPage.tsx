import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Filter, 
  ChevronLeft, 
  ChevronRight, 
  ArrowUpDown, 
  Copy, 
  Check, 
  AlertCircle,
  Eye,
  Lock
} from 'lucide-react';
import { analyticsApi } from '../services/api';
import { StudentExplorerItem, StudentExplorerResponse } from '../types';
import { DetailDrawer } from '../components/DetailDrawer';

export const StudentExplorerPage: React.FC = () => {
  const [data, setData] = useState<StudentExplorerResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters & Pagination State
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [selectedModule, setSelectedModule] = useState<string>('');
  const [selectedTier, setSelectedTier] = useState<string>('');
  const [selectedType, setSelectedType] = useState<string>('TMA');
  const [search, setSearch] = useState<string>('');
  const [sortBy, setSortBy] = useState<string>('predicted_risk_score');
  const [sortDir, setSortDir] = useState<string>('desc');

  // Drawer state
  const [selectedStudent, setSelectedStudent] = useState<StudentExplorerItem | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const fetchRecords = async () => {
    try {
      setLoading(true);
      const res = await analyticsApi.getStudentExplorer({
        page,
        pageSize,
        module: selectedModule || undefined,
        riskTier: selectedTier || undefined,
        assessmentType: selectedType || undefined,
        search: search || undefined,
        sortBy,
        sortDir,
      });
      setData(res);
      setError(null);
    } catch (err: any) {
      console.error('Failed to load student explorer records:', err);
      setError(err.message || 'Error fetching records');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecords();
  }, [page, pageSize, selectedModule, selectedTier, selectedType, sortBy, sortDir]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchRecords();
  };

  const handleCopy = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(id);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleSortToggle = (col: string) => {
    if (sortBy === col) {
      setSortDir(sortDir === 'desc' ? 'asc' : 'desc');
    } else {
      setSortBy(col);
      setSortDir('desc');
    }
    setPage(1);
  };

  return (
    <div className="p-8 space-y-6">
      {/* Top Filter and Search Bar */}
      <div className="enterprise-card p-5 space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          {/* Search Form */}
          <form onSubmit={handleSearchSubmit} className="flex-1 max-w-md relative">
            <Search size={16} className="absolute left-3 top-3 text-slate-400" />
            <input
              type="text"
              placeholder="Search assessment ID or student pseudonym..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2 border border-border-subtle rounded-md text-xs bg-slate-50 focus:bg-white focus:outline-none focus:ring-1 focus:ring-primary-blue text-text-primary"
            />
          </form>

          {/* Quick Filter Selectors */}
          <div className="flex flex-wrap items-center gap-3 text-xs">
            {/* Module Filter */}
            <div className="flex items-center space-x-1.5">
              <span className="text-text-muted font-medium">Module:</span>
              <select
                value={selectedModule}
                onChange={(e) => {
                  setSelectedModule(e.target.value);
                  setPage(1);
                }}
                className="border border-border-subtle rounded px-2.5 py-1.5 text-xs bg-white text-text-primary focus:outline-none"
              >
                <option value="">All Modules (AAA–GGG)</option>
                {['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'].map((m) => (
                  <option key={m} value={m}>Module {m}</option>
                ))}
              </select>
            </div>

            {/* Risk Tier Filter */}
            <div className="flex items-center space-x-1.5">
              <span className="text-text-muted font-medium">Risk Tier:</span>
              <select
                value={selectedTier}
                onChange={(e) => {
                  setSelectedTier(e.target.value);
                  setPage(1);
                }}
                className="border border-border-subtle rounded px-2.5 py-1.5 text-xs bg-white text-text-primary focus:outline-none"
              >
                <option value="">All Risk Tiers</option>
                <option value="HIGH_RISK">High Risk (Alerted)</option>
                <option value="MEDIUM_RISK">Medium Risk (Watchlist)</option>
                <option value="LOW_RISK">Low Risk (On Track)</option>
                <option value="SUPPRESSED">Suppressed (CMA)</option>
              </select>
            </div>

            {/* Assessment Type Filter */}
            <div className="flex items-center space-x-1.5">
              <span className="text-text-muted font-medium">Type:</span>
              <select
                value={selectedType}
                onChange={(e) => {
                  setSelectedType(e.target.value);
                  setPage(1);
                }}
                className="border border-border-subtle rounded px-2.5 py-1.5 text-xs bg-white text-text-primary focus:outline-none"
              >
                <option value="TMA">TMA (Actionable Only)</option>
                <option value="CMA">CMA (Research Telemetry)</option>
                <option value="">All Types (50,014 Snapshots)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Informational Guidance Ribbon */}
        <div className="pt-3 border-t border-border-subtle flex items-center justify-between text-[11px] text-text-muted">
          <span className="flex items-center space-x-1.5">
            <Lock size={12} className="text-slate-500" />
            <span>Strict Privacy Enforcement: All IDs are keyed HMAC-SHA256 digests. Raw student IDs never exposed.</span>
          </span>
          <span className="tabular-nums">
            Total Matching: {data?.total_records.toLocaleString() || 0} snapshots
          </span>
        </div>
      </div>

      {/* Main Data Table */}
      <div className="enterprise-card overflow-hidden">
        {loading ? (
          <div className="p-12 text-center text-text-muted text-xs animate-pulse">
            Loading student assessment snapshots from chronological holdout...
          </div>
        ) : error ? (
          <div className="p-8 text-center text-risk-high-text bg-risk-high-surface text-xs space-y-2">
            <AlertCircle size={20} className="mx-auto text-risk-high-base" />
            <p className="font-semibold">{error}</p>
          </div>
        ) : data && data.records.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left enterprise-table">
              <thead>
                <tr>
                  <th>Pseudonymized Student ID</th>
                  <th>Module</th>
                  <th>Assessment</th>
                  <th 
                    className="cursor-pointer hover:text-primary-blue"
                    onClick={() => handleSortToggle('scheduled_date')}
                  >
                    <div className="flex items-center space-x-1">
                      <span>Scheduled Day</span>
                      <ArrowUpDown size={12} />
                    </div>
                  </th>
                  <th
                    className="cursor-pointer hover:text-primary-blue"
                    onClick={() => handleSortToggle('predicted_risk_score')}
                  >
                    <div className="flex items-center space-x-1">
                      <span>Predicted Risk</span>
                      <ArrowUpDown size={12} />
                    </div>
                  </th>
                  <th>Risk Tier</th>
                  <th>Operational Threshold</th>
                  <th>Governance Action</th>
                  <th>Outreach</th>
                  <th className="text-right">Inspect</th>
                </tr>
              </thead>
              <tbody>
                {data.records.map((rec, i) => {
                  const isHigh = rec.risk_tier === 'HIGH_RISK';
                  const isMed = rec.risk_tier === 'MEDIUM_RISK';
                  const isSupp = rec.risk_tier === 'SUPPRESSED';

                  return (
                    <tr
                      key={i}
                      onClick={() => setSelectedStudent(rec)}
                      className="cursor-pointer hover:bg-slate-50 transition-colors"
                    >
                      <td className="font-mono text-xs font-semibold text-text-primary">
                        <div className="flex items-center space-x-1.5">
                          <span>{rec.pseudonymized_student_id}</span>
                          <button
                            onClick={(e) => handleCopy(rec.pseudonymized_student_id, e)}
                            className="text-slate-400 hover:text-slate-600 p-0.5 rounded"
                            title="Copy ID"
                          >
                            {copiedId === rec.pseudonymized_student_id ? (
                              <Check size={12} className="text-emerald-600" />
                            ) : (
                              <Copy size={12} />
                            )}
                          </button>
                        </div>
                      </td>
                      <td>
                        <span className="inline-block px-1.5 py-0.5 rounded text-[11px] font-semibold bg-slate-100 text-slate-800">
                          {rec.code_module}
                        </span>
                      </td>
                      <td className="text-xs text-text-secondary">
                        {rec.assessment_type} #{rec.id_assessment}
                      </td>
                      <td className="text-xs text-text-secondary tabular-nums">
                        Day {rec.scheduled_date}
                      </td>
                      <td className="text-xs font-bold tabular-nums">
                        <span
                          className={
                            isHigh
                              ? 'text-risk-high-base'
                              : isMed
                              ? 'text-risk-mod-base'
                              : isSupp
                              ? 'text-slate-500'
                              : 'text-risk-low-base'
                          }
                        >
                          {(rec.predicted_risk_score * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td>
                        <span
                          className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold border ${
                            isHigh
                              ? 'bg-risk-high-surface text-risk-high-text border-risk-high-border'
                              : isMed
                              ? 'bg-risk-mod-surface text-risk-mod-text border-risk-mod-border'
                              : isSupp
                              ? 'bg-risk-suppressed-surface text-risk-suppressed-text border-risk-suppressed-border'
                              : 'bg-risk-low-surface text-risk-low-text border-risk-low-border'
                          }`}
                        >
                          {rec.risk_tier.replace('_', ' ')}
                        </span>
                      </td>
                      <td className="text-xs font-mono text-text-secondary">
                        {rec.operational_threshold_applied !== null
                          ? `${(rec.operational_threshold_applied * 100).toFixed(0)}%`
                          : 'Suppressed'}
                      </td>
                      <td>
                        <span className="text-[11px] text-text-secondary">
                          {rec.governance_action.replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td>
                        <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-600">
                          Prohibited
                        </span>
                      </td>
                      <td className="text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedStudent(rec);
                          }}
                          className="text-primary-blue hover:text-primary-hover p-1 rounded hover:bg-slate-100 transition-colors"
                          title="View Case Snapshot"
                        >
                          <Eye size={15} />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-12 text-center text-text-muted text-xs">
            No student snapshots matching the selected criteria.
          </div>
        )}

        {/* Pagination Bar */}
        {data && (
          <div className="px-6 py-4 border-t border-border-subtle bg-slate-50 flex items-center justify-between text-xs">
            <div className="text-text-muted">
              Showing <span className="font-semibold tabular-nums text-text-primary">{(page - 1) * pageSize + 1}</span> to{' '}
              <span className="font-semibold tabular-nums text-text-primary">
                {Math.min(page * pageSize, data.total_records)}
              </span>{' '}
              of <span className="font-semibold tabular-nums text-text-primary">{data.total_records.toLocaleString()}</span> records
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={() => setPage(Math.max(1, page - 1))}
                disabled={page <= 1}
                className="px-2.5 py-1.5 border border-border-subtle rounded bg-white disabled:opacity-40 text-text-secondary hover:bg-slate-100 transition-colors flex items-center space-x-1"
              >
                <ChevronLeft size={14} />
                <span>Previous</span>
              </button>

              <span className="font-semibold tabular-nums px-2">
                Page {page} of {data.total_pages}
              </span>

              <button
                onClick={() => setPage(Math.min(data.total_pages, page + 1))}
                disabled={page >= data.total_pages}
                className="px-2.5 py-1.5 border border-border-subtle rounded bg-white disabled:opacity-40 text-text-secondary hover:bg-slate-100 transition-colors flex items-center space-x-1"
              >
                <span>Next</span>
                <ChevronRight size={14} />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Side Detail Drawer */}
      <DetailDrawer
        student={selectedStudent}
        onClose={() => setSelectedStudent(null)}
      />
    </div>
  );
};

