import {
  CohortSummary,
  RiskDistributionResponse,
  StudentExplorerResponse,
  ModulesComparisonResponse,
  ExplainabilityResponse,
  GovernanceAuditResponse,
  MonitoringResponse,
  PredictionItemInput,
  BatchPredictionResponse,
} from '../types';

const API_BASE = '/api/v1';

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`API Request to ${url} failed with HTTP ${res.status}`);
  }
  return res.json();
}

async function postJson<T>(url: string, body: any): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`API Request to ${url} failed with HTTP ${res.status}: ${errorText}`);
  }
  return res.json();
}

export const analyticsApi = {
  getCohortSummary: (): Promise<CohortSummary> => {
    return fetchJson<CohortSummary>(`${API_BASE}/analytics/cohort-summary`);
  },

  getRiskDistribution: (): Promise<RiskDistributionResponse> => {
    return fetchJson<RiskDistributionResponse>(`${API_BASE}/analytics/risk-distribution`);
  },

  getStudentExplorer: (params: {
    page?: number;
    pageSize?: number;
    module?: string;
    riskTier?: string;
    assessmentType?: string;
    search?: string;
    sortBy?: string;
    sortDir?: string;
  }): Promise<StudentExplorerResponse> => {
    const q = new URLSearchParams();
    if (params.page) q.set('page', params.page.toString());
    if (params.pageSize) q.set('page_size', params.pageSize.toString());
    if (params.module) q.set('module', params.module);
    if (params.riskTier) q.set('risk_tier', params.riskTier);
    if (params.assessmentType) q.set('assessment_type', params.assessmentType);
    if (params.search) q.set('search', params.search);
    if (params.sortBy) q.set('sort_by', params.sortBy);
    if (params.sortDir) q.set('sort_dir', params.sortDir);

    return fetchJson<StudentExplorerResponse>(`${API_BASE}/analytics/student-explorer?${q.toString()}`);
  },

  getModulesComparison: (): Promise<ModulesComparisonResponse> => {
    return fetchJson<ModulesComparisonResponse>(`${API_BASE}/analytics/modules-comparison`);
  },

  getGlobalImportance: (): Promise<ExplainabilityResponse> => {
    return fetchJson<ExplainabilityResponse>(`${API_BASE}/explainability/global-importance`);
  },

  getGovernanceAudit: (): Promise<GovernanceAuditResponse> => {
    return fetchJson<GovernanceAuditResponse>(`${API_BASE}/governance/audit-records`);
  },

  getMonitoringDrift: (): Promise<MonitoringResponse> => {
    return fetchJson<MonitoringResponse>(`${API_BASE}/monitoring/psi-feature-drift`);
  },

  predictBatch: (predictions: PredictionItemInput[]): Promise<BatchPredictionResponse> => {
    return postJson<BatchPredictionResponse>(`${API_BASE}/predict/batch`, { predictions });
  },
};


