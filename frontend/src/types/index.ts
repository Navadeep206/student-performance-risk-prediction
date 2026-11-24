export type NavTab = 
  | 'overview'
  | 'student-explorer'
  | 'modules'
  | 'explainability'
  | 'governance'
  | 'monitoring';

export interface CohortSummary {
  enrolled_learners_2014j: number;
  total_prediction_snapshots: number;
  actionable_tma_alerts: number;
  unique_learners_alerted: number;
  alert_coverage_pct: number;
  mean_alerts_per_alerted_learner: number;
  max_alerts_per_learner: number;
  estimated_weekly_alert_load: number;
  raw_cma_alerts_suppressed: number;
  cma_suppression_workload_reduction_pct: number;
  tma_alert_rate_pct: number;
  overall_alert_rate_pct: number;
  milestone_alert_distribution: {
    TMA_1_Onboarding: number;
    TMA_2_Midterm: number;
    TMA_3_Capstone: number;
  };
  operational_mode: string;
  outreach_permitted: boolean;
}

export interface RiskDistributionBin {
  bin_label: string;
  min_prob: number;
  max_prob: number;
  tma_count: number;
  cma_count: number;
  total_count: number;
  tma_percentage: number;
  cma_percentage: number;
}

export interface RiskDistributionResponse {
  total_snapshots: number;
  tma_snapshots: number;
  cma_snapshots: number;
  bins: RiskDistributionBin[];
  threshold_markers: {
    module_ggg_calibrated_tma: number;
    standard_modules_tma: number;
  };
  tier_counts: {
    HIGH_RISK: number;
    MEDIUM_RISK: number;
    LOW_RISK: number;
    SUPPRESSED: number;
  };
}

export interface StudentExplorerItem {
  pseudonymized_student_id: string;
  code_module: string;
  code_presentation: string;
  id_assessment: number;
  assessment_type: string;
  scheduled_date: number;
  assessment_weight: number;
  predicted_risk_score: number;
  risk_tier: string;
  operational_threshold_applied: number | null;
  predicted_at_risk: boolean;
  governance_action: string;
  suppress_automated_outreach: boolean;
  student_outreach_enabled: boolean;
  advising_recommendation: string;
  is_novel_student: boolean;
  influential_features: Record<string, any>;
}

export interface StudentExplorerResponse {
  total_records: number;
  page: number;
  page_size: number;
  total_pages: number;
  records: StudentExplorerItem[];
}

export interface ModuleMetricItem {
  code_module: string;
  faculty_domain: string;
  benchmark_snapshots: number;
  benchmark_failures: number;
  benchmark_failure_rate_pct: number;
  holdout_2014j_snapshots: number;
  holdout_2014j_failures: number;
  holdout_2014j_failure_rate_pct: number;
  actionable_tma_snapshots_2014j: number;
  governed_tma_alerts_2014j: number;
  governed_tma_alert_rate_pct: number;
  dominant_milestone_risk: string;
}

export interface ModuleGGGMatchedEvaluation {
  matched_snapshots_population: number;
  unique_learners_count: number;
  positive_failures_count: number;
  baseline_failure_prevalence_pct: number;
  baseline_pr_auc: number;
  remediated_pr_auc: number;
  pr_auc_delta: number;
  baseline_roc_auc: number;
  remediated_roc_auc: number;
  global_tau_020_baseline_recall: number;
  global_tau_020_remediated_recall: number;
  calibrated_tau_010_baseline_recall: number;
  calibrated_tau_010_remediated_recall: number;
  recall_gain_at_calibrated_threshold_pct: number;
  baseline_additional_detected_cases: number;
  remediated_additional_detected_cases: number;
  additional_at_risk_captured_count: number;
  cma_quizzes_total_count: number;
  cma_quizzes_weight: number;
  decoupling_rationale: string;
}

export interface ModulesComparisonResponse {
  benchmark_total_snapshots: number;
  benchmark_total_failures: number;
  benchmark_overall_prevalence_pct: number;
  holdout_2014j_total_snapshots: number;
  holdout_2014j_total_failures: number;
  holdout_2014j_overall_prevalence_pct: number;
  holdout_2014j_actionable_tma_snapshots: number;
  holdout_2014j_governed_tma_alerts: number;
  modules: ModuleMetricItem[];
  module_ggg_matched_evaluation: ModuleGGGMatchedEvaluation;
}

export interface FeatureImportanceItem {
  feature_name: string;
  display_name: string;
  mean_abs_shap: number;
  category: string;
  risk_impact_direction: string;
}

export interface ExplainabilityResponse {
  model_name: string;
  explainability_method: string;
  total_features_evaluated: number;
  features: FeatureImportanceItem[];
  methodological_guidance: string[];
}

export interface GovernanceAuditResponse {
  operational_mode: string;
  allow_student_outreach: boolean;
  cma_telemetry_suppressed: boolean;
  threshold_policy: Record<string, any>;
  pseudonymization_method: string;
  prohibited_features_enforced: string[];
  recent_audit_records_count: number;
  recent_records: Record<string, any>[];
}

export interface DriftMetricItem {
  metric_name: string;
  grouped_cv_value: number;
  chronological_2014j_value: number;
  drift_delta: number;
  drift_pct: number;
  status: string;
}

export interface FeaturePSIDriftItem {
  feature_name: string;
  psi_value: number;
  drift_severity: string;
  commentary: string;
}

export interface MonitoringResponse {
  prospective_cohort: string;
  total_snapshots: number;
  novel_students_share_pct: number;
  performance_generalization_gap: DriftMetricItem[];
  top_feature_psi_drift: FeaturePSIDriftItem[];
  calibration_metrics: Record<string, number>;
  ingestion_and_telemetry_limitations: string[];
}

export interface PredictionItemInput {
  id_student: number | string;
  code_module: string;
  code_presentation: string;
  id_assessment: number;
  target_assessment_type: string;
  scheduled_date: number;
  target_assessment_weight?: number;
  num_previous_scheduled_assessments?: number;
  registration_timing_days?: number | null;
  vle_total_clicks_to_date?: number;
  vle_active_days_to_date?: number;
  vle_core_clicks_to_date?: number;
  vle_forum_clicks_to_date?: number;
  vle_quiz_clicks_to_date?: number;
  vle_days_since_last_activity?: number | null;
  vle_clicks_last_7d?: number;
  vle_clicks_last_14d?: number;
  vle_activity_trend_ratio?: number | null;
  num_prior_submissions?: number;
  prior_submission_lateness_mean?: number | null;
  num_prior_scored_assessments?: number;
  mean_prior_score?: number | null;
  std_prior_score?: number | null;
  prior_at_risk_rate?: number | null;
  recent_prior_score?: number | null;
  [key: string]: any;
}

export interface SinglePredictionData {
  prediction_id: string;
  timestamp_utc: string;
  pseudonymized_student_id: string;
  code_module: string;
  code_presentation: string;
  id_assessment: number;
  assessment_type: string;
  scheduled_date: number;
  raw_risk_probability: number;
  risk_tier: string;
  operational_threshold_applied: number | null;
  predicted_at_risk: boolean;
  governance_action: string;
  suppress_automated_outreach: boolean;
  student_outreach_enabled: boolean;
  advising_recommendation: string;
  model_name: string;
  model_version: string;
  feature_pipeline_version: string;
  evaluation_status: string;
}

export interface BatchItemResult {
  item_index: number;
  success: boolean;
  data: SinglePredictionData | null;
  error: string | null;
}

export interface BatchPredictionResponse {
  total_requested: number;
  successful_count: number;
  failed_count: number;
  results: BatchItemResult[];
}


