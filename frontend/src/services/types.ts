export interface UploadResponse {
  image_id: string;
  filename: string;
  url: string;
}

export interface AskResponse {
  result_id: string;
  status: string;
}

export type StepStatus = "supported" | "hallucinated" | "uncertain";

export interface VisualEvidence {
  region_id: string;
  bbox: number[];
  image_base64?: string;
  caption: string;
  confidence: number;
}

export interface TextEvidence {
  text: string;
  source: string;
  confidence: number;
}

export interface EntityExtraction {
  entities: string[];
  objects: string[];
  attributes: string[];
  relationships: string[];
  claims: string[];
}

export interface ReasoningStep {
  step_index: number;
  step: string;
  status: StepStatus;
  confidence: number;
  supported: boolean;
  hallucination_type: string;
  evidence: string;
  visual_evidence: VisualEvidence[];
  textual_evidence: TextEvidence[];
  attribution: string;
  extraction: EntityExtraction;
}

export interface PipelineStageInfo {
  key: string;
  label: string;
  order: number;
}

export interface StageTiming {
  stage: string;
  label: string;
  started_at?: number | null;
  completed_at?: number | null;
  duration_seconds?: number | null;
}

export interface PipelineStatus {
  stage: string;
  progress: number;
  message: string;

  // Actual backend pipeline definition.
  stages: PipelineStageInfo[];

  // Current stage position.
  stage_index: number;
  total_stages: number;
  completed_stages: number;

  // Timing.
  started_at?: number | null;
  elapsed_seconds: number;
  current_stage_elapsed_seconds: number;

  // ETA.
  estimated_remaining_seconds?: number | null;
  eta_confidence: "calculating" | "weighted" | "observed" | "complete" | "unavailable";

  // Timing information for completed/current stages.
  stage_timings: StageTiming[];
}

export interface HeraResult {
  result_id: string;
  image_id: string;
  image_url: string;
  question: string;
  original_cot: string;
  attributed_cot: string;
  corrected_cot: string;
  final_answer: string;
  hallucination_score: number;
  confidence_score: number;
  steps: ReasoningStep[];
  pipeline_status: PipelineStatus;
}

export interface HealthResponse {
  status: string;
  version: string;
  gpu_available: boolean;
  demo_mode: boolean;
  model_loaded: boolean;
}