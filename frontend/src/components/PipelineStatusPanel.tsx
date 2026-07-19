import type { PipelineStatus } from "../services/types";

const STAGE_LABELS: Record<string, string> = {
  upload: "Upload",
  cot_generation: "CoT Generation",
  step_segmentation: "Step Segmentation",
  evidence_retrieval: "Evidence Retrieval",
  step_verification: "Step Verification",
  hallucination_detection: "Hallucination Detection",
  evidence_attribution: "Evidence Attribution",
  reasoning_correction: "Reasoning Correction",
  complete: "Complete",
  error: "Error",
};

interface PipelineStatusPanelProps {
  status: PipelineStatus | null;
  loading?: boolean;
}

export default function PipelineStatusPanel({ status, loading }: PipelineStatusPanelProps) {
  const stages = [
    "cot_generation",
    "step_segmentation",
    "evidence_retrieval",
    "step_verification",
    "hallucination_detection",
    "evidence_attribution",
    "reasoning_correction",
    "complete",
  ];

  const currentStage = status?.stage || "cot_generation";
  const progress = status?.progress || 0;
  const currentIndex = stages.indexOf(currentStage);

  return (
    <div className="card h-full">
      <h3 className="text-lg font-semibold mb-4">Pipeline Status</h3>

      {loading && (
        <div className="mb-4">
          <div className="flex justify-between text-sm text-slate-400 mb-1">
            <span>{status?.message || "Processing..."}</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-2">
            <div
              className="bg-gradient-to-r from-hera-primary to-hera-secondary h-2 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      <ul className="space-y-3">
        {stages.map((stage, i) => {
          const isComplete = currentIndex > i || currentStage === "complete";
          const isCurrent = stage === currentStage;
          const isError = currentStage === "error" && isCurrent;

          return (
            <li key={stage} className="flex items-center gap-3">
              <div
                className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${
                  isError
                    ? "bg-red-500/20 text-red-400 border border-red-500"
                    : isComplete
                    ? "bg-green-500/20 text-green-400 border border-green-500"
                    : isCurrent
                    ? "bg-hera-primary/20 text-indigo-300 border border-hera-primary animate-pulse"
                    : "bg-slate-700 text-slate-500 border border-slate-600"
                }`}
              >
                {isComplete ? "✓" : isError ? "✗" : i + 1}
              </div>
              <span
                className={`text-sm ${
                  isCurrent ? "text-white font-medium" : isComplete ? "text-slate-300" : "text-slate-500"
                }`}
              >
                {STAGE_LABELS[stage] || stage}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
