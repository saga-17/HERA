import { useEffect, useRef, useState } from "react";
import type { PipelineStatus } from "../services/types";

interface PipelineStatusPanelProps {
  status: PipelineStatus | null;
  loading?: boolean;
}

const FALLBACK_STAGES = [
  { key: "cot_generation", label: "CoT Generation", order: 1 },
  { key: "step_segmentation", label: "Step Segmentation", order: 2 },
  { key: "evidence_retrieval", label: "Evidence Retrieval", order: 3 },
  { key: "step_verification", label: "Step Verification", order: 4 },
  { key: "hallucination_detection", label: "Hallucination Detection", order: 5 },
  { key: "evidence_attribution", label: "Evidence Attribution", order: 6 },
  { key: "reasoning_correction", label: "Reasoning Correction", order: 7 },
  { key: "complete", label: "Complete", order: 8 },
];

const STAGE_ESTIMATES: Record<string, number> = {
  cot_generation: 20,
  step_segmentation: 2,
  evidence_retrieval: 30,
  step_verification: 15,
  hallucination_detection: 3,
  evidence_attribution: 5,
  reasoning_correction: 15,
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function getStageIndex(status: PipelineStatus | null, currentStage: string): number {
  if (!status?.stages?.length) {
    return 0;
  }

  const index = status.stages.findIndex((stage) => stage.key === currentStage);
  return index >= 0 ? index : 0;
}

function getStageWindow(
  stages: { key: string; order: number }[],
  totalStages: number,
  stageKey: string,
): { lower: number; upper: number } {
  if (!stages.length || totalStages <= 0) {
    return { lower: 0, upper: 12.5 };
  }

  const stage = stages.find((candidate) => candidate.key === stageKey);
  const order = Number(stage?.order ?? 0);
  if (order <= 0) {
    return { lower: 0, upper: 12.5 };
  }

  return {
    lower: ((order - 1) / totalStages) * 100,
    upper: (order / totalStages) * 100,
  };
}

export default function PipelineStatusPanel({
  status,
  loading,
}: PipelineStatusPanelProps) {
  const currentStage = status?.stage || "cot_generation";
  const stages = status?.stages?.length ? status.stages : FALLBACK_STAGES;
  const totalStages = status?.total_stages || stages.length || FALLBACK_STAGES.length;
  const currentIndex = getStageIndex(status, currentStage);
  const completedStages =
    status?.completed_stages ?? (currentStage === "complete" ? totalStages : currentIndex);

  const isComplete = currentStage === "complete";
  const isError = currentStage === "error";
  const isCancelled = currentStage === "cancelled";

  const [displayProgress, setDisplayProgress] = useState<number>(() =>
    clamp(status?.progress ?? 0, 0, 100)
  );

  const lastStageRef = useRef<string>(currentStage);
  const stageStartRef = useRef<number | null>(null);

  useEffect(() => {
    if (!status && !loading) {
      return;
    }

    if (!status) {
      setDisplayProgress(0);
      return;
    }

    if (isComplete) {
      setDisplayProgress(100);
      return;
    }

    if (isError || isCancelled) {
      const frozenValue = clamp(status.progress ?? 0, 0, 100);
      setDisplayProgress(frozenValue);
      return;
    }

    const stageWindow = getStageWindow(stages, totalStages, currentStage);
    const authoritativeProgress = clamp(status.progress ?? 0, 0, 100);
    const estimateSeconds = Math.max(status.current_stage_elapsed_seconds || STAGE_ESTIMATES[currentStage] || 10, 1);

    if (lastStageRef.current !== currentStage) {
      lastStageRef.current = currentStage;
      stageStartRef.current = Date.now();
    }

    const stageStartMs = stageStartRef.current ?? Date.now();
    let rafId = 0;

    const tick = () => {
      const elapsedMs = Date.now() - stageStartMs;
      const elapsedRatio = clamp(elapsedMs / 1000 / estimateSeconds, 0, 1);
      const softProgress = stageWindow.lower + elapsedRatio * (stageWindow.upper - stageWindow.lower);
      const capProgress = Math.min(softProgress, stageWindow.upper);
      const nextValue = Math.max(capProgress, authoritativeProgress);

      setDisplayProgress((previous) => {
        const candidate = clamp(nextValue, 0, 100);
        return Math.max(previous, candidate);
      });

      if (authoritativeProgress >= stageWindow.upper - 0.05) {
        setDisplayProgress(authoritativeProgress);
        return;
      }

      rafId = window.requestAnimationFrame(tick);
    };

    tick();

    return () => {
      if (rafId) {
        window.cancelAnimationFrame(rafId);
      }
    };
  }, [
    status,
    loading,
    currentStage,
    stages,
    totalStages,
    isComplete,
    isError,
    isCancelled,
  ]);

  const progress = clamp(displayProgress, 0, 100);

  return (
    <div className="card h-full">
      <h3 className="text-lg font-semibold mb-4">Pipeline Status</h3>

      {(loading || status) && (
        <div className="mb-6">
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm text-slate-300">
              {isComplete
                ? "HERA Pipeline Complete"
                : isError
                ? "HERA Pipeline Error"
                : isCancelled
                ? "HERA Pipeline Cancelled"
                : "HERA Pipeline Running"}
            </span>

            <span className="text-sm font-semibold text-white">{Math.round(progress)}%</span>
          </div>

          <div className="w-full bg-slate-700 rounded-full h-3 overflow-hidden">
            <div
              className={`h-3 rounded-full transition-all duration-150 ${
                isError
                  ? "bg-red-500"
                  : isCancelled
                  ? "bg-amber-500"
                  : isComplete
                  ? "bg-green-500"
                  : "bg-gradient-to-r from-hera-primary to-hera-secondary"
              }`}
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="mt-4">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400">Stage</span>
              <span className="text-xs text-slate-400">
                {completedStages} / {totalStages}
              </span>
            </div>

            <div className="mt-1 text-sm font-medium text-white">
              {status?.stages?.[currentIndex]?.label ||
                currentStage
                  .replace(/_/g, " ")
                  .replace(/\b\w/g, (char) => char.toUpperCase())}
            </div>

            {!isComplete && !isError && !isCancelled && (
              <div className="text-xs text-slate-500 mt-1">
                Stage {currentIndex + 1} of {totalStages}
              </div>
            )}
          </div>

          {status?.message && (
            <div className="mt-3 text-xs text-slate-400">{status.message}</div>
          )}

          {isCancelled && (
            <div className="mt-4 rounded-lg bg-amber-500/10 border border-amber-500/30 px-3 py-2">
              <span className="text-sm text-amber-300 font-medium">Pipeline cancelled.</span>
            </div>
          )}
        </div>
      )}

      {stages.length > 0 && (
        <ul className="space-y-3">
          {stages.map((stage, index) => {
            const stageTiming = status?.stage_timings?.find(
              (timing) => timing.stage === stage.key
            );

            const hasDuration =
              stageTiming?.duration_seconds !== null &&
              stageTiming?.duration_seconds !== undefined &&
              Number.isFinite(stageTiming.duration_seconds);

            const isCurrent = stage.key === currentStage;
            const isCompleted = isComplete || hasDuration || index < currentIndex;
            const isRunningStage =
              !isComplete && !isError && !isCancelled && isCurrent && !hasDuration;
            const isPending = !isCompleted && !isRunningStage && !isError && !isCancelled;
            const isFailure = isError && isCurrent;
            const isCancelledStage = isCancelled && isCurrent;

            const iconClass = isFailure
              ? "bg-red-500/20 text-red-400 border border-red-500"
              : isCompleted
              ? "bg-green-500/20 text-green-400 border border-green-500"
              : isRunningStage
              ? "bg-hera-primary/20 text-indigo-300 border border-hera-primary animate-pulse"
              : isCancelledStage
              ? "bg-amber-500/20 text-amber-300 border border-amber-500"
              : isPending
              ? "bg-slate-700 text-slate-500 border border-slate-600"
              : "bg-slate-700 text-slate-500 border border-slate-600";

            const textClass = isFailure
              ? "text-red-300"
              : isCompleted
              ? "text-slate-300"
              : isRunningStage
              ? "text-white font-medium"
              : isCancelledStage
              ? "text-amber-300"
              : isPending
              ? "text-slate-500"
              : "text-slate-500";

            const iconLabel = isFailure
              ? "✗"
              : isCompleted
              ? "✓"
              : isCancelledStage
              ? "−"
              : isRunningStage
              ? "◉"
              : "○";

            return (
              <li key={stage.key} className="flex items-center gap-3">
                <div
                  className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${iconClass}`}
                >
                  {iconLabel}
                </div>

                <div className="flex-1 min-w-0">
                  <div className={`text-sm ${textClass}`}>{stage.label}</div>

                  {hasDuration && (
                    <div className="text-xs text-slate-500 mt-0.5">
                      Stage completed
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}

      {stages.length === 0 && (
        <div className="text-sm text-slate-500">Waiting for pipeline stage information...</div>
      )}
    </div>
  );
}
