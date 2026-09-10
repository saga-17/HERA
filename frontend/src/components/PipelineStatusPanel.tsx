import type { PipelineStatus } from "../services/types";

interface PipelineStatusPanelProps {
  status: PipelineStatus | null;
  loading?: boolean;
}

function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return "--:--";
  }

  const totalSeconds = Math.max(0, Math.floor(seconds));

  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const secs = totalSeconds % 60;

  if (hours > 0) {
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(
      2,
      "0"
    )}:${String(secs).padStart(2, "0")}`;
  }

  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(
    2,
    "0"
  )}`;
}

function getStageIndex(
  status: PipelineStatus | null,
  currentStage: string
): number {
  if (!status?.stages?.length) {
    return 0;
  }

  const index = status.stages.findIndex(
    (stage) => stage.key === currentStage
  );

  return index >= 0 ? index : 0;
}

export default function PipelineStatusPanel({
  status,
  loading,
}: PipelineStatusPanelProps) {
  const currentStage = status?.stage || "cot_generation";

  const stages = status?.stages || [];

  const progress = Math.max(
    0,
    Math.min(100, status?.progress ?? 0)
  );

  const currentIndex = getStageIndex(
    status,
    currentStage
  );

  const totalStages =
    status?.total_stages ||
    stages.length ||
    0;

  const completedStages =
    status?.completed_stages ??
    (currentStage === "complete"
      ? totalStages
      : currentIndex);

  const isComplete = currentStage === "complete";
  const isError = currentStage === "error";

  const elapsedSeconds =
    status?.elapsed_seconds ?? 0;

  const etaSeconds =
    status?.estimated_remaining_seconds;

  const showEta =
    !isComplete &&
    !isError &&
    etaSeconds !== null &&
    etaSeconds !== undefined &&
    Number.isFinite(etaSeconds);

  return (
    <div className="card h-full">
      <h3 className="text-lg font-semibold mb-4">
        Pipeline Status
      </h3>

      {/* ============================================================
          Active pipeline summary
          ============================================================ */}
      {(loading || status) && (
        <div className="mb-6">
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm text-slate-300">
              {isComplete
                ? "HERA Pipeline Complete"
                : isError
                ? "HERA Pipeline Error"
                : "HERA Pipeline Running"}
            </span>

            <span className="text-sm font-semibold text-white">
              {Math.round(progress)}%
            </span>
          </div>

          {/* Progress bar */}
          <div className="w-full bg-slate-700 rounded-full h-3 overflow-hidden">
            <div
              className={`h-3 rounded-full transition-all duration-500 ${
                isError
                  ? "bg-red-500"
                  : isComplete
                  ? "bg-green-500"
                  : "bg-gradient-to-r from-hera-primary to-hera-secondary"
              }`}
              style={{
                width: `${progress}%`,
              }}
            />
          </div>

          {/* Current stage */}
          <div className="mt-4">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400">
                Stage
              </span>

              <span className="text-xs text-slate-400">
                {completedStages} / {totalStages}
              </span>
            </div>

            <div className="mt-1 text-sm font-medium text-white">
              {status?.stages?.[currentIndex]?.label ||
                currentStage
                  .replace(/_/g, " ")
                  .replace(/\b\w/g, (char) =>
                    char.toUpperCase()
                  )}
            </div>

            {!isComplete && !isError && (
              <div className="text-xs text-slate-500 mt-1">
                Stage {currentIndex + 1} of {totalStages}
              </div>
            )}
          </div>

          {/* Message */}
          {status?.message && (
            <div className="mt-3 text-xs text-slate-400">
              {status.message}
            </div>
          )}

          {/* Timing information */}
          <div className="grid grid-cols-2 gap-3 mt-4">
            <div className="rounded-lg bg-slate-800/60 border border-slate-700 p-3">
              <div className="text-xs text-slate-500">
                Elapsed
              </div>

              <div className="text-sm font-semibold text-white mt-1">
                {formatDuration(elapsedSeconds)}
              </div>
            </div>

            <div className="rounded-lg bg-slate-800/60 border border-slate-700 p-3">
              <div className="text-xs text-slate-500">
                {isComplete
                  ? "Total Time"
                  : "Estimated Remaining"}
              </div>

              <div className="text-sm font-semibold text-white mt-1">
                {isComplete
                  ? formatDuration(elapsedSeconds)
                  : showEta
                  ? formatDuration(etaSeconds)
                  : "Calculating..."}
              </div>
            </div>
          </div>

          {/* Completion message */}
          {isComplete && (
            <div className="mt-4 rounded-lg bg-green-500/10 border border-green-500/30 px-3 py-2">
              <span className="text-sm text-green-400 font-medium">
                Completed in {formatDuration(elapsedSeconds)}
              </span>
            </div>
          )}
        </div>
      )}

      {/* ============================================================
          Actual backend stage list
          ============================================================ */}
      {stages.length > 0 && (
        <ul className="space-y-3">
          {stages.map((stage, index) => {
            const isCurrent =
              stage.key === currentStage;

            const stageTiming =
              status?.stage_timings?.find(
                (timing) => timing.stage === stage.key
              );

            const stageCompleted =
              stageTiming?.duration_seconds !== null &&
              stageTiming?.duration_seconds !== undefined;

            const isStageComplete =
              isComplete ||
              stageCompleted ||
              index < currentIndex;

            const isStageError =
              isError && isCurrent;

            return (
              <li
                key={stage.key}
                className="flex items-center gap-3"
              >
                <div
                  className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${
                    isStageError
                      ? "bg-red-500/20 text-red-400 border border-red-500"
                      : isStageComplete
                      ? "bg-green-500/20 text-green-400 border border-green-500"
                      : isCurrent
                      ? "bg-hera-primary/20 text-indigo-300 border border-hera-primary animate-pulse"
                      : "bg-slate-700 text-slate-500 border border-slate-600"
                  }`}
                >
                  {isStageError
                    ? "✗"
                    : isStageComplete
                    ? "✓"
                    : index + 1}
                </div>

                <div className="flex-1 min-w-0">
                  <div
                    className={`text-sm ${
                      isCurrent
                        ? "text-white font-medium"
                        : isStageComplete
                        ? "text-slate-300"
                        : "text-slate-500"
                    }`}
                  >
                    {stage.label}
                  </div>

                  {/* Show actual completed stage duration */}
                  {stageCompleted && (
                    <div className="text-xs text-slate-500 mt-0.5">
                      {formatDuration(
                        stageTiming?.duration_seconds
                      )}
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}

      {/* Backward-compatible fallback if an old result has no stage metadata */}
      {stages.length === 0 && (
        <div className="text-sm text-slate-500">
          Waiting for pipeline stage information...
        </div>
      )}
    </div>
  );
}