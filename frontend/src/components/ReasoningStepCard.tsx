import type { ReasoningStep } from "../services/types";

interface ReasoningStepCardProps {
  step: ReasoningStep;
}

const statusConfig = {
  supported: {
    icon: "✓",
    label: "Supported",
    bg: "bg-green-500/10 border-green-500/30",
    text: "text-green-400",
    badge: "bg-green-500/20 text-green-300",
  },
  hallucinated: {
    icon: "✗",
    label: "Hallucinated",
    bg: "bg-red-500/10 border-red-500/30",
    text: "text-red-400",
    badge: "bg-red-500/20 text-red-300",
  },
  uncertain: {
    icon: "~",
    label: "Uncertain",
    bg: "bg-yellow-500/10 border-yellow-500/30",
    text: "text-yellow-400",
    badge: "bg-yellow-500/20 text-yellow-300",
  },
};

export default function ReasoningStepCard({ step }: ReasoningStepCardProps) {
  const config = statusConfig[step.status] || statusConfig.uncertain;

  return (
    <div className={`rounded-xl border p-5 ${config.bg}`}>
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="flex items-center gap-2">
          <span className={`text-lg font-bold ${config.text}`}>{config.icon}</span>
          <h4 className="font-semibold">Step {step.step_index + 1}</h4>
          <span className={`text-xs px-2 py-0.5 rounded-full ${config.badge}`}>
            {config.label}
          </span>
        </div>
        <div className="text-right shrink-0">
          <div className="text-xs text-slate-400">Confidence</div>
          <div className={`font-mono font-bold ${config.text}`}>
            {(step.confidence * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      <p className="text-slate-200 mb-4 leading-relaxed">{step.step}</p>

      {step.hallucination_type && step.hallucination_type !== "none" && (
        <div className="text-xs text-red-300 mb-3">
          Type: {step.hallucination_type.replace(/_/g, " ")}
        </div>
      )}

      {step.attribution && (
        <div className="text-sm text-slate-400 mb-4 italic">{step.attribution}</div>
      )}

      {/* Entity extraction */}
      {(step.extraction.objects.length > 0 || step.extraction.attributes.length > 0) && (
        <div className="flex flex-wrap gap-2 mb-4">
          {step.extraction.objects.map((obj) => (
            <span key={obj} className="text-xs bg-slate-700/50 px-2 py-1 rounded">
              🎯 {obj}
            </span>
          ))}
          {step.extraction.attributes.map((attr) => (
            <span key={attr} className="text-xs bg-slate-700/50 px-2 py-1 rounded">
              🏷 {attr}
            </span>
          ))}
        </div>
      )}

      {/* Visual evidence */}
      {step.visual_evidence.length > 0 && (
        <div className="mt-3">
          <div className="text-xs text-slate-400 mb-2">Visual Evidence</div>
          <div className="flex gap-3 flex-wrap">
            {step.visual_evidence.map((ve) => (
              <div key={ve.region_id} className="relative">
                {ve.image_base64 && (
                  <img
                    src={`data:image/png;base64,${ve.image_base64}`}
                    alt={ve.caption}
                    className="w-24 h-24 object-cover rounded-lg border border-slate-600"
                  />
                )}
                <div className="text-xs text-slate-500 mt-1 max-w-24 truncate">
                  {ve.caption}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Textual evidence */}
      {step.textual_evidence.length > 0 && (
        <div className="mt-3">
          <div className="text-xs text-slate-400 mb-1">Textual Evidence</div>
          {step.textual_evidence.slice(0, 2).map((te, i) => (
            <div key={i} className="text-xs text-slate-400 bg-slate-800/50 rounded p-2 mb-1">
              <span className="text-slate-500">[{te.source}]</span> {te.text.slice(0, 150)}
              {te.text.length > 150 ? "..." : ""}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
