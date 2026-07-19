import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import ReasoningStepCard from "../components/ReasoningStepCard";
import { getImageUrl, getResult } from "../services/api";
import type { HeraResult } from "../services/types";

export default function ResultPage() {
  const { resultId } = useParams<{ resultId: string }>();
  const [result, setResult] = useState<HeraResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!resultId) return;
    getResult(resultId)
      .then(setResult)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [resultId]);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-16 text-center">
        <div className="animate-pulse text-slate-400">Loading results...</div>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-16 text-center">
        <p className="text-red-400 mb-4">{error || "Result not found"}</p>
        <Link to="/inference" className="btn-primary">Try Again</Link>
      </div>
    );
  }

  const hallucinationPct = (result.hallucination_score * 100).toFixed(0);
  const confidencePct = (result.confidence_score * 100).toFixed(0);

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold">Results</h1>
        <Link to="/inference" className="btn-secondary">New Inference</Link>
      </div>

      {/* Summary cards */}
      <div className="grid md:grid-cols-4 gap-4 mb-8">
        <div className="card text-center">
          <div className="text-xs text-slate-400 mb-1">Hallucination Score</div>
          <div className={`text-3xl font-bold ${result.hallucination_score > 0.3 ? "text-red-400" : "text-green-400"}`}>
            {hallucinationPct}%
          </div>
        </div>
        <div className="card text-center">
          <div className="text-xs text-slate-400 mb-1">Confidence Score</div>
          <div className="text-3xl font-bold text-indigo-400">{confidencePct}%</div>
        </div>
        <div className="card text-center">
          <div className="text-xs text-slate-400 mb-1">Total Steps</div>
          <div className="text-3xl font-bold">{result.steps.length}</div>
        </div>
        <div className="card text-center">
          <div className="text-xs text-slate-400 mb-1">Hallucinated Steps</div>
          <div className="text-3xl font-bold text-red-400">
            {result.steps.filter((s) => !s.supported).length}
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mb-8">
        {/* Image + Question */}
        <div className="card">
          <h3 className="font-semibold mb-3">Input</h3>
          <img
            src={getImageUrl(result.image_id)}
            alt="Uploaded"
            className="w-full max-h-64 object-contain rounded-lg mb-4"
          />
          <p className="text-slate-300">
            <span className="text-slate-500">Q:</span> {result.question}
          </p>
        </div>

        {/* Final Answer */}
        <div className="card">
          <h3 className="font-semibold mb-3">Final Grounded Answer</h3>
          <p className="text-lg text-green-300 leading-relaxed">{result.final_answer}</p>
        </div>
      </div>

      {/* CoT sections */}
      <div className="grid lg:grid-cols-2 gap-6 mb-8">
        <div className="card">
          <h3 className="font-semibold mb-3">Original Chain-of-Thought</h3>
          <pre className="text-sm text-slate-300 whitespace-pre-wrap font-sans leading-relaxed max-h-64 overflow-y-auto">
            {result.original_cot}
          </pre>
        </div>
        <div className="card">
          <h3 className="font-semibold mb-3">Evidence-Attributed CoT</h3>
          <pre className="text-sm text-slate-300 whitespace-pre-wrap font-sans leading-relaxed max-h-64 overflow-y-auto">
            {result.attributed_cot}
          </pre>
        </div>
      </div>

      {result.corrected_cot && (
        <div className="card mb-8">
          <h3 className="font-semibold mb-3">Corrected Reasoning</h3>
          <pre className="text-sm text-slate-300 whitespace-pre-wrap font-sans leading-relaxed">
            {result.corrected_cot}
          </pre>
        </div>
      )}

      {/* Step-by-step visualization */}
      <h2 className="text-2xl font-bold mb-4">Reasoning Steps</h2>
      <div className="space-y-4">
        {result.steps.map((step) => (
          <ReasoningStepCard key={step.step_index} step={step} />
        ))}
      </div>
    </div>
  );
}
