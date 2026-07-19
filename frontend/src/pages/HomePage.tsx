import { Link } from "react-router-dom";

export default function HomePage() {
  return (
    <div className="max-w-7xl mx-auto px-4 py-16">
      <div className="text-center mb-16">
        <div className="inline-flex items-center gap-2 bg-hera-primary/10 border border-hera-primary/30 rounded-full px-4 py-1.5 text-sm text-indigo-300 mb-6">
          Team 7 — HERA-VLM Research Project
        </div>
        <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-white via-indigo-200 to-purple-300 bg-clip-text text-transparent">
          Hallucination Evidence Retrieval
          <br />& Attribution for VLMs
        </h1>
        <p className="text-xl text-slate-400 max-w-2xl mx-auto mb-8">
          Reduce vision-language model hallucinations through step-level evidence retrieval,
          verification, and grounded Chain-of-Thought reasoning.
        </p>
        <Link to="/inference" className="btn-primary text-lg px-8 py-3 inline-block">
          Start Inference →
        </Link>
      </div>

      <div className="grid md:grid-cols-3 gap-6 mb-16">
        {[
          {
            title: "Step-Level Verification",
            desc: "Decompose CoT into individual steps and verify each against multimodal evidence.",
            icon: "🔍",
          },
          {
            title: "Evidence Attribution",
            desc: "Ground every claim with retrieved visual regions and textual evidence citations.",
            icon: "📎",
          },
          {
            title: "Hallucination Detection",
            desc: "Detect object, attribute, relationship, scene, and reasoning hallucinations.",
            icon: "⚠️",
          },
        ].map((feature) => (
          <div key={feature.title} className="card hover:border-hera-primary/30 transition-colors">
            <div className="text-3xl mb-4">{feature.icon}</div>
            <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
            <p className="text-slate-400 text-sm">{feature.desc}</p>
          </div>
        ))}
      </div>

      <div className="card">
        <h2 className="text-2xl font-bold mb-6">Pipeline Overview</h2>
        <div className="flex flex-wrap items-center justify-center gap-2 text-sm">
          {[
            "Image + Question",
            "Base VLM CoT",
            "Step Segmentation",
            "Evidence Retrieval",
            "Step Verification",
            "Hallucination Detection",
            "Attribution",
            "Grounded Answer",
          ].map((stage, i, arr) => (
            <div key={stage} className="flex items-center gap-2">
              <span className="bg-slate-700 px-3 py-1.5 rounded-lg whitespace-nowrap">{stage}</span>
              {i < arr.length - 1 && <span className="text-slate-500">→</span>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
