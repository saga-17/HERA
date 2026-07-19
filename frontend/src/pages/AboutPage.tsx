export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <h1 className="text-4xl font-bold mb-8">About HERA-VLM</h1>

      <div className="space-y-8">
        <section className="card">
          <h2 className="text-xl font-semibold mb-4">Research Overview</h2>
          <p className="text-slate-300 leading-relaxed">
            HERA-VLM (Hallucination Evidence Retrieval and Attribution for Vision Language Models)
            is a research project built on the CaVe-VLM-CoT framework. It addresses the critical
            problem of hallucinations in vision-language model outputs by decomposing Chain-of-Thought
            reasoning into verifiable steps, retrieving multimodal evidence, and attributing each
            claim to its supporting sources.
          </p>
        </section>

        <section className="card">
          <h2 className="text-xl font-semibold mb-4">Hallucination Categories</h2>
          <div className="grid sm:grid-cols-2 gap-3">
            {[
              "Object Hallucination",
              "Attribute Hallucination",
              "Relationship Hallucination",
              "Scene Hallucination",
              "Commonsense Hallucination",
              "Reasoning Hallucination",
            ].map((cat) => (
              <div key={cat} className="flex items-center gap-2 text-slate-300">
                <span className="text-red-400">●</span> {cat}
              </div>
            ))}
          </div>
        </section>

        <section className="card">
          <h2 className="text-xl font-semibold mb-4">Architecture</h2>
          <p className="text-slate-300 mb-4">
            Built on CaVe-VLM-CoT with extensions for step-level verification and evidence attribution:
          </p>
          <ul className="space-y-2 text-slate-400">
            <li>• <strong className="text-slate-200">CoT Generator</strong> — Base VLM generates step-by-step reasoning</li>
            <li>• <strong className="text-slate-200">Evidence Retriever</strong> — Visual region + textual evidence retrieval</li>
            <li>• <strong className="text-slate-200">Hallucination Detector</strong> — Cross-encoder step verification</li>
            <li>• <strong className="text-slate-200">Evidence Attributor</strong> — Citation injection and grounding</li>
            <li>• <strong className="text-slate-200">Reasoning Corrector</strong> — Remove/fix hallucinated steps</li>
          </ul>
        </section>

        <section className="card">
          <h2 className="text-xl font-semibold mb-4">Models & Hardware</h2>
          <p className="text-slate-300 leading-relaxed">
            Optimized for RTX 4060 (8GB VRAM) with 4-bit quantization via bitsandbytes.
            Default model: Qwen2-VL-7B-Instruct. Also supports LLaVA-7B and Phi-3 Vision.
            CPU fallback available when GPU is unavailable.
          </p>
        </section>

        <section className="card">
          <h2 className="text-xl font-semibold mb-4">Team</h2>
          <p className="text-slate-300">Team 7 — HERA-VLM Research Project</p>
          <p className="text-slate-500 text-sm mt-2">Based on CaVe-VLM-CoT paper implementation</p>
        </section>
      </div>
    </div>
  );
}
