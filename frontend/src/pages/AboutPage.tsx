export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <h1 className="text-4xl font-bold mb-8">About HERA-VLM</h1>

      <div className="space-y-8">
        <section className="card">
          <h2 className="text-xl font-semibold mb-4">What HERA-VLM is</h2>
          <p className="text-slate-300 leading-relaxed">
            HERA-VLM (Hallucination Evidence Retrieval and Attribution for Vision Language Models)
            is a research-oriented reasoning pipeline for reducing unsupported claims in multimodal
            model responses. It analyzes visual input, decomposes reasoning into verifiable steps,
            retrieves supporting evidence, and checks whether each conclusion is grounded in what is
            actually visible.
          </p>
        </section>

        <section className="card">
          <h2 className="text-xl font-semibold mb-4">The problem it solves</h2>
          <p className="text-slate-300 leading-relaxed">
            Vision-language models can produce fluent but ungrounded explanations. In high-stakes
            tasks, this creates a risk of confident but incorrect reasoning. HERA-VLM addresses that
            by turning reasoning into evidence-aware steps that can be verified before a final answer
            is accepted.
          </p>
        </section>

        <section className="card">
          <h2 className="text-xl font-semibold mb-4">Hallucination reduction</h2>
          <div className="grid sm:grid-cols-2 gap-3">
            {[
              "Object hallucination detection",
              "Attribute verification",
              "Relationship checking",
              "Scene and context validation",
              "Reasoning quality review",
              "Evidence-grounded correction",
            ].map((item) => (
              <div key={item} className="flex items-center gap-2 text-slate-300">
                <span className="text-red-400">●</span> {item}
              </div>
            ))}
          </div>
        </section>

        <section className="card">
          <h2 className="text-xl font-semibold mb-4">Evidence-grounded reasoning</h2>
          <p className="text-slate-300 mb-4">
            The pipeline generates a chain-of-thought, retrieves relevant visual and textual evidence,
            verifies each reasoning step, and attributes the result back to the supporting sources.
            This allows the system to distinguish between confident observations and unsupported
            inferences.
          </p>
          <ul className="space-y-2 text-slate-400">
            <li>• <strong className="text-slate-200">CoT Generator</strong> — Produces step-by-step reasoning from the input image and question</li>
            <li>• <strong className="text-slate-200">Evidence Retriever</strong> — Gathers supporting visual and textual evidence for each step</li>
            <li>• <strong className="text-slate-200">Hallucination Detector</strong> — Flags unsupported or weakly grounded claims</li>
            <li>• <strong className="text-slate-200">Evidence Attributor</strong> — Connects reasoning back to the evidence it depends on</li>
            <li>• <strong className="text-slate-200">Reasoning Corrector</strong> — Produces a cleaner, grounded conclusion</li>
          </ul>
        </section>

        <section className="card">
          <h2 className="text-xl font-semibold mb-4">Project purpose</h2>
          <p className="text-slate-300 leading-relaxed">
            HERA-VLM is intended to make multimodal AI reasoning more transparent, verifiable, and
            trustworthy. By grounding model outputs in evidence and explicitly identifying where
            reasoning is weak or unsupported, it supports more reliable interpretation of visual
            content and better downstream decision-making.
          </p>
        </section>
      </div>
    </div>
  );
}
