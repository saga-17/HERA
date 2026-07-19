"""Evidence attribution for reasoning steps."""

from __future__ import annotations

import re
from typing import Optional

from backend.api.schemas import ReasoningStepResult, StepStatus


class EvidenceAttributor:
    """
    Ground each reasoning step with retrieved evidence citations.
    Adapted from CaVe-VLM-CoT citation_injector.
    """

    def attribute_steps(self, steps: list[ReasoningStepResult]) -> str:
        """Build evidence-attributed Chain-of-Thought from verified steps."""
        attributed_lines = []

        for step in steps:
            prefix = self._status_prefix(step.status)
            citations = self._format_citations(step)
            line = f"{prefix} Step {step.step_index + 1}: {step.step}"
            if citations:
                line += f" {citations}"
            line += f" [Confidence: {step.confidence:.2f}]"
            attributed_lines.append(line)

        return "\n".join(attributed_lines)

    def _status_prefix(self, status: StepStatus) -> str:
        if status == StepStatus.SUPPORTED:
            return "✓"
        if status == StepStatus.HALLUCINATED:
            return "✗"
        return "~"

    def _format_citations(self, step: ReasoningStepResult) -> str:
        citations = []
        for i, ve in enumerate(step.visual_evidence[:1]):
            citations.append(f"[Visual Evidence {i + 1}]")
        for i, te in enumerate(step.textual_evidence[:1]):
            citations.append(f"[Text Evidence {i + 1}]")
        return " ".join(citations)

    def inject_citations_into_step(self, step: ReasoningStepResult) -> str:
        """Add inline citations to a single step (post-hoc injection pattern)."""
        text = step.step
        if step.visual_evidence and "[Visual" not in text:
            text += f" [Visual Evidence: {step.visual_evidence[0].caption}]"
        if step.textual_evidence and "[Text" not in text:
            text += f" [Text Evidence: {step.textual_evidence[0].text[:80]}...]"
        return text


evidence_attributor = EvidenceAttributor()
