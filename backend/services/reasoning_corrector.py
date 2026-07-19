"""Reasoning correction for hallucinated steps."""

from __future__ import annotations

import re
from typing import Optional

from backend.api.schemas import ReasoningStepResult, StepStatus
from backend.config import settings
from backend.utils.step_parser import parse_cot_sections


class ReasoningCorrector:
    """
    Generate corrected reasoning by removing or fixing hallucinated steps.
    Adapted from CaVe-VLM-CoT verifier feedback loop.
    """

    def correct(
        self,
        original_cot: str,
        steps: list[ReasoningStepResult],
        question: str,
    ) -> tuple[str, str]:
        """
        Produce corrected CoT and final grounded answer.
        Returns (corrected_cot, final_answer).
        """
        supported_steps = [s for s in steps if s.status == StepStatus.SUPPORTED]
        hallucinated_count = sum(1 for s in steps if s.status == StepStatus.HALLUCINATED)

        corrected_lines = ["<CORRECTED_REASONING>"]
        for step in steps:
            if step.status == StepStatus.HALLUCINATED:
                corrected_lines.append(
                    f"Step {step.step_index + 1} [REMOVED — {step.hallucination_type.value}]: "
                    f"Original claim '{step.step[:80]}...' was unsupported."
                )
            elif step.status == StepStatus.UNCERTAIN:
                corrected_lines.append(
                    f"Step {step.step_index + 1} [UNCERTAIN]: {step.step} "
                    f"(low confidence: {step.confidence:.2f})"
                )
            else:
                corrected_lines.append(f"Step {step.step_index + 1}: {step.step}")

        corrected_lines.append("</CORRECTED_REASONING>")

        final_answer = self._generate_final_answer(
            original_cot, supported_steps, question, hallucinated_count
        )
        corrected_lines.append(f"\nFinal Answer: {final_answer}")

        return "\n".join(corrected_lines), final_answer

    def _generate_final_answer(
        self,
        original_cot: str,
        supported_steps: list[ReasoningStepResult],
        question: str,
        hallucinated_count: int,
    ) -> str:
        sections = parse_cot_sections(original_cot)
        original_answer = sections.get("final_answer", "")

        if not supported_steps:
            return "Unable to provide a grounded answer — all reasoning steps were unsupported."

        if hallucinated_count == 0 and original_answer:
            return original_answer

        # Synthesize answer from supported steps only
        key_observations = [s.step for s in supported_steps[:3]]
        synthesis = " ".join(key_observations)

        # Try to extract a concise answer
        if original_answer and hallucinated_count <= 1:
            cleaned = re.sub(
                r"(?:wearing|has|with)\s+\w+",
                "",
                original_answer,
                flags=re.IGNORECASE,
            ).strip()
            if cleaned:
                return cleaned

        if len(synthesis) > 200:
            synthesis = synthesis[:200] + "..."

        return f"Based on verified evidence: {synthesis}"


reasoning_corrector = ReasoningCorrector()
