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
        original_answer = sections.get("final_answer", "").strip()

        total_steps = len(supported_steps) + hallucinated_count
        verification = self._determine_verification(supported_steps, hallucinated_count, total_steps)
        confidence = self._calculate_confidence(supported_steps)

        if not supported_steps:
            return (
                "Answer:\nThe available visual evidence is insufficient to determine this.\n\n"
                f"Verification:\n{verification}\n\n"
                f"Confidence:\n{confidence}%\n\n"
                "Evidence:\nNo supported visual evidence was found for this claim."
            )

        if hallucinated_count == 0 and original_answer:
            answer_text = self._clean_answer_text(original_answer)
            return self._format_answer_block(answer_text, verification, confidence, supported_steps)

        synthesized = self._synthesize_answer_from_steps(supported_steps)
        if not synthesized:
            answer_text = self._clean_answer_text(original_answer) if original_answer else "The available visual evidence supports the described scene."
        else:
            answer_text = synthesized

        return self._format_answer_block(answer_text, verification, confidence, supported_steps)

    def _clean_answer_text(self, answer: str) -> str:
        cleaned = answer.strip()
        cleaned = re.sub(r"^\s*(?:Final Answer|Answer|Conclusion)\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned or cleaned.endswith("..."):
            return "The available visual evidence supports the described scene."
        return cleaned

    def _synthesize_answer_from_steps(self, supported_steps: list[ReasoningStepResult]) -> str:
        parts: list[str] = []
        for step in supported_steps[:3]:
            cleaned = re.sub(r"\s+", " ", step.step).strip()
            if cleaned:
                parts.append(cleaned)

        if not parts:
            return ""

        synthesis = " ".join(parts)
        synthesis = synthesis.rstrip(". ") + "."

        if len(synthesis.split()) <= 25:
            return synthesis

        return synthesis

    def _determine_verification(
        self,
        supported_steps: list[ReasoningStepResult],
        hallucinated_count: int,
        total_steps: int,
    ) -> str:
        if not supported_steps:
            return "UNSUPPORTED" if hallucinated_count > 0 else "INCONCLUSIVE"
        if total_steps and len(supported_steps) == total_steps:
            return "SUPPORTED"
        if hallucinated_count > 0:
            return "SUPPORTED" if supported_steps else "UNSUPPORTED"
        return "SUPPORTED"

    def _calculate_confidence(self, supported_steps: list[ReasoningStepResult]) -> int:
        if not supported_steps:
            return 0
        average = sum(step.confidence for step in supported_steps) / len(supported_steps)
        return max(0, min(99, int(round(average * 100))))

    def _format_answer_block(
        self,
        answer_text: str,
        verification: str,
        confidence: int,
        supported_steps: list[ReasoningStepResult],
    ) -> str:
        evidence_text = "; ".join(
            step.evidence.strip()
            for step in supported_steps[:3]
            if step.evidence and step.evidence.strip()
        )
        if not evidence_text:
            evidence_text = "The visual evidence in the image supports the described conclusion."

        return (
            "Answer:\n"
            f"{answer_text}\n\n"
            "Verification:\n"
            f"{verification}\n\n"
            "Confidence:\n"
            f"{confidence}%\n\n"
            "Evidence:\n"
            f"{evidence_text}"
        )


reasoning_corrector = ReasoningCorrector()
