"""Step-level verification and hallucination detection."""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

from backend.api.schemas import (
    HallucinationType,
    ReasoningStepResult,
    StepStatus,
    TextEvidence,
    VisualEvidence,
)
from backend.config import settings
from backend.models.model_manager import model_manager
from backend.utils.step_parser import classify_hallucination_type, extract_entities

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Verify each reasoning step against retrieved evidence.
    Adapted from CaVe-VLM-CoT verifier and citation_injector logic.
    """

    def verify_step(
        self,
        step_index: int,
        step_text: str,
        visual_evidence: list[VisualEvidence],
        textual_evidence: list[TextEvidence],
    ) -> ReasoningStepResult:
        extraction = extract_entities(step_text)

        if settings.demo_mode:
            return self._demo_verify(step_index, step_text, visual_evidence, textual_evidence, extraction)

        confidence, supported, issue = self._score_step(step_text, visual_evidence, textual_evidence)

        if confidence >= settings.supported_threshold:
            status = StepStatus.SUPPORTED
            hallucination_type = HallucinationType.NONE
        elif confidence <= settings.hallucinated_threshold:
            status = StepStatus.HALLUCINATED
            hallucination_type = HallucinationType(classify_hallucination_type(step_text, issue))
        else:
            status = StepStatus.UNCERTAIN
            hallucination_type = HallucinationType.REASONING

        evidence_summary = self._build_evidence_summary(visual_evidence, textual_evidence)

        return ReasoningStepResult(
            step_index=step_index,
            step=step_text,
            status=status,
            confidence=round(confidence, 3),
            supported=supported,
            hallucination_type=hallucination_type,
            evidence=evidence_summary,
            visual_evidence=visual_evidence,
            textual_evidence=textual_evidence,
            attribution=self._build_attribution(status, visual_evidence, textual_evidence, issue),
            extraction=extraction,
        )

    def _score_step(
        self,
        step_text: str,
        visual_evidence: list[VisualEvidence],
        textual_evidence: list[TextEvidence],
    ) -> tuple[float, bool, str]:
        """Cross-encoder scoring of step against all evidence."""
        encoder = model_manager.get_cross_encoder()
        scores: list[float] = []
        issue = ""

        for ve in visual_evidence:
            try:
                score = float(encoder.predict([(step_text, ve.caption)])[0])
                scores.append(score)
            except Exception:
                scores.append(ve.confidence)

        for te in textual_evidence:
            try:
                score = float(encoder.predict([(step_text, te.text[:300])])[0])
                scores.append(score)
            except Exception:
                scores.append(te.confidence)

        if not scores:
            return 0.4, False, "no evidence retrieved"

        # Normalize cross-encoder scores (roughly -10 to +10) to 0-1
        raw_max = max(scores)
        confidence = self._normalize_score(raw_max)
        supported = confidence >= settings.supported_threshold

        if not supported:
            issue = "claim not supported by retrieved visual or textual evidence"

        return confidence, supported, issue

    def _normalize_score(self, raw: float) -> float:
        """Map cross-encoder score to 0-1 confidence."""
        # Sigmoid-like normalization for ms-marco scores
        return float(1.0 / (1.0 + np.exp(-raw)))

    def _demo_verify(
        self,
        step_index: int,
        step_text: str,
        visual_evidence: list[VisualEvidence],
        textual_evidence: list[TextEvidence],
        extraction,
    ) -> ReasoningStepResult:
        """Deterministic demo verification for development without GPU."""
        text_lower = step_text.lower()

        # Simulate hallucination on specific keywords for demo
        hallucination_keywords = ["glasses", "hat", "flying", "swimming", "red car", "two dogs"]
        is_hallucinated = any(kw in text_lower for kw in hallucination_keywords)

        if is_hallucinated:
            status = StepStatus.HALLUCINATED
            confidence = 0.18
            supported = False
            h_type = HallucinationType.ATTRIBUTE
            issue = "attribute not visible in image evidence"
        elif step_index == 0:
            status = StepStatus.SUPPORTED
            confidence = 0.94
            supported = True
            h_type = HallucinationType.NONE
            issue = ""
        else:
            status = StepStatus.SUPPORTED
            confidence = 0.78
            supported = True
            h_type = HallucinationType.NONE
            issue = ""

        return ReasoningStepResult(
            step_index=step_index,
            step=step_text,
            status=status,
            confidence=confidence,
            supported=supported,
            hallucination_type=h_type,
            evidence=self._build_evidence_summary(visual_evidence, textual_evidence),
            visual_evidence=visual_evidence,
            textual_evidence=textual_evidence,
            attribution=self._build_attribution(status, visual_evidence, textual_evidence, issue),
            extraction=extraction,
        )

    def _build_evidence_summary(
        self,
        visual: list[VisualEvidence],
        textual: list[TextEvidence],
    ) -> str:
        parts = []
        for ve in visual[:2]:
            parts.append(f"[Visual] {ve.caption} (conf={ve.confidence})")
        for te in textual[:2]:
            parts.append(f"[Text] {te.text[:150]}...")
        return " | ".join(parts) if parts else "No evidence retrieved"

    def _build_attribution(
        self,
        status: StepStatus,
        visual: list[VisualEvidence],
        textual: list[TextEvidence],
        issue: str,
    ) -> str:
        if status == StepStatus.SUPPORTED:
            sources = []
            if visual:
                sources.append(f"visual region {visual[0].region_id}")
            if textual:
                sources.append(f"textual source ({textual[0].source})")
            return f"Supported by {' and '.join(sources)}."
        if status == StepStatus.HALLUCINATED:
            return f"Hallucinated: {issue}. No supporting evidence found in image or retrieved text."
        return f"Uncertain: partial evidence available. {issue}"


hallucination_detector = HallucinationDetector()
