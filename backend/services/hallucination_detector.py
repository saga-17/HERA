"""Step-level verification and hallucination detection."""

from __future__ import annotations

import logging
import re
import time
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

    A claim is accepted if it is supported by VISUAL evidence OR TEXTUAL
    evidence (not textual evidence alone) — the best single score across both
    evidence pools decides support, and an average of the top matches is used
    so one weak caption can't zero out an otherwise well-supported claim.
    """

    def verify_step(
        self,
        step_index: int,
        step_text: str,
        visual_evidence: list[VisualEvidence],
        textual_evidence: list[TextEvidence],
    ) -> ReasoningStepResult:
        start = time.time()
        logger.info(
            "START StepVerification step=%d | visual_evidence=%d | textual_evidence=%d",
            step_index, len(visual_evidence), len(textual_evidence),
        )

        extraction = extract_entities(step_text)

        if settings.demo_mode:
            result = self._demo_verify(step_index, step_text, visual_evidence, textual_evidence, extraction)
        else:
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

            result = ReasoningStepResult(
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

        logger.info(
            "END StepVerification step=%d | %.3fs | status=%s | confidence=%.3f | type=%s",
            step_index, time.time() - start, result.status, result.confidence, result.hallucination_type,
        )
        return result

    def _score_step(
        self,
        step_text: str,
        visual_evidence: list[VisualEvidence],
        textual_evidence: list[TextEvidence],
    ) -> tuple[float, bool, str]:
        """
        Cross-encoder scoring of step against all evidence (visual captions AND
        textual snippets are pooled together — the claim only needs support from
        ONE of the two evidence types, not both).
        """
        encoder = model_manager.get_cross_encoder()
        visual_scores: list[float] = []
        textual_scores: list[float] = []

        for ve in visual_evidence:
            try:
                score = float(encoder.predict([(step_text, ve.caption)])[0])
                visual_scores.append(score)
            except Exception:
                # Fall back to the retriever's own confidence, mapped back to
                # the encoder's raw scale so it composes correctly below.
                visual_scores.append((ve.confidence * 20) - 10)

        for te in textual_evidence:
            try:
                score = float(encoder.predict([(step_text, te.text[:300])])[0])
                textual_scores.append(score)
            except Exception:
                textual_scores.append((te.confidence * 20) - 10)

        all_scores = visual_scores + textual_scores
        if not all_scores:
            return 0.4, False, "no evidence retrieved"

        # Use the average of the top-2 scores (across BOTH evidence pools)
        # rather than a single max. This is more robust than one lucky/unlucky
        # match while still letting either evidence type carry the claim.
        top = sorted(all_scores, reverse=True)[:2]
        raw_score = sum(top) / len(top)
        confidence = self._normalize_score(raw_score)
        supported = confidence >= settings.supported_threshold

        issue = ""
        if not supported:
            issue = "claim not supported by retrieved visual or textual evidence"

        return confidence, supported, issue

    def _normalize_score(self, raw: float) -> float:
        """
        Map cross-encoder relevance score to a 0-1 confidence.

        ms-marco cross-encoders are trained for query/passage RANKING and
        typically output raw logits roughly in [-11, 11], where scores well
        above 0 indicate strong relevance and scores near/below 0 indicate
        weak or no relevance. A bare sigmoid centered at 0 was previously used,
        which is directionally correct but was starving out real matches
        because evidence text was meaningless (see evidence_retriever fix).
        With meaningful evidence text this normalization now produces a
        sensible spread; we also clip to avoid float edge cases.
        """
        return float(np.clip(1.0 / (1.0 + np.exp(-raw)), 0.0, 1.0))

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