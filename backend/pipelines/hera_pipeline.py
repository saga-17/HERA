"""HERA-VLM end-to-end inference pipeline."""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from backend.api.schemas import (
    HeraResult,
    PipelineStage,
    PipelineStatus,
)
from backend.config import settings
from backend.services.cot_generator import cot_generator
from backend.services.evidence_attributor import evidence_attributor
from backend.services.evidence_retriever import evidence_retriever
from backend.services.hallucination_detector import hallucination_detector
from backend.services.reasoning_corrector import reasoning_corrector
from backend.utils.image_utils import load_image
from backend.utils.step_parser import parse_cot_sections, segment_reasoning_steps

logger = logging.getLogger(__name__)


class HeraPipeline:
    """
    Full HERA-VLM pipeline:
    Image + Question → CoT → Step Segmentation → Evidence Retrieval →
    Verification → Hallucination Detection → Attribution → Correction → Answer
    """

    def __init__(self) -> None:
        self._results: dict[str, HeraResult] = {}
        self._status_callbacks: dict[str, Callable] = {}

    def get_result(self, result_id: str) -> Optional[HeraResult]:
        if result_id in self._results:
            return self._results[result_id]

        result_path = settings.results_dir / f"{result_id}.json"
        if result_path.exists():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            return HeraResult(**data)

        return None

    def _save_result(self, result: HeraResult) -> None:
        self._results[result.result_id] = result
        result_path = settings.results_dir / f"{result.result_id}.json"
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    def _update_status(self, result_id: str, stage: PipelineStage, progress: float, message: str) -> None:
        """
        Update pipeline status AND persist it immediately.

        Previously this only mutated the in-memory `_results` dict without
        saving to disk, so anything reading results from disk (a different
        worker process, a restarted server, etc.) would not see intermediate
        stage progress until the very end of the run — the frontend would
        appear to "jump straight to results". Saving on every update fixes
        stage-by-stage progress reporting end to end.
        """
        if result_id in self._results:
            self._results[result_id].pipeline_status = PipelineStatus(
                stage=stage, progress=progress, message=message
            )
            self._save_result(self._results[result_id])
            logger.info("[%s] STAGE -> %s (%.0f%%): %s", result_id, stage.value, progress, message)

    def run(
        self,
        image_path: str,
        image_id: str,
        question: str,
        result_id: Optional[str] = None,
    ) -> HeraResult:
        result_id = result_id or str(uuid.uuid4())
        pipeline_start = time.time()
        image = load_image(image_path)

        result = HeraResult(
            result_id=result_id,
            image_id=image_id,
            image_url=f"/api/images/{image_id}",
            question=question,
            original_cot="",
            attributed_cot="",
            corrected_cot="",
            final_answer="",
            hallucination_score=0.0,
            confidence_score=0.0,
            pipeline_status=PipelineStatus(stage=PipelineStage.COT_GENERATION, progress=10, message="Generating CoT..."),
        )
        self._results[result_id] = result
        self._save_result(result)

        try:
            # Stage 1: CoT Generation
            stage_start = time.time()
            logger.info("[%s] START CoTGeneration", result_id)
            self._update_status(result_id, PipelineStage.COT_GENERATION, 15, "Generating Chain-of-Thought...")
            original_cot = cot_generator.generate(image, question)
            result.original_cot = original_cot
            self._save_result(result)
            logger.info(
                "[%s] END CoTGeneration | %.2fs | chars=%d",
                result_id, time.time() - stage_start, len(original_cot),
            )

            # Parse structured sections once. The <OBSERVATIONS> text is the
            # VLM's real, grounded description of the image ("holding bouquet
            # of flowers", etc). Previously this was only extracted as a
            # fallback when step segmentation failed, so the retriever never
            # got to use it as evidence. Now it's always extracted and threaded
            # into every step's evidence retrieval below.
            sections = parse_cot_sections(original_cot)
            observations = sections.get("observations", "")
            logger.info(
                "[%s] Parsed CoT sections: observations=%d chars, reasoning=%s, conclusion=%s",
                result_id, len(observations), "observations" in sections, "conclusion" in sections,
            )

            # Stage 2: Step Segmentation
            stage_start = time.time()
            logger.info("[%s] START StepSegmentation", result_id)
            self._update_status(result_id, PipelineStage.STEP_SEGMENTATION, 30, "Segmenting reasoning steps...")
            reasoning_text = sections.get("reasoning", original_cot)
            step_texts = segment_reasoning_steps(reasoning_text)
            if not step_texts:
                step_texts = segment_reasoning_steps(original_cot) or [original_cot[:500]]
            logger.info(
                "[%s] END StepSegmentation | %.2fs | reasoning_count=%d",
                result_id, time.time() - stage_start, len(step_texts),
            )

            # Stage 3-5: Per-step retrieval, verification, attribution
            logger.info("[%s] START EvidenceRetrieval+Verification (%d steps)", result_id, len(step_texts))
            loop_start = time.time()
            verified_steps = []
            for i, step_text in enumerate(step_texts):
                progress = 30 + (50 * (i + 1) / len(step_texts))
                self._update_status(
                    result_id,
                    PipelineStage.EVIDENCE_RETRIEVAL,
                    progress,
                    f"Processing step {i + 1}/{len(step_texts)}...",
                )

                visual_ev, textual_ev = evidence_retriever.retrieve_for_step(
                    step_text, image, question, i, observations=observations
                )

                self._update_status(
                    result_id,
                    PipelineStage.STEP_VERIFICATION,
                    progress + 5,
                    f"Verifying step {i + 1}/{len(step_texts)}...",
                )

                step_result = hallucination_detector.verify_step(
                    i, step_text, visual_ev, textual_ev
                )
                verified_steps.append(step_result)

            result.steps = verified_steps
            self._save_result(result)
            logger.info(
                "[%s] END EvidenceRetrieval+Verification | %.2fs | steps=%d",
                result_id, time.time() - loop_start, len(verified_steps),
            )

            # Stage 6: Hallucination Detection (aggregate)
            stage_start = time.time()
            logger.info("[%s] START HallucinationDetection", result_id)
            self._update_status(
                result_id,
                PipelineStage.HALLUCINATION_DETECTION,
                85,
                "Computing hallucination scores...",
            )
            hallucinated = sum(1 for s in verified_steps if not s.supported)
            supported_count = len(verified_steps) - hallucinated
            result.hallucination_score = round(hallucinated / max(len(verified_steps), 1), 3)
            result.confidence_score = round(
                sum(s.confidence for s in verified_steps) / max(len(verified_steps), 1),
                3,
            )
            self._save_result(result)
            logger.info(
                "[%s] END HallucinationDetection | %.2fs | supported=%d | hallucinated=%d | "
                "hallucination_score=%.3f | confidence_score=%.3f",
                result_id, time.time() - stage_start, supported_count, hallucinated,
                result.hallucination_score, result.confidence_score,
            )

            # Stage 7: Evidence Attribution
            stage_start = time.time()
            logger.info("[%s] START EvidenceAttribution", result_id)
            self._update_status(
                result_id,
                PipelineStage.EVIDENCE_ATTRIBUTION,
                90,
                "Building evidence-attributed CoT...",
            )
            result.attributed_cot = evidence_attributor.attribute_steps(verified_steps)
            self._save_result(result)
            logger.info(
                "[%s] END EvidenceAttribution | %.2fs | chars=%d",
                result_id, time.time() - stage_start, len(result.attributed_cot),
            )

            # Stage 8: Reasoning Correction
            stage_start = time.time()
            logger.info("[%s] START ReasoningCorrection", result_id)
            self._update_status(
                result_id,
                PipelineStage.REASONING_CORRECTION,
                95,
                "Generating corrected reasoning...",
            )
            corrected_cot, final_answer = reasoning_corrector.correct(
                original_cot, verified_steps, question
            )
            result.corrected_cot = corrected_cot
            result.final_answer = final_answer
            logger.info(
                "[%s] END ReasoningCorrection | %.2fs | final_answer=%r",
                result_id, time.time() - stage_start, final_answer[:120],
            )

            result.pipeline_status = PipelineStatus(
                stage=PipelineStage.COMPLETE,
                progress=100,
                message="Pipeline complete",
            )
            self._save_result(result)
            logger.info(
                "[%s] Pipeline complete | total=%.2fs | steps=%d | supported=%d | hallucinated=%d",
                result_id, time.time() - pipeline_start, len(verified_steps), supported_count, hallucinated,
            )

        except Exception as e:
            logger.exception("[%s] Pipeline failed: %s", result_id, e)
            result.pipeline_status = PipelineStatus(
                stage=PipelineStage.ERROR,
                progress=0,
                message=str(e),
            )
            self._save_result(result)

        return result


hera_pipeline = HeraPipeline()