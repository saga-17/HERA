"""HERA-VLM end-to-end inference pipeline."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from backend.api.schemas import (
    HeraResult,
    PipelineStage,
    PipelineStageInfo,
    PipelineStatus,
    StageTiming,
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

    Image + Question
        → CoT Generation
        → Step Segmentation
        → Evidence Retrieval
        → Step Verification
        → Hallucination Detection
        → Evidence Attribution
        → Reasoning Correction
        → Complete
    """

    # These are the actual executable stages in the current backend.
    #
    # The frontend receives this definition from PipelineStatus, so it does
    # not need to duplicate the pipeline definition.
    PIPELINE_STAGES: tuple[PipelineStageInfo, ...] = (
        PipelineStageInfo(
            key=PipelineStage.COT_GENERATION.value,
            label="CoT Generation",
            order=1,
        ),
        PipelineStageInfo(
            key=PipelineStage.STEP_SEGMENTATION.value,
            label="Step Segmentation",
            order=2,
        ),
        PipelineStageInfo(
            key=PipelineStage.EVIDENCE_RETRIEVAL.value,
            label="Evidence Retrieval",
            order=3,
        ),
        PipelineStageInfo(
            key=PipelineStage.STEP_VERIFICATION.value,
            label="Step Verification",
            order=4,
        ),
        PipelineStageInfo(
            key=PipelineStage.HALLUCINATION_DETECTION.value,
            label="Hallucination Detection",
            order=5,
        ),
        PipelineStageInfo(
            key=PipelineStage.EVIDENCE_ATTRIBUTION.value,
            label="Evidence Attribution",
            order=6,
        ),
        PipelineStageInfo(
            key=PipelineStage.REASONING_CORRECTION.value,
            label="Reasoning Correction",
            order=7,
        ),
        PipelineStageInfo(
            key=PipelineStage.COMPLETE.value,
            label="Complete",
            order=8,
        ),
    )

    # Approximate stage weights in seconds.
    #
    # These are NOT displayed as fake progress. They are only used to
    # estimate ETA when no timing history exists yet.
    #
    # Once actual timings are available, observed timings take precedence.
    DEFAULT_STAGE_ESTIMATES: dict[PipelineStage, float] = {
        PipelineStage.COT_GENERATION: 20.0,
        PipelineStage.STEP_SEGMENTATION: 2.0,
        PipelineStage.EVIDENCE_RETRIEVAL: 30.0,
        PipelineStage.STEP_VERIFICATION: 15.0,
        PipelineStage.HALLUCINATION_DETECTION: 3.0,
        PipelineStage.EVIDENCE_ATTRIBUTION: 5.0,
        PipelineStage.REASONING_CORRECTION: 15.0,
    }

    def __init__(self) -> None:
        self._results: dict[str, HeraResult] = {}
        self._status_callbacks: dict[str, Callable] = {}
        self._cancelled: dict[str, threading.Event] = {}
        self._active_requests: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def _request_key(self, image_id: str, question: str) -> tuple[str, str]:
        return (image_id, (question or "").strip())

    def register_result_for_request(
        self,
        image_id: str,
        question: str,
        result_id: str,
    ) -> str | None:
        key = self._request_key(image_id, question)
        with self._lock:
            active = self._active_requests.get(key)
            if active is not None:
                return active
            self._active_requests[key] = result_id
            return None

    def get_active_result_for_request(
        self,
        image_id: str,
        question: str,
    ) -> Optional[str]:
        active = self._active_requests.get(self._request_key(image_id, question))
        if active is not None:
            return active
        return None

    def has_active_run_for_request(
        self,
        image_id: str,
        question: str,
    ) -> bool:
        return self.get_active_result_for_request(image_id, question) is not None

    def release_active_request(
        self,
        image_id: str,
        question: str,
        result_id: Optional[str] = None,
    ) -> None:
        key = self._request_key(image_id, question)
        with self._lock:
            active = self._active_requests.get(key)
            if active is None:
                return
            if result_id is not None and active != result_id:
                return
            self._active_requests.pop(key, None)

    def is_cancelled(self, result_id: str) -> bool:
        event = self._cancelled.get(result_id)
        return bool(event is not None and event.is_set())

    def _mark_cancelled(
        self,
        result_id: str,
        message: str = "Pipeline cancelled by user.",
    ) -> Optional[HeraResult]:
        result = self._results.get(result_id)
        if result is None:
            return None

        started_at = result.pipeline_status.started_at or time.time()
        current_stage = result.pipeline_status.stage
        elapsed = round(max(0.0, time.time() - started_at), 3)

        result.pipeline_status = PipelineStatus(
            stage=PipelineStage.CANCELLED,
            progress=max(0.0, min(100.0, float(result.pipeline_status.progress))),
            message=message,
            stages=list(self.PIPELINE_STAGES),
            stage_index=self._stage_index(current_stage)
            if current_stage in {stage.value for stage in PipelineStage if stage != PipelineStage.CANCELLED}
            else len(self.PIPELINE_STAGES),
            total_stages=len(self.PIPELINE_STAGES),
            completed_stages=sum(
                1
                for timing in result.pipeline_status.stage_timings
                if timing.duration_seconds is not None
            ),
            started_at=started_at,
            elapsed_seconds=elapsed,
            current_stage_elapsed_seconds=0.0,
            estimated_remaining_seconds=0.0,
            eta_confidence="unavailable",
            stage_timings=result.pipeline_status.stage_timings,
        )

        self._save_result(result)
        return result

    def _check_cancelled(
        self,
        result_id: str,
        *,
        message: str = "Pipeline cancelled by user.",
    ) -> bool:
        if self.is_cancelled(result_id):
            logger.warning("[%s] Cancellation requested: %s", result_id, message)
            self._mark_cancelled(result_id, message)
            return True
        return False

    def cancel(self, result_id: str) -> Optional[HeraResult]:
        event = self._cancelled.get(result_id)
        if event is not None:
            event.set()

        result = self._results.get(result_id)
        if result is None:
            return None

        self._mark_cancelled(result_id, "Pipeline cancelled by user.")

        image_id = result.image_id
        question = result.question
        self.release_active_request(image_id, question, result_id)
        return self._results.get(result_id)

    # ------------------------------------------------------------------
    # Basic result persistence
    # ------------------------------------------------------------------

    def get_result(self, result_id: str) -> Optional[HeraResult]:
        if result_id in self._results:
            return self._results[result_id]

        result_path = settings.results_dir / f"{result_id}.json"

        if result_path.exists():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            result = HeraResult(**data)
            self._results[result_id] = result
            return result

        return None

    def _save_result(self, result: HeraResult) -> None:
        self._results[result.result_id] = result

        result_path = settings.results_dir / f"{result.result_id}.json"
        result_path.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Pipeline metadata
    # ------------------------------------------------------------------

    def _stage_index(self, stage: PipelineStage) -> int:
        """
        Return the 1-based stage number for an actual pipeline stage.

        COMPLETE is stage 8 of 8.
        ERROR is not treated as a normal pipeline stage.
        """
        for stage_info in self.PIPELINE_STAGES:
            if stage_info.key == stage.value:
                return stage_info.order

        return 0

    def _stage_label(self, stage: PipelineStage) -> str:
        for stage_info in self.PIPELINE_STAGES:
            if stage_info.key == stage.value:
                return stage_info.label

        return stage.value.replace("_", " ").title()

    def _default_status(
        self,
        stage: PipelineStage,
        *,
        progress: float = 0.0,
        message: str = "",
        started_at: Optional[float] = None,
    ) -> PipelineStatus:
        return PipelineStatus(
            stage=stage,
            progress=progress,
            message=message,
            stages=list(self.PIPELINE_STAGES),
            stage_index=self._stage_index(stage),
            total_stages=len(self.PIPELINE_STAGES),
            completed_stages=0,
            started_at=started_at,
            elapsed_seconds=0.0,
            current_stage_elapsed_seconds=0.0,
            estimated_remaining_seconds=None,
            eta_confidence="calculating",
            stage_timings=[],
        )

    # ------------------------------------------------------------------
    # Timing helpers
    # ------------------------------------------------------------------

    def _find_stage_timing(
        self,
        status: PipelineStatus,
        stage: PipelineStage,
    ) -> Optional[StageTiming]:
        for timing in status.stage_timings:
            if timing.stage == stage:
                return timing

        return None

    def _start_stage_timing(
        self,
        status: PipelineStatus,
        stage: PipelineStage,
        now: float,
    ) -> None:
        existing = self._find_stage_timing(status, stage)

        if existing is not None:
            existing.started_at = existing.started_at or now
            existing.completed_at = None
            existing.duration_seconds = None
            return

        status.stage_timings.append(
            StageTiming(
                stage=stage,
                label=self._stage_label(stage),
                started_at=now,
                completed_at=None,
                duration_seconds=None,
            )
        )

    def _finish_stage_timing(
        self,
        status: PipelineStatus,
        stage: PipelineStage,
        now: float,
    ) -> float:
        timing = self._find_stage_timing(status, stage)

        if timing is None:
            return 0.0

        if timing.started_at is None:
            timing.started_at = now

        duration = max(0.0, now - timing.started_at)

        timing.completed_at = now
        timing.duration_seconds = round(duration, 3)

        return duration

    def _completed_stage_timings(
        self,
        status: PipelineStatus,
    ) -> list[StageTiming]:
        return [
            timing
            for timing in status.stage_timings
            if timing.duration_seconds is not None
        ]

    # ------------------------------------------------------------------
    # ETA calculation
    # ------------------------------------------------------------------

    def _estimate_stage_duration(
        self,
        status: PipelineStatus,
        stage: PipelineStage,
    ) -> tuple[Optional[float], str]:
        """
        Estimate the duration of a stage.

        Priority:
        1. Actual completed timing from the current inference.
        2. Default stage estimate.

        The default is deliberately stage-specific. We never derive ETA
        simply from elapsed_time / percentage.
        """

        timing = self._find_stage_timing(status, stage)

        if timing and timing.duration_seconds is not None:
            return timing.duration_seconds, "observed"

        default = self.DEFAULT_STAGE_ESTIMATES.get(stage)

        if default is not None:
            return default, "weighted"

        return None, "calculating"

    def _estimate_remaining_time(
        self,
        status: PipelineStatus,
        current_stage: PipelineStage,
        current_stage_fraction: float = 0.0,
    ) -> tuple[Optional[float], str]:
        """
        Estimate remaining pipeline time.

        Current stage:
            Uses observed progress within that stage when available.

        Future stages:
            Uses actual historical timing from this inference when
            available, otherwise stage-specific weighted estimates.

        This intentionally does NOT use:
            elapsed / overall_progress
        """

        if current_stage in (PipelineStage.COMPLETE, PipelineStage.ERROR):
            return 0.0, "complete"

        current_estimate, current_source = self._estimate_stage_duration(
            status,
            current_stage,
        )

        remaining = 0.0
        confidence = "weighted"

        # Estimate remaining portion of the current stage.
        if current_estimate is not None:
            fraction = max(0.0, min(1.0, current_stage_fraction))

            if fraction > 0:
                remaining += current_estimate * (1.0 - fraction)
            else:
                remaining += current_estimate

            if current_source == "observed":
                confidence = "observed"

        else:
            # We cannot produce a trustworthy ETA yet.
            return None, "calculating"

        current_order = self._stage_index(current_stage)

        # Add future stage estimates.
        for stage_info in self.PIPELINE_STAGES:
            if stage_info.order <= current_order:
                continue

            future_stage = PipelineStage(stage_info.key)

            if future_stage == PipelineStage.COMPLETE:
                continue

            duration, source = self._estimate_stage_duration(
                status,
                future_stage,
            )

            if duration is None:
                return None, "calculating"

            remaining += duration

            if source == "observed":
                confidence = "observed"

        return max(0.0, remaining), confidence

    # ------------------------------------------------------------------
    # Status updates
    # ------------------------------------------------------------------

    def _update_status(
        self,
        result_id: str,
        stage: PipelineStage,
        progress: float,
        message: str,
        *,
        stage_fraction: Optional[float] = None,
        start_stage: bool = False,
        finish_stage: bool = False,
    ) -> None:
        """
        Update pipeline state and persist it immediately.

        Progress comes from actual backend stage execution.

        For Evidence Retrieval and Step Verification, stage_fraction
        represents the actual number of completed steps inside that stage.
        """

        result = self._results.get(result_id)

        if result is None:
            return

        status = result.pipeline_status
        now = time.time()

        if status.started_at is None:
            status.started_at = now

        # Start a new stage when requested.
        if start_stage:
            self._start_stage_timing(status, stage, now)

        # Finish the stage when requested.
        if finish_stage:
            self._finish_stage_timing(status, stage, now)

        # If a stage has not been explicitly started yet, start it.
        if self._find_stage_timing(status, stage) is None:
            self._start_stage_timing(status, stage, now)

        # Current elapsed time.
        status.elapsed_seconds = round(
            max(0.0, now - status.started_at),
            3,
        )

        current_timing = self._find_stage_timing(status, stage)

        if current_timing and current_timing.started_at is not None:
            if current_timing.duration_seconds is not None:
                status.current_stage_elapsed_seconds = round(
                    current_timing.duration_seconds,
                    3,
                )
            else:
                status.current_stage_elapsed_seconds = round(
                    max(0.0, now - current_timing.started_at),
                    3,
                )

        # Determine completed stages.
        completed_stage_count = sum(
            1
            for timing in status.stage_timings
            if timing.duration_seconds is not None
        )

        # COMPLETE itself is considered the final completed stage.
        if stage == PipelineStage.COMPLETE:
            completed_stage_count = len(self.PIPELINE_STAGES)

        status.completed_stages = min(
            completed_stage_count,
            len(self.PIPELINE_STAGES),
        )

        status.stage = stage
        status.stage_index = self._stage_index(stage)
        status.total_stages = len(self.PIPELINE_STAGES)
        status.progress = max(0.0, min(100.0, progress))
        status.message = message
        status.stages = list(self.PIPELINE_STAGES)

        # ETA.
        if stage == PipelineStage.COMPLETE:
            status.estimated_remaining_seconds = 0.0
            status.eta_confidence = "complete"

        elif stage == PipelineStage.ERROR:
            status.estimated_remaining_seconds = None
            status.eta_confidence = "unavailable"

        else:
            remaining, confidence = self._estimate_remaining_time(
                status,
                stage,
                stage_fraction or 0.0,
            )

            status.estimated_remaining_seconds = (
                round(remaining, 1)
                if remaining is not None
                else None
            )
            status.eta_confidence = confidence

        self._save_result(result)

        logger.info(
            "[%s] STAGE -> %s | %.1f%% | stage=%d/%d | "
            "elapsed=%.2fs | eta=%s | %s",
            result_id,
            stage.value,
            status.progress,
            status.stage_index,
            status.total_stages,
            status.elapsed_seconds,
            (
                f"{status.estimated_remaining_seconds:.1f}s"
                if status.estimated_remaining_seconds is not None
                else "calculating"
            ),
            message,
        )

    # ------------------------------------------------------------------
    # Progress calculation
    # ------------------------------------------------------------------

    def _stage_progress(
        self,
        stage: PipelineStage,
        fraction: float = 0.0,
    ) -> float:
        """
        Convert actual stage execution into overall progress.

        Example:
            Stage 3 of 8, 50% through stage
            => 31.25% overall

        This is stage-based progress, not elapsed-time extrapolation.
        """

        total = len(self.PIPELINE_STAGES)

        if stage == PipelineStage.COMPLETE:
            return 100.0

        order = self._stage_index(stage)

        if order <= 0:
            return 0.0

        fraction = max(0.0, min(1.0, fraction))

        progress = ((order - 1) + fraction) / total * 100.0

        return round(progress, 1)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def run(
        self,
        image_path: str,
        image_id: str,
        question: str,
        result_id: Optional[str] = None,
    ) -> HeraResult:
        result_id = result_id or str(uuid.uuid4())
        pipeline_start = time.time()
        cancel_event = threading.Event()
        self._cancelled[result_id] = cancel_event

        try:
            # Image loading is part of pipeline startup, but the current backend
            # does not expose it as a separate executable stage. We therefore do
            # not fabricate an additional UI stage for it.
            if self._check_cancelled(result_id, message="Pipeline cancelled before startup."):
                return self._results.get(result_id, HeraResult(
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
                ))

            image = load_image(image_path)

            initial_status = self._default_status(
                PipelineStage.COT_GENERATION,
                progress=0.0,
                message="Starting HERA pipeline...",
                started_at=pipeline_start,
            )

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
                pipeline_status=initial_status,
            )

            self._results[result_id] = result
            self._save_result(result)

            if self._check_cancelled(result_id):
                return result

            try:
                # ============================================================
                # Stage 1: CoT Generation
                # ============================================================

                logger.info("[%s] START CoTGeneration", result_id)

                self._update_status(
                    result_id,
                    PipelineStage.COT_GENERATION,
                    self._stage_progress(
                        PipelineStage.COT_GENERATION,
                        0.0,
                    ),
                    "Generating Chain-of-Thought...",
                    stage_fraction=0.0,
                    start_stage=True,
                )

                if self._check_cancelled(result_id):
                    return result

                original_cot = cot_generator.generate(image, question)

                result.original_cot = original_cot
                self._save_result(result)

                self._update_status(
                    result_id,
                    PipelineStage.COT_GENERATION,
                    self._stage_progress(
                        PipelineStage.COT_GENERATION,
                        1.0,
                    ),
                    "Chain-of-Thought generation complete.",
                    stage_fraction=1.0,
                    finish_stage=True,
                )

                if self._check_cancelled(result_id):
                    return result

                # Parse structured sections once.
                sections = parse_cot_sections(original_cot)

                observations = sections.get("observations", "")

                logger.info(
                    "[%s] Parsed CoT sections: observations=%d chars, "
                    "reasoning=%s, conclusion=%s",
                    result_id,
                    len(observations),
                    "observations" in sections,
                    "conclusion" in sections,
                )

                # ============================================================
                # Stage 2: Step Segmentation
                # ============================================================

                logger.info("[%s] START StepSegmentation", result_id)

                self._update_status(
                    result_id,
                    PipelineStage.STEP_SEGMENTATION,
                    self._stage_progress(
                        PipelineStage.STEP_SEGMENTATION,
                        0.0,
                    ),
                    "Segmenting reasoning steps...",
                    stage_fraction=0.0,
                    start_stage=True,
                )

                if self._check_cancelled(result_id):
                    return result

                reasoning_text = sections.get(
                    "reasoning",
                    original_cot,
                )

                step_texts = segment_reasoning_steps(reasoning_text)

                if not step_texts:
                    step_texts = (
                        segment_reasoning_steps(original_cot)
                        or [original_cot[:500]]
                    )

                logger.info(
                    "[%s] Step segmentation produced %d steps",
                    result_id,
                    len(step_texts),
                )

                self._update_status(
                    result_id,
                    PipelineStage.STEP_SEGMENTATION,
                    self._stage_progress(
                        PipelineStage.STEP_SEGMENTATION,
                        1.0,
                    ),
                    f"Segmented {len(step_texts)} reasoning steps.",
                    stage_fraction=1.0,
                    finish_stage=True,
                )

                if self._check_cancelled(result_id):
                    return result

                # ============================================================
                # Stage 3: Evidence Retrieval
                # ============================================================

                logger.info(
                    "[%s] START EvidenceRetrieval (%d steps)",
                    result_id,
                    len(step_texts),
                )

                self._update_status(
                    result_id,
                    PipelineStage.EVIDENCE_RETRIEVAL,
                    self._stage_progress(
                        PipelineStage.EVIDENCE_RETRIEVAL,
                        0.0,
                    ),
                    f"Preparing evidence retrieval for {len(step_texts)} steps...",
                    stage_fraction=0.0,
                    start_stage=True,
                )

                if self._check_cancelled(result_id):
                    return result

                retrieved_evidence: list[tuple] = []

                for i, step_text in enumerate(step_texts):
                    if self._check_cancelled(result_id):
                        return result

                    visual_ev, textual_ev = evidence_retriever.retrieve_for_step(
                        step_text,
                        image,
                        question,
                        i,
                        observations=observations,
                    )

                    retrieved_evidence.append(
                        (
                            visual_ev,
                            textual_ev,
                        )
                    )

                    fraction = (i + 1) / len(step_texts)

                    self._update_status(
                        result_id,
                        PipelineStage.EVIDENCE_RETRIEVAL,
                        self._stage_progress(
                            PipelineStage.EVIDENCE_RETRIEVAL,
                            fraction,
                        ),
                        f"Retrieved evidence for step {i + 1}/{len(step_texts)}.",
                        stage_fraction=fraction,
                    )

                self._update_status(
                    result_id,
                    PipelineStage.EVIDENCE_RETRIEVAL,
                    self._stage_progress(
                        PipelineStage.EVIDENCE_RETRIEVAL,
                        1.0,
                    ),
                    "Evidence retrieval complete.",
                    stage_fraction=1.0,
                    finish_stage=True,
                )

                if self._check_cancelled(result_id):
                    return result

                # ============================================================
                # Stage 4: Step Verification
                # ============================================================

                logger.info(
                    "[%s] START StepVerification (%d steps)",
                    result_id,
                    len(step_texts),
                )

                self._update_status(
                    result_id,
                    PipelineStage.STEP_VERIFICATION,
                    self._stage_progress(
                        PipelineStage.STEP_VERIFICATION,
                        0.0,
                    ),
                    f"Preparing verification for {len(step_texts)} steps...",
                    stage_fraction=0.0,
                    start_stage=True,
                )

                if self._check_cancelled(result_id):
                    return result

                verified_steps = []

                for i, step_text in enumerate(step_texts):
                    if self._check_cancelled(result_id):
                        return result

                    visual_ev, textual_ev = retrieved_evidence[i]

                    step_result = hallucination_detector.verify_step(
                        i,
                        step_text,
                        visual_ev,
                        textual_ev,
                    )

                    verified_steps.append(step_result)

                    fraction = (i + 1) / len(step_texts)

                    self._update_status(
                        result_id,
                        PipelineStage.STEP_VERIFICATION,
                        self._stage_progress(
                            PipelineStage.STEP_VERIFICATION,
                            fraction,
                        ),
                        f"Verified step {i + 1}/{len(step_texts)}.",
                        stage_fraction=fraction,
                    )

                result.steps = verified_steps
                self._save_result(result)

                self._update_status(
                    result_id,
                    PipelineStage.STEP_VERIFICATION,
                    self._stage_progress(
                        PipelineStage.STEP_VERIFICATION,
                        1.0,
                    ),
                    "Step verification complete.",
                    stage_fraction=1.0,
                    finish_stage=True,
                )

                if self._check_cancelled(result_id):
                    return result

                # ============================================================
                # Stage 5: Hallucination Detection
                # ============================================================

                logger.info(
                    "[%s] START HallucinationDetection",
                    result_id,
                )

                self._update_status(
                    result_id,
                    PipelineStage.HALLUCINATION_DETECTION,
                    self._stage_progress(
                        PipelineStage.HALLUCINATION_DETECTION,
                        0.0,
                    ),
                    "Computing hallucination scores...",
                    stage_fraction=0.0,
                    start_stage=True,
                )

                if self._check_cancelled(result_id):
                    return result

                hallucinated = sum(
                    1
                    for step in verified_steps
                    if not step.supported
                )

                supported_count = len(verified_steps) - hallucinated

                result.hallucination_score = round(
                    hallucinated / max(len(verified_steps), 1),
                    3,
                )

                result.confidence_score = round(
                    sum(
                        step.confidence
                        for step in verified_steps
                    ) / max(len(verified_steps), 1),
                    3,
                )

                self._save_result(result)

                self._update_status(
                    result_id,
                    PipelineStage.HALLUCINATION_DETECTION,
                    self._stage_progress(
                        PipelineStage.HALLUCINATION_DETECTION,
                        1.0,
                    ),
                    "Hallucination analysis complete.",
                    stage_fraction=1.0,
                    finish_stage=True,
                )

                if self._check_cancelled(result_id):
                    return result

                # ============================================================
                # Stage 6: Evidence Attribution
                # ============================================================

                logger.info(
                    "[%s] START EvidenceAttribution",
                    result_id,
                )

                self._update_status(
                    result_id,
                    PipelineStage.EVIDENCE_ATTRIBUTION,
                    self._stage_progress(
                        PipelineStage.EVIDENCE_ATTRIBUTION,
                        0.0,
                    ),
                    "Building evidence-attributed CoT...",
                    stage_fraction=0.0,
                    start_stage=True,
                )

                if self._check_cancelled(result_id):
                    return result

                result.attributed_cot = evidence_attributor.attribute_steps(
                    verified_steps
                )

                self._save_result(result)

                self._update_status(
                    result_id,
                    PipelineStage.EVIDENCE_ATTRIBUTION,
                    self._stage_progress(
                        PipelineStage.EVIDENCE_ATTRIBUTION,
                        1.0,
                    ),
                    "Evidence attribution complete.",
                    stage_fraction=1.0,
                    finish_stage=True,
                )

                if self._check_cancelled(result_id):
                    return result

                # ============================================================
                # Stage 7: Reasoning Correction
                # ============================================================

                logger.info(
                    "[%s] START ReasoningCorrection",
                    result_id,
                )

                self._update_status(
                    result_id,
                    PipelineStage.REASONING_CORRECTION,
                    self._stage_progress(
                        PipelineStage.REASONING_CORRECTION,
                        0.0,
                    ),
                    "Generating corrected reasoning...",
                    stage_fraction=0.0,
                    start_stage=True,
                )

                if self._check_cancelled(result_id):
                    return result

                corrected_cot, final_answer = reasoning_corrector.correct(
                    original_cot,
                    verified_steps,
                    question,
                )

                result.corrected_cot = corrected_cot
                result.final_answer = final_answer

                self._save_result(result)

                self._update_status(
                    result_id,
                    PipelineStage.REASONING_CORRECTION,
                    self._stage_progress(
                        PipelineStage.REASONING_CORRECTION,
                        1.0,
                    ),
                    "Corrected reasoning generated.",
                    stage_fraction=1.0,
                    finish_stage=True,
                )

                if self._check_cancelled(result_id):
                    return result

                # ============================================================
                # Stage 8: Complete
                # ============================================================

                self._update_status(
                    result_id,
                    PipelineStage.COMPLETE,
                    100.0,
                    "Pipeline complete.",
                    stage_fraction=1.0,
                )

                # Ensure the total elapsed time is captured at completion.
                result.pipeline_status.elapsed_seconds = round(
                    max(0.0, time.time() - pipeline_start),
                    3,
                )

                result.pipeline_status.current_stage_elapsed_seconds = 0.0
                result.pipeline_status.estimated_remaining_seconds = 0.0
                result.pipeline_status.eta_confidence = "complete"
                result.pipeline_status.completed_stages = len(
                    self.PIPELINE_STAGES
                )
                result.pipeline_status.stage_index = len(
                    self.PIPELINE_STAGES
                )
                result.pipeline_status.total_stages = len(
                    self.PIPELINE_STAGES
                )

                self._save_result(result)

                logger.info(
                    "[%s] Pipeline complete | total=%.2fs | steps=%d | "
                    "supported=%d | hallucinated=%d",
                    result_id,
                    result.pipeline_status.elapsed_seconds,
                    len(verified_steps),
                    supported_count,
                    hallucinated,
                )

            except Exception as e:
                if self.is_cancelled(result_id):
                    logger.info("[%s] Pipeline cancelled during execution.", result_id)
                    return self._mark_cancelled(result_id, "Pipeline cancelled by user.") or result

                logger.exception(
                    "[%s] Pipeline failed: %s",
                    result_id,
                    e,
                )

                result.pipeline_status = PipelineStatus(
                    stage=PipelineStage.ERROR,
                    progress=0,
                    message=str(e),
                    stages=list(self.PIPELINE_STAGES),
                    stage_index=0,
                    total_stages=len(self.PIPELINE_STAGES),
                    completed_stages=sum(
                        1
                        for timing in result.pipeline_status.stage_timings
                        if timing.duration_seconds is not None
                    ),
                    started_at=result.pipeline_status.started_at,
                    elapsed_seconds=round(
                        max(0.0, time.time() - pipeline_start),
                        3,
                    ),
                    current_stage_elapsed_seconds=0.0,
                    estimated_remaining_seconds=None,
                    eta_confidence="unavailable",
                    stage_timings=result.pipeline_status.stage_timings,
                )

                self._save_result(result)

            return result
        finally:
            self._cancelled.pop(result_id, None)
            if result_id in self._results and self._results[result_id].pipeline_status.stage in {
                PipelineStage.COMPLETE,
                PipelineStage.ERROR,
                PipelineStage.CANCELLED,
            }:
                self.release_active_request(
                    self._results[result_id].image_id,
                    self._results[result_id].question,
                    result_id,
                )


hera_pipeline = HeraPipeline()