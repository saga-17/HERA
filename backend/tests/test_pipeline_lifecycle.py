import threading
import unittest
from unittest.mock import patch

from PIL import Image

from backend.api.schemas import (
    EntityExtraction,
    HallucinationType,
    HeraResult,
    PipelineStage,
    PipelineStatus,
    ReasoningStepResult,
    StepStatus,
    TextEvidence,
)
from backend.config import settings
from backend.pipelines.hera_pipeline import HeraPipeline
from backend.services.cot_generator import CoTGenerator
from backend.services.evidence_retriever import EvidenceRetriever
from backend.services.hallucination_detector import HallucinationDetector
from backend.services.reasoning_corrector import ReasoningCorrector


class TestPipelineLifecycle(unittest.TestCase):
    def setUp(self):
        self.pipeline = HeraPipeline()

    def _metric_step(self, status, step_index=0):
        return ReasoningStepResult(
            step_index=step_index,
            step="A claim.",
            status=status,
            confidence=0.73,
            supported=(status == StepStatus.SUPPORTED),
            extraction=EntityExtraction(),
        )

    def test_verification_metrics_cover_supported_and_uncertain_statuses(self):
        cases = (
            ([StepStatus.SUPPORTED, StepStatus.SUPPORTED], (0.0, 1.0, 0, 2)),
            ([StepStatus.SUPPORTED, StepStatus.UNCERTAIN], (0.0, 0.5, 0, 1)),
            ([StepStatus.SUPPORTED, StepStatus.HALLUCINATED], (0.5, 0.5, 1, 1)),
            ([StepStatus.UNCERTAIN, StepStatus.UNCERTAIN], (0.0, 0.0, 0, 0)),
            ([StepStatus.HALLUCINATED, StepStatus.HALLUCINATED], (1.0, 0.0, 2, 0)),
        )
        for statuses, expected in cases:
            with self.subTest(statuses=statuses):
                steps = [self._metric_step(status, i) for i, status in enumerate(statuses)]
                result = self.pipeline._verification_metrics(steps)
                self.assertEqual(result, expected)
                self.assertEqual(result, self.pipeline._verification_metrics(steps))

    def test_status_controls_supported_boolean(self):
        for status, expected in (
            (StepStatus.SUPPORTED, True),
            (StepStatus.HALLUCINATED, False),
            (StepStatus.UNCERTAIN, False),
        ):
            with self.subTest(status=status):
                step = ReasoningStepResult(
                    step_index=0,
                    step="A claim.",
                    status=status,
                    confidence=0.5,
                    supported=not expected,
                )
                self.assertEqual(step.supported, expected)

    def test_uncertain_irrelevant_step_preserves_supported_final_answer(self):
        supported = self._metric_step(StepStatus.SUPPORTED)
        supported.step = "The image shows a dog."
        uncertain = self._metric_step(StepStatus.UNCERTAIN, 1)
        uncertain.step = "A distant sign contains unreadable text."

        _corrected, answer = ReasoningCorrector().correct(
            "", [supported, uncertain], "What animal is shown?"
        )
        self.assertIn("dog", answer.lower())
        self.assertNotIn("sign", answer.lower())

    def test_retrieval_ranker_failure_is_deterministic(self):
        retriever = EvidenceRetriever()
        image = Image.new("RGB", (30, 30), color="white")
        with (
            patch("backend.services.evidence_retriever.settings.demo_mode", False),
            patch(
                "backend.services.evidence_retriever.model_manager.get_cross_encoder",
                side_effect=RuntimeError("ranker unavailable"),
            ),
        ):
            first = retriever._retrieve_visual_evidence(
                "A tree is visible.", image, 0, "A tree is visible."
            )
            second = retriever._retrieve_visual_evidence(
                "A tree is visible.", image, 0, "A tree is visible."
            )
        self.assertEqual(
            [(e.bbox, e.confidence, e.caption) for e in first],
            [(e.bbox, e.confidence, e.caption) for e in second],
        )

    def test_cancel_marks_result_as_cancelled(self):
        result = HeraResult(
            result_id="run-123",
            image_id="img-1",
            image_url="/api/images/img-1",
            question="What is in the image?",
            original_cot="",
            attributed_cot="",
            corrected_cot="",
            final_answer="",
            hallucination_score=0.0,
            confidence_score=0.0,
            pipeline_status=PipelineStatus(
                stage=PipelineStage.COT_GENERATION,
                progress=25.0,
                message="Running",
                stages=[],
                stage_index=1,
                total_stages=8,
                completed_stages=0,
            ),
        )
        self.pipeline._results["run-123"] = result
        self.pipeline._cancelled["run-123"] = threading.Event()

        self.pipeline.cancel("run-123")

        self.assertTrue(self.pipeline.is_cancelled("run-123"))
        self.assertEqual(result.pipeline_status.stage, PipelineStage.CANCELLED)
        self.assertIn("cancelled", result.pipeline_status.message.lower())

    def test_duplicate_guard_detects_active_run_for_same_request(self):
        self.pipeline._active_requests[("img-2", "Describe the scene")] = "run-999"

        self.assertTrue(self.pipeline.has_active_run_for_request("img-2", "Describe the scene"))
        self.assertFalse(self.pipeline.has_active_run_for_request("img-2", "Different question"))

    def test_unstructured_vlm_answer_gets_independent_visual_observations(self):
        generator = CoTGenerator()
        with (
            patch("backend.services.cot_generator.settings.demo_mode", False),
            patch("backend.services.cot_generator.settings.vlm_model_id", "Qwen2-VL-test"),
            patch("backend.services.cot_generator.model_manager.load_vlm", return_value=(object(), object())),
            patch("backend.services.cot_generator.model_manager.unload_vlm"),
            patch.object(
                generator,
                "_generate_qwen",
                side_effect=["Yes, this is a cow.", "A cow stands in a grassy field."],
            ) as generate,
        ):
            response = generator.generate(Image.new("RGB", (16, 16)), "Is this a cow?")

        self.assertIn("<OBSERVATIONS>\nA cow stands in a grassy field.", response)
        self.assertIn("Step 1: Yes, this is a cow.", response)
        self.assertEqual(generate.call_count, 2)
        self.assertNotIn("Is this a cow?", generate.call_args_list[1].args[3])

    def test_verified_visual_claims_answer_general_question_types(self):
        cases = (
            ("Is this a cow?", "The image shows a cow.", "Yes. The image shows a cow."),
            ("Is this a crow?", "The image shows a cow.", "No. The image shows a cow, not a crow."),
            ("Is the animal a cow?", "The animal in the image is a cow.", "Yes. The animal in the image is a cow."),
            ("Is the animal a crow?", "The image shows a cow.", "No. The image shows a cow, not a crow."),
            ("Is this a crow?", "The image shows a crow.", "Yes. The image shows a crow."),
            ("Is this a cat?", "The image shows a dog.", "No. The image shows a dog, not a cat."),
            ("What is the person holding?", "The person is holding a bouquet of flowers.", "bouquet of flowers"),
            ("Describe the image.", "A person stands outdoors holding a bouquet of flowers.", "stands outdoors"),
            ("What color is the car?", "The car is red.", "The car is red."),
            ("How many objects are visible?", "There are three objects visible.", "There are 3 objects."),
            (
                "Is this an airplane?",
                "The image shows a blurry object.",
                "does not establish whether it is an airplane",
            ),
            (
                "Is this a brain tumor?",
                "A lesion may be consistent with a tumor.",
                "cannot establish a diagnosis",
            ),
            (
                "Is this a brain tumor?",
                "A needle appears to enter a lesion.",
                "lesion. The image alone cannot establish a diagnosis.",
            ),
        )
        corrector = ReasoningCorrector()

        for question, claim, expected in cases:
            with self.subTest(question=question, claim=claim):
                step = ReasoningStepResult(
                    step_index=0,
                    step=claim,
                    status=StepStatus.SUPPORTED,
                    confidence=0.95,
                    supported=True,
                    hallucination_type=HallucinationType.NONE,
                    evidence=claim,
                    visual_evidence=[],
                    textual_evidence=[],
                    attribution="Supported by image-grounded observation.",
                    extraction=EntityExtraction(),
                )
                answer = corrector._generate_final_answer(
                    original_cot="<CONCLUSION>Final Answer: This is definitely a helicopter.</CONCLUSION>",
                    supported_steps=[step],
                    question=question,
                    hallucinated_count=0,
                )
                self.assertIn(expected.lower(), answer.lower())
                self.assertNotIn("helicopter", answer.lower())

    def test_hallucination_detector_uses_grounded_observations_not_web_text(self):
        class OverlapEncoder:
            def predict(self, pairs):
                scores = []
                for claim, evidence in pairs:
                    stopwords = {"a", "an", "the", "is", "are", "visible"}
                    claim_words = {word.strip(".,?!;:") for word in claim.lower().split()} - stopwords
                    evidence_words = {word.strip(".,?!;:") for word in evidence.lower().split()} - stopwords
                    scores.append(8.0 if claim_words & evidence_words else -8.0)
                return scores

        detector = HallucinationDetector()
        with patch(
            "backend.services.hallucination_detector.model_manager.get_cross_encoder",
            return_value=OverlapEncoder(),
        ):
            supported = detector.verify_step(
                0,
                "A bouquet is visible.",
                [],
                [TextEvidence(text="A bouquet is visible.", source="visual_observation", confidence=0.9)],
            )
            web_only = detector.verify_step(
                1,
                "A helicopter is visible.",
                [],
                [TextEvidence(text="A helicopter is visible.", source="web_search_1", confidence=1.0)],
            )
            unsupported = detector.verify_step(
                2,
                "A helicopter is visible.",
                [],
                [TextEvidence(text="A cow is visible.", source="visual_observation", confidence=0.9)],
            )

        self.assertEqual(supported.status, StepStatus.SUPPORTED)
        self.assertGreater(supported.confidence, 0.65)
        self.assertIn("visual_observation", supported.attribution)
        self.assertEqual(web_only.status, StepStatus.UNCERTAIN)
        self.assertEqual(web_only.confidence, 0.4)
        self.assertNotIn("web_search", web_only.attribution)
        self.assertEqual(unsupported.status, StepStatus.HALLUCINATED)

    def test_verified_cow_observation_answers_cow_and_crow_questions_consistently(self):
        class GroundingEncoder:
            def predict(self, pairs):
                return [8.0 for _ in pairs]

        with patch(
            "backend.services.hallucination_detector.model_manager.get_cross_encoder",
            return_value=GroundingEncoder(),
        ):
            verified_step = HallucinationDetector().verify_step(
                0,
                "The image shows a cow.",
                [],
                [TextEvidence(text="The image shows a cow.", source="visual_observation", confidence=0.9)],
            )

        self.assertEqual(verified_step.status, StepStatus.SUPPORTED)
        self.assertGreater(verified_step.confidence, 0.65)
        corrector = ReasoningCorrector()
        for question, expected in (
            ("Is this a cow?", "Yes. The image shows a cow."),
            ("Is this a crow?", "No. The image shows a cow, not a crow."),
        ):
            with self.subTest(question=question):
                answer = corrector._generate_final_answer(
                    original_cot="<CONCLUSION>Final Answer: I can't determine that reliably from the image.</CONCLUSION>",
                    supported_steps=[verified_step],
                    question=question,
                    hallucinated_count=0,
                )
                self.assertEqual(answer, expected)

    def test_detector_does_not_require_lexical_overlap_or_call_missing_evidence_hallucinated(self):
        class SemanticEncoder:
            def predict(self, pairs):
                return [8.0 for _ in pairs]

        detector = HallucinationDetector()
        with patch(
            "backend.services.hallucination_detector.model_manager.get_cross_encoder",
            return_value=SemanticEncoder(),
        ):
            paraphrase = detector.verify_step(
                0,
                "A bouquet is carried by someone.",
                [],
                [TextEvidence(text="Flowers are held in their hands.", source="visual_observation", confidence=0.9)],
            )
            no_evidence = detector.verify_step(1, "A helicopter is visible.", [], [])

        self.assertEqual(paraphrase.status, StepStatus.SUPPORTED)
        self.assertEqual(no_evidence.status, StepStatus.UNCERTAIN)
        self.assertEqual(no_evidence.confidence, 0.4)

    def test_missing_observations_do_not_create_self_supporting_visual_evidence(self):
        with (
            patch.object(settings, "demo_mode", False),
            patch.object(settings, "enable_web_search", False),
        ):
            visual, textual = EvidenceRetriever().retrieve_for_step(
                "The image contains a dragon.",
                Image.new("RGB", (32, 32), "white"),
                "What is in the image?",
                0,
                observations="",
            )

        self.assertEqual(visual, [])
        self.assertEqual(textual, [])

    def test_final_answer_is_semantic_and_clean(self):
        test_cases = [
            (
                "Is this a leg?",
                [
                    ReasoningStepResult(
                        step_index=0,
                        step="The object is a dog, not a leg.",
                        status=StepStatus.SUPPORTED,
                        confidence=0.94,
                        supported=True,
                        hallucination_type=HallucinationType.NONE,
                        evidence="Visual evidence shows a dog.",
                        visual_evidence=[],
                        textual_evidence=[],
                        attribution="The animal shape matches a dog.",
                        extraction=EntityExtraction(),
                    )
                ],
                "No. The object is a dog, not a leg.",
            ),
            (
                "What is this?",
                [
                    ReasoningStepResult(
                        step_index=0,
                        step="It is a dog.",
                        status=StepStatus.SUPPORTED,
                        confidence=0.94,
                        supported=True,
                        hallucination_type=HallucinationType.NONE,
                        evidence="The image contains a dog.",
                        visual_evidence=[],
                        textual_evidence=[],
                        attribution="The animal is a dog.",
                        extraction=EntityExtraction(),
                    )
                ],
                "It is a dog.",
            ),
            (
                "What animals are present?",
                [
                    ReasoningStepResult(
                        step_index=0,
                        step="Dogs, rabbits, foxes, beavers, and mice are visible in the scene.",
                        status=StepStatus.SUPPORTED,
                        confidence=0.95,
                        supported=True,
                        hallucination_type=HallucinationType.NONE,
                        evidence="Multiple animals are visible.",
                        visual_evidence=[],
                        textual_evidence=[],
                        attribution="The evidence supports multiple animals.",
                        extraction=EntityExtraction(),
                    )
                ],
                "The image contains dogs, rabbits, foxes, beavers, and mice.",
            ),
            (
                "Is this a dog?",
                [
                    ReasoningStepResult(
                        step_index=0,
                        step="This is a dog.",
                        status=StepStatus.SUPPORTED,
                        confidence=0.93,
                        supported=True,
                        hallucination_type=HallucinationType.NONE,
                        evidence="The image shows a dog.",
                        visual_evidence=[],
                        textual_evidence=[],
                        attribution="The recognized animal is a dog.",
                        extraction=EntityExtraction(),
                    )
                ],
                "Yes. This is a dog.",
            ),
            (
                "How many dogs are there?",
                [
                    ReasoningStepResult(
                        step_index=0,
                        step="There are three dogs in the image.",
                        status=StepStatus.SUPPORTED,
                        confidence=0.91,
                        supported=True,
                        hallucination_type=HallucinationType.NONE,
                        evidence="Three dogs are visible.",
                        visual_evidence=[],
                        textual_evidence=[],
                        attribution="Multiple dogs are visible in the scene.",
                        extraction=EntityExtraction(),
                    )
                ],
                "There are 3 dogs.",
            ),
            (
                "Is the dog next to the person?",
                [
                    ReasoningStepResult(
                        step_index=0,
                        step="The dog is next to the person.",
                        status=StepStatus.SUPPORTED,
                        confidence=0.90,
                        supported=True,
                        hallucination_type=HallucinationType.NONE,
                        evidence="The dog is adjacent to the person.",
                        visual_evidence=[],
                        textual_evidence=[],
                        attribution="The dog and person are side by side.",
                        extraction=EntityExtraction(),
                    )
                ],
                "Yes, the dog is next to the person.",
            ),
            (
                "Is this a leg?",
                [],
                "I can't determine that reliably from the image.",
            ),
        ]

        for question, steps, expected in test_cases:
            with self.subTest(question=question):
                final_answer = ReasoningCorrector()._generate_final_answer(
                    original_cot="<CONCLUSION>\nFinal Answer: The model says it is a dog.\n</CONCLUSION>",
                    supported_steps=steps,
                    question=question,
                    hallucinated_count=0,
                )

                self.assertNotIn("Answer:", final_answer)
                self.assertNotIn("Verification:", final_answer)
                self.assertNotIn("Confidence:", final_answer)
                self.assertNotIn("Evidence:", final_answer)
                self.assertNotIn("Reasoning:", final_answer)
                self.assertNotIn("(conf=", final_answer)
                self.assertNotIn("[Visual]", final_answer)
                self.assertNotIn("[Text]", final_answer)
                self.assertNotIn("...", final_answer)
                self.assertFalse(final_answer.rstrip().endswith("..."))
                self.assertTrue(final_answer.strip().startswith(expected.split()[0]) or expected in final_answer)

                if question == "What animals are present?":
                    self.assertIn("dogs", final_answer.lower())
                if question == "Is this a leg?" and not steps:
                    self.assertEqual(final_answer, expected)
                elif question != "Is this a leg?":
                    self.assertIn(expected.lower(), final_answer.lower())

    def test_hallucinated_cot_does_not_bleed_into_final_answer(self):
        steps = [
            ReasoningStepResult(
                step_index=0,
                step="The image shows a dog in the center of the scene.",
                status=StepStatus.SUPPORTED,
                confidence=0.96,
                supported=True,
                hallucination_type=HallucinationType.NONE,
                evidence="Visual evidence supports the dog.",
                visual_evidence=[],
                textual_evidence=[],
                attribution="The dog is directly visible.",
                extraction=EntityExtraction(),
            )
        ]

        final_answer = ReasoningCorrector()._generate_final_answer(
            original_cot="<CONCLUSION>\nFinal Answer: The dog is a fish and the scene is underwater.\n</CONCLUSION>",
            supported_steps=steps,
            question="What is this animal?",
            hallucinated_count=1,
        )

        self.assertNotIn("fish", final_answer.lower())
        self.assertNotIn("underwater", final_answer.lower())
        self.assertIn("dog", final_answer.lower())
        self.assertNotIn("Verification:", final_answer)
        self.assertNotIn("Confidence:", final_answer)
        self.assertNotIn("Evidence:", final_answer)
        self.assertNotIn("(conf=", final_answer)
        self.assertFalse(final_answer.strip().endswith("..."))

    def test_eta_is_authoritative_and_not_counted_down_by_client(self):
        result = HeraResult(
            result_id="eta-test",
            image_id="img-eta",
            image_url="/api/images/img-eta",
            question="What is in the image?",
            original_cot="",
            attributed_cot="",
            corrected_cot="",
            final_answer="",
            hallucination_score=0.0,
            confidence_score=0.0,
            pipeline_status=PipelineStatus(
                stage=PipelineStage.COT_GENERATION,
                progress=25.0,
                message="Running",
                stages=list(HeraPipeline.PIPELINE_STAGES),
                stage_index=0,
                total_stages=len(HeraPipeline.PIPELINE_STAGES),
                completed_stages=0,
                started_at=1000.0,
                elapsed_seconds=12.0,
                current_stage_elapsed_seconds=8.0,
                estimated_remaining_seconds=24.0,
                eta_confidence="weighted",
            ),
        )

        self.assertGreater(result.pipeline_status.estimated_remaining_seconds or 0, 0)
        self.assertNotEqual(result.pipeline_status.stage, PipelineStage.COMPLETE)

        pipeline = HeraPipeline()
        pipeline._results["eta-test"] = result
        pipeline._update_status(
            "eta-test",
            PipelineStage.COT_GENERATION,
            25.0,
            "Running",
            stage_fraction=0.4,
            start_stage=True,
        )
        self.assertGreaterEqual(result.pipeline_status.estimated_remaining_seconds or 0, 0)
        self.assertNotEqual(result.pipeline_status.stage, PipelineStage.COMPLETE)

        pipeline._update_status(
            "eta-test",
            PipelineStage.COMPLETE,
            100.0,
            "Pipeline complete.",
            stage_fraction=1.0,
        )
        self.assertEqual(result.pipeline_status.stage, PipelineStage.COMPLETE)
        self.assertEqual(result.pipeline_status.progress, 100.0)
        self.assertEqual(result.pipeline_status.estimated_remaining_seconds, 0.0)
        self.assertEqual(result.pipeline_status.eta_confidence, "complete")

    def test_eta_zero_while_running_is_not_treated_as_completion(self):
        status = PipelineStatus(
            stage=PipelineStage.REASONING_CORRECTION,
            progress=90.0,
            message="Finishing up",
            stages=list(HeraPipeline.PIPELINE_STAGES),
            stage_index=6,
            total_stages=len(HeraPipeline.PIPELINE_STAGES),
            completed_stages=6,
            started_at=1000.0,
            elapsed_seconds=110.0,
            current_stage_elapsed_seconds=15.0,
            estimated_remaining_seconds=0.0,
            eta_confidence="weighted",
        )

        self.assertEqual(status.stage, PipelineStage.REASONING_CORRECTION)
        self.assertEqual(status.estimated_remaining_seconds, 0.0)
        self.assertNotEqual(status.stage, PipelineStage.COMPLETE)

    def test_stage_progress_clamps_to_100_and_waits_for_complete_stage(self):
        pipeline = HeraPipeline()

        self.assertEqual(pipeline._stage_progress(PipelineStage.COMPLETE, 0.0), 100.0)
        self.assertEqual(pipeline._stage_progress(PipelineStage.COMPLETE, 1.0), 100.0)
        self.assertLess(pipeline._stage_progress(PipelineStage.COT_GENERATION, 1.0), 100.0)
        self.assertLessEqual(pipeline._stage_progress(PipelineStage.REASONING_CORRECTION, 1.5), 100.0)
        self.assertNotEqual(pipeline._stage_progress(PipelineStage.REASONING_CORRECTION, 0.0), 100.0)


if __name__ == "__main__":
    unittest.main()
