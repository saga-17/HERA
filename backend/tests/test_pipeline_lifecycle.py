import threading
import unittest

from backend.api.schemas import (
    EntityExtraction,
    HallucinationType,
    HeraResult,
    PipelineStage,
    PipelineStatus,
    ReasoningStepResult,
    StepStatus,
)
from backend.pipelines.hera_pipeline import HeraPipeline
from backend.services.reasoning_corrector import ReasoningCorrector


class TestPipelineLifecycle(unittest.TestCase):
    def setUp(self):
        self.pipeline = HeraPipeline()

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
                "Wrong. It is not a leg; it is a dog.",
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
                "Right. It is a dog.",
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


if __name__ == "__main__":
    unittest.main()
