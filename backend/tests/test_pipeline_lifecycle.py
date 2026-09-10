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

    def test_final_answer_is_complete_and_explicitly_verified(self):
        long_step = (
            "The person is holding a bouquet of flowers with multiple pink and white flowers "
            "in the foreground while standing in a bright indoor setting, and the arrangement "
            "occupies a large portion of the frame and is clearly visible as a bouquet."
        )
        steps = [
            ReasoningStepResult(
                step_index=0,
                step=long_step,
                status=StepStatus.SUPPORTED,
                confidence=0.94,
                supported=True,
                hallucination_type=HallucinationType.NONE,
                evidence="Visual evidence clearly shows a bouquet of flowers in the person's hands.",
                visual_evidence=[],
                textual_evidence=[],
                attribution="The bouquet occupies the foreground and matches the described object.",
                extraction=EntityExtraction(),
            )
        ]

        final_answer = ReasoningCorrector()._generate_final_answer(
            original_cot="<CONCLUSION>\nFinal Answer: The person is holding a bouquet of flowers.\n</CONCLUSION>",
            supported_steps=steps,
            question="What is the person holding?",
            hallucinated_count=0,
        )

        self.assertIn("Answer:", final_answer)
        self.assertIn("Verification:", final_answer)
        self.assertIn("SUPPORTED", final_answer)
        self.assertNotIn("...", final_answer)
        self.assertFalse(final_answer.rstrip().endswith("..."))
        self.assertIn("bouquet of flowers", final_answer.lower())


if __name__ == "__main__":
    unittest.main()
