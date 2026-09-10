import threading
import unittest

from backend.api.schemas import HeraResult, PipelineStage, PipelineStatus
from backend.pipelines.hera_pipeline import HeraPipeline


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


if __name__ == "__main__":
    unittest.main()
