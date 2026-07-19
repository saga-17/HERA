"""Pydantic schemas for HERA-VLM REST API."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class HallucinationType(str, Enum):
    OBJECT = "object_hallucination"
    ATTRIBUTE = "attribute_hallucination"
    RELATIONSHIP = "relationship_hallucination"
    SCENE = "scene_hallucination"
    COMMONSENSE = "commonsense_hallucination"
    REASONING = "reasoning_hallucination"
    NONE = "none"


class StepStatus(str, Enum):
    SUPPORTED = "supported"
    HALLUCINATED = "hallucinated"
    UNCERTAIN = "uncertain"


class PipelineStage(str, Enum):
    UPLOAD = "upload"
    COT_GENERATION = "cot_generation"
    STEP_SEGMENTATION = "step_segmentation"
    EVIDENCE_RETRIEVAL = "evidence_retrieval"
    STEP_VERIFICATION = "step_verification"
    HALLUCINATION_DETECTION = "hallucination_detection"
    EVIDENCE_ATTRIBUTION = "evidence_attribution"
    REASONING_CORRECTION = "reasoning_correction"
    COMPLETE = "complete"
    ERROR = "error"


class VisualEvidence(BaseModel):
    region_id: str
    bbox: list[int] = Field(default_factory=list)
    image_base64: Optional[str] = None
    caption: str = ""
    confidence: float = 0.0


class TextEvidence(BaseModel):
    text: str
    source: str = "retrieval"
    confidence: float = 0.0


class EntityExtraction(BaseModel):
    entities: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)


class ReasoningStepResult(BaseModel):
    step_index: int
    step: str
    status: StepStatus
    confidence: float
    supported: bool
    hallucination_type: HallucinationType = HallucinationType.NONE
    evidence: str = ""
    visual_evidence: list[VisualEvidence] = Field(default_factory=list)
    textual_evidence: list[TextEvidence] = Field(default_factory=list)
    attribution: str = ""
    extraction: EntityExtraction = Field(default_factory=EntityExtraction)


class PipelineStatus(BaseModel):
    stage: PipelineStage
    progress: float = 0.0
    message: str = ""


class UploadResponse(BaseModel):
    image_id: str
    filename: str
    url: str


class AskRequest(BaseModel):
    image_id: str
    question: str


class AskResponse(BaseModel):
    result_id: str
    status: str = "processing"


class HeraResult(BaseModel):
    result_id: str
    image_id: str
    image_url: str
    question: str
    original_cot: str
    attributed_cot: str
    corrected_cot: str
    final_answer: str
    hallucination_score: float
    confidence_score: float
    steps: list[ReasoningStepResult] = Field(default_factory=list)
    pipeline_status: PipelineStatus = Field(
        default_factory=lambda: PipelineStatus(stage=PipelineStage.COMPLETE, progress=100.0)
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool
    demo_mode: bool
    model_loaded: bool
