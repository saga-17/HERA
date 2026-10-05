"""HERA-VLM FastAPI route handlers."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    HeraResult,
    UploadResponse,
)
from backend.config import settings
from backend.models.model_manager import model_manager
from backend.pipelines.hera_pipeline import hera_pipeline

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        gpu_available=model_manager.gpu_available,
        demo_mode=settings.demo_mode,
        model_loaded=model_manager.is_loaded,
    )


@router.post("/upload-image", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_id = str(uuid.uuid4())
    ext = Path(file.filename or "image.png").suffix or ".png"
    filename = f"{image_id}{ext}"
    dest = settings.upload_dir / filename

    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return UploadResponse(
        image_id=image_id,
        filename=filename,
        url=f"/api/images/{image_id}",
    )


@router.get("/images/{image_id}")
async def get_image(image_id: str):
    matches = list(settings.upload_dir.glob(f"{image_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(matches[0])


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, background_tasks: BackgroundTasks) -> AskResponse:
    matches = list(settings.upload_dir.glob(f"{request.image_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Image not found. Upload an image first.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    active_result_id = hera_pipeline.get_active_result_for_request(
        request.image_id,
        question,
    )
    if active_result_id is not None:
        return AskResponse(result_id=active_result_id, status="running")

    result_id = str(uuid.uuid4())
    image_path = str(matches[0])

    hera_pipeline.register_result_for_request(
        request.image_id,
        question,
        result_id,
    )

    background_tasks.add_task(
        hera_pipeline.run,
        image_path,
        request.image_id,
        question,
        result_id,
    )

    return AskResponse(result_id=result_id, status="processing")


@router.post("/cancel/{result_id}")
async def cancel_inference(result_id: str) -> dict[str, str]:
    cancelled = hera_pipeline.cancel(result_id)
    if cancelled is None:
        raise HTTPException(status_code=404, detail="Inference not found")
    return {"result_id": result_id, "status": "cancelled"}


@router.post("/ask-sync", response_model=HeraResult)
async def ask_question_sync(request: AskRequest) -> HeraResult:
    """Synchronous inference endpoint for testing."""
    matches = list(settings.upload_dir.glob(f"{request.image_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Image not found")

    return hera_pipeline.run(
        str(matches[0]),
        request.image_id,
        request.question.strip(),
    )


@router.get("/result/{result_id}", response_model=HeraResult)
async def get_result(result_id: str) -> HeraResult:
    result = hera_pipeline.get_result(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return result
