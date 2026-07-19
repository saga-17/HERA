"""HERA-VLM application configuration."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HERA_", env_file=".env", extra="ignore")
    """Runtime settings loaded from environment variables."""

    app_name: str = "HERA-VLM"
    app_version: str = "1.0.0"
    debug: bool = False

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    upload_dir: Path = base_dir / "data" / "uploads"
    results_dir: Path = base_dir / "data" / "results"
    cave_src_dir: Path = base_dir.parent / "cave-vlm-cot-062A" / "src" / "cite_verify_vlm_cot"

    # Model configuration (RTX 4060 8GB optimized)
    vlm_model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    use_4bit: bool = True
    use_cpu_fallback: bool = True
    max_new_tokens: int = 1024
    demo_mode: bool = os.getenv("HERA_DEMO_MODE", "false").lower() == "true"

    # Retrieval
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_top_k: int = 5
    citation_threshold: float = 0.5

    # Verification thresholds
    supported_threshold: float = 0.65
    hallucinated_threshold: float = 0.35

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.results_dir.mkdir(parents=True, exist_ok=True)
