"""Lazy-loading VLM model manager with 4-bit quantization and GPU memory management."""

from __future__ import annotations

import gc
import logging
from typing import Any, Optional

import torch

from backend.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton manager: loads one VLM at a time, caches, frees GPU after use."""

    _instance: Optional[ModelManager] = None

    def __new__(cls) -> ModelManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._model: Any = None
        self._processor: Any = None
        self._model_id: str = ""
        self._device: str = "cpu"
        self._cross_encoder: Any = None
        self._embedding_model: Any = None
        self._initialized = True

    @property
    def device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def gpu_available(self) -> bool:
        return torch.cuda.is_available()

    def _resolve_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if settings.use_cpu_fallback:
            logger.warning("CUDA unavailable — falling back to CPU")
            return "cpu"
        raise RuntimeError("CUDA required but not available")

    def load_vlm(self, model_id: Optional[str] = None) -> tuple[Any, Any]:
        """Load VLM with 4-bit quantization. Returns (model, processor)."""
        model_id = model_id or settings.vlm_model_id

        if self._model is not None and self._model_id == model_id:
            return self._model, self._processor

        self.unload_vlm()
        self._device = self._resolve_device()

        if settings.demo_mode:
            logger.info("Demo mode — skipping VLM load")
            return None, None

        logger.info("Loading VLM %s on %s (4bit=%s)", model_id, self._device, settings.use_4bit)

        if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id:
            return self._load_qwen_vl(model_id)
        if "llava" in model_id.lower():
            return self._load_llava(model_id)
        if "Phi-3" in model_id or "phi-3" in model_id.lower():
            return self._load_phi3_vision(model_id)

        return self._load_qwen_vl(model_id)

    def _load_qwen_vl(self, model_id: str) -> tuple[Any, Any]:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig

        bnb_config = None
        if settings.use_4bit and self._device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if bnb_config:
            kwargs["quantization_config"] = bnb_config
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float32 if self._device == "cpu" else torch.float16
            kwargs["device_map"] = self._device

        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self._model_id = model_id
        return self._model, self._processor

    def _load_llava(self, model_id: str) -> tuple[Any, Any]:
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

        bnb_config = None
        if settings.use_4bit and self._device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if bnb_config:
            kwargs["quantization_config"] = bnb_config
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float32 if self._device == "cpu" else torch.float16
            kwargs["device_map"] = self._device

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self._model_id = model_id
        return self._model, self._processor

    def _load_phi3_vision(self, model_id: str) -> tuple[Any, Any]:
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

        bnb_config = None
        if settings.use_4bit and self._device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if bnb_config:
            kwargs["quantization_config"] = bnb_config
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float32 if self._device == "cpu" else torch.float16
            kwargs["device_map"] = self._device

        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self._model_id = model_id
        return self._model, self._processor

    def get_cross_encoder(self) -> Any:
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._cross_encoder = CrossEncoder(settings.cross_encoder_model, device=device)
        return self._cross_encoder

    def get_embedding_model(self) -> Any:
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(settings.embedding_model)
        return self._embedding_model

    def unload_vlm(self) -> None:
        """Free GPU memory after inference."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._model_id = ""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VLM unloaded, GPU memory freed")

    def generate_kwargs(self) -> dict[str, Any]:
        return {
            "max_new_tokens": settings.max_new_tokens,
            "temperature": 0.1,
            "do_sample": False,
        }


model_manager = ModelManager()
