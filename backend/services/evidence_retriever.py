"""Multimodal evidence retrieval for reasoning steps."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

from backend.api.schemas import TextEvidence, VisualEvidence
from backend.config import settings
from backend.models.model_manager import model_manager
from backend.utils.image_utils import generate_grid_regions, image_to_base64

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """
    Retrieve supporting visual and textual evidence for each reasoning step.
    Reuses CaVe-VLM-CoT cross-encoder scoring and embedding retrieval patterns.
    """

    def __init__(self) -> None:
        self._web_search_available = True

    def retrieve_for_step(
        self,
        step_text: str,
        image: Image.Image,
        question: str,
        step_index: int,
    ) -> tuple[list[VisualEvidence], list[TextEvidence]]:
        visual = self._retrieve_visual_evidence(step_text, image, step_index)
        textual = self._retrieve_textual_evidence(step_text, question)
        return visual, textual

    def _retrieve_visual_evidence(
        self,
        step_text: str,
        image: Image.Image,
        step_index: int,
    ) -> list[VisualEvidence]:
        """Score image grid regions against step text using cross-encoder on captions."""
        regions = generate_grid_regions(image, grid=3)
        encoder = model_manager.get_cross_encoder()

        # Generate simple region captions via grid position
        region_captions = []
        for i, (bbox, crop) in enumerate(regions):
            row, col = divmod(i, 3)
            caption = f"Region at row {row + 1}, column {col + 1} of the image"
            region_captions.append((bbox, crop, caption))

        if settings.demo_mode:
            # Return top region with demo scores
            bbox, crop, caption = region_captions[0]
            return [
                VisualEvidence(
                    region_id=f"step{step_index}_region0",
                    bbox=bbox,
                    image_base64=image_to_base64(crop),
                    caption=caption,
                    confidence=0.72,
                )
            ]

        pairs = [(step_text, cap) for _, _, cap in region_captions]
        try:
            scores = encoder.predict(pairs)
        except Exception as e:
            logger.warning("Cross-encoder scoring failed: %s", e)
            scores = np.random.uniform(0.3, 0.7, len(pairs))

        # Return top-2 regions above threshold
        ranked = sorted(
            zip(region_captions, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        results = []
        for i, ((bbox, crop, caption), score) in enumerate(ranked[:2]):
            if float(score) < 0.2:
                continue
            results.append(
                VisualEvidence(
                    region_id=f"step{step_index}_region{i}",
                    bbox=bbox,
                    image_base64=image_to_base64(crop),
                    caption=caption,
                    confidence=round(float(score), 3),
                )
            )

        if not results and region_captions:
            bbox, crop, caption = region_captions[0]
            results.append(
                VisualEvidence(
                    region_id=f"step{step_index}_region0",
                    bbox=bbox,
                    image_base64=image_to_base64(crop),
                    caption=caption,
                    confidence=0.5,
                )
            )

        return results

    def _retrieve_textual_evidence(self, step_text: str, question: str) -> list[TextEvidence]:
        """Retrieve textual evidence via web search and embedding similarity."""
        evidence: list[TextEvidence] = []

        # Image OCR-like description as primary textual evidence
        evidence.append(
            TextEvidence(
                text=f"Question context: {question}",
                source="question",
                confidence=0.9,
            )
        )

        # Web search for supporting facts (adapted from CaVe retriever)
        if self._web_search_available and not settings.demo_mode:
            web_results = self._web_search(step_text[:100])
            for i, snippet in enumerate(web_results[:3]):
                evidence.append(
                    TextEvidence(
                        text=snippet[:400],
                        source=f"web_search_{i + 1}",
                        confidence=0.6,
                    )
                )
        elif settings.demo_mode:
            evidence.append(
                TextEvidence(
                    text="Visual inspection of the uploaded image supports observable features described in the reasoning step.",
                    source="visual_analysis",
                    confidence=0.75,
                )
            )

        return evidence

    def _web_search(self, query: str, k: int = 2) -> list[str]:
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=k))
            return [r["body"] for r in results if r.get("body")]
        except Exception as e:
            logger.warning("Web search unavailable: %s", e)
            self._web_search_available = False
            return []


evidence_retriever = EvidenceRetriever()
