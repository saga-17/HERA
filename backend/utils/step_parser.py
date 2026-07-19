"""Reasoning step segmentation and entity extraction."""

from __future__ import annotations

import re
from typing import Any

from backend.api.schemas import EntityExtraction


_TAG_PATTERN = re.compile(
    r"</?(?:SUMMARY|CAPTION|REASONING|CONCLUSION|OBSERVATIONS|ANSWER|answer|reasoning|conclusion)>"
)


def clean_reasoning_text(text: str) -> str:
    """Remove XML-like tags from CoT output."""
    text = _TAG_PATTERN.sub("", text)
    text = re.sub(r"\[Text Evidence \d+\]|\[Question Image \d+\]|\[Image ROI \d+\]", "", text)
    return text.strip()


def segment_reasoning_steps(cot_text: str) -> list[str]:
    """
    Decompose Chain-of-Thought into individual reasoning steps.
    Adapted from CaVe-VLM-CoT citation_injector.split_reasoning_into_claims.
    """
    cleaned = clean_reasoning_text(cot_text)

    # Try numbered steps first
    numbered = re.split(r"(?=\n\s*(?:Step\s*)?\d+[\.\):]\s)", cleaned, flags=re.IGNORECASE)
    if len(numbered) > 1:
        steps = []
        for seg in numbered:
            seg = re.sub(r"^\s*(?:Step\s*)?\d+[\.\):]\s*", "", seg.strip(), flags=re.IGNORECASE)
            if len(seg) >= 10:
                steps.append(seg.strip())
        if steps:
            return steps

    # Split on sentence boundaries and bullet points
    segments = re.split(r"(?<=[.!?])\s+|(?=- Step \d)|(?=\n-\s)|(?=\n\s*•\s)", cleaned)
    steps = []
    for seg in segments:
        seg = seg.strip()
        if len(seg) >= 15:
            steps.append(seg)

    if not steps and cleaned:
        steps = [cleaned]

    return steps


def extract_entities(step_text: str) -> EntityExtraction:
    """Extract entities, objects, attributes, relationships, and claims from a step."""
    text = step_text.lower()

    # Objects: nouns after "the/a/an" or capitalized words
    objects = list(set(re.findall(r"\b(?:the|a|an)\s+([a-z]+(?:\s+[a-z]+)?)", text)))
    cap_objects = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", step_text)
    objects.extend(cap_objects)
    objects = list(dict.fromkeys(objects))[:10]

    # Attributes: adjective patterns
    attributes = list(set(re.findall(
        r"\b(is|are|was|were|has|have|wearing|showing|contains?)\s+([a-z]+(?:\s+[a-z]+)?)",
        text,
    )))
    attributes = [a[1] if isinstance(a, tuple) else a for a in attributes][:10]

    # Relationships: prepositional / spatial
    relationships = list(set(re.findall(
        r"\b(on|in|under|above|beside|near|behind|in front of|next to|with|without)\s+(?:the\s+)?([a-z]+)",
        text,
    )))
    relationships = [f"{r[0]} {r[1]}" if isinstance(r, tuple) else r for r in relationships][:10]

    # Entities: all significant nouns
    entities = list(dict.fromkeys(objects + [a for a in attributes if isinstance(a, str)]))[:15]

    # Claims: the step itself as primary claim
    claims = [step_text.strip()] if step_text.strip() else []

    return EntityExtraction(
        entities=entities,
        objects=objects,
        attributes=[str(a) for a in attributes],
        relationships=[str(r) for r in relationships],
        claims=claims,
    )


def classify_hallucination_type(step_text: str, issue: str) -> str:
    """Classify hallucination category based on step content and verification issue."""
    text = (step_text + " " + issue).lower()

    if any(w in text for w in ("relationship", "next to", "on top", "beside", "between", "with")):
        return "relationship_hallucination"
    if any(w in text for w in ("color", "size", "shape", "wearing", "attribute", "property")):
        return "attribute_hallucination"
    if any(w in text for w in ("scene", "background", "environment", "setting", "location")):
        return "scene_hallucination"
    if any(w in text for w in ("because", "therefore", "implies", "reasoning", "logic")):
        return "reasoning_hallucination"
    if any(w in text for w in ("common sense", "typically", "usually", "generally")):
        return "commonsense_hallucination"
    if any(w in text for w in ("object", "not visible", "does not exist", "fabricated", "not in image")):
        return "object_hallucination"

    return "object_hallucination"


def parse_cot_sections(cot_text: str) -> dict[str, Any]:
    """Extract structured sections from VLM CoT output."""
    sections: dict[str, Any] = {}

    for tag in ("OBSERVATIONS", "REASONING", "CONCLUSION", "ANSWER"):
        match = re.search(rf"<{tag}>(.*?)</{tag}>", cot_text, re.IGNORECASE | re.DOTALL)
        if match:
            sections[tag.lower()] = match.group(1).strip()

    # Extract final answer line
    answer_match = re.search(
        r"(?:Final Answer|Answer|Conclusion):\s*(.+?)(?:\n|$)",
        cot_text,
        re.IGNORECASE,
    )
    if answer_match:
        sections["final_answer"] = answer_match.group(1).strip()
    elif "conclusion" in sections:
        sections["final_answer"] = sections["conclusion"]

    return sections
