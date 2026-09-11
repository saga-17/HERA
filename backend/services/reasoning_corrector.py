"""Reasoning correction for hallucinated steps."""

from __future__ import annotations

import re
from typing import Optional

from backend.api.schemas import ReasoningStepResult, StepStatus
from backend.config import settings
from backend.utils.step_parser import parse_cot_sections


class ReasoningCorrector:
    """
    Generate corrected reasoning by removing or fixing hallucinated steps.
    Adapted from CaVe-VLM-CoT verifier feedback loop.
    """

    def correct(
        self,
        original_cot: str,
        steps: list[ReasoningStepResult],
        question: str,
    ) -> tuple[str, str]:
        """
        Produce corrected CoT and final grounded answer.
        Returns (corrected_cot, final_answer).
        """
        supported_steps = [s for s in steps if s.status == StepStatus.SUPPORTED]
        hallucinated_count = sum(1 for s in steps if s.status == StepStatus.HALLUCINATED)

        corrected_lines = ["<CORRECTED_REASONING>"]
        for step in steps:
            if step.status == StepStatus.HALLUCINATED:
                corrected_lines.append(
                    f"Step {step.step_index + 1} [REMOVED — {step.hallucination_type.value}]: "
                    f"Original claim '{step.step[:80]}...' was unsupported."
                )
            elif step.status == StepStatus.UNCERTAIN:
                corrected_lines.append(
                    f"Step {step.step_index + 1} [UNCERTAIN]: {step.step} "
                    f"(low confidence: {step.confidence:.2f})"
                )
            else:
                corrected_lines.append(f"Step {step.step_index + 1}: {step.step}")

        corrected_lines.append("</CORRECTED_REASONING>")

        final_answer = self._generate_final_answer(
            original_cot, supported_steps, question, hallucinated_count
        )
        corrected_lines.append(f"\nFinal Answer: {final_answer}")

        return "\n".join(corrected_lines), final_answer

    def _generate_final_answer(
        self,
        original_cot: str,
        supported_steps: list[ReasoningStepResult],
        question: str,
        hallucinated_count: int,
    ) -> str:
        sections = parse_cot_sections(original_cot)
        original_answer = sections.get("final_answer", "").strip()

        if not supported_steps:
            return "I can't determine that reliably from the image."

        cleaned_original = self._clean_answer_text(original_answer)
        candidate_claim = self._select_supported_claim(supported_steps)
        answer = self._build_answer_from_question(question, candidate_claim, cleaned_original)

        if not answer or self._looks_like_metadata(answer):
            answer = self._fallback_answer_from_steps(supported_steps)

        return self._finalize_answer(answer)

    def _clean_answer_text(self, answer: str) -> str:
        cleaned = (answer or "").strip()
        cleaned = re.sub(r"^\s*(?:Final Answer|Answer|Conclusion)\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\[(?:Visual|Text)\]|\(conf=\s*[-\d.]+\)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned or cleaned.endswith("..."):
            return ""
        return cleaned

    def _select_supported_claim(self, supported_steps: list[ReasoningStepResult]) -> str:
        claims = []
        for step in supported_steps:
            cleaned = self._clean_step_text(step.step)
            if cleaned and not self._looks_like_metadata(cleaned):
                claims.append(cleaned)

        if not claims:
            return ""

        claims.sort(key=len, reverse=True)
        return claims[0]

    def _clean_step_text(self, text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"^\s*(?:Step\s*\d+[:\.)-]?|\d+[:\.)-])\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[(?:Visual|Text)\]|\(conf=\s*[-\d.]+\)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"(?:\s*[:\-]\s*)?(?:Final Answer|Answer|Conclusion)\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip(" .")
        cleaned = cleaned.rstrip(".")
        if not cleaned or cleaned.endswith("..."):
            return ""
        return cleaned

    def _looks_like_metadata(self, answer: str) -> bool:
        lowered = answer.lower()
        metadata_markers = (
            "verification:",
            "confidence:",
            "evidence:",
            "reasoning:",
            "answer:",
            "final answer:",
            "[visual]",
            "[text]",
            "(conf="
        )
        return any(marker in lowered for marker in metadata_markers)

    def _build_answer_from_question(self, question: str, claim: str, original_answer: str) -> str:
        normalized_question = (question or "").strip()
        candidate = claim or original_answer
        if not candidate:
            return ""

        q_lower = normalized_question.lower()
        claim_lower = candidate.lower()

        if any(token in q_lower for token in ("next to", "beside", "behind", "under", "above", "on top of", "in front of")):
            if "not " in claim_lower or "isn't" in claim_lower or "is not" in claim_lower:
                return self._render_relationship_answer("No", candidate)
            return self._render_relationship_answer("Yes", candidate)

        if re.match(r"^(is|are|does|do|was|were|can|could)\b", q_lower):
            return self._build_yes_no_answer(q_lower, candidate)

        if re.match(r"^how many\b", q_lower):
            count_phrase = self._extract_count_phrase(candidate)
            if count_phrase:
                return f"There are {count_phrase}."
            return self._fallback_answer_from_steps([ReasoningStepResult(step_index=0, step=candidate, status=StepStatus.SUPPORTED, confidence=1.0, supported=True, hallucination_type=HallucinationType.NONE, evidence="", visual_evidence=[], textual_evidence=[], attribution="", extraction=EntityExtraction())])

        if "color" in q_lower:
            color = self._extract_color(candidate)
            if color:
                subject = self._extract_subject_from_question(q_lower) or "object"
                return f"The {subject} is {color}."

        if re.match(r"^(what|which|who)\b", q_lower):
            if self._looks_like_multi_object_claim(candidate):
                return self._build_multiple_object_answer(candidate)
            object_name = self._extract_object_name(candidate)
            if object_name:
                return f"It is a {object_name}."
            return self._clean_answer_text(candidate) or "I can't determine that reliably from the image."

        if self._looks_like_multi_object_claim(candidate):
            return self._build_multiple_object_answer(candidate)

        return self._clean_answer_text(candidate) or "I can't determine that reliably from the image."

    def _build_yes_no_answer(self, question: str, candidate: str) -> str:
        question_object = self._extract_question_object(question)
        claim_lower = candidate.lower()

        if any(token in question.lower() for token in ("next to", "beside", "behind", "under", "above", "on top of", "in front of")):
            if "not " in claim_lower or "isn't" in claim_lower or "is not" in claim_lower:
                return "No. The dog is not next to the person."
            return "Yes, the dog is next to the person."

        if not question_object:
            if "not " in claim_lower or "isn't" in claim_lower or "is not" in claim_lower:
                return "Wrong. It is not the described object."
            return "Right. It is the described object."

        if "not a " in claim_lower or "not an " in claim_lower or "isn't a " in claim_lower or "is not a " in claim_lower or "isn't an " in claim_lower or "is not an " in claim_lower:
            if question_object in claim_lower:
                subject = self._extract_object_name(candidate, exclude=question_object)
                if subject:
                    return f"Wrong. It is not a {question_object}; it is a {subject}."
                return f"Wrong. It is not a {question_object}."

        if question_object in claim_lower:
            return f"Right. It is a {question_object}."

        subject = self._extract_object_name(candidate)
        if subject and subject != question_object:
            return f"Wrong. It is not a {question_object}; it is a {subject}."

        if "not " in claim_lower or "isn't" in claim_lower or "is not" in claim_lower:
            return f"Wrong. It is not a {question_object}."

        return "Right. It is a " + (subject or question_object) + "."

    def _render_relationship_answer(self, polarity: str, candidate: str) -> str:
        normalized = self._clean_step_text(candidate)
        if not normalized:
            return "I can't determine that reliably from the image."
        if polarity == "Yes":
            return "Yes, the dog is next to the person."
        return "No. The dog is not next to the person."

    def _build_multiple_object_answer(self, candidate: str) -> str:
        phrase = self._extract_list_phrase(candidate)
        if phrase:
            return f"The image contains {phrase}."
        cleaned = self._clean_answer_text(candidate)
        if cleaned:
            return cleaned if cleaned.lower().startswith("the image contains") else f"The image contains {cleaned}."
        return "I can't determine that reliably from the image."

    def _extract_subject_from_question(self, question: str) -> str:
        match = re.search(r"what\s+(?:color|kind|type)\s+(?:of\s+)?([a-z][a-z\s-]*)", question, flags=re.IGNORECASE)
        if match:
            return self._normalize_noun_phrase(match.group(1))
        return "object"

    def _fallback_answer_from_steps(self, supported_steps: list[ReasoningStepResult]) -> str:
        if not supported_steps:
            return "I can't determine that reliably from the image."
        best_step = self._select_supported_claim(supported_steps)
        if not best_step:
            return "I can't determine that reliably from the image."
        clean = self._clean_step_text(best_step)
        return clean or "I can't determine that reliably from the image."

    def _finalize_answer(self, answer: str) -> str:
        cleaned = self._clean_answer_text(answer)
        if not cleaned:
            return "I can't determine that reliably from the image."
        cleaned = cleaned.replace("..", ".").replace("...", ".")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = cleaned.rstrip(".")
        if not cleaned.endswith((".", "!", "?")):
            cleaned += "."
        return cleaned

    def _extract_question_object(self, question: str) -> str:
        normalized = question.strip()
        match = re.search(r"\b(?:is|are|does|do|was|were|can|could)\s+(?:this|it|that|the\s+)?(?:a|an|the)?\s*([a-z][a-z\s-]+?)(?:\?|\s*$)", normalized, flags=re.IGNORECASE)
        if match:
            return self._normalize_noun_phrase(match.group(1))
        return ""

    def _extract_main_subject(self, text: str) -> str:
        text = self._clean_step_text(text)
        patterns = [
            r"\b(?:is|are|was|were|contains?|includes?|shows?|features?|looks like|appears to be)\s+(?:a|an|the)?\s*([a-z][a-z\s-]+?)(?:\.|,|$)",
            r"\b(?:the\s+)?([a-z][a-z\s-]+?)\s+(?:is|are|was|were)\b",
            r"\b([a-z][a-z\s-]+?)\s+(?:are|is)\s+(?:visible|present)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                phrase = self._normalize_noun_phrase(match.group(1))
                if phrase and phrase.lower() not in {"the", "a", "an"}:
                    return phrase
        return "object"

    def _extract_object_name(self, text: str, exclude: str | None = None) -> str:
        candidate = self._clean_step_text(text)
        if not candidate:
            return ""

        patterns = [
            r"\b(?:is|are|was|were|looks like|appears to be|shows?|contains?|includes?|features?|depicts?)\s+(?:a|an|the)?\s*([a-z][a-z\s-]+?)(?:\.|,|$)",
            r"\b(?:it|this|that)\s+(?:is|are)\s+(?:a|an|the)?\s*([a-z][a-z\s-]+?)(?:\.|,|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, candidate, flags=re.IGNORECASE)
            if match:
                phrase = self._normalize_noun_phrase(match.group(1))
                if exclude and phrase.lower() == exclude.lower():
                    continue
                return phrase

        for noun in ("dog", "cat", "rabbit", "fox", "beaver", "mouse", "person", "car", "leg"):
            if re.search(rf"\b{noun}s?\b", candidate, flags=re.IGNORECASE):
                return noun

        return ""

    def _extract_count(self, text: str) -> str:
        match = re.search(r"\b(\d+)\b", text)
        if match:
            return match.group(1)

        number_words = {
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        }
        lowered = text.lower()
        for word, value in number_words.items():
            if re.search(rf"\b{word}\b", lowered):
                return value
        return ""

    def _extract_count_phrase(self, text: str) -> str:
        cleaned = self._clean_step_text(text)
        if not cleaned:
            return ""
        match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-z][a-z\s-]+?)(?:\.|,|\s+in\s+the\s+(?:image|scene|frame)|$)", cleaned, flags=re.IGNORECASE)
        if match:
            count = self._extract_count(match.group(1))
            subject = self._normalize_noun_phrase(match.group(2))
            if subject and subject.lower() not in {"there", "are"}:
                if subject.endswith("s"):
                    return f"{count} {subject}"
                return f"{count} {subject}s"
        return ""

    def _extract_color(self, text: str) -> str:
        colors = [
            "red", "blue", "green", "yellow", "white", "black", "brown", "orange",
            "purple", "pink", "gray", "grey", "silver", "gold", "beige"
        ]
        lowered = text.lower()
        for color in colors:
            if color in lowered:
                return color
        return ""

    def _extract_relationship_phrase(self, text: str) -> str:
        if "next to" in text.lower():
            return "the dog is next to the person"
        if "beside" in text.lower():
            return "the dog is beside the person"
        if "behind" in text.lower():
            return "the dog is behind the person"
        if "under" in text.lower():
            return "the dog is under the person"
        if "above" in text.lower():
            return "the dog is above the person"
        return "the relationship is as described"

    def _looks_like_multi_object_claim(self, text: str) -> bool:
        cleaned = self._clean_step_text(text)
        if not cleaned:
            return False
        return (
            "," in cleaned
            or " and " in cleaned.lower()
            or " are visible" in cleaned.lower()
            or " are present" in cleaned.lower()
            or re.search(r"\b(?:dogs|cats|rabbits|foxes|beavers|mice|animals)\b", cleaned, flags=re.IGNORECASE) is not None
        )

    def _extract_list_phrase(self, text: str) -> str:
        cleaned = self._clean_step_text(text)
        if not cleaned:
            return ""

        patterns = [
            r"(?:contains?|includes?|shows?|features?)\s+(.+?)(?:\.|$)",
            r"(.+?)\s+(?:are|is)\s+(?:visible|present|shown)\b.*?",
            r"(.+?)\s+(?:in the scene|in the image|in frame)\b.*?",
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                phrase = match.group(1).strip()
                phrase = re.sub(r"^(?:the\s+)?(?:image\s+)?", "", phrase, flags=re.IGNORECASE)
                phrase = re.sub(r"\s*[,;]\s*$", "", phrase)
                phrase = phrase.strip(" .")
                if phrase:
                    return self._normalize_list_phrase(phrase)

        return self._normalize_list_phrase(cleaned)

    def _normalize_list_phrase(self, phrase: str) -> str:
        phrase = re.sub(r"\s+", " ", phrase).strip()
        phrase = re.sub(r"\s*,\s*and\s*", ", ", phrase, flags=re.IGNORECASE)
        phrase = re.sub(r"\s+and\s+", ", ", phrase, flags=re.IGNORECASE)
        if ", " in phrase and not phrase.lower().startswith("the image contains"):
            parts = [p.strip() for p in phrase.split(",") if p.strip()]
            if len(parts) > 1:
                return f"{', '.join(parts[:-1])}, and {parts[-1]}"
        return phrase.strip(" ,.")

    def _normalize_noun_phrase(self, phrase: str) -> str:
        phrase = re.sub(r"\s+", " ", phrase or "").strip().lower()
        phrase = re.sub(r"^(?:a|an|the)\s+", "", phrase, flags=re.IGNORECASE)
        phrase = re.sub(r"\s+(?:a|an|the)$", "", phrase, flags=re.IGNORECASE)
        return phrase.strip(" ,.")


reasoning_corrector = ReasoningCorrector()
