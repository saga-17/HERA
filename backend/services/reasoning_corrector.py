"""Reasoning correction for hallucinated steps."""

from __future__ import annotations

import re

from backend.api.schemas import ReasoningStepResult, StepStatus


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
        if not supported_steps:
            return "I can't determine that reliably from the image."

        candidate_claim = self._select_supported_claim(supported_steps, question)
        answer = self._build_answer_from_question(question, candidate_claim, "")

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

    def _select_supported_claim(
        self,
        supported_steps: list[ReasoningStepResult],
        question: str = "",
    ) -> str:
        question_object = self._extract_question_object(question)
        question_lower = question.lower()
        action = self._extract_question_action(question)
        has_spatial_intent = self._spatial_parts(question) is not None
        scored: list[tuple[int, float, int, str]] = []

        for step in supported_steps:
            cleaned = self._clean_step_text(step.step)
            if cleaned and not self._looks_like_metadata(cleaned):
                relevance = 0
                if question_object and self._contains_phrase(cleaned, question_object):
                    relevance += 3
                if re.match(r"^how many\b", question_lower) and self._extract_count_phrase(cleaned):
                    relevance += 3
                if "color" in question_lower and self._extract_color(cleaned):
                    relevance += 3
                if action and re.search(rf"\b{re.escape(action)}\b", cleaned, re.IGNORECASE):
                    relevance += 3
                if has_spatial_intent and self._spatial_parts(cleaned):
                    relevance += 3
                scored.append((relevance, float(step.confidence), -step.step_index, cleaned))

        if not scored:
            return ""

        return max(scored)[3]

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
        candidate = self._clean_step_text(claim) or self._clean_answer_text(original_answer)
        if not candidate:
            return ""

        q_lower = normalized_question.lower()
        if self._spatial_parts(normalized_question):
            polarity = "No" if self._is_negative_claim(candidate) else "Yes"
            return self._render_relationship_answer(normalized_question, polarity, candidate)

        if re.match(r"^(is|are|does|do|did|was|were|can|could|has|have)\b", q_lower):
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
            return self._clean_answer_text(candidate)

        if self._extract_question_action(q_lower):
            return self._clean_answer_text(candidate)

        if re.match(r"^(describe|summari[sz]e)\b", q_lower):
            return self._clean_answer_text(candidate)

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
        claim = self._clean_answer_text(candidate)
        claim = re.sub(r"^(?:yes|right)[,.:]?\s*", "", claim, flags=re.IGNORECASE)

        if not question_object:
            return f"No. {claim}" if self._is_negative_claim(claim) else claim

        if self._is_target_negated(claim, question_object):
            return f"No. {claim}"

        medical_question = self._is_medical_question(question)
        if self._contains_phrase(claim, question_object):
            if medical_question:
                if self._has_uncertainty(claim):
                    return f"{claim.rstrip('. ')}. An image alone cannot establish a diagnosis."
                return f"The image may show {question_object}, but an image alone cannot establish a diagnosis."
            if self._has_uncertainty(claim):
                return f"{claim.rstrip('. ')}. The image alone does not establish this definitively."
            return f"Yes. {claim}"

        alternative = self._extract_object_name(claim)
        if medical_question:
            return f"{claim.rstrip('. ')}. The image alone cannot establish a diagnosis."
        if alternative and self._is_subject_identification(claim):
            last_word = question_object.split()[-1]
            article = "" if last_word.endswith("s") else "an" if last_word[:1] in "aeiou" else "a"
            target_phrase = f"{article} {question_object}".strip()
            if self._has_uncertainty(claim):
                return f"{claim.rstrip('. ')}. The image does not establish whether it is {target_phrase}."
            return f"No. {claim.rstrip('. ')}, not {target_phrase}."

        return f"The image shows {claim.rstrip('. ')}, but it does not establish whether it is {question_object}."

    def _render_relationship_answer(self, question: str, polarity: str, candidate: str) -> str:
        parts = self._spatial_parts(question) or self._spatial_parts(candidate)
        if not parts:
            return self._clean_answer_text(candidate)
        subject, relation, obj = parts
        if polarity == "Yes":
            return f"Yes, {subject} is {relation} {obj}."
        return f"No. {subject} is not {relation} {obj}."

    def _spatial_parts(self, text: str) -> tuple[str, str, str] | None:
        relations = ("on top of", "in front of", "next to", "beside", "behind", "under", "above")
        stripped = re.sub(
            r"^\s*(?:is|are|was|were)\s+",
            "",
            (text or "").strip().rstrip("?."),
            flags=re.IGNORECASE,
        )
        lowered = stripped.lower()
        for relation in relations:
            index = lowered.find(relation)
            if index == -1:
                continue
            subject = stripped[:index].strip(" ,")
            obj = stripped[index + len(relation):].strip(" ,")
            subject = re.sub(r"^(?:yes,?\s+|no\.?\s+)", "", subject, flags=re.IGNORECASE)
            subject = re.sub(r"\s+(?:is|are|was|were)\s+$", "", subject, flags=re.IGNORECASE).strip()
            obj = re.sub(r"^(?:is|are|was|were)\s+", "", obj, flags=re.IGNORECASE).strip()
            if subject and obj:
                return subject.lower(), relation, obj.lower()
        return None

    def _extract_question_action(self, question: str) -> str:
        match = re.search(
            r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:[a-z-]+\s+)*?([a-z]+ing)\b",
            question or "",
            re.IGNORECASE,
        )
        return match.group(1).lower() if match else ""

    def _contains_phrase(self, text: str, phrase: str) -> bool:
        words = (phrase or "").lower().split()
        if not words:
            return False
        prefix = r"\b" + r"\s+".join(re.escape(word) for word in words[:-1])
        final_word = words[-1]
        final_forms = {final_word}
        if final_word.endswith("ies") and len(final_word) > 3:
            final_forms.add(final_word[:-3] + "y")
        elif final_word.endswith("y"):
            final_forms.add(final_word[:-1] + "ies")
        elif final_word.endswith("s") and not final_word.endswith("ss"):
            final_forms.add(final_word[:-1])
        else:
            final_forms.update((final_word + "s", final_word + "es"))
        ending = "(?:" + "|".join(re.escape(form) for form in final_forms) + ")"
        pattern = (prefix + r"\s+" if prefix else r"\b") + ending + r"\b"
        return re.search(pattern, text or "", re.IGNORECASE) is not None

    def _is_target_negated(self, text: str, target: str) -> bool:
        escaped_target = r"\s+".join(re.escape(word) for word in target.split())
        return re.search(
            rf"\b(?:not|no|never|without|isn't|aren't|doesn't|don't)\b.{{0,20}}\b{escaped_target}\b",
            text or "",
            re.IGNORECASE,
        ) is not None

    def _is_negative_claim(self, text: str) -> bool:
        return re.search(r"\b(?:no|not|never|without|isn't|aren't|doesn't|don't)\b", text or "", re.IGNORECASE) is not None

    def _is_subject_identification(self, text: str) -> bool:
        normalized = re.sub(r"^(?:yes|no)[,.:]?\s*", "", text or "", flags=re.IGNORECASE)
        patterns = (
            r"^(?:(?:the\s+)?(?:image|picture|photo|animal|object|subject)|this|it)\b.{0,40}\b(?:is|appears\s+to\s+be|looks\s+like|shows?|depicts?)\s+(?:a|an|the)\b",
            r"^(?:a|an|the)\s+[a-z][a-z -]+\s+(?:is|are)\s+(?:visible|present|shown)\b",
        )
        return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns)

    def _is_medical_question(self, text: str) -> bool:
        terms = r"\b(?:medical|diagnos\w*|mri|ct\s+scan|x-ray|radiograph|lesion|tumou?r|cancer|fracture|biopsy|brain)\b"
        return re.search(terms, text or "", re.IGNORECASE) is not None

    def _has_uncertainty(self, text: str) -> bool:
        terms = r"\b(?:may|might|possibly|possible|likely|probably|appears?|seems?|suspected|uncertain|consistent\s+with|blurry|blurred|indistinct|unclear|ambiguous|unrecognizable)\b"
        return re.search(terms, text or "", re.IGNORECASE) is not None

    def _build_multiple_object_answer(self, candidate: str) -> str:
        phrase = self._extract_list_phrase(candidate)
        if phrase:
            return f"The image contains {phrase}."
        cleaned = self._clean_answer_text(candidate)
        if cleaned:
            return cleaned if cleaned.lower().startswith("the image contains") else f"The image contains {cleaned}."
        return "I can't determine that reliably from the image."

    def _extract_subject_from_question(self, question: str) -> str:
        match = re.search(
            r"what\s+color\s+(?:is|are)\s+(?:the|a|an)?\s*([a-z][a-z\s-]*?)\??$",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(r"what\s+(?:kind|type)\s+of\s+([a-z][a-z\s-]*)", question, flags=re.IGNORECASE)
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
        normalized = (question or "").strip().rstrip("?.! ").lower()
        match = re.search(r"\b(?:a|an)\s+([a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*)*)$", normalized)
        if match:
            return self._normalize_noun_phrase(match.group(1))
        remainder = re.sub(r"^(?:is|are|does|do|did|was|were|can|could|has|have)\s+", "", normalized)
        remainder = re.sub(r"^(?:(?:this|it|that|there|any|the)\s+)+", "", remainder)
        words = re.findall(r"[a-z][a-z0-9-]*", remainder)
        return words[-1] if words else ""

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

        leading_noun = re.match(r"^(?:a|an|the)\s+([a-z][a-z-]*)\b", candidate, flags=re.IGNORECASE)
        if leading_noun:
            return leading_noun.group(1).lower()

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
        match = re.search(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-z][a-z\s-]*?)(?:\.|,|$)",
            cleaned,
            flags=re.IGNORECASE,
        )
        if match:
            count = self._extract_count(match.group(1))
            subject = self._normalize_noun_phrase(match.group(2))
            subject = re.sub(
                r"\s+(?:visible|present|shown|in the (?:image|scene|frame))\b.*$",
                "",
                subject,
                flags=re.IGNORECASE,
            ).strip()
            if subject and subject.lower() not in {"there", "are"}:
                if count == "1" and subject.endswith("s") and not subject.endswith("ss"):
                    subject = subject[:-1]
                elif count != "1" and not subject.endswith("s"):
                    subject += "s"
                return f"{count} {subject}"
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

    def _looks_like_multi_object_claim(self, text: str) -> bool:
        cleaned = self._clean_step_text(text)
        if not cleaned:
            return False
        return (
            "," in cleaned
            or " and " in cleaned.lower()
            or " are visible" in cleaned.lower()
            or " are present" in cleaned.lower()
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
