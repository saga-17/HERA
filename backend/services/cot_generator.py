"""Chain-of-Thought generation using base VLM."""

from __future__ import annotations

import logging
import re
from typing import Optional

import torch
from PIL import Image

from backend.config import settings
from backend.models.model_manager import model_manager

logger = logging.getLogger(__name__)

COT_PROMPT = """You are a vision-language reasoning assistant. Analyze the image carefully and answer the question with step-by-step Chain-of-Thought reasoning.

Question: {question}

Provide your response in this exact format:

<OBSERVATIONS>
Describe what you see in the image. Reference visual elements explicitly.
</OBSERVATIONS>

<REASONING>
Step 1: [First reasoning step based on observations]
Step 2: [Second reasoning step]
Step 3: [Additional steps as needed]
</REASONING>

<CONCLUSION>
Final Answer: [Your concise grounded answer]
</CONCLUSION>

Be factual. Only describe what is actually visible in the image."""


DEMO_COT = """<OBSERVATIONS>
The image shows a domestic animal with four legs, fur, and a tail. It appears to be sitting in an indoor environment.
</OBSERVATIONS>

<REASONING>
Step 1: The animal has physical characteristics consistent with a dog — four legs, fur coat, and canine facial structure.
Step 2: The animal is positioned on what appears to be a floor surface in an indoor setting.
Step 3: Based on visible features, this is most likely a dog in a home environment.
</REASONING>

<CONCLUSION>
Final Answer: The image shows a dog in an indoor setting.
</CONCLUSION>"""


class CoTGenerator:
    """Generate Chain-of-Thought reasoning from image + question."""

    def generate(self, image: Image.Image, question: str) -> str:
        if settings.demo_mode:
            return DEMO_COT

        model, processor = model_manager.load_vlm()
        if model is None:
            return DEMO_COT

        prompt = COT_PROMPT.format(question=question)

        try:
            if "Qwen2-VL" in settings.vlm_model_id or "Qwen2.5-VL" in settings.vlm_model_id:
                return self._generate_qwen(model, processor, image, prompt)
            if "llava" in settings.vlm_model_id.lower():
                return self._generate_llava(model, processor, image, prompt)
            return self._generate_qwen(model, processor, image, prompt)
        finally:
            model_manager.unload_vlm()

    def _generate_qwen(self, model, processor, image: Image.Image, prompt: str) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if process_vision_info:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(text=[text], images=[image], return_tensors="pt")

        device = next(model.parameters()).device
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **model_manager.generate_kwargs())

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        return processor.decode(new_tokens, skip_special_tokens=True).strip()

    def _generate_llava(self, model, processor, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **model_manager.generate_kwargs())

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        parts = re.split(r"\nassistant\n", decoded, flags=re.IGNORECASE)
        return parts[-1].strip() if len(parts) > 1 else decoded


cot_generator = CoTGenerator()
