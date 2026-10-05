# HERA-VLM Implementation Plan

## Phase 1: Repository Analysis

### CaVe-VLM-CoT Base (`cave-vlm-cot-062A/`)

Five-stage LangGraph agentic-RAG pipeline for ScienceQA:

```
Planner → Retriever → Solver → Citation Injector → Verifier → (retry loop)
```

**Key modules identified:**

| Module | File | Purpose |
|--------|------|---------|
| State schema | `utils.py` | Pydantic State with reasoning_steps, hallucination_details |
| Planner | `extractor/planner.py` | Query decomposition (Qwen2.5-7B) |
| Retriever | `retriever/retriever.py` | Hybrid BM25 + FAISS + web search + cross-encoder reranking |
| Solver | `solver/solver.py` | VLM CoT with [Question Image N] / [Text Evidence N] citations |
| Citation Injector | `citation_injector/citation_injector.py` | Post-hoc claim→evidence alignment |
| Verifier | `verifier/verifier.py` | Second VLM verifies citations, detects hallucinations |
| Prompts | `prompts.py` | SOLVER/VERIFIER prompt templates |
| Evaluations | `evaluations.py` | 26 metrics including CaVeScore |

### Gap Analysis for HERA-VLM

CaVe-VLM-CoT is designed for **ScienceQA multiple-choice** with a knowledge base. HERA-VLM needs:

- Single-image open QA (no choices required)
- Step-level (not answer-level) verification
- Visual region evidence retrieval from uploaded image
- Web dashboard with real-time pipeline status
- RTX 4060 8GB optimization (CaVe uses 32B verifier — too large)

---

## Phase 2: Proposed Architecture

### Backend Services (adapted from CaVe)

```
cot_generator.py       ← solver.py (CoT prompts, VLM inference)
evidence_retriever.py  ← retriever.py (cross-encoder, web search, grid regions)
hallucination_detector.py ← verifier.py (step-level scoring)
evidence_attributor.py ← citation_injector.py (inline citations)
reasoning_corrector.py ← verifier feedback loop (remove bad steps)
```

### Model Strategy (8GB VRAM)

| Role | CaVe Original | HERA Adaptation |
|------|--------------|-----------------|
| Solver/CoT | Llama-3.2V-11B | Qwen2-VL-7B-Instruct (4-bit) |
| Verifier | Qwen2.5-VL-32B | Cross-encoder (ms-marco-MiniLM) |
| Planner | Qwen2.5-7B | Rule-based entity extraction |
| Embeddings | all-MiniLM-L6-v2 | Same (reused) |

Single-model-at-a-time with lazy loading eliminates the need for multiple VLMs simultaneously.

---

## Phase 3: File Structure

```
hera-vlm/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   ├── routes.py          # POST /upload, /ask, GET /result, /health
│   │   └── schemas.py         # ReasoningStepResult, HeraResult, etc.
│   ├── services/
│   │   ├── cot_generator.py
│   │   ├── evidence_retriever.py
│   │   ├── hallucination_detector.py
│   │   ├── evidence_attributor.py
│   │   └── reasoning_corrector.py
│   ├── pipelines/
│   │   └── hera_pipeline.py   # Orchestrates all stages
│   ├── models/
│   │   └── model_manager.py   # 4-bit lazy loading
│   └── utils/
│       ├── step_parser.py     # From citation_injector claim splitting
│       └── image_utils.py     # Grid region extraction
├── frontend/
│   └── src/
│       ├── pages/             # Home, Inference, Result, About
│       ├── components/        # ReasoningStepCard, PipelineStatusPanel
│       ├── hooks/             # useResultPolling
│       └── services/          # API client
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Phase 4: Execution Flow

```
1. User uploads image → POST /api/upload-image → stored in data/uploads/
2. User submits question → POST /api/ask → background task starts
3. Pipeline stages (with status updates):
   a. cot_generator.generate(image, question) → original_cot
   b. step_parser.segment_reasoning_steps(cot) → [step1, step2, ...]
   c. For each step:
      - evidence_retriever.retrieve_for_step() → visual + textual evidence
      - hallucination_detector.verify_step() → status, confidence, type
   d. evidence_attributor.attribute_steps() → attributed_cot
   e. reasoning_corrector.correct() → corrected_cot + final_answer
4. Result saved → GET /api/result/{id}
5. Frontend polls until complete → navigates to Result page
```

---

## Phase 5: Reuse Mapping

| CaVe Function | HERA Usage |
|---------------|-----------|
| `split_reasoning_into_claims()` | `step_parser.segment_reasoning_steps()` |
| `cross_encoder.predict()` | `hallucination_detector._score_step()` |
| `web_search()` | `evidence_retriever._web_search()` |
| `inject_citations()` | `evidence_attributor.attribute_steps()` |
| `parse_hallucination_details()` | `hallucination_detector` + `classify_hallucination_type()` |
| `generate_targeted_feedback()` | `reasoning_corrector.correct()` |
| `prepare_question_images()` | `image_utils.load_image()` |
| `State.reasoning_steps` | `HeraResult.steps: list[ReasoningStepResult]` |

---

## Phase 6: Testing Plan

1. **Demo mode** — `HERA_DEMO_MODE=true`, no GPU, verify full UI flow
2. **API tests** — upload → ask-sync → result JSON schema validation
3. **GPU test** — Qwen2-VL-7B 4-bit on RTX 4060, monitor VRAM
4. **Hallucination demo** — ask about "glasses" on image without glasses → red step
5. **Docker** — `docker-compose up` end-to-end

---

## Status: Implemented

All phases complete. Run `setup.bat` (Windows) or `./setup.sh` (Linux) to start.
