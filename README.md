# HERA-VLM

**Hallucination Evidence Retrieval and Attribution for Vision Language Models**

Team 7 research project — built on [CaVe-VLM-CoT](https://anonymous.4open.science/r/cave-vlm-cot-XXXX) with a production web application for step-level hallucination detection, evidence grounding, and trustworthy VLM answers.

---

## Overview

HERA-VLM reduces hallucinations in vision-language model responses by:

1. Generating Chain-of-Thought reasoning from a base VLM
2. Decomposing reasoning into individual verifiable steps
3. Retrieving multimodal evidence (visual regions + textual sources)
4. Verifying each step against retrieved evidence
5. Detecting and classifying hallucinations
6. Attributing evidence to each step
7. Producing corrected, grounded final answers

---

## Architecture

```
Image + Question
       ↓
Base VLM Inference (CoT Generation)
       ↓
Reasoning Step Segmentation
       ↓
Evidence Retrieval (visual + textual)
       ↓
Step Verification (cross-encoder)
       ↓
Hallucination Detection
       ↓
Evidence Attribution
       ↓
Reasoning Correction
       ↓
Final Grounded Response
```

### Reused from CaVe-VLM-CoT

| CaVe Module | HERA Adaptation |
|---|---|
| `solver/solver.py` | `services/cot_generator.py` — CoT generation |
| `retriever/retriever.py` | `services/evidence_retriever.py` — cross-encoder + web search |
| `verifier/verifier.py` | `services/hallucination_detector.py` — step verification |
| `citation_injector/citation_injector.py` | `services/evidence_attributor.py` — citation injection |
| `utils.py` (State, RoiInfo) | `api/schemas.py` — Pydantic models |
| Verifier feedback loop | `services/reasoning_corrector.py` |

---

## Project Structure

```
hera-vlm/
├── backend/
│   ├── api/              # REST routes + Pydantic schemas
│   ├── services/         # ML services (CoT, retrieval, verification)
│   ├── pipelines/        # End-to-end HERA pipeline
│   ├── models/           # Lazy-loading VLM manager (4-bit)
│   └── utils/            # Step parser, image utilities
├── frontend/
│   ├── src/pages/        # Home, Inference, Result, About
│   ├── src/components/   # Step cards, pipeline status, navbar
│   ├── src/hooks/        # Result polling
│   └── src/services/     # API client
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── setup.sh / setup.bat
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend)
- CUDA GPU optional (RTX 4060 8GB supported with 4-bit quantization)

### Setup

**Windows:**
```bat
setup.bat
```

**Linux/macOS:**
```bash
chmod +x setup.sh && ./setup.sh
```

### Run (Demo Mode — no GPU required)

```bash
# Terminal 1 — Backend
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
set HERA_DEMO_MODE=true     # export on Linux/macOS
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Open **http://localhost:5173**

### Run with GPU (Qwen2-VL-7B)

```bash
# Copy and edit environment
cp .env.example .env
# Set HERA_DEMO_MODE=false

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API Documentation

Interactive docs: **http://localhost:8000/docs**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check + GPU status |
| `POST` | `/api/upload-image` | Upload image (multipart) |
| `POST` | `/api/ask` | Start async inference |
| `POST` | `/api/ask-sync` | Synchronous inference (testing) |
| `GET` | `/api/result/{id}` | Get pipeline result |
| `GET` | `/api/images/{id}` | Serve uploaded image |

### Example Response (Step)

```json
{
  "step": "The animal is a dog.",
  "evidence": "[Visual] Region at row 1, column 1 (conf=0.72)",
  "supported": true,
  "confidence": 0.94,
  "hallucination_type": "none",
  "attribution": "Supported by visual region step0_region0 and textual source (question)."
}
```

---

## Docker

```bash
docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:5173
- API docs: http://localhost:8000/docs

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HERA_DEMO_MODE` | `false` | Skip VLM loading, use demo responses |
| `HERA_VLM_MODEL_ID` | `Qwen/Qwen2-VL-7B-Instruct` | Base VLM model |
| `HERA_USE_4BIT` | `true` | 4-bit quantization (bitsandbytes) |
| `HERA_USE_CPU_FALLBACK` | `true` | Fall back to CPU if no GPU |
| `HERA_SUPPORTED_THRESHOLD` | `0.65` | Confidence threshold for "supported" |
| `HERA_HALLUCINATED_THRESHOLD` | `0.35` | Below this = hallucinated |

---

## Hallucination Categories

- **Object** — claiming objects not in the image
- **Attribute** — wrong colors, sizes, properties
- **Relationship** — incorrect spatial/logical relations
- **Scene** — wrong environment/background
- **Commonsense** — implausible world knowledge
- **Reasoning** — logical inference errors

---

## GPU Optimization (RTX 4060 8GB)

- 4-bit NF4 quantization via `bitsandbytes`
- Lazy model loading — one VLM at a time
- GPU memory freed after each inference
- Cross-encoder runs on GPU when available
- CPU fallback for development

---

## License

Apache-2.0 (inherited from CaVe-VLM-CoT base)

## Citation

```bibtex
@article{hera2026,
  title={HERA-VLM: Hallucination Evidence Retrieval and Attribution for Vision Language Models},
  author={Team 7},
  year={2026},
  note={Built on CaVe-VLM-CoT}
}
```
