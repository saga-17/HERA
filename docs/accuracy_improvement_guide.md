# HERA Accuracy Improvement Workflow

This project follows an evaluation-driven loop instead of repeated blind prompting.

## 1. Run a fixed evaluation set

Use the evaluation data under `backend/evaluation/` and keep a small but representative validation set fixed across iterations.

## 2. Collect failure modes

Categorize every incorrect prediction using the taxonomy in `backend/evaluation/taxonomy.py`.

Examples:
- `OBJECT_HALLUCINATION`
- `ATTRIBUTE_HALLUCINATION`
- `RELATIONSHIP_HALLUCINATION`
- `COUNTING_ERROR`
- `SPATIAL_ERROR`
- `INSUFFICIENT_EVIDENCE`

## 3. Update the dataset

Add strong examples only when they reveal a genuine gap. Do not repeatedly retrain on the exact same example set without a reason.

## 4. Improve the prompt and verification logic

The project improves:
- prompt engineering in `backend/services/cot_generator.py`
- retrieval in `backend/services/evidence_retriever.py`
- verification in `backend/services/hallucination_detector.py`
- final answer generation in `backend/services/reasoning_corrector.py`

## 5. Evaluate before training

Baseline metrics are computed from the `compute_metrics` helper in `backend/evaluation/metrics.py`.

## 6. Train LoRA/QLoRA only when the dataset is meaningful

The project supports a separate training structure for later use, without modifying the production model automatically.

Recommended structure:

```text
training/
  dataset/
  prepare_dataset.py
  train_lora.py
  evaluate.py
  configs/
  checkpoints/
```

## 7. Keep a model version record

Model versions should be tracked separately, for example:
- `base-qwen2-vl-7b`
- `hera-lora-v1`
- `hera-lora-v2`

## 8. Compare against the previous checkpoint

Only keep a new model if it improves validation metrics without degrading other categories.

## 9. Safety rule

When evidence is uncertain, prefer:

> I can't determine that reliably from the image.

This is better than a confident hallucination.
