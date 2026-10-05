# HERA Training Workflow

This folder is intentionally separate from the production inference pipeline.

## Purpose

- prepare and version evaluation data
- track model versions
- train LoRA/QLoRA checkpoints separately from runtime code
- evaluate before promotion to production

## Suggested structure

```text
training/
  dataset/
  prepare_dataset.py
  train_lora.py
  evaluate.py
  configs/
  checkpoints/
```

## Workflow

1. Run evaluation on fixed validation data.
2. Review failure categories.
3. Add only meaningful examples.
4. Train a LoRA/QLoRA checkpoint separately.
5. Evaluate on validation data.
6. Compare against the previous version.
7. Promote the model only if it improves key metrics.

## Important rule

Do not repeatedly fine-tune on the same prompt and do not silently replace the production model. The training loop must be data-driven and versioned.
