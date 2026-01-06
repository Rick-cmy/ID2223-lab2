# ID2223 Lab 2 — LLM Fine-tuning + Public UI

This repository contains the code, training notebook, and a deployed UI for Lab 2.

## Deliverables
- Source code (this GitHub repo)
- Task 2 description (this README)
- Public UI: https://huggingface.co/spaces/cmyRick/ID2223-Lab2

---

## Task 1 — Fine-tuning

### Setup
- Base model: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- Fine-tuning method: LoRA (trainable params ≈ 5.64M, ≈ 0.45%)
- Dataset: `mlabonne/FineTome-100k` (subset: `train[:3000]`)
- Hardware: Google Colab Tesla T4 (15GB)

### Training configuration
- Batch size: 2
- Gradient accumulation: 4 (effective batch size: 8)
- Max steps: 200
- Learning rate: 2e-4
- Optimizer: `adamw_8bit`
- LR scheduler: linear

### Results (runtime / memory)
- Training time: ~583.7s (~9.73 min)
- Peak reserved GPU memory: ~2.203 GB

### Outputs
- Training notebook: `notebooks/finetune.ipynb`
- Checkpoints (examples): `outputs/checkpoint-150`, `outputs/checkpoint-200`
- LoRA adapter: `lora_model/adapter_model.safetensors` + `lora_model/adapter_config.json`

---

## Task 2 — Improving performance / scalability (description)

### (A) Model-centric improvements
- Hyperparameter search: tune LoRA rank `r` (e.g., 8/16) and learning rate (e.g., 1e-4/2e-4) to balance quality vs training cost.
- Enable sequence packing for short samples (`packing=True`) to reduce padding waste and improve throughput.
- Adjust training length (more steps) and use early stopping / validation-based selection to avoid underfitting or overfitting.

### (B) Data-centric improvements
- Data filtering: keep high-quality instruction-following dialogues and remove noisy or off-topic conversations to better match the target UI.
- Add domain-aligned examples (e.g., tutoring style, concise answers, safety constraints) to shape response behavior.
- Deduplicate and normalize formatting (consistent chat template) to reduce dataset variance and stabilize training.

---

## UI
- Hugging Face Spaces (public): https://huggingface.co/spaces/cmyRick/ID2223-Lab2
