# ID2223-lab2

# ID2223 Lab 2 — LLM Fine-tuning + UI

## Task 1: Fine-tuning (what I did)
- Base model: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- Method: LoRA fine-tuning (trainable params ~5.64M, ~0.45%)
- Dataset: `mlabonne/FineTome-100k`, subset `train[:3000]`
- Training setup: batch size 2, gradient accumulation 4 (effective batch size 8), `max_steps=200`, lr `2e-4`, optimizer `adamw_8bit`
- Hardware: Google Colab Tesla T4 (15GB)
- Runtime: ~9.73 minutes (583.7s)
- Peak GPU reserved memory: ~2.203 GB

Artifacts (saved during training):
- Checkpoints: `outputs/checkpoint-150`, `outputs/checkpoint-200`
- LoRA adapter: `lora_model/adapter_model.safetensors` + `lora_model/adapter_config.json`

Notebook:
- See `notebooks/finetune.ipynb`

## Task 2: Improving performance / scalability (description)
### (A) Model-centric improvements (ideas)
- Tune LoRA hyperparameters: try `r` in {8,16} and different learning rates (e.g., 1e-4, 2e-4) to balance quality vs compute.
- Use sequence packing for short samples (`packing=True`) to increase throughput and reduce wasted padding.
- Try different training lengths (more steps / early stopping) to avoid underfitting or overfitting.

### (B) Data-centric improvements (ideas)
- Filter the dataset to match the target UI domain (keep high-quality instruction-following dialogues; remove noisy/irrelevant conversations).
- Add domain-specific examples aligned with the UI use case (e.g., tutoring style, concise answers, safety guidelines).
- Clean and deduplicate samples; enforce a consistent chat template to reduce format variance.

## UI (public URL)
- (to be added) Hugging Face Spaces / Streamlit URL
