# distributed-llm-training

Fine-tuned Llama-3-8B across 4× A10G GPUs. Benchmarked Single GPU → DDP → FSDP → Ray Train.
Measured samples/sec, scaling efficiency, and GPU memory per rank at each stage.

---

## Why this project

Llama-3-8B in FP16 = 16GB. An A10G has 24GB. That leaves 8GB for activations, gradients, and optimizer states — not enough for a standard training run.

**The memory math:**

| Precision | Parameters | Bytes/param | Model size |
|-----------|------------|-------------|------------|
| FP32      | 8B         | 4 bytes     | 32GB       |
| FP16      | 8B         | 2 bytes     | 16GB       |

FP32 doesn't fit on a single A10G. FP16 fits but leaves almost no room.

**What each backend costs:**

| What          | Size   | Why                                              |
|---------------|--------|--------------------------------------------------|
| Model weights | 16GB   | 8B params × 2 bytes (FP16)                       |
| Gradients     | ~16GB  | Same shape as weights                            |
| Optimizer states (AdamW) | ~32GB | Momentum + variance per param = 2× model size |
| Activations   | varies | Depends on batch size and sequence length        |

Total without tricks: well over 24GB. This is the problem each stage solves.

---

## Stages

### Stage 1 — Single GPU Baseline
- Llama-3-8B on 1× A10G, FP16
- Without gradient checkpointing → OOM on first backward (expected)
- With gradient checkpointing → fits, but small batch size, slow

### Stage 2 — PyTorch DDP
- Full model replicated on all 4 GPUs (~16GB/rank)
- Gradients all-reduced after every backward
- Still needs gradient checkpointing
- Target scaling efficiency: 80–88% at 4 GPUs

### Stage 3 — PyTorch FSDP
- Model weights + gradients + optimizer states sharded across 4 GPUs
- Memory drops from ~18GB (DDP) to ~4–6GB per rank
- No gradient checkpointing needed — larger batch, better GPU utilization

### Stage 4 — Ray Train
- Same DDP training, different launcher
- No manual dist.init_process_group
- Native HPO integration via Ray Tune

---

## Benchmark Results

| Backend       | GPUs | samples/sec | Scaling eff. | Mem/rank  | Notes                          |
|---------------|------|-------------|--------------|-----------|--------------------------------|
| Single GPU    | 1    | —           | 100%         | ~18–20GB  | Grad checkpointing required    |
| DDP           | 2    | —           | —%           | ~18–20GB  | Still needs grad checkpointing |
| DDP           | 4    | —           | —%           | ~18–20GB  | Target: 80–88% efficiency      |
| FSDP          | 4    | —           | —%           | ~4–6GB    | No checkpointing, larger batch |
| Ray Train DDP | 4    | —           | ~DDP%        | ~18–20GB  | Simpler setup vs torchrun      |

*Numbers filled in after GPU runs on Vast.ai*

---

## Dataset

Alpaca — 52k instruction-response pairs (Stanford, 2023). Fine-tuned with response-only loss masking: loss computed on response tokens only, instruction tokens masked to -100.

## Stack

- PyTorch 2.2+ (DDP, FSDP built-in)
- HuggingFace Transformers + Datasets
- MLflow (experiment tracking)
- Ray Train + Ray Tune (Stage 4)
- Vast.ai 4× A10G (~$1.40/hr)
