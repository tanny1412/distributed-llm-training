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
- Llama-3-8B on 1× RTX 4090 (24GB), FP16
- Without gradient checkpointing → OOM during forward pass (activations fill 24GB)
- With gradient checkpointing + batch_size=1 → still OOM on backward
- Root cause: 16GB weights + 16GB gradients = 32GB minimum, exceeds 24GB HBM
- Conclusion: full fine-tuning of 8B on a single 24GB GPU is not feasible without QLoRA/LoRA

### Stage 2 — PyTorch DDP
- Full model replicated on all 4 GPUs (~16GB/rank)
- Crashed at `DDP.__init__` — pre-allocates gradient buffer same size as model (15GB). 15GB weights + 15GB buffer = 30GB > 24GB before training even starts.

### Stage 3 — PyTorch FSDP
- Model weights + gradients + optimizer states sharded across 4 GPUs
- 5 attempts before stable training:
  1. Hardcoded `LlamaDecoderLayer` import → FSDP didn't shard (22GB/rank). Newer transformers wraps the class — `isinstance()` always failed.
  2. Fixed with `type(model.model.layers[0])` runtime detection → 11GB/rank, sharding confirmed
  3. No gradient checkpointing → OOM on backward (activations 13GB/rank)
  4. Added gradient checkpointing → training started but `loss nan` from step 10. FP16 overflow in forward pass.
  5. Switched to BF16 → loss stable, training running

### Stage 4 — Ray Train
- Same DDP training, different launcher
- No manual dist.init_process_group
- Native HPO integration via Ray Tune

---

## Benchmark Results

| Backend       | GPUs | samples/sec | Scaling eff. | Mem/rank  | Notes                          |
|---------------|------|-------------|--------------|-----------|--------------------------------|
| Single GPU    | 1    | OOM         | —            | >24GB     | 16GB weights + 16GB grads > 24GB HBM |
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
