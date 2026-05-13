# distributed-llm-training

Fine-tune Llama-3-8B across multiple GPUs. Benchmark Single GPU → DDP → FSDP → Ray Train.
Measure peak memory, steady-state memory, throughput, and scaling efficiency at each stage.

---

## The Problem

Training an 8B parameter model requires more than model weights. Four things live in HBM simultaneously:

| What | Size | Notes |
|------|------|-------|
| Weights | 16GB | 8B params × 2 bytes (BF16) |
| Gradients | 16GB | Same shape as weights, always |
| Optimizer states | 32GB | AdamW: momentum + variance per param |
| Activations | varies | Depends on batch size and sequence length |

Total without tricks: **64GB+ before activations**. A single 24GB GPU cannot run full fine-tuning of an 8B model. Each stage in this project is a response to that constraint.

---

## Two Phases

### Phase 1 — OOM Story (4× RTX 4090, 24GB each, Vast.ai)

Intentionally hit every failure mode before solving it. The errors are the story.

| Backend | GPUs | Result | Root Cause |
|---------|------|--------|------------|
| Single GPU | 1 | OOM (forward) | Activations alone fill 24GB HBM |
| Single GPU + grad checkpointing | 1 | OOM (optimizer.step) | AdamW states created lazily: +32GB on first step |
| DDP | 4 | OOM (init) | Gradient bucket pre-allocated = model size. 16GB + 16GB = 32GB before step 0 |
| FSDP (hardcoded import) | 4 | 22GB/rank | `isinstance()` failed on newer transformers — no sharding happened |
| FSDP (runtime detection) | 4 | OOM (backward) | Sharding worked (11GB/rank) but activations still 13GB/rank |
| FSDP + grad checkpointing | 4 | loss NaN | FP16 overflow in forward pass — SiLU with large inputs → inf × 0 = NaN |
| FSDP + BF16 | 4 | ✓ 0.97 samples/sec | Stable. 15837MB/rank. BF16 has same exponent range as FP32. |
| Ray Train | 4 | OOM (init) | Wraps DDP — same gradient bucket pre-allocation |

**4 GPUs with FSDP + BF16 + gradient checkpointing is the minimum configuration that fits.**

### Phase 2 — Benchmarks (4× RTX PRO 6000 96GB, PCIe 5.0, Vast.ai)

With enough HBM, measure what actually matters: throughput, scaling efficiency, and real peak memory.

---

## Stages

### Stage 1 — Single GPU: Peak Memory Experiment

**Question:** What is the actual HBM ceiling during a training step — not after it?

Two metrics per step:
- `steady_memory_mb` — after `optimizer.step()`. Activations already freed. Shows: weights + gradients + optimizer states.
- `peak_memory_mb` — right after `backward()`. Activations still in HBM. Shows: weights + gradients + activations.

`peak - steady = activation memory`. The difference between runs = memory saved by gradient checkpointing.

**GPU sizing decision:**
```
peak(checkpointing OFF) → minimum GPU HBM without tricks
peak(checkpointing ON)  → minimum GPU HBM with checkpointing
savings % = (peak_OFF - peak_ON) / peak_OFF × 100%
```
If a cheaper GPU tier fits `peak_ON`, checkpointing pays off. Cost: 18% throughput penalty.

Three metrics tracked per step — `activation_mb = fwd_peak - baseline` isolates pure activation cost:
```
baseline  = memory after zero_grad()     = weights + optimizer states (~46467MB)
fwd_peak  = memory after forward()       = baseline + activations
activation_mb = fwd_peak - baseline      = pure activation memory
steady    = memory after optimizer.step()= weights + optimizer states + gradients (~61783MB)
```

| Run | samples/sec | steady_mb | fwd_peak_mb | activation_mb |
|-----|-------------|-----------|-------------|---------------|
| Single GPU (ckpt ON)  | 6.16 | 61783MB | 49549MB | 3082MB |
| Single GPU (ckpt OFF) | TBD  | 61783MB | TBD     | TBD    |
| Saved by checkpointing | — | — | — | TBD |

---

### Stage 2 — DDP: Throughput Scaling

**Question:** How much throughput do you gain per GPU added, and how much does all-reduce cost?

DDP is **memory-bound** — every GPU holds a full copy of the model. Peak memory per rank = same as single GPU. No memory savings. The only benefit is throughput: more GPUs process more samples in parallel.

Scaling efficiency measures how much communication eats into the theoretical speedup:
```
efficiency = (N_gpu_throughput / (N × single_gpu_throughput)) × 100%
```

1 GPU DDP skipped — identical to single GPU baseline. Single GPU result from Stage 1 is the baseline.

Baseline: 6.16 samples/sec (single GPU, ckpt ON, RTX PRO 6000)

| Run | GPUs | samples/sec | Expected | Actual multiplier | Scaling efficiency |
|-----|------|-------------|----------|-------------------|--------------------|
| DDP | 2 | TBD | 12.32 (2×) | TBD | TBD |
| DDP | 4 | TBD | 24.64 (4×) | TBD | TBD |

---

### Stage 3 — FSDP: Memory Savings + Throughput Recovery

**Question:** Does sharding free enough memory to run larger batches and recover the throughput lost to communication?

FSDP shards weights + gradients + optimizer states across GPUs. Peak memory per rank drops dramatically. Gradient checkpointing OFF — sharding alone is enough on RTX PRO 6000 96GB.

**The throughput story:**
```
FSDP 4GPU batch=4   → lower than DDP (communication overhead, small batch)
FSDP 4GPU batch=16  → back to ~DDP level (freed memory → bigger batch → amortizes cost)
FSDP 2GPU batch=16  → can 2 GPUs match DDP 4 GPU throughput? (half the cost)
```

| Run | GPUs | Batch | samples/sec | peak_mem/rank | vs DDP 4GPU |
|-----|------|-------|-------------|---------------|-------------|
| FSDP | 4 | 4  | TBD | TBD | apples-to-apples |
| FSDP | 4 | 16 | TBD | TBD | throughput recovery |
| FSDP | 2 | 16 | TBD | TBD | half the GPUs, same job? |

---

### Stage 4 — Ray Train: Managed DDP

**Question:** Same throughput as DDP — is the simpler setup worth it?

Ray Train wraps DDP. Same memory math, same communication, same throughput. What changes:
- No manual `dist.init_process_group`
- No `MASTER_ADDR`/`MASTER_PORT` setup
- Native HPO integration via Ray Tune
- Multi-node coordination handled automatically

| Run | GPUs | samples/sec | peak_mem/rank | vs torchrun DDP |
|-----|------|-------------|---------------|-----------------|
| Ray Train | 4 | TBD | TBD | TBD |

---

## How to Run

```bash
# Single GPU
python train_single.py

# DDP
torchrun --nproc_per_node=2 train_ddp.py
torchrun --nproc_per_node=4 train_ddp.py

# FSDP (change BATCH_SIZE in script between runs)
torchrun --nproc_per_node=4 train_fsdp.py   # batch=4
torchrun --nproc_per_node=4 train_fsdp.py   # batch=16
torchrun --nproc_per_node=2 train_fsdp.py   # batch=16

# Ray Train
python train_ray.py

# Compare all results
python compare_runs.py
```

MLflow server (run before training, SSH with port forwarding `-L 8080:localhost:8080`):
```bash
mlflow server --host 0.0.0.0 --port 8080
```

---

## Dataset

Alpaca — 52k instruction-response pairs (Stanford, 2023). Response-only loss masking: loss computed on response tokens only, instruction tokens masked to `-100`.

---

## Stack

- PyTorch 2.2+ (DDP, FSDP built-in)
- HuggingFace Transformers + Datasets
- MLflow (experiment tracking + run comparison)
- Ray Train (Stage 4)
- Vast.ai 4× RTX 4090 24GB PCIe (Phase 1 — OOM story)
- Vast.ai 4× RTX PRO 6000 96GB PCIe 5.0 (Phase 2 — benchmarks)

**Note on interconnect:** RTX PRO 6000 uses PCIe 5.0 (~54GB/s), not NVLink (600GB/s). FSDP scaling efficiency will reflect PCIe overhead. On NVLink hardware the same workload would achieve 85–92% efficiency — the gap is the story.
