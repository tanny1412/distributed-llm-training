# Distributed Training — Dev Log

## Project
Fine-tune Llama-3-8B across 4× A10G GPUs. Benchmark Single GPU → DDP → FSDP → Ray Train.
Goal: resume project that can be explained deeply in interviews.

---

## Build Philosophy
Every stage has a story arc:
1. Naive attempt (intentionally broken/constrained)
2. Hit a specific, explainable error
3. Fix it mechanically
4. Show the numbers

Interviewers can tell if you ran the code. The errors are proof.

---

## Design Decisions (Interview Ready)

### Why Alpaca over AG News?
AG News would require adding a classification head on top of a causal LM — that's not what Llama was built for. Alpaca keeps the model doing next-token prediction, which is exactly what the architecture was designed for. Instruction tuning is also what companies actually do in production. The task matches the model.

### Why Llama-3-8B?
16GB in FP16. This is the sweet spot: big enough that DDP is painful (barely fits, needs gradient checkpointing, limits batch size), small enough that FSDP makes the difference dramatically visible (~4GB/rank). A smaller model wouldn't tell the story.

### Why 4× A10G?
Each A10G has 24GB. Under DDP, the full 16GB model sits on every GPU — tight but fits with gradient checkpointing. Under FSDP, each GPU holds ~4GB — the memory drop is dramatic and explainable. The contrast is the point.

### Why Alpaca loss masking?
We only compute loss on the response tokens, not the instruction. We're not teaching the model to predict instructions — we're teaching it to generate good responses given an instruction. This is called response-only loss masking.

**Common confusion:** "If we don't compute loss on instruction tokens, how does the model learn to follow instructions?"

The instruction tokens still go through the forward pass — the model reads every token including the instruction. We just don't penalize the model for predicting them. The response is conditioned on the instruction, so the model has to understand the instruction to get the response tokens right. Over 52k examples it learns: given instruction X as context, correct response looks like Y.

Analogy: you grade a student only on their answers, not on whether they can recite the questions. But they still have to read the questions to write good answers. The instruction is the condition. The response is what we train the model to generate.

**Two types of masking — don't confuse them:**
- Causal attention mask: architectural, always on, training AND inference. Triangular matrix. Each token can only attend to tokens before it.
- Loss mask: training only. Zeros out loss on instruction token positions. Inference has no loss so this doesn't exist there.

**The three arrays — mental model:**
- `input_ids` — what the model READS. Full sequence as token IDs. Padding = `0` (pad token). No -100 anywhere.
- `attention_mask` — what the model is ALLOWED TO ATTEND TO. `1` = real token, `0` = padding. Guards the forward pass.
- `labels` — what the model gets GRADED ON. Same IDs as `input_ids` at response positions. `-100` on instruction + padding. Guards the backward pass.

Padding needs two guards:
- `attention_mask = 0` → model ignores padding in forward pass
- `labels = -100` → loss ignores padding in backward pass

---

## Concepts to Know Cold

### GPU memory hierarchy — how data actually moves
All GPU compute involves HBM ↔ SRAM movement. Always. When any operation runs:
```
HBM → SRAM   read data (weights, activations, inputs)
compute in SRAM   (fast, on-chip)
SRAM → HBM   write result back
```

For training specifically:
```
forward:   HBM → SRAM (read weights + input) → compute activation → SRAM → HBM (save activation)
backward:  HBM → SRAM (read activation back) + HBM → SRAM (read weights) → compute gradient → SRAM → HBM (write gradient)
```

Two types of bottleneck — same hardware, different problem:
- **Bandwidth bottleneck** — too much data movement per step. Classic example: KV cache in attention — each new token reads ALL previous K,V pairs from HBM, tiny compute per byte moved.
- **Capacity bottleneck** — too much data sitting in HBM at once. Training OOM — activations + gradients + weights all in HBM simultaneously = 40GB > 24GB.

OOM is a capacity problem, not a bandwidth problem.

### HBM vs SRAM — always say HBM
- **HBM** (High Bandwidth Memory) — the large memory on the GPU die. This is what nvidia-smi shows. This is what OOMs. Model weights, activations, gradients, optimizer states all live here.
- **SRAM** — tiny on-chip cache inside the GPU (L1/L2). Not where model weights live. Do not confuse with HBM.

Always say HBM when talking about GPU memory in interviews. It signals you know the hardware.

### Enabling gradient checkpointing — two lines
```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda",
    use_cache=False,   # required — conflicts with gradient checkpointing
)
model.gradient_checkpointing_enable()
```

Why `use_cache=False`: KV cache saves attention states to speed up generation. Gradient checkpointing discards intermediate states to save HBM. They conflict — both trying to control what stays in HBM. During training you don't need KV cache anyway (you're not generating token by token). Disable it.

Story arc for Stage 1:
1. Run without gradient checkpointing → OOM on backward (commit this first)
2. Add these two lines → fits → record samples/sec, peak HBM, batch size

### Four things that occupy HBM during training

```
Weights          → 16GB  (loaded at startup, always in HBM)
Activations      → varies (created during forward, freed after backward)
Gradients        → 16GB  (created during backward, freed after optimizer step)
Optimizer states → 32GB  (AdamW: momentum + variance, persist across all steps)
```

They don't all peak at the same time:
```
forward:         weights + activations
backward:        weights + activations (shrinking) + gradients (growing)
optimizer step:  weights + gradients + optimizer states  ← peak
next step start: weights + optimizer states (gradients zeroed by zero_grad)
```

Optimizer states are the silent killer — 32GB that sits in HBM every single step once initialized. This is why batch_size=1 + gradient checkpointing still OOMs on a single 24GB GPU. Before a single activation is computed: 16GB weights + 32GB optimizer states = 48GB.

Gradient checkpointing only saves activation memory. It does nothing for gradients or optimizer states.

### Why reducing batch size doesn't save you from optimizer state OOM

Batch size only controls activation memory. Everything else is fixed per-parameter:

```
Weights:          16GB  ← fixed (8B params × 2 bytes)
Gradients:        16GB  ← fixed (same shape as weights, always)
Optimizer states: 32GB  ← fixed (AdamW momentum + variance per param)
Activations:      varies ← the ONLY thing batch size controls
```

At batch_size=1 with gradient checkpointing, activations drop to ~0.5GB. But that's irrelevant when weights + gradients + optimizer states alone = 64GB.

Reducing batch size moves a small number in an equation that's already 40GB over budget. The three fixed costs dominate. Only sharding (FSDP) spreads those fixed costs across GPUs.

Interview line: "Batch size controls activation memory, which is the smallest of the four. Weights, gradients, and optimizer states are per-parameter — they don't shrink with batch size. That's why single GPU full fine-tuning of 8B is a math problem, not a tuning problem."

### Why batch size affects HBM usage
More samples in a batch = more activations held in HBM simultaneously during forward.

batch_size=4 → 4 sequences × activations per layer = 4× activation memory in HBM
batch_size=1 → 1 sequence worth of activations

Gradients are per-parameter, not per-sample — same size regardless of batch size. But activations scale directly with batch size. Even with gradient checkpointing (which reduces activations), the checkpointed activations still scale with batch. 4 sequences × checkpoints was still too much for 24GB.

Real constraint: batch size is limited by how many sequences' activations fit in HBM simultaneously. On a single 24GB GPU with an 8B model and 512-length sequences, batch_size=2 is the ceiling.

### Why single GPU OOMs on backward (the story)
```
forward pass  → activations saved to HBM (~8GB depending on batch)
backward pass → gradients computed using those activations
              → gradients same size as weights = 16GB
              → total: 16GB weights + 8GB activations + 16GB gradients = 40GB
              → A10G has 24GB HBM → OOM
```
And that's before optimizer states (another 32GB for AdamW). Model loads fine — it's the backward pass that kills it.

Interview line: "The model loaded fine — 16GB fits in 24GB HBM. But the backward pass needs to hold activations AND gradients simultaneously. That pushes us well past 24GB. OOM on first backward, exactly as expected."

### Memory breakdown on a single GPU (why 8GB headroom isn't enough)
- Model weights (FP16): 16GB
- Gradients: ~16GB (same size as weights)
- Optimizer states (AdamW — momentum + variance per param): ~32GB
- Activations: variable, depends on batch size and sequence length
- Total: well over 24GB without tricks

### Gradient checkpointing — how it actually works
Not all activations discarded — checkpoints are strategically kept as anchors:

```
layer 1 → activation saved (checkpoint) ✓
layer 2 → activation discarded
layer 3 → activation discarded
layer 4 → activation saved (checkpoint) ✓
layer 5 → activation discarded
...
```

During backward, when it needs layer 2's activation:
- Finds nearest checkpoint before it (layer 1)
- Reruns forward from layer 1 → layer 2 (small segment, not whole model)
- Gets activation → computes gradient → discards activation

Checkpoints = anchors/starting points for recomputation segments. Not hints, not approximations — full recomputation of a segment.

Memory: only checkpoint activations stay in HBM — O(√layers) instead of O(layers)
Compute: ~30% extra — segments of forward pass rerun during backward

Checkpoints are per-step, not persistent:
```
step 1: forward → checkpoints created → backward → checkpoints released → optimizer step
step 2: forward → new checkpoints created → backward → released → optimizer step
```
Fresh set every step. HBM only holds one step's worth of checkpoints at a time. After backward, gradients written to HBM, checkpoints released. After optimizer.step(), gradients released. Only weights remain before next step.

### Gradient checkpointing
Discards activations after the forward pass, recomputes them during backprop.
- Memory saving: ~60% of activation memory
- Compute cost: ~30% more (forward pass runs twice)
- Use when: memory-constrained, have compute to spare

### Gradient checkpointing — memory vs compute tradeoff (measured)

Gradient checkpointing is not free. It trades memory for compute. On hardware where memory is the binding constraint, it's mandatory. On hardware where memory is abundant, it's a pure throughput tax.

**A100 SXM 80GB experiment** — same model, same code, two runs:

| Run | Checkpointing | samples/sec | gpu_memory_mb |
|-----|--------------|-------------|---------------|
| single-gpu | ON  | 4.61 | 61783MB |
| single-gpu-no-checkpointing | OFF | 5.45 | 61783MB |

**Speedup**: 18.2% faster without checkpointing. Less than the theoretical ~30% because at 80GB scale other factors (weight loading, memory bandwidth) partially limit throughput too.

**Memory identical**: 61783MB in both runs. Not because checkpointing doesn't save activation memory — it does. But because of *when* we measure.

`gpu_memory_mb()` calls `torch.cuda.memory_allocated()` — current allocations at the exact moment we log, which is after `optimizer.step()`. By then, activations are already long gone.

Exact order every step:
```
zero_grad()       ← clears previous gradients
forward()         ← activations created in HBM
backward()        ← activations consumed layer by layer, freed as we go
                     activations GONE by the time backward() returns
optimizer.step()  ← weights updated using optimizer states
gpu_memory_mb()   ← we log HERE
```

Activations are freed **during** backward — as it walks each layer, it reads the activation, computes the gradient, then releases the activation. `optimizer.step()` never sees them. So at logging time, only the fixed costs remain:

```
Weights:          ~16GB
Gradients:        ~16GB  (not yet zeroed — zero_grad runs next step)
Optimizer states: ~32GB
Total:            ~64GB  ← same regardless of checkpointing
```

To capture the difference, you'd need `torch.cuda.max_memory_allocated()` — peak since last reset. That would show the spike during the forward→backward window where checkpointing actually changes memory. Our `memory_allocated()` at logging time is steady-state, after the window closes.

### Isolating activation memory — updated train_single.py

Added `peak_memory_mb` logging by resetting peak stats at the start of each step and capturing right after `backward()`:

```python
torch.cuda.reset_peak_memory_stats()   # reset counter at step start
optimizer.zero_grad()
outputs = model(...)
loss.backward()
peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2   # caught here — activations still in HBM
optimizer.step()
# gpu_memory_mb() = memory_allocated() logged after — steady-state, activations gone
```

Two metrics per step:
```
steady_memory_mb  = weights + gradients + optimizer states   (activations freed during backward)
peak_memory_mb    = weights + gradients + activations        (caught right after backward)

peak - steady         = activation memory for this run
peak(OFF) - peak(ON)  = memory saved by gradient checkpointing
```

Results after re-running both versions:

| Run | samples/sec | peak_memory_mb | steady_memory_mb | activation_memory |
|-----|-------------|---------------|-----------------|-------------------|
| single-gpu (checkpointing ON)  | 4.61 | TBD | ~61783MB | TBD |
| single-gpu (checkpointing OFF) | 5.45 | TBD | ~61783MB | TBD |
| Saved by checkpointing         | —    | —   | —        | TBD |

### FSDP batch size experiment — recovering throughput with freed memory

FSDP sharding frees HBM that DDP couldn't use. The freed space lets you run larger batches. Larger batches amortize the communication overhead (AllGather + ReduceScatter is a fixed cost per step — spread over more samples it hurts less per sample).

**The story:**
```
DDP  4 GPUs, batch=4  → X samples/sec  (memory full, can't go higher)
FSDP 4 GPUs, batch=4  → lower than DDP (communication overhead, same small batch)
FSDP 4 GPUs, batch=16 → back to ~X or higher (bigger batch amortizes communication cost)
FSDP 2 GPUs, batch=16 → can 2 GPUs match DDP 4 GPU throughput?
```

The last line is the real argument: same throughput as DDP at 4 GPUs, using only 2 GPUs. Half the cost.

**FSDP run plan:**
```
torchrun --nproc_per_node=4 train_fsdp.py  (batch=4)   ← apples-to-apples vs DDP
torchrun --nproc_per_node=4 train_fsdp.py  (batch=16)  ← recover throughput with freed memory
torchrun --nproc_per_node=2 train_fsdp.py  (batch=16)  ← can 2 GPUs match DDP 4 GPU throughput?
```

**Interview line**: "FSDP isn't just a memory tool. By freeing HBM through sharding, you can run larger batches, which amortizes the communication overhead. We showed FSDP at 2 GPUs matching DDP at 4 GPUs in throughput — same job, half the GPU cost."

### compare_runs.py — auto-calculates everything after runs complete

After each stage, run `python compare_runs.py` on the pod (MLflow server must be running). Reads all runs from MLflow and prints two tables:

**Memory table** — all runs:
```
peak_memory_mb · steady_memory_mb · activation_memory (peak − steady)
checkpointing savings % = (peak_OFF − peak_ON) / peak_OFF × 100%
```

**Scaling table** — DDP and FSDP only:
```
expected = single_gpu_throughput × world_size
actual multiplier = run_throughput / single_gpu_throughput
efficiency % = (actual / expected) × 100%
```

Missing runs show TBD — script works at any stage, table fills progressively as runs complete.

Single GPU and Ray Train show throughput only — scaling efficiency not calculated for them (single GPU has no scaling, Ray Train uses DDP math).

### Why not Prometheus + Grafana for tracking savings %

Prometheus and Grafana are designed for continuous service monitoring — scrape metrics every N seconds from a long-running service. A training run that completes in 2 minutes doesn't fit that model. Adding them here would be infrastructure for show, not because the problem requires it. Interviewers notice the difference.

MLflow is the right tool: it's built for experiment tracking and run comparison. After both runs complete, select both in the MLflow UI → Compare → plots `peak_memory_mb` side by side. Savings % computed from those two numbers.

If this were a production training job running continuously (days, weeks), Prometheus makes sense — scrape GPU utilization, memory pressure, throughput in real time, alert when something degrades. For a discrete benchmark, MLflow handles it completely.

Interview line if asked: "We used MLflow for experiment tracking and compared runs in the UI. Prometheus/Grafana would make sense for a long-running production job where you need real-time alerting — for a benchmark with discrete runs, MLflow is the right fit."

### Why peak memory is the number that matters for GPU sizing

`steady_memory_mb` tells you what HBM the run uses between steps — weights + gradients + optimizer states. `peak_memory_mb` tells you the maximum HBM needed at any single moment during the run — the forward+backward window where activations are also in HBM.

OOMs happen at peak, not steady-state. A run can have a 40GB steady-state and peak at 70GB during backward — if your GPU has 64GB, it crashes mid-backward even though it would have "fit" otherwise.

**`peak_memory_mb` is the number you size your GPU against.**

```
peak (checkpointing OFF) → minimum GPU HBM needed without checkpointing
peak (checkpointing ON)  → minimum GPU HBM needed with checkpointing
```

Real decision: if checkpointing OFF peaks at 70GB → need 80GB GPU (A100 SXM). If checkpointing ON brings peak to 45GB → can use a 48GB GPU (A6000) which is cheaper. The throughput penalty (18%) is the price you pay for the cheaper hardware.

**The decision framework:**
```
run checkpointing OFF → peak = X GB  (what you need without tricks)
run checkpointing ON  → peak = Y GB  (what you need with checkpointing)

savings % = (X - Y) / X × 100%
```

Then look at GPU tiers:
```
if X = 70GB → need A100 80GB
if Y = 45GB → can drop to A6000 48GB or L40S 48GB  ← cheaper per hour
```

The savings % tells you whether the GPU downgrade is possible. You already know the cost: 18% throughput penalty. So the decision becomes:

"Checkpointing saves Z% peak memory. That lets us use a cheaper GPU. We pay 18% in throughput for that. Worth it or not depends on the job — cost-sensitive training takes the cheaper GPU, time-sensitive training keeps the faster one."

Interview line: "We tracked peak memory specifically for GPU sizing decisions. Steady-state memory tells you nothing about OOM risk — a run can crash mid-backward even if steady-state would fit. Peak right after backward is the true ceiling."

Why the difference in samples/sec:
- **Checkpointing ON**: activations discarded during forward. Backward reruns segments of forward to recompute them. Extra compute per step → lower throughput.
- **Checkpointing OFF**: activations stored during forward. Backward reads them directly. No recomputation → faster, same memory here because activations aren't the dominant cost.

On A100 with 80GB, neither run OOMs — checkpointing saves memory we don't need. The only effect is the compute overhead showing up as lower samples/sec.

On a 24GB GPU it's the opposite story — without checkpointing activations alone push past 24GB on the forward pass, so there's no choice.

**Interview line**: "Gradient checkpointing is a memory-compute tradeoff. On 24GB hardware it was mandatory — without it we OOM. On A100 with 80GB the memory was never the constraint, so checkpointing was pure compute overhead. We measured it: 18% throughput penalty. The peak vs steady-state memory split shows exactly how much activation memory checkpointing saves — peak catches activations still in HBM right after backward, steady-state is after they're freed."

### The training loop
zero_grad → forward pass → compute loss → backward → optimizer step

### torchrun — what it injects and why
Running `torchrun --nproc_per_node=4 train_ddp.py` spawns 4 processes and injects into each:
```
LOCAL_RANK  = 0, 1, 2, 3     # which GPU on this machine
RANK        = 0, 1, 2, 3     # global rank (same as local on single node)
WORLD_SIZE  = 4               # total processes across all nodes
MASTER_ADDR = localhost       # where rank 0 listens for initial handshake
MASTER_PORT = random free port
```

You never calculate any of this — torchrun injects it, you read `os.environ["LOCAL_RANK"]` etc.

Multi-node (2 machines, 4 GPUs each):
```
node 0: LOCAL_RANK 0,1,2,3 → RANK 0,1,2,3
node 1: LOCAL_RANK 0,1,2,3 → RANK 4,5,6,7
WORLD_SIZE = 8
```

### MASTER_ADDR and MASTER_PORT — the meeting point
All processes need to find each other before training starts. They connect to rank 0 at MASTER_ADDR:MASTER_PORT for initial handshake. After that, communicate directly via NCCL (GPU to GPU). Master only needed for startup coordination.

Single node: always localhost, torchrun picks free port automatically. Never set manually.

Multi-node: MASTER_ADDR must be IP of first machine — set manually. torchrun can't know the other machine's IP. This is the main complexity of multi-node setup. Ray Train handles this automatically — one of its main selling points.

All processes across all nodes connect to rank 0 at startup:
```
node 0, rank 0 → listens at 192.168.1.1:29500  (master)
node 0, rank 1 → connects to 192.168.1.1:29500
...
node 1, rank 4 → connects to 192.168.1.1:29500
node 1, rank 7 → connects to 192.168.1.1:29500
```
All 8 check in with rank 0. Once all connected → training starts. If any fails to show up → everyone waits → timeout. After handshake, rank 0 is no longer special — all ranks communicate directly via NCCL for gradient all-reduce.

### Why we only track training loss, not validation
Benchmarking training backends, not model quality. Training loss confirms the model is learning. Throughput and memory are the actual metrics.

Three numbers tell the whole story across all 4 stages:
- `loss` → is training healthy?
- `samples_per_sec` → how fast?
- `gpu_memory_mb` → how much HBM?

Interview line: "We're benchmarking training backends, not model quality. Training loss confirms the model is learning. Throughput and memory are the actual metrics we care about."

### Four types of parallelism — know all four

**1. Data Parallelism (DDP)**
Same full model on every GPU, different data shard per GPU. All-reduce gradients after backward. Simple, works when model fits on one GPU.

**2. Fully Sharded Data Parallelism (FSDP / ZeRO-3)**
Data parallel but shards weights + gradients + optimizer states across GPUs. Each GPU holds ~1/N of everything. What we built in Stage 3.

**3. Tensor Parallelism**
Splits individual weight matrices across GPUs. One layer's matrix sliced — GPU 0 gets columns 0-512, GPU 1 gets 512-1024. Needs very fast interconnect (NVLink). Used by Megatron-LM.

**4. Pipeline Parallelism**
Splits model by layers across GPUs. GPU 0 runs layers 1-8, GPU 1 runs 9-16, etc. Each GPU processes a micro-batch then passes activations to the next GPU.

```
DDP/FSDP  → split DATA across GPUs
Tensor    → split one LAYER across GPUs
Pipeline  → split LAYERS across GPUs
```

Production LLM training (GPT-4, Llama) uses all simultaneously = 3D parallelism (data + tensor + pipeline). Our project covers the first two — most common in MLOps interviews.

**3D Parallelism — how the three dimensions interact:**
```
Data parallelism:    multiple REPLICAS, each gets different full batch
Pipeline parallelism: within one replica, batch → micro-batches → flow through layer stages (nodes)
Tensor parallelism:  within one node, 4 GPUs compute ONE micro-batch together (split computation)
```

Micro-batch flow is ONE direction only:
- Forward: node0 → node1 → node2 → node3
- Backward: gradients flow back node3 → node2 → node1 → node0
- A micro-batch from node2 never goes to node1. Only gradients go backwards.

Multiple micro-batches in pipeline simultaneously — different stages, different micro-batches, same moment:
```
time 4: node3(mb1) → node2(mb2) → node1(mb3) → node0(mb4)  ← all busy
```

⚠ Still need to revisit pipeline parallelism flow — come back to this.

**Pipeline Parallelism — micro-batches:**
Pipeline bubble = GPUs idle waiting for previous GPU. Fixed with micro-batches — split one batch into smaller chunks so multiple micro-batches flow through pipeline simultaneously:
```
GPU 0 (layers 1-8):   [mb1] [mb2] [mb3] [mb4]
GPU 1 (layers 9-16):        [mb1] [mb2] [mb3] [mb4]
GPU 2 (layers 17-24):             [mb1] [mb2] [mb3] [mb4]
GPU 3 (layers 25-32):                   [mb1] [mb2] [mb3] [mb4]
```
Different micro-batches, different layers, same moment in time. Each micro-batch still flows through layers sequentially — but multiple micro-batches in pipeline simultaneously keep GPUs busy.

**3D Parallelism (GPT-3, 1024 A100s):**
- 8-way tensor parallelism within a node
- 8-way pipeline parallelism across nodes
- 16-way data parallelism across replicas
- 8 × 8 × 16 = 1024 GPUs

Production LLM training (GPT-4, Llama) uses all simultaneously = 3D parallelism (data + tensor + pipeline). Our project covers the first two — most common in MLOps interviews.

**FSDP vs Tensor Parallelism — the real difference:**

FSDP — partial storage, briefly reconstructs full matrix to compute:
```
partial storage → all-gather → full matrix in HBM → compute → discard non-owned shards
```

Tensor Parallelism — partial storage, NEVER reconstructs full matrix:
```
GPU 0: partial weights → compute partial result Y0
GPU 1: partial weights → compute partial result Y1
combine: Y = concat(Y0, Y1)   ← full matrix never exists on any GPU
```

Key: FSDP saves HBM between steps. Tensor parallelism splits the actual computation — full weight matrix never lives on any single GPU at any moment.

Why tensor parallelism needs NVLink: combining partial results happens mid-computation, requires extremely low latency. FSDP's all-gather can tolerate slightly higher latency (happens once per layer, not mid-multiply).

### Mixed precision — simple version
Two number formats:
```
FP32 → 4 bytes per number → precise, slow, more HBM
FP16 → 2 bytes per number → less precise, fast, less HBM
```

Mixed precision = FP16 for speed, FP32 where precision matters:
```
weights during compute  → FP16  (fast forward/backward)
weight updates          → FP32  (precise — small gradients don't disappear)
```

MixedPrecision in FSDP:
```python
param_dtype=torch.float16    # store/compute weights in FP16
reduce_dtype=torch.float16   # communicate gradients in FP16
buffer_dtype=torch.float16   # everything else in FP16
```

All FP16 = maximum speed, minimum HBM. Fine for 200 step benchmark.
Risk: small gradient values underflow to zero in FP16. For production (weeks of training) → use FP32 for reduce_dtype to prevent gradient precision loss during all-reduce.

### Mixed precision — short version
FP16 for compute (fast, less HBM). FP32 for master weights (precise, prevents underflow over many steps). Two copies of weights: FP32 master + FP16 working copy. Cast to FP16 before forward, update FP32 master after backward.

### ShardingStrategy options
```
NO_SHARD      → DDP memory (no sharding, just FSDP wrapper)
SHARD_GRAD_OP → shard gradients + optimizer states, full weights (ZeRO-2)
FULL_SHARD    → shard everything: weights + gradients + optimizer states (ZeRO-3) ← we use this
```
FULL_SHARD = ~4GB/rank for 8B model. Maximum memory saving. The dramatic drop that makes the story.

### Why FSDP throughput is low on consumer GPUs

0.97 samples/sec across 4× RTX 4090 is slow. Three reasons:

1. **Gradient checkpointing** — recomputes every activation during backward. ~30% extra compute per step on top of normal training.

2. **PCIe not NVLink** — RTX 4090s on Vast.ai are connected via PCIe (16GB/s), not NVLink (600GB/s). FSDP does AllGather + ReduceScatter for every layer (32 layers × 2 = 64 communication rounds per step). Over PCIe each round is slow — most of wall time is waiting for communication, not computing.

3. **batch_size=4** — communication cost is fixed per step regardless of batch size. Small batch = same communication overhead spread over fewer samples = high overhead per sample. Larger batch amortizes the cost.

On NVLink A100s with batch_size=16: same model, 5-10× higher throughput. The 4090s are consumer GPUs — fast compute (TFLOPS), slow interconnect (PCIe).

Interview line: "The absolute throughput isn't the point — the relative scaling and memory story are. On NVLink hardware the numbers look completely different."

### FSDP throughput — compute might suffer
FSDP has more communication than DDP:
```
DDP:  AllReduce after backward (one sync point)
FSDP: AllGather before forward + ReduceScatter after backward (two sync points)
```

More sync points = compute waits for communication = dead time.

FSDP hides this two ways:
1. Prefetching (BackwardPrefetch) — while computing layer 5, all-gather layer 6 in background. Overlap compute and communication.
2. Larger batch — communication cost fixed per step. Bigger batch = same cost spread over more samples = lower overhead per sample.

DDP batch=4 (HBM constrained) vs FSDP batch=16 (HBM freed). Whether throughput improves depends on batch gain vs communication overhead. On NVLink — yes. On PCIe — marginal.

Interview answer: "FSDP isn't about raw speed. It's about fitting larger models and enabling larger batches. The memory story is the point." We measure it, we don't claim it.

### auto_wrap_policy vs ShardingStrategy — two different things
```
auto_wrap_policy  → WHERE/WHEN to shard (at which module boundaries)
ShardingStrategy  → WHAT to shard within those boundaries
```

`transformer_auto_wrap_policy` with `LlamaDecoderLayer` = shard at each decoder layer boundary.
`ShardingStrategy.FULL_SHARD` = within those boundaries, shard weights + gradients + optimizer states.

Together: each LlamaDecoderLayer has its weights, gradients, and optimizer states all sharded across 4 GPUs.

`FULL_SHARD` without `wrap_policy` = shard everything but as one giant 16GB unit (all-gather full model before forward).
`wrap_policy` without `FULL_SHARD` = shard at layer boundaries but maybe not everything inside.
Both needed together for maximum memory savings.

### Why transformer_auto_wrap_policy matters
Without wrap policy: FSDP treats entire model as one unit → all-gather full 16GB before every forward → defeats the purpose.

With wrap policy at LlamaDecoderLayer boundaries:
```
LlamaDecoderLayer(0)  → one FSDP unit, sharded across 4 GPUs
LlamaDecoderLayer(1)  → one FSDP unit, sharded across 4 GPUs
...
LlamaDecoderLayer(31) → one FSDP unit, sharded across 4 GPUs
```
Before forward on layer 5 → all-gather only layer 5's weights → compute → discard → move to layer 6.
Only one layer's weights in HBM at a time, not the whole model.

`functools.partial` pre-bakes `transformer_layer_cls={LlamaDecoderLayer}` so FSDP can call the policy without extra arguments.

### FSDP wrap policy — why layer boundaries
FSDP shards at LlamaDecoderLayer boundaries. This is NOT model parallelism:

Model/Pipeline Parallelism:
```
GPU 0 owns layers 0-7  → only GPU 0 ever computes those layers (sequential)
GPU 1 owns layers 8-15 → only GPU 1 ever computes those layers
```

FSDP:
```
GPU 0 owns 1/4 of EVERY layer's weights
GPU 1 owns 1/4 of EVERY layer's weights
...
before layer 5: all-gather → all 4 GPUs get full layer 5 weights
all 4 GPUs compute layer 5 together on same input
discard non-owned shards → back to 1/4 each
```

Layer boundary = where FSDP decides what to gather/discard, NOT where computation splits. Every GPU runs every layer. All 4 compute the same layer simultaneously after gathering.

`transformer_auto_wrap_policy` tells FSDP to shard at each `LlamaDecoderLayer`. Decoder layer is a self-contained unit — safe to shard between them, not inside them.

### FSDP vs DDP — key difference
DDP: full model on every GPU → 16GB weights + full gradients + full optimizer states per rank
FSDP: shards weights + gradients + optimizer states → ~4GB/rank

Three things FSDP shards (not just weights):
1. Model weights
2. Gradients
3. Optimizer states (AdamW momentum + variance — 2x model size)

Result: no gradient checkpointing needed in FSDP. HBM freed up enough to run larger batch sizes.

### device_map — DDP vs FSDP vs single GPU
- Single GPU: `device_map="cuda"` → loads to cuda:0 (only one GPU, fine)
- DDP: `device_map={"": device}` → loads to this rank's specific GPU. `"cuda"` would put all 4 processes on cuda:0 — collision.
- FSDP: no device_map at all → load to CPU first, FSDP shards and distributes to GPUs itself. device_map conflicts with FSDP sharding.

### dist.destroy_process_group()
Counterpart to dist.init_process_group(). Releases NCCL communication resources and closes connections between processes. Without it, processes may hang instead of exiting cleanly. Always clean up what you initialized.

```python
dist.init_process_group()    # open
# ... training ...
dist.destroy_process_group() # close
```

### DistributedSampler — why it exists
Without it: all 4 ranks iterate the full dataset → every example processed 4 times → gradients 4x inflated → training wrong.

With it: dataset split into non-overlapping shards, each rank gets its own:
```
52k examples, 4 ranks:
rank 0 → examples 0, 4, 8, 12, ...
rank 1 → examples 1, 5, 9, 13, ...
rank 2 → examples 2, 6, 10, 14, ...
rank 3 → examples 3, 7, 11, 15, ...
```
Each example processed exactly once globally per epoch, by different ranks.

`num_replicas=world_size` → split into 4 shards  
`rank=rank` → this process gets shard number rank  
`shuffle=True` → randomize before splitting each epoch  

Can't use `shuffle=True` in DataLoader AND a sampler together — sampler handles shuffling itself. Use `DataLoader(dataset, sampler=sampler)`.

### DDP implementation details

**tracker.update(batch_size * world_size)**: Each rank only sees its own batch. Multiply by world_size to get global samples processed per step. Without this, samples/sec is 4x too low and scaling efficiency calculations are wrong.

**if rank == 0 for MLflow**: Only rank 0 runs logging code. Ranks 1,2,3 skip it entirely — nothing is sent to rank 0. Rank 0's metrics are representative because after all-reduce all ranks have identical gradients.

**Straggler problem**: All-reduce is a barrier — every rank waits for the slowest one. One slow GPU tanks the whole run. Negligible on homogeneous hardware (all A10G). Ray Train handles stragglers better — another reason to prefer it for large clusters.

**ThroughputTracker per process**: Each of 4 processes creates its own tracker with its own clock. Only rank 0's tracker matters for logging — but must account for all ranks' work via world_size multiplication.

### DDP — what's synchronized and what's not
```
forward:    each rank sees different batch → different loss, different activations
            NO synchronization

backward:   each rank computes own gradients from own loss
            NO synchronization yet

all-reduce: gradients averaged across all ranks
            → every rank now has IDENTICAL gradients

optimizer:  identical update on every rank → weights stay in sync
```

Why only rank 0 logs to MLflow: all 4 processes run the same script. Without a rank check, all 4 log simultaneously → 4 identical copies of every metric → charts broken. Rank 0 is just the designated logger by convention. Metrics are identical after all-reduce so it doesn't matter which rank logs.

Loss logged is rank 0's local loss (its batch) — not averaged across ranks. Good enough for tracking trends. The gradient all-reduce is what makes training correct, not the logged loss value.

### Three NCCL operations for LLM training

**DDP → AllReduce**
Each GPU has full model replica. After backward, gradients averaged and synchronized across all GPUs before optimizer step.

**FSDP → AllGather**
Model weights sharded across GPUs. Before each forward pass, GPUs temporarily AllGather full weights for computation, then release non-local shards to save HBM.
```
gather weights → forward/backward compute → release weights
```

**FSDP → ReduceScatter**
After backward, gradients averaged and immediately scattered back into shards. Each GPU keeps only the gradient shard corresponding to its owned weights.
```
reduce gradients → scatter gradient shards → optimizer step
```

Summary:
- DDP = replicate full model, synchronize full gradients
- FSDP = shard model, gather for compute, reshard after compute

### Why AllReduce is named that way
"All" = every process both sends AND receives (not one-to-many or many-to-one).
"Reduce" = applies a reduction operation (sum then /world_size for average).

Other collective operations:
- broadcast   → rank 0 sends, everyone receives (one → all)
- reduce      → everyone sends, rank 0 receives (all → one)
- all-reduce  → everyone sends, everyone receives (all → all)
- all-gather  → everyone sends their piece, everyone gets all pieces (FSDP forward pass)
- reduce-scatter → everyone sends, result scattered as shards (FSDP backward pass)

### Why DDP needs all-reduce
Each GPU sees a different data shard and computes different gradients. If they update independently, models diverge (GPU 0 learns sports, GPU 1 learns politics). All-reduce averages gradients across all GPUs before the weight update — every GPU applies the same delta, weights stay in sync.

### DDP vs FSDP
- DDP: full model on every GPU. Gradient all-reduce after backward. 16GB/rank. Needs gradient checkpointing.
- FSDP: shards weights + gradients + optimizer states. Each GPU holds ~4GB. No checkpointing needed. Larger batch possible.

### device_map="cuda" vs model.to("cuda")
`model.to("cuda")` loads weights into CPU RAM first, then copies to GPU. For 16GB model = 16GB CPU RAM + 16GB GPU VRAM occupied simultaneously = 32GB CPU RAM needed during transfer.

`device_map="cuda"` streams weights directly onto GPU during loading. CPU RAM never accumulates the full model. Right way to load large models.

Interview line: "`device_map='cuda'` loads weights directly to GPU. `model.to('cuda')` loads to CPU first then copies — you need double the memory during transfer. For a 16GB model that's 16GB of CPU RAM you don't want to waste."

### Why tokenizer.pad_token = tokenizer.eos_token
Llama is a generative model — never needed to pad sequences during pretraining. So its tokenizer has BOS and EOS but no PAD token. We need padding to batch sequences to the same length. Fix: reuse EOS as the pad token. Safe because attention_mask=0 and labels=-100 both neutralize padding positions — the model never actually uses those tokens.

Must be set BEFORE passing tokenizer to AlpacaDataset. Python passes objects by reference — same tokenizer object inside the dataset. Set it late and the dataset has the same broken tokenizer.

Interview line: "Llama has no dedicated pad token — it's a generative model, never needed one. We reuse EOS. The model never sees those positions anyway because attention mask and labels both zero them out."

### Small details that matter in interviews
- `batch['input_ids'].shape[0]` not hardcoded batch_size — last batch of an epoch may be smaller if dataset doesn't divide evenly. `.shape[0]` gives real count every time. Hardcoding would overcount.
- `gpu_memory_mb()` called after `optimizer.step()` not before — memory is at peak after full step (activations + gradients + optimizer states all in memory simultaneously).
- Log every 10 steps not every step — MLflow logging has overhead, every step would slow training.

### metrics.py — what it does and why
Two things to track: `start_time` (wall clock when training started) and `total_samples` (incremented by batch_size after every optimizer.step()).

`time.perf_counter()` not `time.time()` — more precise for benchmarking.

`gpu_memory_mb()`:
- `torch.cuda.is_available()` guard — CPU locally has no GPU, calling memory_allocated() without it crashes. Returns 0.0 on CPU, real number on GPU. Same code works everywhere.
- `/ 1024**2` — converts bytes to MB. torch returns bytes, nvidia-smi shows MB, keep them consistent. 4GB in bytes is unreadable.

Why log real memory: the DDP vs FSDP memory story (18-20GB → 4-6GB per rank) has to come from real numbers. Without this, the comparison is just a claim. With it, you have a graph to show in an interview.

### Why samples/sec and not just loss
Loss tells you how well the model is learning. It says nothing about how fast. Two runs can have identical loss curves but one finishes in 2 hours, the other in 8. For distributed training, speed is the whole point — you're adding GPUs to go faster.

Why not steps/sec: batch size changes between runs. DDP batch_size=4 vs FSDP batch_size=16 — steps aren't comparable. Samples/sec normalizes for batch size. Always measuring the same thing: how much real data moved through the model per second.

### What "processed" means
One full training step = forward → loss → backward → optimizer step. Increment sample count AFTER optimizer.step(), not during forward. Backward pass is often slower — counting only forward would lie about throughput.

1 step = 1 batch = N samples. If batch_size=8, one step = 8 samples processed. After 200 steps = 1600 samples.
samples_per_sec = total_samples / elapsed_time

### Scaling efficiency formula
efficiency = (N_gpu_throughput / (N × single_gpu_throughput)) × 100%
Target: 80–88% at 4 GPUs. Below 70% = bottleneck elsewhere (usually data loading).

### What instruction tuning changes
Base Llama is a text completion machine — give it "The capital of France" and it continues "is Paris. The capital of Germany is Berlin...". After Alpaca fine-tuning it learns the instruction-following behavior: when it sees the `### Instruction / ### Response` format, it answers directly instead of completing text.

### How we evaluate for this project
We are NOT evaluating model quality — we're benchmarking training efficiency. Loss going down confirms training is healthy. Samples/sec, GPU memory, and scaling efficiency are the metrics that matter here. Proper model quality evaluation (MT-Bench, AlpacaEval) is out of scope.

Interview answer: "The project's goal was benchmarking training backends, not maximizing model quality. We tracked loss to confirm training was healthy, and throughput/memory to compare backends."

### Dataset mental model — works for any model
Rule: check what the model's `forward()` accepts → that's what `__getitem__` must return.

How to check:
- `help(model.forward)` in terminal → shows every argument, types, what's optional
- HuggingFace docs → same but with descriptions. Look for "tokens with -100 are ignored" — that's where the -100 convention is documented.

Different models, different inputs:
- BERT classification: `input_ids`, `attention_mask`, `labels` (scalar per example)
- ViT image classification: `pixel_values`, `labels` (no input_ids — images not text)
- Whisper speech-to-text: `input_features`, `labels` (mel spectrogram in, transcript out)
- Llama instruction tuning: `input_ids`, `attention_mask`, `labels` (per-token, -100 masked)

### Dataset + DataLoader pattern — always the same
```
Dataset.__init__()     → gets everything ready (load files, format strings, store metadata)
Dataset.__getitem__()  → called by DataLoader on demand, one example at a time
DataLoader             → orchestrates when and how __getitem__ is called
```
`__init__` = prepare. `__getitem__` = serve. `DataLoader` = the one who asks.

Tokenization is lazy — happens in `__getitem__`, not `__init__`. If you tokenized all 52k at init you'd consume huge memory before training even starts. `__init__` only formats strings and stores them. DataLoader triggers `__getitem__` per batch during the training loop.

Every dataset you write follows this pattern. Only the contents change.

### alpaca.py — what the file does end to end
Takes raw Alpaca JSON → formats into instruction/response string → tokenizes → masks instruction + padding in labels → returns three tensors per example. DataLoader stacks N examples into [batch_size, 512]. That batch goes directly into the model. File is never touched again after this.

Why format at `__init__` not `__getitem__`: formatting is cheap (string substitution), tokenization is expensive. Pre-format all 52k once upfront, tokenize on demand per step.

Why two templates (with/without `### Input:`): some Alpaca examples have extra context, some don't. Empty `### Input:` section wastes tokens and confuses the model. Use the right template per example.

`instruction` vs `input` in Alpaca:
- `instruction` = what to do ("Translate the following sentence to French")
- `input` = what to do it to ("The weather is nice today")
- Some tasks need both, some are self-contained (write a poem — no data needed)

Regardless of template, everything ends up in the same three tensors. Template differences only matter during string formatting. After tokenization it's all just numbers.

Loss masking rule: everything before `### Response:\n` → `-100`. Always. If instruction + input both exist, both get masked. If only instruction, only that gets masked. One line of code handles both cases — it just finds `### Response:\n` and masks everything before it.

---

### Memory hierarchy — model lives everywhere
```
Disk (SSD)     → permanent storage. Model downloaded here (~16GB). Persists.
CPU RAM        → transit. from_pretrained() loads model here first. Freed after move to HBM.
HBM/VRAM       → working memory. Model lives here during training. Where OOMs happen.
SRAM (on-chip) → compute cache. Tiny chunks brought here for actual math, result written back to HBM.
```

All four need to be big enough:
- Not enough disk → can't download model
- Not enough CPU RAM → can't load model (need ~32GB for 16GB model + buffer)
- Not enough HBM → OOM during training
- SRAM is fixed hardware — can't change it, just affects compute speed

On Vast.ai: check GPU count, HBM per GPU, CPU RAM, and disk. All four matter.

### Single node vs multi node — what actually triggers multi-node
Not the technique — the model size vs hardware capacity.

```
Single node, single GPU  → model fits on one GPU
Single node, multi GPU   → DDP, FSDP, even tensor parallelism possible here
Multi node               → when you need more GPUs than one machine has
```

Tensor parallelism can run on single node (8× A100 within one machine). Pipeline parallelism becomes necessary when model doesn't fit even across all GPUs on one node.

Real trigger for multi-node: model too big for one node → go multi-node → pipeline parallelism necessary → 3D parallelism for maximum efficiency.

Our project: single node, 4× A10G. 8B model fits with FSDP. No multi-node needed.
GPT-4 scale: hundreds of nodes → 3D parallelism mandatory.

The line is model size vs hardware capacity, not the technique itself.

### PCIe vs NVLink — why interconnect speed matters for FSDP

Two interconnect types for multi-GPU communication:

```
PCIe 5.0 x16  →  ~54 GB/s   consumer and workstation GPUs (RTX, RTX PRO)
NVLink        →  600 GB/s   data center GPUs (A100, H100, H200)
```

FSDP does AllGather + ReduceScatter for every transformer layer — 32 layers × 2 = 64 communication rounds per step. Over PCIe, each round is slow. Most of wall time becomes waiting for communication, not computing. Scaling efficiency drops significantly vs NVLink hardware.

DDP and single GPU runs are unaffected — no inter-GPU communication.

**Why PCIe benchmarks are still valuable:**

Showing 60% scaling efficiency on PCIe and explaining why is a stronger interview answer than showing 88% on NVLink without understanding it. The bottleneck is visible in the data. The explanation — AllGather latency per layer × 64 rounds × slow interconnect — is exactly what an interviewer wants to hear.

The story: "On PCIe 5.0 our FSDP scaling efficiency was X%. The bottleneck is AllGather + ReduceScatter over 64 communication rounds per step at 54GB/s. On NVLink at 600GB/s the same workload would achieve 85–92% efficiency — 11× faster interconnect directly translates to less idle time per step."

**Hardware used:**
- Phase 1 (OOM story): 4× RTX 4090 24GB, PCIe, Vast.ai
- Phase 2 (benchmarks): 4× RTX PRO 6000 96GB, PCIe 5.0, Vast.ai

96GB VRAM means no memory pressure — all configurations fit without tricks. Pure benchmark numbers.

## Project Scope — What We Did and Didn't Build

**In scope:**
- Single GPU baseline
- PyTorch DDP (data parallelism)
- PyTorch FSDP / ZeRO-3 (sharded data parallelism)
- Ray Train (managed DDP)

**Out of scope — and why:**
Tensor parallelism, pipeline parallelism, 3D parallelism require multi-node NVLink clusters (Megatron-LM, DeepSpeed). Not something you run on 4× A10G. Claiming you did would be a red flag.

Interview answer when asked about tensor/pipeline parallelism:
> "I didn't implement those — they require multi-node NVLink clusters. I understand how they work conceptually, but my project focused on DDP → FSDP which is what most MLOps roles actually use day to day."

## GPU Setup — How to Connect

### Vast.ai + VS Code Remote SSH
1. Create Vast.ai account → add credits
2. Select template: PyTorch (Vast) → set container size to 60GB
3. Filter: 4x RTX 4090 (24GB VRAM each) — perfect for OOM story
4. Rent instance → wait for "Running"
5. Generate SSH key locally if needed: `ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""`
6. Copy public key: `cat ~/.ssh/id_rsa.pub`
7. Paste into Vast.ai → Manage SSH Keys → ADD SSH KEY
8. VS Code → Cmd+Shift+P → "Remote-SSH: Connect to Host" → `ssh -p <port> root@<ip>`
9. Open terminal → verify: `nvidia-smi`

### Instance details (session 1)
- 4x RTX 4090, 24564MB HBM each
- $0.018/hr
- CUDA 12.6, Driver 560.35.05
- Connected via: `ssh -p 19722 root@175.155.64.227 -L 8080:localhost:8080`

### Vast.ai tmux behavior — and how to disable it
Vast.ai auto-attaches every SSH connection (including VS Code Remote terminals) to the same tmux session. Every new terminal tab in VS Code drops into the same tmux — can't use keyboard shortcuts because macOS intercepts them.

Fix: disable auto-tmux permanently:
```bash
touch ~/.no_auto_tmux
```
Then reconnect. Every SSH session after that is a plain shell.

Workflow after fix: two plain terminal tabs, both SSH'd in. One runs `watch -n 1 nvidia-smi`, other runs experiments. No tmux navigation needed.

Why the `-L 8080:localhost:8080` flag: forwards port 8080 so MLflow UI is accessible at `localhost:8080` in the local browser during experiments.

## GPU Day Process (Day 2)
1. Rent 4× A10G on Vast.ai
2. Clone repo, `pip install -r requirements.txt`
3. Run `train_single.py` → screenshot the OOM
4. Add checkpointing → run → record numbers
5. Run `train_ddp.py` at 1, 2, 4 GPUs → record scaling efficiency
6. Run `train_fsdp.py` → watch memory drop → record numbers
7. Run `train_ray.py` → compare to DDP
8. Fill benchmark table in README with real numbers

## GPU Run — What Actually Happened

### Stage 1 — Single GPU (3 attempts)

**Attempt 1: No gradient checkpointing**
- Model loaded fine (16GB fits in 24GB HBM)
- OOM during **forward pass** — activations alone filled 24GB
- `23.06 GiB allocated, tried to allocate 16 MiB`

**Attempt 2: Gradient checkpointing + batch_size=4**
- OOM during **backward pass**
- Gradient checkpointing reduces activations but not gradients (16GB) or weights (16GB)
- 32GB minimum before optimizer states

**Attempt 3: batch_size=1**
- Still OOM — crashes inside `optimizer.step()` on the very first step
- AdamW creates momentum + variance buffers lazily on first call (not at model load time)
- Sequence:
  ```
  backward()       → 16GB gradients now in HBM. Total: 16GB weights + 16GB grads = 32GB
  optimizer.step() → allocates 32GB momentum + variance buffers HERE → total 64GB → OOM
                     weight update never runs
  ```
- Conclusion: full fine-tuning of 8B on 24GB is mathematically impossible without QLoRA

---

### Stage 2 — DDP (1 attempt)

**Attempt 1: torchrun --nproc_per_node=4**
- OOM at `DDP.__init__` — never reached step 0
- DDP pre-allocates gradient bucket = full model size (15GB) upfront
- 15GB weights + 15GB gradient buffer = 30GB > 24GB

---

### Stage 3 — FSDP (5 attempts)

**Attempt 1: Hardcoded LlamaDecoderLayer import**
- 22GB/rank — same as no sharding
- Newer transformers wraps `LlamaDecoderLayer` → `isinstance()` always False → FSDP treats whole model as one unit → AllGather full 16GB before every forward

**Attempt 2: Runtime class detection**
- `decoder_layer_cls = type(model.model.layers[0])`
- Memory: 22GB → 11GB/rank. Sharding confirmed.
- OOM on backward — activations without gradient checkpointing = 13GB/rank
- Math: 4GB weights + 13GB activations + 4GB gradients + 8GB optimizer = 29GB

**Attempt 3: Added gradient checkpointing**
- Training started, all 4 GPUs at 100% compute, peak 21GB/rank
- `loss nan` from step 10 — FP16 overflow in forward pass (SiLU with large inputs → inf × 0 = NaN)

**Attempt 4: Added gradient clipping**
- Still NaN — clipping is post-backward, overflow was in forward pass itself

**Attempt 5: Switched to BF16**
- BF16 has same exponent range as FP32 (8 bits vs FP16's 5)
- Loss immediately stable: 1.33 → 1.39 → 1.52 → running

---

## Stages

### Stage 1 — Single GPU Baseline (complete)
- [x] train_single.py WITHOUT gradient checkpointing → OOM during forward pass (activations fill 24GB HBM)
- [x] Added gradient checkpointing + batch_size=1 → still OOM on backward
- Root cause: 16GB weights + 16GB gradients = 32GB minimum, exceeds 24GB. Optimizer states add another 32GB.
- Conclusion: full fine-tuning of 8B on a single 24GB GPU is not feasible. OOM is the story.
- Screenshot saved: screenshots/stage1_single_gpu_oom.png

### BF16 vs FP16 — why BF16 is preferred for training

```
FP16:  1 sign + 5 exponent + 10 mantissa bits  → max value: 65504
BF16:  1 sign + 8 exponent + 7 mantissa bits   → max value: ~3.4 × 10^38 (same as FP32)
FP32:  1 sign + 8 exponent + 23 mantissa bits  → max value: ~3.4 × 10^38
```

FP16 has only 5 exponent bits → tiny representable range → easy to overflow to inf during forward pass (SiLU activation with large inputs → inf × 0 = NaN). NaN propagates through all subsequent operations and corrupts weights permanently.

BF16 has 8 exponent bits (same as FP32) → same range, impossible to overflow where FP32 wouldn't. Less mantissa precision (7 bits vs 10) but training is robust to that. 

RTX 4090, A100, H100 all support BF16 natively. BF16 is the default format for modern LLM training. FP16 requires GradScaler to be safe; BF16 does not.

Switch: `torch_dtype=torch.bfloat16` + `MixedPrecision(param_dtype=torch.bfloat16, ...)`. Loss immediately stable.

### NaN loss in FSDP — FP16 gradient overflow

Symptom: loss=1.3 at step 0, then `loss nan` from step 10 onward. Training broken.

Cause: all-FP16 mixed precision (`param_dtype=float16, reduce_dtype=float16`). FP16 max value is 65504. During backward, gradients can spike past that → overflow to `inf` → NaN propagates through all subsequent operations.

Fix: gradient clipping before optimizer step. FSDP has its own method — can't use `torch.nn.utils.clip_grad_norm_` because gradients are sharded:
```python
loss.backward()
model.clip_grad_norm_(1.0)  # FSDP-aware — gathers sharded grads, clips, rescatters
optimizer.step()
```

Why `model.clip_grad_norm_` not `torch.nn.utils.clip_grad_norm_`: standard PyTorch clipping computes the global grad norm by reading all `.grad` attributes. With FSDP, gradients are sharded — each rank only has its own shard. The global norm requires an AllReduce across ranks. `model.clip_grad_norm_` handles that communication internally.

Max norm = 1.0: industry standard. If gradient vector magnitude exceeds 1.0, all gradients scaled down proportionally. Prevents spikes without killing signal.

### FSDP — why activations still OOM even after sharding

FSDP shards weights + gradients + optimizer states across ranks. But **activations are not sharded** — each rank stores full activations for its own batch.

Memory per rank without gradient checkpointing (batch_size=4, seq_len=512):
```
Weights (sharded):         4GB   ← 16GB / 4 ranks
Activations (NOT sharded): ~13GB ← 32 layers × ~400MB each
Gradients (sharded):       4GB
Optimizer states (sharded):8GB
Total:                     ~29GB > 24GB → OOM on backward
```

nvidia-smi confirmed: 11GB/rank after wrap policy fix (down from 22GB with broken wrap policy), still OOM.

Fix: `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`

`use_reentrant=False` is required for FSDP — the default reentrant checkpointing uses Python autograd hooks that conflict with FSDP's own hooks. Non-reentrant uses torch.autograd.graph instead, which is FSDP-safe.

With gradient checkpointing: activations drop from ~13GB to ~2GB. Total ~18GB → fits in 24GB.

### FSDP wrap policy — why hardcoded imports break

Original code:
```python
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
```

Problem: newer transformers versions rename or restructure model classes. If `LlamaDecoderLayer` doesn't match the actual class in the loaded model, FSDP finds no layers to wrap and treats the entire 16GB model as one FSDP unit. Result: AllGather reconstructs the full model before every forward pass — same memory as no sharding. nvidia-smi shows 22GB per rank instead of ~4GB.

Fix: detect the actual decoder layer class at runtime:
```python
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
decoder_layer_cls = type(model.model.layers[0])  # whatever class it actually is
wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={decoder_layer_cls})
```

Works regardless of transformers version. The model is already loaded on CPU before wrapping, so `model.model.layers[0]` is always accessible.

Interview line: "We detected the actual decoder layer class at runtime rather than hardcoding the import — newer transformers versions restructure internals and a stale import silently breaks FSDP sharding. The symptom is 22GB/rank instead of 4GB."

### Stage 2 — PyTorch DDP
- [ ] train_ddp.py — full model on every rank, torchrun launcher
- [ ] Run at 1, 2, 4 GPUs — record scaling efficiency
- [ ] Show nvidia-smi: ~16–20GB on every rank

### Stage 3 — PyTorch FSDP (complete)
- [x] train_fsdp.py — sharded weights/grads/optimizer states
- Results: 0.97 samples/sec, 15837MB/rank, loss stable 1.33→1.32 over 200 steps
- BF16 + gradient checkpointing (use_reentrant=False) + gradient clipping (1.0)
- Memory higher than theoretical 4GB because grad checkpointing recomputation + optimizer states peak during backward

### Stage 3 — PyTorch FSDP
- [ ] Remove gradient checkpointing — show it fits without it
- [ ] Show memory drop: ~18GB (DDP) → ~4–6GB (FSDP)
- [ ] Increase batch size until OOM — compare max batch vs DDP

### Stage 4 — Ray Train (complete)
- [x] train_ray.py — OOM at DDP.__init__, same as torchrun DDP
- Same gradient bucket pre-allocation problem. Ray Train wraps DDP, doesn't change the memory math.

### FSDP scaling test — why only 4 GPUs work on 24GB

Tested FSDP at 1, 2, 4 GPUs:

**1 GPU**: FSDP auto-switches to `NO_SHARD` (nothing to shard across) → same as single GPU → OOM

**2 GPUs**: per rank peak = 8GB shard + 8GB AllGather + 8GB gradients + 16GB optimizer states + 2GB activations = ~42GB → OOM

**4 GPUs**: per rank peak = 4GB + 4GB + 4GB + 8GB + 2GB = ~22GB → fits (observed 21GB)

4 GPUs is the **minimum** that fits Llama-3-8B on 24GB hardware. Not a choice — a constraint.

Interview line: "On 24GB GPUs with an 8B model, 4-way FSDP isn't an optimization — it's the only configuration that fits. Single GPU, DDP, Ray Train, FSDP 1-GPU, FSDP 2-GPU all OOM. The memory math dictates the minimum GPU count."

### Stage 4 — Ray Train
- [ ] train_ray.py — same DDP but no manual dist.init_process_group
- [ ] ray_tune.py — HPO sweep in ~10 lines
- [ ] Compare setup complexity and throughput vs torchrun

### Checkpointing — how long training jobs survive crashes

Save model + optimizer state to disk periodically. Resume from last checkpoint if process crashes.

```python
# Save every N steps
if step % 500 == 0 and rank == 0:
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'checkpoint_step_{step}.pt')

# Resume
checkpoint = torch.load('checkpoint_step_500.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_step = checkpoint['step']
```

Must save optimizer state, not just weights. AdamW has momentum + variance per parameter — if you only save weights, optimizer starts cold and training stutters for hundreds of steps until momentum rebuilds.

At GPT-4 scale (1024 GPUs): checkpoint every few hundred steps to cloud storage (S3). Hardware failures at that scale are expected. A node dies → spin up replacement → load latest checkpoint → continue. Without checkpointing, a failure at step 50,000 means restarting from zero.

FSDP uses `FSDP.state_dict()` not regular `.state_dict()` — weights are sharded across ranks, you need FSDP's API to gather and save them correctly.

Not in our project scope (200 steps = minutes), but know it cold for interviews.

---

## Benchmark Table (fill in after GPU runs)

| Backend       | GPUs | samples/sec | Scaling eff. | Mem/rank  | Notes                          |
|---------------|------|-------------|--------------|-----------|--------------------------------|
| Single GPU    | 1    | —           | 100%         | ~18–20GB  | Grad checkpointing required    |
| DDP           | 2    | —           | —%           | ~18–20GB  | Still needs grad checkpointing |
| DDP           | 4    | —           | —%           | ~18–20GB  | Target: 80–88% efficiency      |
| FSDP          | 4    | —           | —%           | ~4–6GB    | No checkpointing, larger batch |
| Ray Train DDP | 4    | —           | ~DDP%        | ~18–20GB  | Simpler setup vs torchrun      |
