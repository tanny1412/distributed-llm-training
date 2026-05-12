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

### Mixed precision — short version
FP16 for compute (fast, less HBM). FP32 for master weights (precise, prevents underflow over many steps). Two copies of weights: FP32 master + FP16 working copy. Cast to FP16 before forward, update FP32 master after backward.

### ShardingStrategy options
```
NO_SHARD      → DDP memory (no sharding, just FSDP wrapper)
SHARD_GRAD_OP → shard gradients + optimizer states, full weights (ZeRO-2)
FULL_SHARD    → shard everything: weights + gradients + optimizer states (ZeRO-3) ← we use this
```
FULL_SHARD = ~4GB/rank for 8B model. Maximum memory saving. The dramatic drop that makes the story.

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

## Stages

### Stage 1 — Single GPU Baseline (current)
- [ ] Write train_single.py WITHOUT gradient checkpointing (will OOM on GPU — intentional)
- [ ] Add gradient checkpointing, record samples/sec, peak memory, batch size
- [ ] Log to MLflow: loss, samples/sec, GPU memory, batch size
- Dataset: Alpaca (52k instruction-response pairs, response-only loss masking)

### Stage 2 — PyTorch DDP
- [ ] train_ddp.py — full model on every rank, torchrun launcher
- [ ] Run at 1, 2, 4 GPUs — record scaling efficiency
- [ ] Show nvidia-smi: ~16–20GB on every rank

### Stage 3 — PyTorch FSDP
- [ ] train_fsdp.py — sharded weights/grads/optimizer states
- [ ] Remove gradient checkpointing — show it fits without it
- [ ] Show memory drop: ~18GB (DDP) → ~4–6GB (FSDP)
- [ ] Increase batch size until OOM — compare max batch vs DDP

### Stage 4 — Ray Train
- [ ] train_ray.py — same DDP but no manual dist.init_process_group
- [ ] ray_tune.py — HPO sweep in ~10 lines
- [ ] Compare setup complexity and throughput vs torchrun

---

## Benchmark Table (fill in after GPU runs)

| Backend       | GPUs | samples/sec | Scaling eff. | Mem/rank  | Notes                          |
|---------------|------|-------------|--------------|-----------|--------------------------------|
| Single GPU    | 1    | —           | 100%         | ~18–20GB  | Grad checkpointing required    |
| DDP           | 2    | —           | —%           | ~18–20GB  | Still needs grad checkpointing |
| DDP           | 4    | —           | —%           | ~18–20GB  | Target: 80–88% efficiency      |
| FSDP          | 4    | —           | —%           | ~4–6GB    | No checkpointing, larger batch |
| Ray Train DDP | 4    | —           | ~DDP%        | ~18–20GB  | Simpler setup vs torchrun      |
