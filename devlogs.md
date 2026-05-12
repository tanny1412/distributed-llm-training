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
