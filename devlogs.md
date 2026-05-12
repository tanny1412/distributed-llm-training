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

### Memory breakdown on a single GPU (why 8GB headroom isn't enough)
- Model weights (FP16): 16GB
- Gradients: ~16GB (same size as weights)
- Optimizer states (AdamW — momentum + variance per param): ~32GB
- Activations: variable, depends on batch size and sequence length
- Total: well over 24GB without tricks

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
