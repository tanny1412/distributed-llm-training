import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import mlflow
import functools

from data.alpaca import AlpacaDataset
from utils.metrics import ThroughputTracker, gpu_memory_mb

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MAX_LENGTH = 512
BATCH_SIZE = 4   # start conservative, increase after confirming FSDP sharding works
LR = 2e-5
MAX_STEPS = 200
LOG_EVERY = 10


def train():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # load on CPU first — FSDP will shard and distribute to GPUs
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    # use_reentrant=False required for FSDP compatibility
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # detect actual decoder layer class at runtime — avoids transformers version mismatches
    decoder_layer_cls = type(model.model.layers[0])

    # wrap policy — FSDP shards at the transformer layer boundary
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={decoder_layer_cls},
    )

    # mixed precision — compute in FP16, keep master weights in FP32
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,  # BF16 all-reduce is stable — no overflow risk
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # shard weights + grads + optimizer states
        device_id=device,
    )

    dataset = AlpacaDataset(tokenizer, max_length=MAX_LENGTH)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    tracker = ThroughputTracker()

    if rank == 0:
        mlflow.set_experiment("distributed-training")
        mlflow.start_run(run_name=f"fsdp-{world_size}gpu-batch{BATCH_SIZE}")
        mlflow.log_params({
            "model": MODEL_ID,
            "batch_size": BATCH_SIZE,
            "world_size": world_size,
            "max_length": MAX_LENGTH,
            "lr": LR,
            "backend": "fsdp",
            "gradient_checkpointing": True,
        })

    model.train()
    step = 0

    for batch in dataloader:
        if step >= MAX_STEPS:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        model.clip_grad_norm_(1.0)  # FSDP-aware clipping — prevents FP16 overflow
        optimizer.step()

        tracker.update(input_ids.shape[0] * world_size)

        if step % LOG_EVERY == 0 and rank == 0:
            mlflow.log_metrics({
                "loss": loss.item(),
                "samples_per_sec": tracker.samples_per_sec(),
                "gpu_memory_mb": gpu_memory_mb(),
            }, step=step)
            print(f"step {step} | loss {loss.item():.4f} | "
                  f"{tracker.samples_per_sec():.2f} samples/sec | "
                  f"{gpu_memory_mb():.0f} MB")

        step += 1

    if rank == 0:
        mlflow.end_run()
        print(f"\nDone. Final: {tracker.samples_per_sec():.2f} samples/sec")

    dist.destroy_process_group()


if __name__ == "__main__":
    train()
