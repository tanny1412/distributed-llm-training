import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import mlflow

from data.alpaca import AlpacaDataset
from utils.metrics import ThroughputTracker, gpu_memory_mb

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MAX_LENGTH = 512
BATCH_SIZE = 4
LR = 2e-5
MAX_STEPS = 200
LOG_EVERY = 10


def train():
    # torchrun injects these — each process gets a different local_rank
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # initialize communication between all processes
    dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # full model on every rank — DDP replicates, 16GB per rank
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        use_cache=False,
    )
    
    model.gradient_checkpointing_enable()

    # wrap model with DDP — handles all-reduce automatically after backward
    model = DDP(model, device_ids=[local_rank])

    dataset = AlpacaDataset(tokenizer, max_length=MAX_LENGTH)

    # DistributedSampler ensures each rank gets a different data shard
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    tracker = ThroughputTracker()

    # only rank 0 logs — all ranks have identical metrics after all-reduce
    if rank == 0:
        mlflow.set_experiment("distributed-training")
        mlflow.start_run(run_name=f"ddp-{world_size}gpu")
        mlflow.log_params({
            "model": MODEL_ID,
            "batch_size": BATCH_SIZE,
            "world_size": world_size,
            "max_length": MAX_LENGTH,
            "lr": LR,
            "backend": "ddp",
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
        loss.backward()       # DDP all-reduces gradients automatically here
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tracker.update(input_ids.shape[0] * world_size)  # global samples = local × world_size

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
