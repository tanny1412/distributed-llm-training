import functools
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import mlflow
import ray
from ray import train as ray_train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

from data.alpaca import AlpacaDataset
from utils.metrics import ThroughputTracker, gpu_memory_mb

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MAX_LENGTH = 512
BATCH_SIZE = 16
LR = 2e-5
MAX_STEPS = 200
LOG_EVERY = 10


def train_func(_config):
    rank = ray_train.get_context().get_world_rank()
    world_size = ray_train.get_context().get_world_size()
    local_rank = ray_train.get_context().get_local_rank()

    device = torch.device(f"cuda:{local_rank}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # load on CPU — FSDP distributes to GPUs
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    # FSDP instead of DDP — Ray still handles process group
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
    )

    dataset = AlpacaDataset(tokenizer, max_length=MAX_LENGTH)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    tracker = ThroughputTracker()

    if rank == 0:
        mlflow.set_experiment("distributed-training")
        mlflow.start_run(run_name="ray-fsdp-4gpu")
        mlflow.log_params({
            "model": MODEL_ID,
            "batch_size": BATCH_SIZE,
            "world_size": world_size,
            "backend": "ray-fsdp",
            "sharding": "FULL_SHARD",
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
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


if __name__ == "__main__":
    ray.init()

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={},
        scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    )
    
    trainer.fit()
