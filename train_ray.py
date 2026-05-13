import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import mlflow
import ray
from ray import train as ray_train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

from data.alpaca import AlpacaDataset
from utils.metrics import ThroughputTracker, gpu_memory_mb

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MAX_LENGTH = 512
BATCH_SIZE = 4
LR = 2e-5
MAX_STEPS = 200
LOG_EVERY = 10


def train_func(_config):
    # Ray injects rank/world_size — no manual dist.init_process_group needed
    rank = ray_train.get_context().get_world_rank()
    world_size = ray_train.get_context().get_world_size()
    local_rank = ray_train.get_context().get_local_rank()

    device = torch.device(f"cuda:{local_rank}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        use_cache=False,
    )
    
    model.gradient_checkpointing_enable()

    # Ray Train prepares the model for distributed training
    model = ray_train.torch.prepare_model(model)

    dataset = AlpacaDataset(tokenizer, max_length=MAX_LENGTH)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    tracker = ThroughputTracker()

    if rank == 0:
        mlflow.set_tracking_uri("http://localhost:8080")
        mlflow.set_experiment("distributed-training")
        mlflow.start_run(run_name="ray-train-4gpu")
        mlflow.log_params({
            "model": MODEL_ID,
            "batch_size": BATCH_SIZE,
            "world_size": world_size,
            "max_length": MAX_LENGTH,
            "lr": LR,
            "backend": "ray-train",
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
        torch.cuda.reset_peak_memory_stats()
        baseline_mb = torch.cuda.memory_allocated() / 1024**2

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        forward_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        activation_mb = forward_peak_mb - baseline_mb

        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tracker.update(input_ids.shape[0] * world_size)

        if step % LOG_EVERY == 0:
            metrics = {
                "loss": loss.item(),
                "samples_per_sec": tracker.samples_per_sec(),
                "gpu_memory_mb": gpu_memory_mb(),
                "forward_peak_mb": forward_peak_mb,
                "activation_mb": activation_mb,
            }
            ray_train.report(metrics)   # must be called by ALL ranks — it's a sync barrier
            if rank == 0:
                mlflow.log_metrics(metrics, step=step)
                print(f"step {step} | loss {loss.item():.4f} | "
                      f"{tracker.samples_per_sec():.2f} samples/sec | "
                      f"steady {gpu_memory_mb():.0f} MB | "
                      f"fwd_peak {forward_peak_mb:.0f} MB | activations {activation_mb:.0f} MB")

        step += 1

    if rank == 0:
        mlflow.end_run()
        print(f"\nDone. Final: {tracker.samples_per_sec():.2f} samples/sec")


if __name__ == "__main__":
    ray.init()

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={},
        scaling_config=ScalingConfig(
            num_workers=4,
            use_gpu=True,
        ),
    )

    trainer.fit()
