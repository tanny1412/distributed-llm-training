import torch
import mlflow
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.alpaca import AlpacaDataset
from utils.metrics import ThroughputTracker, gpu_memory_mb

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MAX_LENGTH = 512
BATCH_SIZE = 4
LR = 2e-5
MAX_STEPS = 200
LOG_EVERY = 10


def train():
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # load in FP16 — 8B x 2 bytes = 16GB, fits on A10G (24GB)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    # NOTE: no gradient checkpointing here — this will OOM on first backward

    dataset = AlpacaDataset(tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    tracker = ThroughputTracker()

    mlflow.set_experiment("distributed-training")
    with mlflow.start_run(run_name="single-gpu-no-checkpointing"):
        mlflow.log_params({
            "model": MODEL_ID,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "lr": LR,
            "gradient_checkpointing": False,
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
            optimizer.step()

            tracker.update(input_ids.shape[0])

            if step % LOG_EVERY == 0:
                mlflow.log_metrics({
                    "loss": loss.item(),
                    "samples_per_sec": tracker.samples_per_sec(),
                    "gpu_memory_mb": gpu_memory_mb(),
                }, step=step)
                print(f"step {step} | loss {loss.item():.4f} | "
                      f"{tracker.samples_per_sec():.2f} samples/sec | "
                      f"{gpu_memory_mb():.0f} MB")

            step += 1

        print(f"\nDone. Final: {tracker.samples_per_sec():.2f} samples/sec")


if __name__ == "__main__":
    train()
