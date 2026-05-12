from torch.utils.data import Dataset
from datasets import load_dataset
import torch


PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

PROMPT_TEMPLATE_NO_INPUT = """### Instruction:
{instruction}

### Response:
{output}"""


def format_example(example):
    if example.get("input", "").strip():
        return PROMPT_TEMPLATE.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    return PROMPT_TEMPLATE_NO_INPUT.format(
        instruction=example["instruction"],
        output=example["output"],
    )


class AlpacaDataset(Dataset):
    def __init__(self, tokenizer, max_length=512, split="train"):
        raw = load_dataset("tatsu-lab/alpaca", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = [format_example(ex) for ex in raw]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]

        # find where response starts so we can mask instruction tokens in loss
        response_marker = "### Response:\n"
        response_start_char = text.find(response_marker) + len(response_marker)

        # tokenize full sequence
        full = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = full["input_ids"].squeeze(0)
        attention_mask = full["attention_mask"].squeeze(0)

        # tokenize just the instruction portion to find where response tokens begin
        instruction_part = text[:response_start_char]
        instruction_len = len(
            self.tokenizer(
                instruction_part,
                add_special_tokens=False,
            )["input_ids"]
        )

        # labels = input_ids but -100 on instruction tokens (ignored by cross-entropy loss)
        labels = input_ids.clone()
        labels[:instruction_len] = -100          # mask instruction
        labels[attention_mask == 0] = -100       # mask padding

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
