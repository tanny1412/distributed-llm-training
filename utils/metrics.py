import time
import torch


class ThroughputTracker:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.total_samples = 0

    def update(self, batch_size):
        self.total_samples += batch_size

    def samples_per_sec(self):
        elapsed = time.perf_counter() - self.start_time
        return self.total_samples / elapsed


def gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0
