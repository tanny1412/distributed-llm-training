import mlflow

EXPERIMENT_NAME = "distributed-training"

RUN_NAMES = [
    "single-gpu",
    "single-gpu-no-checkpointing",
    "ddp-2gpu",
    "ddp-4gpu",
    "fsdp-4gpu-batch4",
    "fsdp-4gpu-batch16",
    "fsdp-2gpu-batch16",
    "ray-train-4gpu",
]

METRICS = ["samples_per_sec", "peak_memory_mb", "gpu_memory_mb"]


def get_runs(experiment_name):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return {}

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time ASC"],
    )

    results = {}
    for run in runs:
        name = run.data.tags.get("mlflow.runName", run.info.run_id)
        metrics = {m: run.data.metrics.get(m) for m in METRICS}
        params = run.data.params
        results[name] = {"metrics": metrics, "params": params}
    return results


def print_table(results):
    single = results.get("single-gpu", {}).get("metrics", {}).get("samples_per_sec")

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Memory table
    print("\n--- Memory (MB) ---")
    print(f"{'Run':<35} {'peak_memory_mb':>16} {'steady_memory_mb':>18} {'activation_mem':>16}")
    print("-" * 87)
    for name in RUN_NAMES:
        if name not in results:
            continue
        m = results[name]["metrics"]
        peak = m.get("peak_memory_mb")
        steady = m.get("gpu_memory_mb")
        activation = (peak - steady) if peak and steady else None
        print(f"{name:<35} {fmt(peak):>16} {fmt(steady):>18} {fmt(activation):>16}")

    # Peak memory checkpointing comparison
    peak_on = results.get("single-gpu", {}).get("metrics", {}).get("peak_memory_mb")
    peak_off = results.get("single-gpu-no-checkpointing", {}).get("metrics", {}).get("peak_memory_mb")
    if peak_on and peak_off:
        savings_pct = (peak_off - peak_on) / peak_off * 100
        print(f"\n  Checkpointing saves {savings_pct:.1f}% peak memory ({peak_off:.0f}MB → {peak_on:.0f}MB)")
        print(f"  GPU sizing: without checkpointing need >{peak_off:.0f}MB, with checkpointing need >{peak_on:.0f}MB")

    # Throughput table (all runs)
    print("\n--- Throughput ---")
    print(f"{'Run':<35} {'samples/sec':>12}")
    print("-" * 49)
    for name in RUN_NAMES:
        if name not in results:
            continue
        tput = results[name]["metrics"].get("samples_per_sec")
        print(f"{name:<35} {fmt(tput):>12}")

    # Scaling efficiency (DDP and FSDP only)
    SCALING_RUNS = ["ddp-2gpu", "ddp-4gpu", "fsdp-4gpu-batch4", "fsdp-4gpu-batch16", "fsdp-2gpu-batch16"]
    print("\n--- Scaling Efficiency (vs single-gpu baseline) ---")
    print(f"{'Run':<35} {'samples/sec':>12} {'expected':>10} {'actual mult':>12} {'efficiency':>12}")
    print("-" * 85)
    for name in SCALING_RUNS:
        if name not in results:
            continue
        m = results[name]["metrics"]
        tput = m.get("samples_per_sec")
        params = results[name]["params"]
        world_size = int(params.get("world_size", 1))

        if single and tput:
            expected = single * world_size
            actual_mult = tput / single
            efficiency = (tput / expected) * 100
            print(f"{name:<35} {fmt(tput):>12} {fmt(expected):>10} {actual_mult:>11.2f}x {efficiency:>11.1f}%")
        else:
            print(f"{name:<35} {'TBD':>12} {'—':>10} {'—':>12} {'—':>12}")

    print("\n" + "=" * 80)


def fmt(val):
    if val is None:
        return "TBD"
    return f"{val:.2f}"


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:8080")
    results = get_runs(EXPERIMENT_NAME)
    if results:
        print_table(results)
