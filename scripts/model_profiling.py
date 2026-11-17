import os, glob, time, json
import numpy as np
import torch
import pytorch_lightning as pl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 36
N_FEAT = 12

def load_model_class():
    import importlib.util
    for candidate in ["train_v5.py", "train_v4_fixed.py", "train_v3.py"]:
        if os.path.exists(candidate):
            spec = importlib.util.spec_from_file_location("train_module", candidate)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for cls_name in ["DualStreamV5", "DualStreamV4", "DualStreamV3"]:
                if hasattr(mod, cls_name):
                    return getattr(mod, cls_name), candidate
    raise RuntimeError("Could not find model class")

def find_checkpoint():
    candidates = []
    for folder in ["ckpt5", "ckpt4_fixed", "ckpt4", "ckpt3", "ckpt2", "checkpoints"]:
        if os.path.isdir(folder):
            candidates.extend(glob.glob(os.path.join(folder, "*.ckpt")))
    if not candidates:
        raise FileNotFoundError("No checkpoint found")
    return max(candidates, key=os.path.getmtime)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def model_size_mb(model):
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)

def estimate_flops(model, x):
    """Approximate FLOPs by counting multiply-add operations from model structure."""
    flops = 0
    B, T, F = x.shape
    # Walk through modules
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.LSTM):
            # LSTM: 4 gates * (input + hidden) * hidden_dim * 2 (mul+add) * T * B * 2 (bidir)
            in_dim = mod.input_size
            hid_dim = mod.hidden_size
            bidir = 2 if mod.bidirectional else 1
            flops += 4 * (in_dim + hid_dim) * hid_dim * 2 * T * B * bidir
        elif isinstance(mod, torch.nn.Linear):
            # Linear: 2 * in * out * B (mul+add)
            flops += 2 * mod.in_features * mod.out_features * B
    return flops

def measure_latency(model, x, n_warmup=50, n_runs=200):
    """Measure inference latency on GPU."""
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(x)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))  # ms
            else:
                t0 = time.perf_counter()
                _ = model(x)
                times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times), np.percentile(times, 99)

def measure_throughput(model, x, batch_sizes, duration_s=2.0):
    """Measure windows/second at different batch sizes."""
    results = {}
    for bs in batch_sizes:
        x_batch = x[:bs].repeat((bs + len(x) - 1) // len(x), 1, 1)[:bs]
        # Warmup
        with torch.no_grad():
            for _ in range(20):
                _ = model(x_batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        # Timed
        count = 0
        t_start = time.perf_counter()
        with torch.no_grad():
            while time.perf_counter() - t_start < duration_s:
                _ = model(x_batch)
                count += bs
        elapsed = time.perf_counter() - t_start
        results[bs] = count / elapsed
    return results

def architecture_summary(model):
    """Print a readable architecture summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("MODEL ARCHITECTURE SUMMARY")
    lines.append("=" * 70)
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0 and hasattr(mod, 'weight'):
            params = sum(p.numel() for p in mod.parameters())
            if isinstance(mod, torch.nn.LSTM):
                shape = f"input={mod.input_size}, hidden={mod.hidden_size}, bidir={mod.bidirectional}"
            elif isinstance(mod, torch.nn.Linear):
                shape = f"{mod.in_features} -> {mod.out_features}"
            else:
                shape = str(list(mod.weight.shape))
            lines.append(f"  {name:45s} {params:>10,} params  [{shape}]")
    return "\n".join(lines)

def main():
    print("=" * 70)
    print("MODEL PROFILING REPORT")
    print("=" * 70)
    print(f"DEVICE: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ModelClass, src_file = load_model_class()
    print(f"Model class loaded from: {src_file} ({ModelClass.__name__})")

    ckpt_path = find_checkpoint()
    print(f"Checkpoint: {ckpt_path}")
    model = ModelClass.load_from_checkpoint(ckpt_path, map_location=DEVICE).to(DEVICE).eval()

    # --- Basic stats ---
    total_params, trainable_params = count_parameters(model)
    size_mb = model_size_mb(model)

    print("\n" + "=" * 70)
    print("BASIC STATISTICS")
    print("=" * 70)
    print(f"  Total parameters:     {total_params:>12,}")
    print(f"  Trainable parameters: {trainable_params:>12,}")
    print(f"  Model size (MB):      {size_mb:>12.2f}")
    print(f"  Sequence length:      {SEQ_LEN:>12d} (6 hours @ 10-min)")
    print(f"  Input features:       {N_FEAT:>12d}")

    # --- Architecture summary ---
    print("\n" + architecture_summary(model))

    # --- Dummy input ---
    x = torch.randn(1, SEQ_LEN, N_FEAT, device=DEVICE, dtype=torch.float32)
    flops = estimate_flops(model, x)
    print(f"\n  FLOPs per window (approx): {flops:>12,}")
    print(f"  MFLOPs per window:         {flops/1e6:>12.2f}")

    # --- Latency ---
    print("\n" + "=" * 70)
    print("INFERENCE LATENCY (per window, ms)")
    print("=" * 70)
    batch_sizes = [1, 8, 32, 128, 512, 2048, 8192]
    latency_rows = []
    for bs in batch_sizes:
        x_batch = torch.randn(bs, SEQ_LEN, N_FEAT, device=DEVICE, dtype=torch.float32)
        mean, std, p99 = measure_latency(model, x_batch, n_warmup=30, n_runs=100)
        latency_rows.append((bs, mean, std, p99, mean / bs))
        print(f"  batch={bs:>5d}:  mean={mean:>7.3f} ms  std={std:.3f}  p99={p99:.3f}  per_window={mean/bs:.4f} ms")

    # --- Throughput ---
    print("\n" + "=" * 70)
    print("THROUGHPUT (windows / second)")
    print("=" * 70)
    tp = measure_throughput(model, x, batch_sizes=[32, 128, 512, 2048, 8192], duration_s=2.0)
    for bs, wps in tp.items():
        print(f"  batch={bs:>5d}:  {wps:>12,.0f} windows/s")

    # --- Memory ---
    print("\n" + "=" * 70)
    print("GPU MEMORY FOOTPRINT")
    print("=" * 70)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated() / 1e6
        with torch.no_grad():
            xb = torch.randn(8192, SEQ_LEN, N_FEAT, device=DEVICE)
            _ = model(xb)
        peak = torch.cuda.max_memory_allocated() / 1e6
        after = torch.cuda.memory_allocated() / 1e6
        print(f"  Before inference:  {before:>8.2f} MB")
        print(f"  After inference:   {after:>8.2f} MB")
        print(f"  Peak memory:       {peak:>8.2f} MB")
    else:
        print("  (GPU not available — skipping memory stats)")

    # --- Save ---
    report = dict(
        device=str(DEVICE),
        model_class=ModelClass.__name__,
        checkpoint=ckpt_path,
        total_params=total_params,
        trainable_params=trainable_params,
        model_size_mb=round(size_mb, 2),
        flops_per_window=flops,
        latency_ms=[{"batch": bs, "mean": mean, "std": std, "p99": p99, "per_window": mean/bs}
                    for bs, mean, std, p99, _ in latency_rows],
        throughput_wps={int(bs): int(wps) for bs, wps in tp.items()},
    )
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/model_profiling.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nSaved: results/metrics/model_profiling.json")

if __name__ == "__main__":
    main()