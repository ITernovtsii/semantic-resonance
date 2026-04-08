#!/usr/bin/env python3
"""
Experiment 001: Multi-seed matched routing comparison.

Runs 4 models x 2 extra seeds = 8 training runs.
Each run generates a YAML config, trains, and runs analysis.

Usage:
  # Run a specific batch (pair of models on GPU 1 and GPU 2):
  python3 experiments/001_multi_seed/run_multi_seed.py --batch 1

  # Run a single specific model:
  python3 experiments/001_multi_seed/run_multi_seed.py --model SRA_B --seed 42 --gpu 1

  # Dry run (print configs without training):
  python3 experiments/001_multi_seed/run_multi_seed.py --batch 1 --dry-run

Batch schedule (sequential on GPU 0):
  Batch 1: SRA_B seed=42 then StdMoE_B seed=42 (~7h)
  Batch 2: SRA_E seed=42 then StdMoE_E seed=42 (~7h)
  Batch 3: SRA_B seed=137 then StdMoE_B seed=137 (~7h)
  Batch 4: SRA_E seed=137 then StdMoE_E seed=137 (~7h)
"""
import argparse
import os
import subprocess
import sys
import yaml
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiments", "001_multi_seed")
OUTPUT_BASE = os.path.join(EXPERIMENT_DIR, "outputs")

# ============================================================
# Model definitions
# ============================================================
# All share: 256 experts, d_ff=256, 4 layers, d_model=512, 8 heads,
# bandpass S6 masked (0.0005-0.0040), alpha=0.4, tau=10.0, 8 epochs
MODELS = {
    "SRA_B": {
        "description": "Cosine routing, K=1->4 at epoch 3",
        "routing_type": "cosine",
        "progressive_schedule": [
            {"epoch": 1, "top_k": 1},
            {"epoch": 3, "top_k": 4},
        ],
        "initial_top_k": 1,
    },
    "SRA_E": {
        "description": "Cosine routing, K=2->4 at epoch 3",
        "routing_type": "cosine",
        "progressive_schedule": [
            {"epoch": 1, "top_k": 2},
            {"epoch": 3, "top_k": 4},
        ],
        "initial_top_k": 2,
    },
    "StdMoE_B": {
        "description": "Linear routing, K=1->4 at epoch 3",
        "routing_type": "linear",
        "progressive_schedule": [
            {"epoch": 1, "top_k": 1},
            {"epoch": 3, "top_k": 4},
        ],
        "initial_top_k": 1,
    },
    "StdMoE_E": {
        "description": "Linear routing, K=2->4 at epoch 3",
        "routing_type": "linear",
        "progressive_schedule": [
            {"epoch": 1, "top_k": 2},
            {"epoch": 3, "top_k": 4},
        ],
        "initial_top_k": 2,
    },
}

SEEDS = [42, 137]  # seed=19 already exists

BATCHES = {
    1: [("SRA_B", 42, 0), ("StdMoE_B", 42, 0)],
    2: [("SRA_E", 42, 0), ("StdMoE_E", 42, 0)],
    3: [("SRA_B", 137, 0), ("StdMoE_B", 137, 0)],
    4: [("SRA_E", 137, 0), ("StdMoE_E", 137, 0)],
}


def generate_config(model_name: str, seed: int) -> dict:
    """Generate a full YAML config for a given model and seed."""
    model_def = MODELS[model_name]
    run_name = f"{model_name}_seed{seed}"
    output_dir = os.path.join(OUTPUT_BASE, run_name)

    config = {
        "project": {
            "name": "SR-Research-WT103",
            "experiment_name": f"Exp001 {run_name}",
            "output_dir": output_dir,
            "seed": seed,
        },
        "hardware": {
            "use_amp": True,
            "amp_dtype": "bf16",
        },
        "data": {
            "dataset_name": "wikitext-103",
            "data_dir": os.path.join(PROJECT_ROOT, "data", "raw", "wikitext-103"),
            "processed_dir": os.path.join(PROJECT_ROOT, "data", "processed", "wikitext-103"),
            "tokenizer_path": os.path.join(PROJECT_ROOT, "data", "tokenizers", "sra_bpe_32k_wt103.json"),
            "vocab_size": 32000,
            "max_seq_length": 256,
        },
        "model": {
            "d_model": 512,
            "n_layers": 4,
            "n_heads": 8,
            "d_ff": 256,
            "dropout": 0.1,
            "gradient_checkpointing": True,
            "activation": "gelu",
            "max_seq_length": 256,
            "csr": {
                "enabled": True,
                "num_experts": 256,
                "top_k": model_def["initial_top_k"],
                "anchor_init": "batch" if model_def["routing_type"] == "cosine" else "orthogonal",
                "routing_type": model_def["routing_type"],
                "temperature_scaling": True,
                "temperature_init": 10.0,
                "progressive_top_k": {
                    "enabled": True,
                    "schedule": model_def["progressive_schedule"],
                },
            },
        },
        "losses": {
            "balance_weight": 0.4,
            "loss_type": "bandpass",
            "bandpass_min_pct": 0.0005,
            "bandpass_max_pct": 0.0040,
        },
        "training": {
            "batch_size_per_gpu": 128 if model_def["routing_type"] == "cosine" else 64,
            "gradient_accumulation_steps": 1 if model_def["routing_type"] == "cosine" else 2,
            "epochs": 8,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1.0e-08,
            "scheduler": "cosine",
            "warmup_steps": 4000,
        },
        "logging": {
            "use_wandb": True,
            "log_interval": 100,
            "eval_interval": 2000,
        },
    }
    return config


def save_config(config: dict, model_name: str, seed: int) -> str:
    """Save config to YAML file, return path."""
    config_dir = os.path.join(EXPERIMENT_DIR, "configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"{model_name}_seed{seed}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path


def launch_training(config_path: str, gpu: int, dry_run: bool = False) -> subprocess.Popen | None:
    """Launch training on a specific GPU. Returns the process handle."""
    cmd = [
        "accelerate", "launch",
        "--config_file", os.path.join(PROJECT_ROOT, "configs", "accelerate", "1gpu.yaml"),
        os.path.join(PROJECT_ROOT, "scripts", "train.py"),
        "--config", config_path,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    run_name = os.path.basename(config_path).replace(".yaml", "")
    log_path = os.path.join(EXPERIMENT_DIR, "outputs", f"{run_name}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"  CMD: CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}")
    print(f"  LOG: {log_path}")

    if dry_run:
        print("  [DRY RUN - not launching]")
        return None

    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
                            cwd=PROJECT_ROOT)
    print(f"  PID: {proc.pid}")
    return proc


def launch_analysis(output_dir: str, gpu: int):
    """Run analyze.py on a completed training output."""
    checkpoint = os.path.join(output_dir, "best_model.pt")
    if not os.path.exists(checkpoint):
        print(f"  WARNING: No checkpoint at {checkpoint}, skipping analysis")
        return

    cmd = [
        "python3", os.path.join(PROJECT_ROOT, "scripts", "analyze.py"),
        "--checkpoint", checkpoint,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    run_name = os.path.basename(output_dir)
    log_path = os.path.join(EXPERIMENT_DIR, "outputs", f"{run_name}_analysis.log")

    print(f"  Analyzing {run_name}...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=PROJECT_ROOT)
    with open(log_path, "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)

    if result.returncode == 0:
        print(f"  Analysis complete: {log_path}")
    else:
        print(f"  Analysis FAILED (exit {result.returncode}): {log_path}")


def run_batch(batch_num: int, dry_run: bool = False):
    """Run a batch of models. Sequential on same GPU, parallel on different GPUs."""
    if batch_num not in BATCHES:
        print(f"Invalid batch number: {batch_num}. Valid: 1-4")
        sys.exit(1)

    runs = BATCHES[batch_num]
    print(f"\n{'='*60}")
    print(f"BATCH {batch_num}: {len(runs)} runs (sequential on same GPU)")
    print(f"{'='*60}")

    # Check if runs use different GPUs (can parallelize) or same GPU (must serialize)
    gpus_used = set(gpu for _, _, gpu in runs)
    parallel = len(gpus_used) > 1

    if dry_run:
        for model_name, seed, gpu in runs:
            print(f"\n--- {model_name} seed={seed} on GPU {gpu} ---")
            config = generate_config(model_name, seed)
            config_path = save_config(config, model_name, seed)
            print(f"  Config: {config_path}")
            print(f"  Output: {config['project']['output_dir']}")
            launch_training(config_path, gpu, dry_run=True)
        print("\n[DRY RUN COMPLETE]")
        return

    start = time.time()

    if parallel:
        # Launch all in parallel, wait for all
        processes = []
        configs = []
        for model_name, seed, gpu in runs:
            print(f"\n--- {model_name} seed={seed} on GPU {gpu} ---")
            config = generate_config(model_name, seed)
            config_path = save_config(config, model_name, seed)
            configs.append((model_name, seed, gpu, config["project"]["output_dir"]))
            proc = launch_training(config_path, gpu)
            if proc:
                processes.append((proc, model_name, seed))

        print(f"\nWaiting for {len(processes)} parallel runs...")
        for proc, model_name, seed in processes:
            proc.wait()
            elapsed = time.time() - start
            status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
            print(f"  {model_name} seed={seed}: {status} ({elapsed/3600:.1f}h)")
    else:
        # Run sequentially on same GPU
        configs = []
        for i, (model_name, seed, gpu) in enumerate(runs):
            print(f"\n--- [{i+1}/{len(runs)}] {model_name} seed={seed} on GPU {gpu} ---")
            config = generate_config(model_name, seed)
            config_path = save_config(config, model_name, seed)
            configs.append((model_name, seed, gpu, config["project"]["output_dir"]))
            print(f"  Config: {config_path}")
            print(f"  Output: {config['project']['output_dir']}")

            proc = launch_training(config_path, gpu)
            if proc:
                proc.wait()
                elapsed = time.time() - start
                status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                print(f"  {model_name} seed={seed}: {status} ({elapsed/3600:.1f}h)")

                # Run analysis immediately after each training
                if proc.returncode == 0:
                    launch_analysis(config["project"]["output_dir"], gpu)

    total_time = time.time() - start
    print(f"\nBatch {batch_num} complete in {total_time/3600:.1f}h")

    # Run analysis for parallel runs (sequential already analyzed above)
    if parallel:
        print("\nRunning analysis...")
        for model_name, seed, gpu, output_dir in configs:
            launch_analysis(output_dir, gpu)


def run_single(model_name: str, seed: int, gpu: int, dry_run: bool = False):
    """Run a single model training."""
    if model_name not in MODELS:
        print(f"Invalid model: {model_name}. Valid: {list(MODELS.keys())}")
        sys.exit(1)

    print(f"\n--- {model_name} seed={seed} on GPU {gpu} ---")
    config = generate_config(model_name, seed)
    config_path = save_config(config, model_name, seed)
    print(f"  Config: {config_path}")
    print(f"  Output: {config['project']['output_dir']}")

    proc = launch_training(config_path, gpu, dry_run)
    if proc:
        print(f"\nWaiting for training to complete...")
        proc.wait()
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"  Result: {status}")

        if proc.returncode == 0:
            launch_analysis(config["project"]["output_dir"], gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 001: Multi-seed matched routing comparison")
    parser.add_argument("--batch", type=int, help="Run a batch (1-4)")
    parser.add_argument("--all", action="store_true", help="Run all 4 batches sequentially (~28h)")
    parser.add_argument("--model", type=str, help="Run a single model (SRA_B, SRA_E, StdMoE_B, StdMoE_E)")
    parser.add_argument("--seed", type=int, help="Seed for single model run")
    parser.add_argument("--gpu", type=int, default=0, help="GPU for single model run (default: 0)")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without launching")
    args = parser.parse_args()

    if args.all:
        for b in range(1, 5):
            run_batch(b, args.dry_run)
        print(f"\n{'='*60}")
        print("ALL BATCHES COMPLETE")
        print(f"{'='*60}")
    elif args.batch:
        run_batch(args.batch, args.dry_run)
    elif args.model and args.seed:
        run_single(args.model, args.seed, args.gpu, args.dry_run)
    else:
        print("Usage:")
        print("  --batch N        Run batch N (1-4), each runs 2 models in parallel")
        print("  --model M --seed S --gpu G   Run a single model")
        print()
        print("Batches:")
        for b, runs in BATCHES.items():
            desc = " + ".join(f"{m} seed={s} GPU{g}" for m, s, g in runs)
            print(f"  Batch {b}: {desc}")
        print()
        print("Existing results (seed=19):")
        for name, mdef in MODELS.items():
            print(f"  {name}: {mdef['description']}")
