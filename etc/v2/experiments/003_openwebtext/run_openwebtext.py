#!/usr/bin/env python3
"""
Experiment 003: Cross-dataset validation on OpenWebText.

Trains SRA E (cosine) and StdMoE E (linear) on OpenWebText subset (~100M tokens)
to validate that routing properties generalize beyond WikiText-103.

Usage:
  # Step 1: Prepare data (downloads, tokenizes, chunks OpenWebText)
  python3 experiments/003_openwebtext/run_openwebtext.py --prepare-data

  # Step 2: Train both models in parallel (GPU 1 + GPU 2)
  python3 experiments/003_openwebtext/run_openwebtext.py --train

  # Step 3: Run analysis on checkpoints
  python3 experiments/003_openwebtext/run_openwebtext.py --analyze

  # Or run everything sequentially:
  python3 experiments/003_openwebtext/run_openwebtext.py --all

  # Dry run (print configs without training):
  python3 experiments/003_openwebtext/run_openwebtext.py --train --dry-run
"""
import argparse
import logging
import os
import subprocess
import sys
import time
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiments", "003_openwebtext")
OUTPUT_BASE = os.path.join(EXPERIMENT_DIR, "outputs")
CONFIG_DIR = os.path.join(EXPERIMENT_DIR, "configs")

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "openwebtext")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "openwebtext")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "data", "tokenizers", "sra_bpe_32k_owt.json")

# OpenWebText subset size: calibrated to match WikiText-103's 404,842 training chunks
# 104K docs → 398,749 train chunks (4.26 chunks/train_doc, 90% train split)
# 404,842 / 4.26 / 0.9 ≈ 105,589 → use 106K for slight margin
OWT_NUM_DOCS = 106_000

SEED = 19

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
# Model definitions (E schedule only)
# ============================================================
MODELS = {
    "SRA_E": {
        "description": "Cosine routing, K=2->4 at epoch 3",
        "routing_type": "cosine",
        "progressive_schedule": [
            {"epoch": 1, "top_k": 2},
            {"epoch": 3, "top_k": 4},
        ],
        "initial_top_k": 2,
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


# ============================================================
# Data Preparation
# ============================================================
def prepare_data():
    """Download OpenWebText subset, train tokenizer, and preprocess."""
    logging.info("=" * 60)
    logging.info("STEP 1: Preparing OpenWebText data")
    logging.info("=" * 60)

    # Check if already processed
    train_path = os.path.join(PROCESSED_DIR, "train")
    if os.path.exists(train_path):
        logging.info(f"Processed data already exists at {PROCESSED_DIR}. Skipping.")
        logging.info("Delete the directory to re-process.")
        return

    import pandas as pd
    import pyarrow as pa
    from datasets import load_dataset, Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer
    from tqdm import tqdm

    # 1. Load OpenWebText subset
    logging.info(f"Loading first {OWT_NUM_DOCS:,} documents from OpenWebText...")
    dataset = load_dataset("openwebtext", split=f"train[:{OWT_NUM_DOCS}]",
                           trust_remote_code=True)
    total_docs = len(dataset)
    logging.info(f"Loaded {total_docs:,} documents")

    # Estimate token count (rough: ~4 chars per token)
    total_chars = sum(len(doc['text']) for doc in tqdm(dataset, desc="Counting chars"))
    est_tokens = total_chars // 4
    logging.info(f"Estimated tokens: ~{est_tokens / 1e6:.0f}M ({total_chars:,} chars)")

    # 2. Split: 90% train, 5% validation, 5% test
    logging.info("Splitting dataset (90/5/5)...")
    splits = dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = splits['train']
    val_test = splits['test'].train_test_split(test_size=0.5, seed=SEED)
    split_datasets = {
        'train': train_dataset,
        'validation': val_test['train'],
        'test': val_test['test'],
    }
    for name, ds in split_datasets.items():
        logging.info(f"  {name}: {len(ds):,} documents")

    # 3. Train tokenizer
    logging.info("Training BPE tokenizer (32K vocab)...")
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)

    if os.path.exists(TOKENIZER_PATH):
        logging.info(f"Tokenizer already exists at {TOKENIZER_PATH}. Loading.")
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    else:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=32000,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )

        def training_corpus():
            for i in range(0, len(split_datasets['train']), 1000):
                batch = split_datasets['train'][i:i + 1000]['text']
                processed = [t for t in batch if t.strip()]
                if processed:
                    yield processed

        tokenizer.train_from_iterator(training_corpus(), trainer=trainer)
        tokenizer.save(TOKENIZER_PATH)
        logging.info(f"Tokenizer saved to {TOKENIZER_PATH}")

    # 4. Preprocess each split
    max_seq_length = 256
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for split_name, split_data in split_datasets.items():
        logging.info(f"\nProcessing {split_name} split ({len(split_data):,} docs)...")
        processed_examples = []

        for example in tqdm(split_data, desc=f"Tokenizing {split_name}"):
            text = example['text'].strip()
            if not text:
                continue

            encoding = tokenizer.encode(text, add_special_tokens=False)
            tokens = encoding.ids

            for i in range(0, len(tokens), max_seq_length):
                chunk = tokens[i:i + max_seq_length]
                if len(chunk) < max_seq_length // 2:
                    continue
                if len(chunk) < max_seq_length:
                    chunk = chunk + [0] * (max_seq_length - len(chunk))
                processed_examples.append({
                    "input_ids": chunk,
                    "token_indices": list(range(len(chunk))),
                })

        hf_dataset = Dataset(pa.Table.from_pandas(pd.DataFrame(processed_examples)))
        split_path = os.path.join(PROCESSED_DIR, split_name)
        hf_dataset.save_to_disk(split_path)
        logging.info(f"Saved {split_name}: {len(hf_dataset):,} examples to {split_path}")

    logging.info("\nData preparation complete!")


# ============================================================
# Config Generation
# ============================================================
def generate_config(model_name: str) -> dict:
    """Generate a full YAML config for a given model on OpenWebText."""
    model_def = MODELS[model_name]
    run_name = f"{model_name}_owt"
    output_dir = os.path.join(OUTPUT_BASE, run_name)

    config = {
        "project": {
            "name": "SR-Research-OWT",
            "experiment_name": f"Exp003 {run_name}",
            "output_dir": output_dir,
            "seed": SEED,
        },
        "hardware": {
            "use_amp": True,
            "amp_dtype": "bf16",
        },
        "data": {
            "dataset_name": "openwebtext",
            "data_dir": DATA_DIR,
            "processed_dir": PROCESSED_DIR,
            "tokenizer_path": TOKENIZER_PATH,
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
            "batch_size_per_gpu": 128,
            "gradient_accumulation_steps": 1,
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


def save_config(config: dict, model_name: str) -> str:
    """Save config to YAML file, return path."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    config_path = os.path.join(CONFIG_DIR, f"{model_name}_owt.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path


# ============================================================
# Training
# ============================================================
def launch_training(config_path: str, gpu: int, dry_run: bool = False):
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
    log_path = os.path.join(OUTPUT_BASE, f"{run_name}.log")
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


def train_models(dry_run: bool = False):
    """Train SRA_E on GPU 1 and StdMoE_E on GPU 2 in parallel."""
    # Verify data exists (skip check for dry-run)
    train_path = os.path.join(PROCESSED_DIR, "train")
    if not os.path.exists(train_path) and not dry_run:
        logging.error(f"Processed data not found at {train_path}")
        logging.error("Run with --prepare-data first.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("STEP 2: Training models (parallel on GPU 1 + GPU 2)")
    print("=" * 60)

    gpu_assignments = {"SRA_E": 2, "StdMoE_E": 1}
    processes = []

    for model_name, gpu in gpu_assignments.items():
        print(f"\n--- {model_name} on GPU {gpu} ---")
        config = generate_config(model_name)
        config_path = save_config(config, model_name)
        print(f"  Config: {config_path}")
        print(f"  Output: {config['project']['output_dir']}")

        proc = launch_training(config_path, gpu, dry_run)
        if proc:
            processes.append((proc, model_name, gpu, config["project"]["output_dir"]))

    if dry_run:
        print("\n[DRY RUN COMPLETE]")
        return

    # Wait for both to finish
    start = time.time()
    print(f"\nWaiting for {len(processes)} parallel runs...")
    for proc, model_name, gpu, output_dir in processes:
        proc.wait()
        elapsed = time.time() - start
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"  {model_name} (GPU {gpu}): {status} ({elapsed / 3600:.1f}h)")

    total_time = time.time() - start
    print(f"\nTraining complete in {total_time / 3600:.1f}h")


# ============================================================
# Analysis
# ============================================================
def analyze_models():
    """Run analyze.py on both checkpoints."""
    print("\n" + "=" * 60)
    print("STEP 3: Running analysis")
    print("=" * 60)

    gpu_assignments = {"SRA_E": 2, "StdMoE_E": 1}

    for model_name, gpu in gpu_assignments.items():
        output_dir = os.path.join(OUTPUT_BASE, f"{model_name}_owt")
        checkpoint = os.path.join(output_dir, "best_model.pt")

        if not os.path.exists(checkpoint):
            print(f"  WARNING: No checkpoint at {checkpoint}, skipping {model_name}")
            continue

        print(f"\n--- Analyzing {model_name} ---")
        cmd = [
            "python3", os.path.join(PROJECT_ROOT, "scripts", "analyze.py"),
            "--checkpoint", checkpoint,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        log_path = os.path.join(OUTPUT_BASE, f"{model_name}_owt_analysis.log")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=PROJECT_ROOT)
        with open(log_path, "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        if result.returncode == 0:
            print(f"  Analysis complete: {log_path}")
        else:
            print(f"  Analysis FAILED (exit {result.returncode}): {log_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 003: Cross-dataset validation on OpenWebText"
    )
    parser.add_argument("--prepare-data", action="store_true",
                        help="Download and preprocess OpenWebText subset")
    parser.add_argument("--train", action="store_true",
                        help="Train SRA_E + StdMoE_E in parallel (GPU 1 + GPU 2)")
    parser.add_argument("--analyze", action="store_true",
                        help="Run analysis on completed checkpoints")
    parser.add_argument("--all", action="store_true",
                        help="Run all steps: prepare-data, train, analyze")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without launching training")
    args = parser.parse_args()

    if not any([args.prepare_data, args.train, args.analyze, args.all]):
        parser.print_help()
        print("\nSteps:")
        print("  1. --prepare-data   Download & preprocess OpenWebText (CPU, ~30 min)")
        print("  2. --train          Train 2 models in parallel (GPU 1+2, ~12h)")
        print("  3. --analyze        Run IC/EP analysis (~15 min)")
        print("  4. --all            Run all steps sequentially")
        sys.exit(0)

    if args.all or args.prepare_data:
        prepare_data()

    if args.all or args.train:
        train_models(args.dry_run)

    if (args.all or args.analyze) and not args.dry_run:
        analyze_models()

    print("\n" + "=" * 60)
    print("EXPERIMENT 003 COMPLETE")
    print("=" * 60)
