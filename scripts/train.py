import argparse
import logging
import os
import torch
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.utils.config_utils import load_config, ConfigDict
from src.data.dataset import create_dataloaders
from src.models.sra import SRA
from src.training.trainer import SRATrainer
from tokenizers import Tokenizer


def set_seed(seed):
    """Set seed for reproducibility across ranks (Accelerate if available)."""
    try:
        from accelerate.utils import set_seed as accel_set_seed
        accel_set_seed(int(seed), device_specific=True)
        return
    except Exception:
        pass
    # Fallback legacy seeding
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(config_path, resume_checkpoint=None):
    if resume_checkpoint:
        logging.info(f"Starting SRA Training Pipeline with checkpoint resume from: {resume_checkpoint}")
    else:
        logging.info("Starting SRA Training Pipeline.")

    # 1. Load Configuration
    config_dict = load_config(config_path)
    if not config_dict:
        logging.error("Failed to load configuration.")
        return
    config = ConfigDict(config_dict)

    # Set seed
    set_seed(config.project.seed)

    # 2. Prepare Output Directory
    os.makedirs(config.project.output_dir, exist_ok=True)

    # 3. Load Tokenizer
    try:
        tokenizer = Tokenizer.from_file(config.data.tokenizer_path)
        logging.info(f"Tokenizer loaded. Vocab size: {tokenizer.get_vocab_size()}")
        # Ensure config vocab size matches tokenizer
        if config.data.vocab_size != tokenizer.get_vocab_size():
            logging.warning(f"Config vocab size mismatch. Updating config to {tokenizer.get_vocab_size()}.")
            config.data.vocab_size = tokenizer.get_vocab_size()

    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}. Ensure prepare_data.py was run.")
        return

    # 4. Initialize DataLoaders
    try:
        # The create_dataloaders function handles loading the processed Arrow datasets
        train_loader, val_loader, _ = create_dataloaders(
            config.data,
            batch_size=config.training.batch_size_per_gpu
        )
        logging.info(f"DataLoaders initialized. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except Exception as e:
        logging.error(f"Failed to initialize DataLoaders. Ensure data is preprocessed. Error: {e}")
        return

    # 5. Initialize Model
    try:
        model = SRA(config.model, config.data.vocab_size)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model initialized. Total trainable parameters: {num_params:,}")

        if config.model.csr.enabled:
            logging.info(
                f"SRA Architecture: CSR (MoE) is ENABLED. Experts: {config.model.csr.num_experts}, Top-K: {config.model.csr.top_k}")
        else:
            logging.info("Baseline Architecture: CSR is DISABLED (Dense FFN).")
    except Exception as e:
        logging.error(f"Failed to initialize SRA model: {e}", exc_info=True)
        return

    # 6. Initialize Trainer
    try:
        # The Trainer handles Accelerate initialization, Optimizer, Scheduler, and the training loop
        # Pass tokenizer for sample generation
        trainer = SRATrainer(config, model, train_loader, val_loader, tokenizer=tokenizer)
        logging.info("Trainer initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Trainer: {e}", exc_info=True)
        return

    # 6.5. Load Checkpoint if Resuming
    if resume_checkpoint:
        try:
            if not os.path.exists(resume_checkpoint):
                logging.error(f"Checkpoint file not found: {resume_checkpoint}")
                return

            trainer.load_checkpoint(resume_checkpoint)
            logging.info("Successfully loaded checkpoint for resume training.")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}", exc_info=True)
            return

    # 7. Start Training
    logging.info("--- Launching Training (use accelerate launch for multi-GPU/optimized setup) ---")
    try:
        trainer.train()
        logging.info("--- Training Completed Successfully ---")
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    except Exception as e:
        logging.error(f"Training failed with error: {e}", exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the SRA model.")
    parser.add_argument("--config", type=str, default="configs/sra_base.yaml",
                        help="Path to the configuration YAML file.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint file to resume training from.")
    args = parser.parse_args()
    main(args.config, args.resume)
