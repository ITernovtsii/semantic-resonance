import torch
from torch.utils.data import Dataset, DataLoader
import logging
import os
from datasets import load_from_disk

class SimpleTokenizedDataset(Dataset):
    def __init__(self, dataset_path):
        logging.info(f"Loading dataset from {dataset_path}...")
        try:
            self.data = load_from_disk(dataset_path)
            logging.info(f"Loaded {len(self.data)} examples.")
        except Exception as e:
            logging.error(f"Failed to load dataset from {dataset_path}. Error: {e}")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.data:
             raise IndexError("Dataset is empty or failed to load.")

        item = self.data[idx]

        # We only need input_ids
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)

        inputs = {
            'input_ids': input_ids[:-1],
            'labels': input_ids[1:].clone()
        }
        return inputs

def create_dataloaders(data_config, batch_size, num_workers=None):
    if num_workers is None:
        nw = getattr(data_config, 'num_workers', None)
        num_workers = int(nw) if nw is not None else 4
    processed_dir = data_config.processed_dir

    # Check that the directory exists
    if not os.path.exists(processed_dir):
        logging.error(f"Processed data directory not found: {processed_dir}. Please run prepare_data.py first.")
        raise FileNotFoundError(f"Directory not found: {processed_dir}")

    try:
        # Use the simplified dataset class
        train_dataset = SimpleTokenizedDataset(os.path.join(processed_dir, "train"))
        val_dataset = SimpleTokenizedDataset(os.path.join(processed_dir, "validation"))
        test_dataset = SimpleTokenizedDataset(os.path.join(processed_dir, "test"))
    except Exception as e:
        logging.error(f"Error initializing datasets: {e}")
        raise

    # Worker seeding for determinism across ranks
    def seed_worker(worker_id):
        try:
            import numpy as np, random
            worker_seed = torch.initial_seed() % (2**32)
            np.random.seed(worker_seed + worker_id)
            random.seed(worker_seed + worker_id)
        except Exception:
            pass

    extra_worker_args = {}
    if num_workers and num_workers > 0:
        extra_worker_args.update({"persistent_workers": True, "prefetch_factor": 2, "worker_init_fn": seed_worker})
    else:
        extra_worker_args.update({"worker_init_fn": seed_worker})

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, **extra_worker_args
    )

    # For validation and test, shuffle=False
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, **extra_worker_args)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, **extra_worker_args)

    return train_loader, val_loader, test_loader
