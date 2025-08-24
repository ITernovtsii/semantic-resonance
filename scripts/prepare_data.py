import argparse
import logging
import os
import re

import pandas as pd
import pyarrow as pa
import yaml
from datasets import load_dataset, Dataset
from sacremoses import MosesDetokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Moses detokenizer globally
moses_detokenizer = MosesDetokenizer(lang='en')


# --- Utilities ---

def load_config(config_path):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading or parsing config file {config_path}: {e}")
        return None
    return config


def detokenize_wikitext(text):
    """
    Fully detokenizes WikiText using Moses detokenizer.
    Handles special tokens like @-@, @.@ and applies Moses detokenization.
    """
    if not text or not text.strip():
        return text

    # First handle WikiText-specific artifacts (these are not standard Moses)
    # Handle @-@ tokens (split hyphens) - most common in WikiText
    text = re.sub(r'\s*@-@\s*', '-', text)

    # Handle @.@ tokens (split dots)
    text = re.sub(r'\s*@\.@\s*', '.', text)

    # Handle @,@ tokens (split commas)
    text = re.sub(r'\s*@,@\s*', ',', text)

    # Handle @/@ tokens (split slashes) if present
    text = re.sub(r'\s*@/@\s*', '/', text)

    # Now apply Moses detokenization for proper spacing and punctuation
    # Moses expects a list of tokens
    tokens = text.split()
    text = moses_detokenizer.detokenize(tokens)

    return text


def clean_wikitext_content(text):
    """
    Additional cleaning specific to WikiText format.
    Removes section markers, handles <unk> tokens, and other WikiText artifacts.
    """
    if not text:
        return text

    # Remove section markers (e.g., = = Section Name = =, = Title =)
    text = re.sub(r'^\s*=+\s*.*?\s*=+\s*$', '', text, flags=re.MULTILINE)

    # Optionally handle <unk> tokens (unknown words in WikiText)
    # You can either remove them or replace with a placeholder, we remove
    text = text.replace('<unk>', '')

    # Remove multiple consecutive newlines and spaces
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def get_training_corpus(dataset, detokenize=False):
    """
    Generator to yield batches of text for tokenizer training.

    Args:
        dataset: The dataset to iterate over
        detokenize: Whether to apply detokenization (for WikiText datasets)
    """
    for i in range(0, len(dataset), 1000):
        batch = dataset[i: i + 1000]["text"]

        # Process each text in the batch
        processed_batch = []
        for text in batch:
            if text.strip():
                if detokenize:
                    text = detokenize_wikitext(text)
                    text = clean_wikitext_content(text)
                if text.strip():  # Check again after cleaning
                    processed_batch.append(text)

        if processed_batch:
            yield processed_batch


# --- Main Preparation Steps ---

def train_tokenizer(config, dataset, is_wikitext=False):
    """
    Trains and saves BPE tokenizer.

    Args:
        config: Configuration dictionary
        dataset: Training dataset
        is_wikitext: Whether the dataset is WikiText (needs detokenization)
    """
    logging.info("--- 1. Tokenizer Training ---")
    tokenizer_path = config['data']['tokenizer_path']
    vocab_size = config['data']['vocab_size']

    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)

    if os.path.exists(tokenizer_path):
        logging.info(f"Tokenizer already exists at {tokenizer_path}. Loading.")
        return Tokenizer.from_file(tokenizer_path)

    logging.info(f"Training new BPE tokenizer with vocab size {vocab_size}...")
    if is_wikitext:
        logging.info("WikiText detected - will detokenize Moses artifacts before training")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Define base special tokens
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    # Train with detokenization if needed
    tokenizer.train_from_iterator(
        get_training_corpus(dataset, detokenize=is_wikitext),
        trainer=trainer
    )

    tokenizer.save(tokenizer_path)
    logging.info(f"Tokenizer trained and saved to {tokenizer_path}")
    return tokenizer


def preprocess_data(config, dataset_split, tokenizer, is_wikitext=False):
    """
    Basic preprocessing: detokenization (if needed), tokenization and chunking.

    Args:
        config: Configuration dictionary
        dataset_split: Dataset split to process
        tokenizer: Trained tokenizer
        is_wikitext: Whether to apply WikiText detokenization
    """
    logging.info("--- 2. Data Preprocessing ---")
    max_seq_length = config['data']['max_seq_length']

    processed_examples = []

    for example in tqdm(dataset_split, desc="Processing examples"):
        text = example['text'].strip()
        if not text:
            continue

        # Detokenize if this is WikiText
        if is_wikitext:
            text = detokenize_wikitext(text)
            text = clean_wikitext_content(text)
            if not text:  # Skip if cleaning resulted in empty text
                continue

        encoding = tokenizer.encode(text, add_special_tokens=False)
        tokens = encoding.ids

        # For long texts, split into chunks of max_seq_length
        for i in range(0, len(tokens), max_seq_length):
            chunk_tokens = tokens[i:i + max_seq_length]

            # Skip chunks that are too small
            if len(chunk_tokens) < max_seq_length // 2:
                continue

            # Pad if necessary
            if len(chunk_tokens) < max_seq_length:
                chunk_tokens = chunk_tokens + [0] * (max_seq_length - len(chunk_tokens))

            # Create example with metadata
            processed_example = {
                "input_ids": chunk_tokens,
                "token_indices": list(range(len(chunk_tokens))),
            }

            # Add metadata from original example (for cosmopedia)
            for field in ['prompt', 'audience', 'format', 'seed_data', 'token_length']:
                if field in example:
                    processed_example[field] = example[field]

            processed_examples.append(processed_example)

    return processed_examples


def main(config_path):
    config = load_config(config_path)
    if not config:
        return

    # 1. Load Dataset
    dataset_name_config = config['data'].get('dataset_name', 'cosmopedia-v2')
    is_wikitext = False

    logging.info(f"Loading dataset: {dataset_name_config}")
    try:
        if dataset_name_config == 'cosmopedia-v2':
            dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train")
            # Split dataset into train, validation, test (90%, 5%, 5%)
            splits = dataset.train_test_split(test_size=0.1)
            train_dataset = splits['train']
            test_valid = splits['test'].train_test_split(test_size=0.5)
            dataset = {
                'train': train_dataset,
                'validation': test_valid['train'],
                'test': test_valid['test']
            }
        elif dataset_name_config.startswith('wikitext'):
            is_wikitext = True
            # Handle WikiText datasets
            if dataset_name_config == 'wikitext-103':
                hf_dataset_name = "wikitext"
                hf_config_name = "wikitext-103-raw-v1"
            elif dataset_name_config == 'wikitext-2':
                hf_dataset_name = "wikitext"
                hf_config_name = "wikitext-2-raw-v1"
            else:
                logging.error(f"Unknown dataset name in config: {dataset_name_config}")
                return

            logging.info(f"Loading WikiText dataset - will apply Moses detokenization")
            dataset = load_dataset(hf_dataset_name, hf_config_name)
        else:
            logging.error(f"Unknown dataset name in config: {dataset_name_config}")
            return
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # 2. Train Tokenizer
    tokenizer = train_tokenizer(config, dataset['train'], is_wikitext=is_wikitext)

    # 3. Preprocessing and Saving
    processed_data_dir = config['data']['processed_dir']
    os.makedirs(processed_data_dir, exist_ok=True)

    for split in ['train', 'validation', 'test']:
        logging.info(f"\n--- Processing {split} split ---")
        processed_split_data = preprocess_data(
            config,
            dataset[split],
            tokenizer,
            is_wikitext=is_wikitext
        )

        # Convert to HuggingFace Dataset (Arrow format) for efficient storage
        # This allows fast data loading during training.
        hf_dataset = Dataset(pa.Table.from_pandas(pd.DataFrame(processed_split_data)))

        split_path = os.path.join(processed_data_dir, split)
        hf_dataset.save_to_disk(split_path)
        logging.info(f"Successfully saved processed {split} split (Examples: {len(hf_dataset)}) to {split_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare data for SRA model.")
    parser.add_argument("--config", type=str, default="configs/sra_base.yaml",
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()
    main(args.config)
