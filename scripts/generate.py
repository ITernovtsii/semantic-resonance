# python3 scripts/generate.py --config configs/sra_wikitext103-ablation1-full-sre-base.yaml --checkpoint outputs_wt103-ablation/1-full/best_model.pt --tokenizer data/tokenizers/sra_bpe_32k_wt103.json --demo
# python3 scripts/generate.py --config configs/sra_wikitext103-ablation2-noHSE.yaml --checkpoint outputs_wt103-ablation/2-noHSE/best_model.pt --tokenizer data/tokenizers/sra_bpe_32k_wt103.json --demo
# python3 scripts/generate.py --config configs/sra_wikitext103-ablation3-no-dispersion.yaml --checkpoint outputs_wt103-ablation/3-noDispersion/best_model.pt --tokenizer data/tokenizers/sra_bpe_32k_wt103.json --demo
# python3 scripts/generate.py --config configs/sra_wikitext103-ablation4-standard-MoE.yaml --checkpoint outputs_wt103-ablation/4-standard-MoE/best_model.pt --tokenizer data/tokenizers/sra_bpe_32k_wt103.json --demo
# python3 scripts/generate.py --config configs/sra_wikitext103-ablation5-dense-baseline.yaml --checkpoint outputs_wt103-ablation/5-dense-baseline/best_model.pt --tokenizer data/tokenizers/sra_bpe_32k_wt103.json --demo
import torch
import argparse
import logging
import sys
import os
import time
from tokenizers import Tokenizer

# Ensure the src directory is in PYTHONPATH for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Assuming the project structure allows these imports
from src.models.sra import SRA
from src.utils.config_utils import load_config, convert_to_dot_notation

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextGenerator:
    """
    A class to handle model loading, tokenization, and text generation.
    """
    def __init__(self, config, checkpoint_path, tokenizer_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.max_seq_len = self.config.model.get('max_seq_length', 256) - 1 # Reserve one token for generation

        # Load Tokenizer
        logging.info(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = self.tokenizer.get_vocab_size()

        # Load Model
        logging.info("Initializing model...")
        self.model = SRA(self.config.model, vocab_size)

        logging.info(f"Loading model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Handle potential DataParallel or DDP wrappers
        state_dict = checkpoint['model_state_dict']
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k # remove `module.`
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
        logging.info("Model loaded successfully and moved to device.")

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=0.7, top_k=40, repetition_penalty=1.2):
        """
        Generates text based on a given prompt.
        This method is adapted from the SRATrainer._generate_samples method.
        """
        try:
            # 1. Tokenize the prompt
            prompt_encoding = self.tokenizer.encode(prompt)
            input_ids_tensor = torch.tensor(prompt_encoding.ids, dtype=torch.long).unsqueeze(0).to(self.device)

            # 2. Autoregressive Generation Loop
            for _ in range(max_new_tokens):
                # Truncate input to model's max sequence length
                current_ids = input_ids_tensor[:, -self.max_seq_len:]

                # Approximate HSE indices for the current sequence
                token_indices, word_indices, sentence_indices = self._approximate_hse_indices(current_ids[0].tolist())

                # Prepare model inputs
                inputs = {
                    'input_ids': current_ids,
                    'token_indices': torch.tensor(token_indices, dtype=torch.long).unsqueeze(0).to(self.device),
                    'word_indices': torch.tensor(word_indices, dtype=torch.long).unsqueeze(0).to(self.device),
                    'sentence_indices': torch.tensor(sentence_indices, dtype=torch.long).unsqueeze(0).to(self.device),
                }

                # Forward pass
                logits, _ = self.model(inputs)
                next_token_logits = logits[:, -1, :]

                # Apply Repetition Penalty
                if repetition_penalty > 1.0 and input_ids_tensor.shape[1] > 0:
                    generated_ids = input_ids_tensor[0]
                    # Ensure indices are within the vocab size
                    valid_ids = generated_ids[generated_ids < next_token_logits.size(-1)]
                    if valid_ids.numel() > 0:
                        scores = next_token_logits[0, valid_ids]
                        penalized_scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
                        next_token_logits[0, valid_ids] = penalized_scores

                # Apply temperature and Top-K sampling
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

                # Sample the next token
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append the new token
                input_ids_tensor = torch.cat([input_ids_tensor, next_token], dim=1)

                # Stop if EOS token is generated
                if next_token.item() == self.tokenizer.token_to_id("[EOS]"):
                    break

            # Decode the final sequence
            return self.tokenizer.decode(input_ids_tensor[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

        except Exception as e:
            logging.error(f"Error during generation for prompt '{prompt}': {e}", exc_info=True)
            return "Generation failed."

    def _approximate_hse_indices(self, input_ids_list):
        """
        Approximates HSE indices. Adapted from SRATrainer.
        """
        text = self.tokenizer.decode(input_ids_list, skip_special_tokens=True)
        encoding = self.tokenizer.encode(text)

        if len(encoding.ids) != len(input_ids_list):
            logging.warning("Token mismatch during HSE approximation. Using basic indices.")
            token_indices = list(range(len(input_ids_list)))
            return token_indices, token_indices, [0] * len(input_ids_list)

        token_indices = list(range(len(encoding.ids)))

        word_indices = []
        current_word_idx = 0
        for wid in encoding.word_ids:
            if wid is not None:
                current_word_idx = wid
            word_indices.append(current_word_idx)

        sentence_indices = []
        current_sentence_idx = 0
        for i, token_id in enumerate(encoding.ids):
            sentence_indices.append(current_sentence_idx)
            token_str = self.tokenizer.id_to_token(token_id)
            if token_str and any(punc in token_str for punc in '.!?'):
                if i < len(encoding.ids) - 1:
                    current_sentence_idx += 1

        # Ensure all lists have the same length as the input
        final_len = len(input_ids_list)
        return token_indices[:final_len], word_indices[:final_len], sentence_indices[:final_len]


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained SRA model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model config YAML file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pt) file.")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json", help="Path to the tokenizer.json file.")
    parser.add_argument("--demo", action="store_true", help="Run with 10 predefined demo prompts instead of interactive mode.")
    parser.add_argument("--max_new_tokens", type=int, default=75, help="Maximum number of new tokens to generate.")

    args = parser.parse_args()

    # Load configuration
    config_dict = load_config(args.config)
    config = convert_to_dot_notation(config_dict)

    # Initialize generator
    generator = TextGenerator(config, args.checkpoint, args.tokenizer)

    if args.demo:
        # --- Demo Mode ---
        logging.info("Running in --demo mode...")
        prompts = [
            "The history of the computer began with",
            "Science is the pursuit of knowledge and understanding of the",
            "In the future, technology will likely allow us to",
            "The main purpose of government is to",
            "According to recent studies, the Earth's climate is",
            "The development of artificial intelligence raises questions about",
            "Climate change has resulted in more frequent",
            "The capital of France is Paris, which is known for",
            "Modern society depends on a complex web of",
            "Research has shown that regular exercise can"
        ]

        # Start timing for demo outputs
        demo_start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            print("-" * 50)
            print(f"DEMO PROMPT {i+1}/10")
            print(f"PROMPT: {prompt}")
            generated_text = generator.generate(prompt, max_new_tokens=args.max_new_tokens)
            print(f"GENERATED: {generated_text}")
        
        # End timing and calculate total time
        demo_end_time = time.time()
        total_demo_time = demo_end_time - demo_start_time
        
        print("-" * 50)
        print(f"TOTAL TIME FOR DEMO OUTPUTS: {total_demo_time:.2f} seconds")
        print("-" * 50)

    else:
        # --- Interactive Mode ---
        logging.info("Running in interactive mode. Type 'exit' or 'quit' to stop.")
        while True:
            prompt = input("Enter your prompt: ")
            if prompt.lower() in ['exit', 'quit']:
                break

            print("...generating...")
            generated_text = generator.generate(prompt, max_new_tokens=args.max_new_tokens)
            print("-" * 50)
            print(f"PROMPT: {prompt}")
            print(f"GENERATED: {generated_text}")
            print("-" * 50)

if __name__ == "__main__":
    main()
