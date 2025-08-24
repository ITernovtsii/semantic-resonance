import torch
import torch.optim as optim
from accelerate import Accelerator
from transformers import get_scheduler
import math
import logging
import os
import random
import numpy as np
from tqdm.auto import tqdm
from src.training.losses import calculate_sra_losses

class SRATrainer:
    # Added tokenizer for generation
    def __init__(self, config, model, train_loader, val_loader, tokenizer=None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.epochs = config.training.epochs

        # Initialize Accelerator
        try:
            self.accelerator = Accelerator(
                mixed_precision=(config.hardware.amp_dtype if getattr(config.hardware, 'use_amp', False) else "no"),
                gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                log_with=("wandb" if getattr(config.logging, 'use_wandb', False) else "tensorboard"),
                project_dir=config.project.output_dir
            )
        except AttributeError as e:
            # Handle rare Accelerate state mismatch (_mixed_precision missing)
            if "_mixed_precision" in str(e):
                logging.warning("Accelerate state mismatch detected; falling back to safe Accelerator initialization.")
                try:
                    from accelerate.state import AcceleratorState
                    try:
                        AcceleratorState._reset_state()
                    except Exception:
                        pass
                except Exception:
                    pass
                # retry after _reset_state
                self.accelerator = Accelerator(
                    # mixed_precision=(config.hardware.amp_dtype if getattr(config.hardware, 'use_amp', False) else "no"),
                    gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                    log_with=("wandb" if getattr(config.logging, 'use_wandb', False) else "tensorboard"),
                    project_dir=config.project.output_dir
                )
            else:
                raise

        self.device = self.accelerator.device
        # Optimizer is initialized based on the initial model structure
        self.optimizer = self._configure_optimizers()
        self.lr_scheduler = self._configure_scheduler()

        # Prepare everything with accelerate. DeepSpeed initializes here.
        self.model, self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler
        )

        # Initialize Tracking
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_ppl = float('inf')
        self._init_tracking()

    def _init_tracking(self):
        # Initialize W&B or TensorBoard
        if self.accelerator.is_main_process:
            # Ensure config is serializable
            run_config = self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config
            self.accelerator.init_trackers(
                project_name=self.config.project.name,
                config=run_config,
                init_kwargs={"wandb": {"name": self.config.project.experiment_name}}
            )
            logging.info(f"Initialized tracking. Experiment: {self.config.project.experiment_name}")

    def _configure_optimizers(self):
        # Decoupled Weight Decay (AdamW)
        # Exclude biases, LayerNorm weights, and potentially Semantic Anchors from decay
        no_decay = ["bias", "LayerNorm.weight", "ln_f.weight", "ln1.weight", "ln2.weight", "semantic_anchors"]

        # Always creates exactly 2 groups
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.training.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # Check for 'fused' implementation availability (faster on modern GPUs like RTX 3090)
        use_fused = torch.cuda.is_available() and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames

        try:
            lr = float(self.config.training.learning_rate)
            beta1 = float(self.config.training.adam_beta1)
            beta2 = float(self.config.training.adam_beta2)
            epsilon = float(self.config.training.adam_epsilon)
        except (TypeError, ValueError) as e:
            logging.error(f"Error casting optimizer parameters to float. Check config values. Error: {e}")
            raise
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=(beta1, beta2),
            eps=epsilon,
            fused=use_fused
        )
        return optimizer

    def _configure_scheduler(self):
        # Calculate total training steps
        num_update_steps_per_epoch = math.ceil(
            len(self.train_loader) / self.config.training.gradient_accumulation_steps)
        num_training_steps = num_update_steps_per_epoch * self.epochs

        lr_scheduler = get_scheduler(
            name=self.config.training.scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    def train(self):
        """Main training loop."""
        logging.info("Starting training...")

        # Rely on the restored self.current_epoch instead of calculating from global_step.
        # This ensures we resume in the correct epoch even if the batch size or GPU count has changed.
        # Accelerate handles skipping the already processed batches within that epoch via the restored sampler state.

        start_epoch = self.current_epoch

        # Update logging message for clarity
        if self.global_step > 0:
            # The epoch logging reflects the epoch based on the training progress, regardless of the current configuration.
            logging.info(f"Resuming training from Epoch {start_epoch + 1} (Global step {self.global_step})")

        # Iterate starting from the restored epoch
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            logging.info(f"--- Epoch {epoch + 1}/{self.epochs} ---")

            # Training phase
            self._train_epoch(epoch)

            # Validation phase
            val_metrics = self.validate()
            self.accelerator.log(val_metrics, step=self.global_step)

            # Generation Callback (only on main process)
            if self.accelerator.is_main_process:
                # We must unwrap the model to access custom methods like generate_simple
                try:
                    # unwrap_model() can trigger DeepSpeed initialization if installed
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    self._generate_samples(unwrapped_model)
                except Exception as e:
                    logging.error(
                        f"Error during generation callback (potentially DeepSpeed/CUDA environment issue). Training continues. Error: {e}")
                    if "CUDA_HOME" in str(e) or "DeepSpeed" in str(type(e)) or "MissingCUDAException" in str(
                            type(e).__name__):
                        logging.warning(
                            "The error seems related to DeepSpeed. Ensure CUDA_HOME is set or uninstall DeepSpeed if not needed.")

            # Checkpointing
            if val_metrics["val/perplexity"] < self.best_val_ppl:
                self.best_val_ppl = val_metrics["val/perplexity"]
                self.save_checkpoint(f"best_model.pt")
                logging.info(f"New best model saved with PPL: {self.best_val_ppl:.4f}")

        self.accelerator.end_training()
        logging.info("Training finished.")

    def _train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Training Ep {epoch + 1}", disable=not self.accelerator.is_main_process)

        # Calculate intervals for generation
        total_batches = len(self.train_loader)
        generation_intervals = [round(total_batches * p) for p in (0.35, 0.70)] # twice per epoch
        batch_idx = 0

        for batch in pbar:
            # The accelerator handles the context for gradient accumulation
            with self.accelerator.accumulate(self.model):

                inputs = {'input_ids': batch['input_ids']}
                labels = batch['labels']

                # Forward Pass
                logits, aux_data_list = self.model(inputs)

                # Loss Calculation (Centralized)
                loss_dict = calculate_sra_losses(
                    logits, labels, aux_data_list, self.config.losses, self.config.model.csr.enabled
                )
                loss = loss_dict["total_loss"]

                # Backward Pass
                self.accelerator.backward(loss)

                # Optimization Step
                if self.accelerator.sync_gradients:
                    # Gradient Clipping
                    if self.config.training.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)  # Efficient zero_grad
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.logging.log_interval == 0:
                        def get_scalar_value(value):
                            """Helper function to extract scalar value from tensor or float"""
                            if hasattr(value, 'item'):
                                return value.item()
                            return float(value)

                        metrics = {
                            "train/loss": loss.item(),
                            "train/lm_loss": get_scalar_value(loss_dict["lm_loss"]),
                            "train/lr": self.lr_scheduler.get_last_lr()[0],
                        }
                        if "aux_loss" in loss_dict:
                            metrics["train/aux_loss"] = get_scalar_value(loss_dict["aux_loss"])
                        if "z_loss" in loss_dict:
                            metrics["train/z_loss"] = get_scalar_value(loss_dict["z_loss"])
                        if "balance_loss" in loss_dict:
                            metrics["train/balance_loss"] = get_scalar_value(loss_dict["balance_loss"])
                        if "dispersion_loss" in loss_dict:
                            metrics["train/dispersion_loss"] = get_scalar_value(loss_dict["dispersion_loss"])

                        self.accelerator.log(metrics, step=self.global_step)
                        pbar.set_postfix(
                            {"Loss": f"{loss.item():.4f}", "LR": f"{self.lr_scheduler.get_last_lr()[0]:.2e}"})

            # Increment batch counter and check for generation intervals
            batch_idx += 1

            # Generation Callback every 25% of batch (only on main process)
            if self.accelerator.is_main_process and batch_idx in generation_intervals:
                # We must unwrap the model to access custom methods like generate_simple
                try:
                    # unwrap_model() can trigger DeepSpeed initialization if installed
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    self._generate_samples(unwrapped_model)
                except Exception as e:
                    logging.error(
                        f"Error during generation callback (potentially DeepSpeed/CUDA environment issue). Training continues. Error: {e}")
                    if "CUDA_HOME" in str(e) or "DeepSpeed" in str(type(e)) or "MissingCUDAException" in str(
                            type(e).__name__):
                        logging.warning(
                            "The error seems related to DeepSpeed. Ensure CUDA_HOME is set or uninstall DeepSpeed if not needed.")

    @torch.no_grad()
    def validate(self):
        """Validation loop."""
        self.model.eval()
        total_lm_loss = 0.0
        pbar = tqdm(self.val_loader, desc="Validating", disable=not self.accelerator.is_main_process)

        for batch in pbar:
            inputs = {'input_ids': batch['input_ids']}
            labels = batch['labels']

            logits, aux_data_list = self.model(inputs)

            # We only care about LM loss for validation perplexity
            loss_dict = calculate_sra_losses(
                logits, labels, aux_data_list, self.config.losses, False  # Disable aux losses for val PPL calculation
            )
            lm_loss = loss_dict["lm_loss"]

            # Gather losses from all processes (important for distributed training)
            gathered_loss = self.accelerator.gather(lm_loss)
            total_lm_loss += gathered_loss.mean().item()

        avg_lm_loss = total_lm_loss / len(self.val_loader)

        # Calculate Perplexity
        try:
            perplexity = math.exp(avg_lm_loss)
        except OverflowError:
            perplexity = float('inf')

        logging.info(f"Validation Loss: {avg_lm_loss:.4f}, Perplexity: {perplexity:.4f}")
        return {"val/loss": avg_lm_loss, "val/perplexity": perplexity}

    @torch.no_grad()
    def _generate_samples(self, model, max_new_tokens=50, temperature=0.7, repetition_penalty=1.2, top_k=50):
        """Generates text samples for qualitative assessment."""
        if self.tokenizer is None:
            return

        logging.info("Generating text samples...")
        was_training = model.training
        model.eval()

        # Define prompt contexts relevant to WikiText
        prompts = [
            "The history of the",
            "Science is defined as",
            "In the future, technology will",
            "The main purpose of",
            "According to recent studies,",
            "The development of artificial intelligence",
            "Climate change has resulted in",
            "The capital of France is",
            "Modern society depends on",
            "Research has shown that 1 + 1 ="
        ]

        generated_texts = []
        max_context_len = self.config.data.max_seq_length - 1

        for prompt in prompts:
            try:
                # Tokenize the prompt
                prompt_encoding = self.tokenizer.encode(prompt)
                input_ids_tensor = torch.tensor(prompt_encoding.ids, dtype=torch.long).unsqueeze(0).to(self.device)

                # Autoregressive Generation Loop
                for _ in range(max_new_tokens):
                    # Prepare inputs: Truncate context to max length
                    current_ids = input_ids_tensor[:, -max_context_len:]
                    inputs = {
                        'input_ids': current_ids,
                    }

                    # 4. Forward pass
                    logits, _ = model(inputs)
                    next_token_logits = logits[:, -1, :]

                    # 5. Apply Repetition Penalty
                    if repetition_penalty > 1.0:
                        # Assuming batch size 1 (as input_ids_tensor is initialized with unsqueeze(0))
                        generated_ids = input_ids_tensor[0]
                        # Only consider non-padded tokens for repetition penalty
                        pad_token_id = self.tokenizer.token_to_id(
                            "[PAD]") if "[PAD]" in self.tokenizer.get_vocab() else 0
                        non_pad_mask = generated_ids != pad_token_id
                        unique_ids = generated_ids[non_pad_mask].unique()

                        if unique_ids.max() < next_token_logits.size(-1):
                            scores = next_token_logits[0, unique_ids]
                            penalized_scores = torch.where(
                                scores < 0,
                                scores * repetition_penalty,
                                scores / repetition_penalty
                            )
                            next_token_logits[0, unique_ids] = penalized_scores

                    # 6. Apply temperature and Top-K
                    next_token_logits = next_token_logits / temperature
                    if top_k > 0:
                        k = min(top_k, next_token_logits.size(-1))
                        v, _ = torch.topk(next_token_logits, k)
                        next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

                    # 7. Sample and Append
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    input_ids_tensor = torch.cat([input_ids_tensor, next_token], dim=1)

                # Decode the final sequence
                generated_text = self.tokenizer.decode(input_ids_tensor[0].tolist(), skip_special_tokens=True)
                generated_texts.append((prompt, generated_text))

            except Exception as e:
                logging.error(f"Error during generation for prompt '{prompt}': {e}")
                continue

        # Log generated texts to W&B and console
        # We check if W&B is initialized via accelerator's loggers
        if self.config.logging.use_wandb and self.accelerator.is_main_process:
            try:
                import wandb
                # Log as a W&B Table for better visualization
                table = wandb.Table(columns=["Step", "Prompt", "Generated Text"])
                for prompt, text in generated_texts:
                    # Check if text exists before adding
                    if text:
                        table.add_data(self.global_step, prompt, text)

                # Safely get the underlying W&B run object (unwrap=True)
                # This is the recommended way to log complex objects (tables) with Accelerate
                wandb_tracker = self.accelerator.get_tracker("wandb", unwrap=True)

                # Log table directly through wandb run object
                if hasattr(wandb_tracker, "log"):
                    wandb_tracker.log({"generation_samples": table}, step=self.global_step)
                else:
                    logging.warning("Could not access W&B tracker log method.")

            except ImportError:
                logging.warning("W&B configured but 'wandb' library not installed.")
            except Exception as e:
                # Catch potential issues if the tracker wasn't initialized correctly
                logging.warning(f"Failed to log generation table to W&B: {e}")

        logging.info("--- Generation Samples ---")
        for prompt, text in generated_texts:
            if text:
                logging.info(f"[PROMPT]: {prompt}\n[GENERATED]: {text}\n")
        logging.info("--------------------------")

        if was_training:
            model.train()

    def save_checkpoint(self, filename):
        """Saves the model checkpoint."""
        # Ensure output directory exists
        os.makedirs(self.config.project.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.project.output_dir, filename)

        self.accelerator.wait_for_everyone()

        # unwrap_model() is crucial to save the raw model, not the DDP/FSDP wrapper
        # unwrapped_model = self.accelerator.unwrap_model(self.model)
        try:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        except Exception as e:
            logging.error(
                f"Could not unwrap model during checkpointing (DeepSpeed issue?): {e}. Attempting to save wrapped model.")
            unwrapped_model = self.model  # Fallback

        # Gather model state dict robustly (Accelerate-aware when available)
        try:
            if hasattr(self.accelerator, "get_state_dict"):
                model_state_dict = self.accelerator.get_state_dict(self.model)
            else:
                model_state_dict = unwrapped_model.state_dict()
        except Exception as e:
            logging.warning(f"Accelerate get_state_dict failed ({e}); falling back to unwrapped_model.state_dict().")
            model_state_dict = unwrapped_model.state_dict()

        # RNG states for reproducible resume
        rng_state = {
            'python_random': random.getstate(),
            'numpy_random': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        # GradScaler state if fp16 is used by Accelerate
        scaler_state = None
        try:
            if hasattr(self.accelerator, 'scaler') and self.accelerator.scaler is not None:
                scaler_state = self.accelerator.scaler.state_dict()
        except Exception as e:
            logging.warning(f"Could not serialize GradScaler state: {e}")

        # Optionally save full accelerate state (handles ZeRO optimizer partitioning, RNG, etc.)
        # Initialize accel_state_dir to None to prevent NameError
        accel_state_dir = None
        try:
            base = os.path.splitext(os.path.basename(filename))[0]
            candidate_dir = os.path.join(self.config.project.output_dir, f"{base}_accelerate_state")
            # Ensure directory exists only on the main process
            if self.accelerator.is_main_process:
                os.makedirs(candidate_dir, exist_ok=True)
            self.accelerator.wait_for_everyone()
            # accelerator.save_state handles saving everything, including registered trackers if any.
            self.accelerator.save_state(candidate_dir)
            accel_state_dir = candidate_dir
        except Exception as e:
            logging.warning(
                f"accelerator.save_state failed or unavailable: {e}. Proceeding without accelerate state directory.")

        # Get the optimizer state dict. We need the base optimizer if Accelerate wrapped it.
        try:
            base_optimizer = self.optimizer
            # Check if it's an AcceleratedOptimizer wrapper (common pattern)
            if hasattr(self.optimizer, 'optimizer') and not hasattr(self.optimizer, 'add_param_group'):
                base_optimizer = self.optimizer.optimizer
            optimizer_state_dict = base_optimizer.state_dict()
        except Exception as e:
            logging.warning(f"Failed to get optimizer state_dict: {e}. Saving without optimizer state.")
            optimizer_state_dict = None

        # Save metadata file (best_model.pt)
        state = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config,
            'best_val_ppl': self.best_val_ppl,
            'rng_state': rng_state,
            'scaler_state_dict': scaler_state,
            'accelerate_state_dir': accel_state_dir,
        }

        self.accelerator.save(state, checkpoint_path)

        if self.accelerator.is_main_process:
            logging.info(f"Checkpoint saved to {checkpoint_path}")
            # Check if accel_state_dir is not None before using it
            if accel_state_dir and os.path.isdir(accel_state_dir):
                logging.info(f"Accelerate state saved to {accel_state_dir}")

    def load_checkpoint(self, checkpoint_path):
        """Loads the model checkpoint and restores training state."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        logging.info(f"Loading checkpoint from {checkpoint_path}")

        # Load the checkpoint file (metadata loaded to CPU first for inspection)
        try:
            # Load minimally first to inspect structure AND use for metadata restoration later
            checkpoint_meta = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            logging.error(f"Failed to load checkpoint file: {e}")
            raise

        # Get the unwrapped model instance
        try:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        except Exception as e:
            logging.error(f"Could not unwrap model during checkpoint loading: {e}. Using wrapped model.")
            unwrapped_model = self.model

        logging.info("Fixed-mode checkpoint load: dynamic expansion is disabled.")
        expanded = False
        # ----------------------------------------------------------

        # 3. Load the state (Model weights, Optimizer state, RNG, etc.)

        # Prefer loading full Accelerate state if available
        loaded_via_accelerate = False

        # Detect Accelerate directory
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_basename = os.path.splitext(os.path.basename(checkpoint_path))[0]
        accel_dir_expected = os.path.join(checkpoint_dir, f"{checkpoint_basename}_accelerate_state")

        accel_dir_from_meta = checkpoint_meta.get('accelerate_state_dir', None) or checkpoint_meta.get(
            'accelerator_state_dir', None)

        # Use the path found in metadata or the expected path
        accel_dir = None
        if accel_dir_from_meta and os.path.isdir(accel_dir_from_meta):
            accel_dir = accel_dir_from_meta
        elif os.path.isdir(accel_dir_expected):
            accel_dir = accel_dir_expected

        if accel_dir:
            try:
                # This restores model, optimizer, scheduler, RNG, AND data samplers.
                self.accelerator.load_state(accel_dir)
                loaded_via_accelerate = True
                logging.info(f"Accelerate state loaded from {accel_dir}")
            except Exception as e:
                logging.warning(
                    f"Failed to load accelerate state from {accel_dir}: {e}. Falling back to manual state restoration.")
                loaded_via_accelerate = False

        # Manual Fallback Loading Path (Handles Non-DeepSpeed or if Accelerate load_state failed)
        if not loaded_via_accelerate:
            # If accelerator.load_state failed, we must load the checkpoint onto the target device now.
            try:
                # If we are here, we need the full checkpoint loaded on the device for manual restoration
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                checkpoint_meta = checkpoint  # Use the device-loaded checkpoint for metadata if needed later
            except Exception as e:
                logging.error(f"Failed to load checkpoint onto device {self.device} for manual restore: {e}")
                raise

            # Load model state with robust filtering (handles DDP prefixes, dynamic MoE growth, shape mismatches)
            ckpt_state = checkpoint['model_state_dict']

            # 1) Clean 'module.' prefixes if present
            cleaned_state = {}
            for k, v in ckpt_state.items():
                if k.startswith('module.'):
                    cleaned_state[k[7:]] = v
                else:
                    cleaned_state[k] = v

            # 2) Filter out unexpected and shape-mismatched parameters
            model_state = unwrapped_model.state_dict()
            filtered_state = {}
            unexpected_keys = []
            shape_mismatch_keys = []
            for k, v in cleaned_state.items():
                if k in model_state:
                    if model_state[k].shape == v.shape:
                        filtered_state[k] = v
                    else:
                        shape_mismatch_keys.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                else:
                    unexpected_keys.append(k)
            missing_keys = [k for k in model_state.keys() if k not in filtered_state]

            # 3) Load non-strictly with filtered dict
            unwrapped_model.load_state_dict(filtered_state, strict=False)

            # 4) Log informative summary
            if unexpected_keys:
                logging.warning(
                    f"Ignored {len(unexpected_keys)} unexpected keys from checkpoint (e.g., dynamic experts).")
            if shape_mismatch_keys:
                logging.warning(f"Ignored {len(shape_mismatch_keys)} keys with shape mismatches.")
            if missing_keys:
                logging.warning(
                    f"Missing {len(missing_keys)} keys relative to model (initialized from model defaults).")

            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logging.info("Optimizer state loaded from checkpoint")
                except Exception as e:
                    logging.warning(
                        f"Failed to load optimizer state_dict (likely due to architecture/param changes): {e}. Optimizer will be reinitialized.")

            # Load scheduler state if available
            if 'lr_scheduler_state_dict' in checkpoint:
                try:
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                    logging.info("Learning rate scheduler state loaded from checkpoint")
                except Exception as e:
                    logging.warning(f"Failed to load LR scheduler state_dict: {e}")

            # Restore RNG states if available (ensure correct device/dtype)
            rng_state = checkpoint.get('rng_state', None)
            if rng_state is not None:
                try:
                    if rng_state.get('python_random') is not None:
                        random.setstate(rng_state['python_random'])
                    if rng_state.get('numpy_random') is not None:
                        np.random.set_state(rng_state['numpy_random'])
                    if rng_state.get('torch_cpu') is not None:
                        cpu_state = rng_state['torch_cpu']
                        if not isinstance(cpu_state, torch.Tensor):
                            cpu_state = torch.tensor(cpu_state, dtype=torch.uint8)
                        else:
                            cpu_state = cpu_state.detach().cpu().to(torch.uint8)
                        torch.set_rng_state(cpu_state)
                    if rng_state.get('torch_cuda_all') is not None and torch.cuda.is_available():
                        states = rng_state['torch_cuda_all']
                        # Apply per-device for robustness
                        if isinstance(states, (list, tuple)):
                            for i, s in enumerate(states):
                                if i >= torch.cuda.device_count():
                                    break
                                if not isinstance(s, torch.Tensor):
                                    s = torch.tensor(s, dtype=torch.uint8, device=f'cuda:{i}')
                                else:
                                    s = s.detach().to(device=f'cuda:{i}', dtype=torch.uint8)
                                torch.cuda.set_rng_state(s, device=i)
                        else:
                            # Single state provided; apply to device 0
                            s = states
                            if not isinstance(s, torch.Tensor):
                                s = torch.tensor(s, dtype=torch.uint8, device='cuda:0')
                            else:
                                s = s.detach().to(device='cuda:0', dtype=torch.uint8)
                            torch.cuda.set_rng_state(s, device=0)
                except Exception as e:
                    logging.warning(f"Failed to restore RNG states from checkpoint: {e}")

            # Restore GradScaler if present
            try:
                if 'scaler_state_dict' in checkpoint and hasattr(self.accelerator,
                                                                 'scaler') and self.accelerator.scaler is not None:
                    self.accelerator.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    logging.info("GradScaler state restored from checkpoint")
            except Exception as e:
                logging.warning(f"Failed to restore GradScaler state: {e}")
            pass

        # Restore training metadata from the metadata file (best_model.pt)
        # This is necessary because global_step and current_epoch are managed by the trainer script.
        self.global_step = checkpoint_meta.get('global_step', 0)
        self.best_val_ppl = checkpoint_meta.get('best_val_ppl', float('inf'))

        # Restore the epoch number explicitly
        self.current_epoch = checkpoint_meta.get('current_epoch', 0)

        # Update the final logging message
        logging.info(
            f"Checkpoint loaded successfully. Resuming from global step {self.global_step} (Epoch {self.current_epoch + 1})")
        logging.info(f"Best validation perplexity so far: {self.best_val_ppl:.4f}")

        # Clean up CPU metadata copy
        del checkpoint_meta
        if 'checkpoint' in locals():
            del checkpoint
