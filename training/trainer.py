"""Training loop with MPS optimization."""

import os
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import ModelConfig, TrainingConfig
from model.gpt import GPT


class Trainer:
    """Trainer class for GPT model with MPS support."""

    def __init__(
        self,
        model: GPT,
        train_dataloader: DataLoader,
        config: TrainingConfig,
        eval_dataloader: Optional[DataLoader] = None
    ):
        """
        Args:
            model: GPT model instance.
            train_dataloader: Training data loader.
            config: Training configuration.
            eval_dataloader: Optional evaluation data loader.
        """
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Set device
        self.device = self._get_device()
        print(f"Using device: {self.device}")

        # Move model to device
        self.model = model.to(self.device)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def _get_device(self) -> torch.device:
        """Get the appropriate device."""
        if self.config.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "ln" in name or "layernorm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        return AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        total_steps = len(self.train_dataloader) * self.config.max_epochs

        if self.config.max_steps is not None:
            total_steps = min(total_steps, self.config.max_steps)

        def lr_lambda(step):
            # Linear warmup
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            # Cosine decay
            progress = (step - self.config.warmup_steps) / max(
                1, total_steps - self.config.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}",
            leave=True
        )

        for batch_idx, (input_ids, labels) in enumerate(progress_bar):
            # Move to device
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

            # Update tracking
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}"
            })

            # Logging
            if self.global_step % self.config.log_interval == 0:
                avg_loss = total_loss / num_batches
                print(f"\nStep {self.global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")

            # Evaluation
            if self.eval_dataloader and self.global_step % self.config.eval_interval == 0:
                eval_loss = self.evaluate()
                print(f"Eval loss: {eval_loss:.4f}")
                self.model.train()

            # Checkpointing
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint()

            # Max steps check
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def evaluate(self) -> float:
        """Evaluate the model.

        Returns:
            Average evaluation loss.
        """
        if self.eval_dataloader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for input_ids, labels in self.eval_dataloader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input_ids=input_ids, labels=labels)
                total_loss += outputs["loss"].item()
                num_batches += 1

        return total_loss / max(1, num_batches)

    def train(self) -> None:
        """Run full training loop."""
        print(f"Starting training for {self.config.max_epochs} epochs")
        print(f"Total parameters: {self.model.count_parameters():,}")

        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            avg_loss = self.train_epoch()

            print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

            # Save checkpoint at end of epoch
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint("best_model.pt")
                print(f"New best model saved with loss: {avg_loss:.4f}")

            # Check max steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                print("Reached max_steps, stopping training")
                break

        print("Training complete!")

    def save_checkpoint(self, filename: str = None) -> None:
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_step_{self.global_step}.pt"

        path = os.path.join(self.config.checkpoint_dir, filename)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": self.model.config
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        print(f"Checkpoint loaded from: {path}")
        print(f"Resuming from step {self.global_step}, epoch {self.epoch}")
