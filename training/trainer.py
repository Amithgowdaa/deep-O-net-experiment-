"""
Trainer: handles training loop, logging, checkpointing.

Designed to be experiment-agnostic — takes model, dataloaders, config.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any

from training.losses import get_loss


class Trainer:
    """
    Generic trainer for DeepONet variants.

    Args:
        model: DeepONet (or any variant)
        config: dict from YAML config
        device: torch device
        output_dir: where to save checkpoints and logs
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        output_dir: str,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        train_cfg = config["training"]

        self.loss_fn = get_loss("mse")
        self.eval_loss_fn = get_loss("relative_l2")

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=train_cfg["lr"]
        )

        sched_cfg = train_cfg.get("lr_scheduler", {})
        if sched_cfg.get("type") == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_cfg["epochs"],
                eta_min=sched_cfg.get("min_lr", 1e-5),
            )
        else:
            self.scheduler = None

        self.epochs = train_cfg["epochs"]
        self.log_every = train_cfg["log_every"]
        self.save_every = train_cfg["save_every"]

        self.history = {"train_loss": [], "test_loss": [], "epoch": []}

    def _make_loader(self, branch, trunk, target, shuffle=True) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(branch, dtype=torch.float32),
            torch.tensor(trunk,  dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )
        return DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=shuffle,
            pin_memory=(self.device.type == "cuda"),
            num_workers=2,
        )

    def train(self, train_data, test_data):
        """
        Args:
            train_data: tuple (branch, trunk, target) — numpy arrays
            test_data:  tuple (branch, trunk, target) — numpy arrays
        """
        train_loader = self._make_loader(*train_data)
        test_loader  = self._make_loader(*test_data, shuffle=False)

        print(f"Training on {self.device} | "
              f"{self.config['training']['epochs']} epochs | "
              f"batch size {self.config['training']['batch_size']}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 60)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for branch_b, trunk_b, target_b in train_loader:
                branch_b = branch_b.to(self.device)
                trunk_b  = trunk_b.to(self.device)
                target_b = target_b.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(branch_b, trunk_b)
                loss = self.loss_fn(pred, target_b)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if self.scheduler is not None:
                self.scheduler.step()

            avg_train_loss = epoch_loss / len(train_loader)

            if epoch % self.log_every == 0:
                test_loss = self._evaluate(test_loader)
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0

                print(f"Epoch {epoch:6d} | "
                      f"Train MSE: {avg_train_loss:.4e} | "
                      f"Test RelL2: {test_loss:.4e} | "
                      f"LR: {lr:.2e} | "
                      f"Time: {elapsed:.1f}s")

                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(avg_train_loss)
                self.history["test_loss"].append(test_loss)

            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

        self._save_checkpoint(epoch, final=True)
        print("Training complete.")
        return self.history

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        preds, targets = [], []
        for branch_b, trunk_b, target_b in loader:
            branch_b = branch_b.to(self.device)
            trunk_b  = trunk_b.to(self.device)
            pred = self.model(branch_b, trunk_b).cpu()
            preds.append(pred)
            targets.append(target_b)
        preds   = torch.cat(preds)
        targets = torch.cat(targets)
        return self.eval_loss_fn(preds, targets).item()

    def _save_checkpoint(self, epoch: int, final: bool = False):
        name = "final.pt" if final else f"epoch_{epoch:06d}.pt"
        path = os.path.join(self.output_dir, name)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.config,
        }, path)
        if final:
            print(f"Saved final checkpoint → {path}")
