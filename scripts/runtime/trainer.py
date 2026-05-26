from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from modules.nn import mask_ratio_loss
from scripts.runtime.metrics import accuracy


class ClassificationTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: SGD,
        scheduler: LambdaLR,
        device: torch.device,
        output_dir: Path,
        config: dict[str, Any],
        amp_enabled: bool,
        target_mask_ratio: float,
        rho: float,
        epochs: int,
        log_interval: int,
        eval_interval: int,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.config = config
        self.amp_enabled = amp_enabled
        self.target_mask_ratio = target_mask_ratio
        self.rho = rho
        self.epochs = epochs
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
        self.best_top1 = -1.0

    @staticmethod
    def format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def log_step(
        self,
        split: str,
        epoch: int,
        step: int,
        total_steps: int,
        epoch_started_at: float,
        totals: dict[str, float],
        seen: int,
    ) -> None:
        elapsed = time.perf_counter() - epoch_started_at
        eta = (elapsed / max(1, step)) * max(0, total_steps - step)
        prefix = (
            f"[{split}] epoch {epoch}/{self.epochs} step {step}/{total_steps} "
            f"epoch_eta={self.format_duration(eta)}"
        )
        fields = [
            f"total_loss={totals['loss'] / seen:.4f}",
            f"ce_loss={totals['ce'] / seen:.4f}",
            f"ratio_loss={totals['ratio'] / seen:.4f}",
            f"top1={totals['top1'] / seen:.2f}",
            f"mask_mean={totals['mask_mean'] / seen:.4f}",
        ]
        if "top5" in totals:
            fields.append(f"top5={totals['top5'] / seen:.2f}")
        print(f"{prefix} {' '.join(fields)}")

    def fit(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "resolved_config.json").write_text(
            json.dumps(self.config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        metrics_path = self.output_dir / "metrics.csv"
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_one_epoch(epoch)

            val_metrics: dict[str, float] = {}
            if self.eval_interval > 0 and epoch % self.eval_interval == 0:
                val_metrics = self.evaluate(epoch)
                if val_metrics["top1"] > self.best_top1:
                    self.best_top1 = val_metrics["top1"]
                    self.save_checkpoint(self.output_dir / "best.pt", epoch)

            self.scheduler.step()
            self.write_metrics(metrics_path, epoch, train_metrics, val_metrics)
            self.save_checkpoint(self.output_dir / "last.pt", epoch)
            self.print_epoch_summary(epoch, train_metrics, val_metrics)

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "ce": 0.0, "ratio": 0.0, "top1": 0.0, "mask_mean": 0.0}
        seen = 0
        epoch_started_at = time.perf_counter()
        total_steps = len(self.train_loader)

        for step, (images, target) in enumerate(self.train_loader, 1):
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits, aux = self.model(images, return_aux=True)
                ce_loss = F.cross_entropy(logits, target)
                ratio_loss = mask_ratio_loss(aux["mask"], self.target_mask_ratio)
                loss = ce_loss + self.rho * ratio_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = target.numel()
            top1 = accuracy(logits.detach(), target, (1,))[0]
            seen += batch_size
            totals["loss"] += loss.item() * batch_size
            totals["ce"] += ce_loss.item() * batch_size
            totals["ratio"] += ratio_loss.item() * batch_size
            totals["top1"] += top1.item() * batch_size
            totals["mask_mean"] += aux["mask"].detach().mean().item() * batch_size

            if self.log_interval > 0 and (step % self.log_interval == 0 or step == total_steps):
                self.log_step("train", epoch, step, total_steps, epoch_started_at, totals, seen)

        return {key: value / max(1, seen) for key, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        totals = {
            "loss": 0.0,
            "ce": 0.0,
            "ratio": 0.0,
            "top1": 0.0,
            "top5": 0.0,
            "mask_mean": 0.0,
        }
        seen = 0
        epoch_started_at = time.perf_counter()
        total_steps = len(self.val_loader)

        for step, (images, target) in enumerate(self.val_loader, 1):
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits, aux = self.model(images, return_aux=True)
                ce_loss = F.cross_entropy(logits, target)
                ratio_loss = mask_ratio_loss(aux["mask"], self.target_mask_ratio)
                loss = ce_loss + self.rho * ratio_loss

            top1, top5 = accuracy(logits, target, (1, 5))
            batch_size = target.numel()
            seen += batch_size
            totals["loss"] += loss.item() * batch_size
            totals["ce"] += ce_loss.item() * batch_size
            totals["ratio"] += ratio_loss.item() * batch_size
            totals["top1"] += top1.item() * batch_size
            totals["top5"] += top5.item() * batch_size
            totals["mask_mean"] += aux["mask"].mean().item() * batch_size

            if self.log_interval > 0 and (step % self.log_interval == 0 or step == total_steps):
                self.log_step("eval", epoch, step, total_steps, epoch_started_at, totals, seen)

        return {key: value / max(1, seen) for key, value in totals.items()}

    def write_metrics(
        self,
        path: Path,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        current_lrs = [group["lr"] for group in self.optimizer.param_groups]
        row: dict[str, Any] = {
            "epoch": epoch,
            "lr_backbone": current_lrs[0],
            "lr_classifier": current_lrs[1],
            "lr_defender": current_lrs[2],
            "train_loss": train_metrics["loss"],
            "train_ce": train_metrics["ce"],
            "train_ratio": train_metrics["ratio"],
            "train_top1": train_metrics["top1"],
            "train_mask_mean": train_metrics["mask_mean"],
            "val_loss": val_metrics.get("loss", ""),
            "val_ce": val_metrics.get("ce", ""),
            "val_ratio": val_metrics.get("ratio", ""),
            "val_top1": val_metrics.get("top1", ""),
            "val_top5": val_metrics.get("top5", ""),
            "val_mask_mean": val_metrics.get("mask_mean", ""),
            "best_top1": self.best_top1,
        }
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_top1": self.best_top1,
            },
            path,
        )

    def print_epoch_summary(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        val_summary = ""
        if val_metrics:
            val_summary = (
                f" val_loss={val_metrics['loss']:.4f}"
                f" val_top1={val_metrics['top1']:.2f}"
                f" val_top5={val_metrics['top5']:.2f}"
                f" val_mask_mean={val_metrics['mask_mean']:.4f}"
            )
        print(
            f"epoch={epoch}/{self.epochs} train_loss={train_metrics['loss']:.4f} "
            f"train_top1={train_metrics['top1']:.2f}"
            f" train_mask_mean={train_metrics['mask_mean']:.4f}{val_summary}"
        )
