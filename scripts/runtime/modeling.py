from __future__ import annotations

import math
from typing import Any

from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from models.wrapper import build_resnet50_afl
from scripts.runtime.config import deep_get


def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if name not in {"resnet50", "resnet50_afl"}:
        raise ValueError(f"Unsupported model: {name}")
    return build_resnet50_afl(num_classes=num_classes, pretrained=pretrained)


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> SGD:
    if deep_get(config, "optimizer.type", "sgd") != "sgd":
        raise ValueError("Only SGD is supported for the initial training loop.")

    backbone_params = []
    classifier_params = []
    defender_params = list(model.defender.parameters())  # type: ignore[attr-defined]

    backbone = model.backbone  # type: ignore[attr-defined]
    for module in [
        backbone.conv1,
        backbone.bn1,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
    ]:
        backbone_params.extend(module.parameters())
    classifier_params.extend(backbone.layer4.parameters())
    classifier_params.extend(backbone.fc.parameters())

    return SGD(
        [
            {"params": backbone_params, "lr": float(deep_get(config, "lr.backbone", 1e-4))},
            {"params": classifier_params, "lr": float(deep_get(config, "lr.classifier", 1e-3))},
            {"params": defender_params, "lr": float(deep_get(config, "lr.defender", 1e-3))},
        ],
        momentum=float(deep_get(config, "optimizer.momentum", 0.9)),
        weight_decay=float(deep_get(config, "optimizer.weight_decay", 1e-4)),
    )


def build_scheduler(optimizer: SGD, config: dict[str, Any]) -> LambdaLR:
    epochs = int(deep_get(config, "train.epochs", 1))
    warmup_epochs = int(deep_get(config, "scheduler.warmup_epochs", 0))
    min_lr = float(deep_get(config, "scheduler.min_lr", 0.0))
    scheduler_type = str(deep_get(config, "scheduler.type", "cosine"))
    if scheduler_type != "cosine":
        raise ValueError("Only cosine scheduler is supported for the initial training loop.")

    def make_lr_lambda(base_lr: float):
        min_factor = min(1.0, min_lr / base_lr) if base_lr > 0.0 else 0.0

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            span = max(1, epochs - warmup_epochs)
            progress = min(1.0, max(0.0, (epoch - warmup_epochs + 1) / span))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return lr_lambda

    return LambdaLR(
        optimizer,
        lr_lambda=[make_lr_lambda(group["lr"]) for group in optimizer.param_groups],
    )

