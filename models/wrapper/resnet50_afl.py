from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.resnet import ResNet

from modules.nn import FrequencyDefender


class ResNet50AFL(nn.Module):
    """ResNet-50 wrapper with a single FrequencyDefender after layer3."""

    def __init__(self, backbone: ResNet, defender: FrequencyDefender) -> None:
        super().__init__()
        self.backbone = backbone
        self.defender = defender

    def _forward_to_layer3(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        return self.backbone.layer3(x)

    def _forward_from_layer3(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.backbone.fc(x)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = self._forward_to_layer3(x)
        defended, mask = self.defender(features, return_mask=True)
        logits = self._forward_from_layer3(defended)

        if return_aux:
            return logits, {"mask": mask}
        return logits


def build_resnet50_afl(
    num_classes: int = 1000,
    pretrained: bool = True,
    defender_hidden_channels: int = 64,
    min_gate: float = 0.1,
    temperature: float = 1.0,
) -> ResNet50AFL:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = resnet50(weights=weights)
    if backbone.fc.out_features != num_classes:
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

    defender = FrequencyDefender(
        hidden_channels=defender_hidden_channels,
        min_gate=min_gate,
        temperature=temperature,
    )
    return ResNet50AFL(backbone=backbone, defender=defender)

