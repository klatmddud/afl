from __future__ import annotations

import torch
from torch import nn


class FrequencyDefender(nn.Module):
    """Amplitude-only frequency gate for a single intermediate feature map."""

    def __init__(
        self,
        hidden_channels: int = 64,
        min_gate: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= min_gate <= 1.0:
            raise ValueError("min_gate must be in [0, 1].")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")

        self.min_gate = min_gate
        self.temperature = temperature
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def spectrum_features(self, spectrum: torch.Tensor) -> torch.Tensor:
        amplitude = torch.abs(spectrum)
        return torch.log1p(amplitude).mean(dim=1, keepdim=True)

    def predict_mask(self, spectrum: torch.Tensor) -> torch.Tensor:
        features = self.spectrum_features(spectrum)
        logits = self.mask_predictor(features)
        return torch.sigmoid(logits / self.temperature)

    def apply_gate(self, spectrum: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gate = self.min_gate + (1.0 - self.min_gate) * mask
        return gate * spectrum

    def apply_gate_and_ifft(
        self,
        spectrum: torch.Tensor,
        mask: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        gated = self.apply_gate(spectrum, mask)
        return torch.fft.irfft2(gated, s=spatial_size, dim=(-2, -1))

    def forward(
        self,
        features: torch.Tensor,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        spatial_size = features.shape[-2:]
        spectrum = torch.fft.rfft2(features, dim=(-2, -1))
        mask = self.predict_mask(spectrum)
        defended = self.apply_gate_and_ifft(spectrum, mask, spatial_size)
        if return_mask:
            return defended, mask
        return defended


def mask_ratio_loss(mask: torch.Tensor, target_ratio: float) -> torch.Tensor:
    return (mask.mean() - target_ratio).abs()

