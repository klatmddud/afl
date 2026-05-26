from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.wrapper import build_resnet50_afl


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet50_afl(pretrained=False).to(device)
    model.eval()

    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        logits, aux = model(x, return_aux=True)

    print(f"device={device}")
    print(f"logits={tuple(logits.shape)}")
    print(f"mask={tuple(aux['mask'].shape)}")
    print(f"mask_mean={aux['mask'].mean().item():.4f}")


if __name__ == "__main__":
    main()
