from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime.config import deep_get, load_dotenv, load_simple_yaml, resolve_device, set_seed
from scripts.runtime.data import build_imagefolder_loaders
from scripts.runtime.logging import TeeOutput
from scripts.runtime.modeling import build_model, build_optimizer, build_scheduler
from scripts.runtime.trainer import ClassificationTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AFL classification model.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    config = load_simple_yaml(args.config)

    seed = int(args.seed if args.seed is not None else deep_get(config, "seed", 42))
    set_seed(seed)
    device = resolve_device(str(args.device or deep_get(config, "device", "auto")))
    output_dir = args.output_dir or Path(str(deep_get(config, "output_dir", "runs/train")))
    output_dir.mkdir(parents=True, exist_ok=True)

    with TeeOutput(output_dir / "train.log"):
        train_loader, val_loader, num_classes = build_imagefolder_loaders(config)
        pretrained = bool(deep_get(config, "model.pretrained", True))
        model = build_model(args.model, num_classes=num_classes, pretrained=pretrained).to(device)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        print(f"device={device}")
        print(f"model={args.model} num_classes={num_classes}")
        print(f"train_batches={len(train_loader)} val_batches={len(val_loader)}")
        print(f"output_dir={output_dir}")
        print(f"log_file={output_dir / 'train.log'}")

        trainer = ClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=output_dir,
            config=config,
            amp_enabled=bool(deep_get(config, "amp", True)) and device.type == "cuda",
            target_mask_ratio=float(deep_get(config, "afl.target_mask_ratio", 0.5)),
            rho=float(deep_get(config, "afl.rho", 0.05)),
            epochs=int(deep_get(config, "train.epochs", 1)),
            log_interval=int(deep_get(config, "train.log_interval", 20)),
            eval_interval=int(deep_get(config, "train.eval_interval", 1)),
        )
        trainer.fit()


if __name__ == "__main__":
    main()
