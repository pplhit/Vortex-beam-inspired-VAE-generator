from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from train import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate images from Gaussian noise through the vortex optical decoder")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="generated.png")
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument("--pixel-pitch", type=float, default=8e-6)
    p.add_argument("--waist", type=float, default=0.45e-3)
    p.add_argument("--no-basis-orthonormalization", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    out = model.generate(args.num_samples, device)
    grid = make_grid(out["image"].cpu(), nrow=max(1, int(args.num_samples**0.5)), padding=2)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.squeeze(0), cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved generated samples to {path}")


if __name__ == "__main__":
    main()
