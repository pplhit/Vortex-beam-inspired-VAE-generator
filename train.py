from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from optics import build_diffractive_network
from vortex_vae import (
    ConvVAEEncoder,
    IntensityDetector,
    VortexLatentMapper,
    VortexOpticalVAE,
    build_lg_basis,
    vae_loss,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the vortex-beam-inspired optical VAE")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--save-dir", type=str, default="./checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--reconstruction", choices=["mse", "bce"], default="mse")
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument("--pixel-pitch", type=float, default=8e-6)
    p.add_argument("--waist", type=float, default=0.45e-3)
    p.add_argument("--no-basis-orthonormalization", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def build_model(args: argparse.Namespace, device: torch.device) -> VortexOpticalVAE:
    basis = build_lg_basis(
        height=args.grid_size,
        width=args.grid_size,
        pixel_pitch=args.pixel_pitch,
        waist=args.waist,
        device=device,
        orthonormalize=not args.no_basis_orthonormalization,
    )
    mapper = VortexLatentMapper(basis)
    encoder = ConvVAEEncoder(latent_dim=mapper.latent_dim)
    optical_decoder = build_diffractive_network(smoke_test=args.smoke_test)
    detector = IntensityDetector(output_size=(28, 28), normalize="max")
    return VortexOpticalVAE(encoder, mapper, optical_decoder, detector).to(device)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = datasets.MNIST(args.data_root, train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = build_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        meter = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
        progress = tqdm(loader, desc=f"Epoch {epoch:03d}/{args.epochs:03d}")

        for x, _ in progress:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            losses = vae_loss(
                out["recon"], x, out["mu"], out["logvar"],
                beta=args.beta, reconstruction=args.reconstruction,
            )
            losses["loss"].backward()
            optimizer.step()

            for key in meter:
                meter[key] += losses[key].item()
            progress.set_postfix(loss=f"{losses['loss'].item():.3f}", kl=f"{losses['kl'].item():.3f}")

        n = len(loader)
        print(" | ".join(f"{k}={v / n:.4f}" for k, v in meter.items()))
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args)},
            save_dir / "latest.pt",
        )


if __name__ == "__main__":
    main()
