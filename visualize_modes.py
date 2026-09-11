from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch

from vortex_vae import build_lg_basis, default_mode_list


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize one LG/OAM basis mode")
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument("--pixel-pitch", type=float, default=8e-6)
    p.add_argument("--waist", type=float, default=0.45e-3)
    p.add_argument("--index", type=int, default=0)
    args = p.parse_args()

    basis = build_lg_basis(
        args.grid_size,
        args.grid_size,
        args.pixel_pitch,
        args.waist,
        orthonormalize=False,
    )
    modes = default_mode_list()
    if not 0 <= args.index < len(modes):
        raise ValueError(f"index must be in [0, {len(modes) - 1}]")

    field = basis[args.index].cpu()
    p_idx, ell = modes[args.index]

    plt.figure(figsize=(5, 4))
    plt.imshow(torch.abs(field).square(), cmap="gray")
    plt.title(f"LG mode: p={p_idx}, ell={ell} | intensity")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 4))
    plt.imshow(torch.angle(field), cmap="twilight", vmin=-torch.pi, vmax=torch.pi)
    plt.title(f"LG mode: p={p_idx}, ell={ell} | phase")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
