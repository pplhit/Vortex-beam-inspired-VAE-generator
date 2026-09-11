from __future__ import annotations

import math
import torch
import torch.nn as nn


class VortexLatentMapper(nn.Module):
    """Map a real Gaussian latent vector into a coherent LG/OAM field."""

    def __init__(self, basis: torch.Tensor, normalize_expected_power: bool = True) -> None:
        super().__init__()
        if basis.ndim != 3 or not torch.is_complex(basis):
            raise ValueError("basis must be complex with shape [K, H, W]")
        self.register_buffer("basis", basis)
        self.num_modes = basis.shape[0]
        self.normalize_expected_power = normalize_expected_power

    @property
    def latent_dim(self) -> int:
        return 2 * self.num_modes

    def latent_to_coefficients(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"Expected z with shape [B, {self.latent_dim}], got {tuple(z.shape)}")
        real = z[:, 0::2]
        imag = z[:, 1::2]
        coeff = (real + 1j * imag) / math.sqrt(2.0)
        if self.normalize_expected_power:
            coeff = coeff / math.sqrt(self.num_modes)
        return coeff

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coeff = self.latent_to_coefficients(z)
        field = torch.einsum("bk,khw->bhw", coeff, self.basis)
        return field.unsqueeze(1), coeff
