from __future__ import annotations

import torch
import torch.nn as nn


class VortexOpticalVAE(nn.Module):
    """VAE with a vortex-mode embedding and a diffractive optical decoder."""

    def __init__(self, encoder: nn.Module, vortex_mapper: nn.Module, optical_decoder: nn.Module, detector: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.vortex_mapper = vortex_mapper
        self.optical_decoder = optical_decoder
        self.detector = detector

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode_latent(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        input_field, coefficients = self.vortex_mapper(z)
        output_field = self.optical_decoder(input_field)
        image = self.detector(output_field)
        return {
            "image": image,
            "coefficients": coefficients,
            "input_field": input_field,
            "output_field": output_field,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode_latent(z)
        return {"recon": decoded.pop("image"), "mu": mu, "logvar": logvar, "z": z, **decoded}

    @torch.no_grad()
    def generate(self, batch_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        z = torch.randn(batch_size, self.vortex_mapper.latent_dim, device=device)
        decoded = self.decode_latent(z)
        return {"image": decoded.pop("image"), "z": z, **decoded}
