"""Adapter point for the user's existing diffractive network / ASM code.

Required contract:
    input : complex tensor [B, 1, H, W]
    output: complex tensor [B, 1, H, W]
"""

import torch
import torch.nn as nn


class IdentityOpticalDecoder(nn.Module):
    """Smoke-test backend. Do NOT use this as the research optical decoder."""

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return field


def build_diffractive_network(*, smoke_test: bool = False) -> nn.Module:
    if smoke_test:
        return IdentityOpticalDecoder()

    # Replace with your existing D2NN / angular spectrum model, e.g.
    # from my_d2nn import DiffractiveNetwork
    # return DiffractiveNetwork(...)
    raise NotImplementedError(
        "Insert your existing diffractive network / angular-spectrum implementation "
        "in optics/user_diffractive_network.py, or run with --smoke-test."
    )
