import torch

from vortex_vae.lg_modes import build_lg_basis
from vortex_vae.vortex_mapper import VortexLatentMapper


def test_vortex_mapper_shapes_and_finite_values():
    basis = build_lg_basis(32, 32, pixel_pitch=8e-6, waist=0.12e-3, device="cpu")
    mapper = VortexLatentMapper(basis)
    z = torch.randn(4, mapper.latent_dim)
    field, coeff = mapper(z)

    assert field.shape == (4, 1, 32, 32)
    assert coeff.shape == (4, basis.shape[0])
    assert torch.is_complex(field)
    assert torch.isfinite(torch.view_as_real(field)).all()
