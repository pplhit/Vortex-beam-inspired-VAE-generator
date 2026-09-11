from __future__ import annotations

import torch


def generalized_laguerre(p: int, alpha: int, x: torch.Tensor) -> torch.Tensor:
    """Evaluate the generalized Laguerre polynomial L_p^alpha(x)."""
    if p < 0:
        raise ValueError("p must be non-negative")
    if p == 0:
        return torch.ones_like(x)
    if p == 1:
        return 1.0 + alpha - x

    l_nm2 = torch.ones_like(x)
    l_nm1 = 1.0 + alpha - x
    for n in range(2, p + 1):
        l_n = ((2 * n - 1 + alpha - x) * l_nm1 - (n - 1 + alpha) * l_nm2) / n
        l_nm2, l_nm1 = l_nm1, l_n
    return l_nm1


def make_lg_mode(
    height: int,
    width: int,
    pixel_pitch: float,
    waist: float,
    p: int,
    ell: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a discretely normalized LG_p^ell mode at its waist plane."""
    y = (torch.arange(height, device=device, dtype=dtype) - (height - 1) / 2) * pixel_pitch
    x = (torch.arange(width, device=device, dtype=dtype) - (width - 1) / 2) * pixel_pitch
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    r = torch.sqrt(xx.square() + yy.square())
    phi = torch.atan2(yy, xx)
    abs_ell = abs(ell)
    rho2 = 2.0 * r.square() / waist**2

    radial = (torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype)) * r / waist) ** abs_ell
    laguerre = generalized_laguerre(p, abs_ell, rho2)
    gaussian = torch.exp(-r.square() / waist**2)
    amplitude = radial * laguerre * gaussian
    phase = torch.polar(torch.ones_like(phi), ell * phi)

    mode = amplitude.to(phase.dtype) * phase
    return mode / torch.sqrt(torch.sum(torch.abs(mode).square()) + 1e-12)


def default_mode_list() -> list[tuple[int, int]]:
    """32 complex modes corresponding to a 64-D real latent space."""
    modes = [(0, ell) for ell in range(-8, 9)]
    modes += [(1, ell) for ell in range(-7, 8)]
    assert len(modes) == 32
    return modes


def orthonormalize_discrete_basis(basis: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Whiten a sampled modal basis so B B^H ~= I on the finite grid."""
    if basis.ndim != 3 or not torch.is_complex(basis):
        raise ValueError("basis must be a complex tensor with shape [K, H, W]")

    k, h, w = basis.shape
    b = basis.reshape(k, h * w)
    gram = b @ b.conj().transpose(0, 1)
    eigvals, eigvecs = torch.linalg.eigh(gram)
    inv_diag = torch.diag(torch.rsqrt(eigvals.clamp_min(eps))).to(eigvecs.dtype)
    inv_sqrt = eigvecs @ inv_diag @ eigvecs.conj().transpose(0, 1)
    return (inv_sqrt @ b).reshape(k, h, w)


def build_lg_basis(
    height: int,
    width: int,
    pixel_pitch: float,
    waist: float,
    mode_list: list[tuple[int, int]] | None = None,
    *,
    device: torch.device | str = "cpu",
    orthonormalize: bool = True,
) -> torch.Tensor:
    """Build a complex LG/OAM basis with shape [K, H, W]."""
    if mode_list is None:
        mode_list = default_mode_list()

    basis = torch.stack(
        [
            make_lg_mode(height, width, pixel_pitch, waist, p, ell, device=device)
            for p, ell in mode_list
        ],
        dim=0,
    )
    return orthonormalize_discrete_basis(basis) if orthonormalize else basis
