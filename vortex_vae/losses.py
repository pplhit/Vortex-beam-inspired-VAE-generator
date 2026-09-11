import torch
import torch.nn.functional as F


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0,
    reconstruction: str = "mse",
) -> dict[str, torch.Tensor]:
    """Negative ELBO for a diagonal-Gaussian VAE."""
    batch = target.shape[0]

    if reconstruction == "mse":
        per_element = F.mse_loss(recon, target, reduction="none")
    elif reconstruction == "bce":
        per_element = F.binary_cross_entropy(recon.clamp(1e-6, 1 - 1e-6), target, reduction="none")
    else:
        raise ValueError("reconstruction must be 'mse' or 'bce'")

    rec = per_element.reshape(batch, -1).sum(dim=1).mean()
    kl = (-0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=1)).mean()
    total = rec + beta * kl
    return {"loss": total, "reconstruction": rec, "kl": kl}
