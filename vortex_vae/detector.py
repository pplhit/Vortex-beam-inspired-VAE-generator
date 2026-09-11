import torch
import torch.nn as nn
import torch.nn.functional as F


class IntensityDetector(nn.Module):
    """Convert a complex output field into a normalized intensity image."""

    def __init__(self, output_size: tuple[int, int] = (28, 28), normalize: str = "max") -> None:
        super().__init__()
        if normalize not in {"max", "energy", "none"}:
            raise ValueError("normalize must be one of: max, energy, none")
        self.output_size = output_size
        self.normalize = normalize

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim == 3:
            field = field.unsqueeze(1)
        if field.ndim != 4 or field.shape[1] != 1:
            raise ValueError("field must have shape [B, H, W] or [B, 1, H, W]")

        intensity = torch.abs(field).square()
        intensity = F.adaptive_avg_pool2d(intensity, self.output_size)

        if self.normalize == "max":
            intensity = intensity / (intensity.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        elif self.normalize == "energy":
            intensity = intensity / (intensity.sum(dim=(-2, -1), keepdim=True) + 1e-8)
        return intensity
