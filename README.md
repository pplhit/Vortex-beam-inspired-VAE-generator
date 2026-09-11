# Vortex-beam-inspired VAE generator

A variational optical generator in which a standard Gaussian VAE latent variable is embedded into a Laguerre-Gaussian / orbital-angular-momentum (LG/OAM) modal space and decoded by a diffractive optical network.

## Core idea

The probabilistic part is an ordinary VAE:

\[
q_\phi(z|x)=\mathcal{N}(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))),
\qquad
z=\mu+\sigma\odot\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
\]

The optical embedding pairs two real Gaussian latent coordinates into one complex modal coefficient:

\[
c_m=\frac{z_{2m}+i z_{2m+1}}{\sqrt{2K}},
\]

and synthesizes a coherent structured-light field

\[
U_0(x,y)=\sum_{m=1}^{K}c_m\,LG_{p_m}^{\ell_m}(x,y).
\]

A diffractive neural network then implements the optical decoder:

\[
U_{out}=D_\theta(U_0), \qquad \hat{x}=|U_{out}|^2.
\]

During generation the encoder is removed completely:

```text
Gaussian noise z ~ N(0,I)
        |
        v
complex OAM/LG coefficients
        |
        v
coherent vortex-mode field
        |
        v
diffractive optical decoder
        |
        v
output intensity image
```

## Repository structure

```text
.
├── vortex_vae/
│   ├── encoder.py          # q_phi(z|x): mu and log-variance
│   ├── lg_modes.py         # LG/OAM basis construction + discrete whitening
│   ├── vortex_mapper.py    # Gaussian latent -> complex modal field
│   ├── detector.py         # complex field -> intensity image
│   ├── model.py            # full VortexOpticalVAE
│   └── losses.py           # negative ELBO
├── optics/
│   └── user_diffractive_network.py
│                            # adapter for your existing D2NN / ASM code
├── train.py
├── generate.py
├── visualize_modes.py
└── tests/
```

## Default modal space

The default basis contains 32 complex modes:

- `p = 0`, `ell = -8, ..., +8`  -> 17 modes
- `p = 1`, `ell = -7, ..., +7`  -> 15 modes

Two real latent variables form each complex coefficient, giving a 64-dimensional real VAE latent space.

Both positive and negative topological charges are included. The `ell=0` modes are intentionally retained to avoid forcing every input field to contain a central vortex null.

## Why the mapping preserves the VAE formulation

The KL divergence is still computed in the original real latent space:

\[
D_{KL}\left[q_\phi(z|x)\,\|\,\mathcal{N}(0,I)\right].
\]

The deterministic mapping `z -> U0` is simply the first part of the decoder. Therefore no change to the mathematical definition of a VAE is required.

For a discretely orthonormal modal basis, the complex-pair mapping also approximately preserves latent energy geometry:

\[
\|U_0\|_2^2 \approx \sum_m |c_m|^2.
\]

`lg_modes.py` optionally whitens the sampled basis through its Gram matrix to compensate for finite aperture and pixel sampling.

## Installation

```bash
git clone https://github.com/pplhit/Vortex-beam-inspired-VAE-generator.git
cd Vortex-beam-inspired-VAE-generator
pip install -r requirements.txt
```

## Connect your existing diffraction / angular-spectrum code

Edit only:

```text
optics/user_diffractive_network.py
```

The required interface is:

```python
# input : complex [B, 1, H, W]
# output: complex [B, 1, H, W]
Uout = optical_decoder(U0)
```

For example:

```python
from my_d2nn import DiffractiveNetwork


def build_diffractive_network(*, smoke_test=False):
    if smoke_test:
        return IdentityOpticalDecoder()
    return DiffractiveNetwork(...)
```

If your existing ASM layer already uses `[B, C, H, W]` complex tensors, no tensor-layout modification should be necessary.

## Smoke test

Before inserting the real optical decoder, verify that the whole software pipeline is connected correctly:

```bash
python train.py --smoke-test --epochs 1
```

The identity backend is only for software verification and is **not** the research optical decoder.

## Training with the real optical decoder

After implementing `build_diffractive_network()`:

```bash
python train.py \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-4 \
  --beta 1.0
```

The default objective is the negative ELBO:

\[
\mathcal{L}=\mathcal{L}_{rec}+\beta\mathcal{L}_{KL}.
\]

No GAN or adversarial loss is used.

## Generation

```bash
python generate.py \
  --checkpoint checkpoints/latest.pt \
  --num-samples 16 \
  --output outputs/generated.png
```

In this path the encoder is unused. Samples are produced directly from `z ~ N(0,I)`.

## Visualize individual vortex modes

```bash
python visualize_modes.py --index 0
```

## Recommended first experiments

1. MNIST reconstruction and random generation.
2. Compare `LG/OAM modal embedding` against `reshape + upsample` latent encoding.
3. Compare against a learned FC latent-to-field mapping.
4. Plot modal power spectra `|c_m|^2` alongside generated images.
5. Test the effect of discrete basis whitening.
6. Extend to Fashion-MNIST after confirming that the passive optical decoder has sufficient capacity.

## Notes on physical implementation

The synthesized field generally contains both amplitude and phase. A phase-only SLM therefore requires a complex-field encoding method (for example double-phase or an equivalent holographic encoding strategy) before the field enters the diffractive decoder.

## Current status

The VAE, LG/OAM modal embedding, loss, detector, training loop and generation path are implemented. The repository intentionally leaves the physical diffraction module as an adapter because propagation distance, wavelength, pixel pitch, padding/cropping and phase-mask parameterization depend on the user's existing optical simulation code.
