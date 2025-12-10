# ManifoldAgnostic

A vector field-based diffusion model for 3D point cloud generation that represents arbitrary-dimensional manifolds (1D curves, 2D surfaces, 3D solids) embedded in R³.

## Overview

This repository implements the theoretical framework from "Vector Field Diffusion Models for Point Cloud Generation on Manifolds." Unlike traditional approaches (GANs, normalizing flows), our method learns a continuous time-dependent vector field that transports points from noise to data manifolds, offering several key advantages:

- **Manifold Support**: Accurately represents distributions concentrated on lower-dimensional manifolds in R³, which standard diffusion models and invertible flows struggle with
- **Arbitrary Resolution**: Generate any number of points at inference time—the vector field is defined continuously over space and time
- **Theoretical Guarantees**: Provably converges to manifold-supported distributions (unlike invertible flows which require the manifold hypothesis to be violated)

## Key Contributions

### Theoretical

1. **Manifold Representation Theorem**: Diffusion models with non-invertible forward processes can exactly target singular distributions on lower-dimensional manifolds, whereas invertible flows cannot (Proposition 2)

2. **Attracting Vector Field**: The learned denoising vector field provably attracts points onto the correct manifold, with the score field aligning with manifold normals (Lemma 1)

3. **Resolution Independence**: The model supports arbitrary output resolution without retraining (Proposition 1)

### Methodological

- Forward diffusion via Variance-Preserving SDE that perturbs points off the data manifold
- Learned reverse process parameterized as a time-dependent vector field v_θ(x, t)
- Training via denoising score matching objective
- Point-wise evaluation enabling permutation invariance and scalable generation

## Architecture

```
Forward Process:  x(0) ~ p_data  →  x(T) ~ N(0, σ²I)
                  [manifold]        [isotropic noise]
                       ↓
                  dx = -½β(t)x dt + √β(t) dw

Reverse Process:  x(T) ~ N(0, σ²I)  →  x(0) ~ p_data
                       ↓
                  dx/dt = v_θ(x, t)
                  [learned vector field]
```

## Mathematical Framework

### Forward SDE (Variance-Preserving)
```
dx = -½β(t)x dt + √β(t) dw
```

### Reverse ODE (Probability Flow)
```
dx/dt = -½β(t)x - ½β(t)∇_x log p_t(x)
```

### Training Objective
```
L(θ) = E_{t,x(0),ε}[λ(t) ||v_θ(x(t), t) - v̄(x(t), t)||²]
```

## Comparison with Prior Work

| Method | Manifold Support | Resolution Flexible | Invertible |
|--------|-----------------|---------------------|------------|
| PointFlow (NF) | ✗ (requires dequantization) | ✗ | ✓ |
| Point Cloud DDPM | Partial | ✗ | ✗ |
| ShapeGF | ✓ | ✓ | ✗ |
| **Ours** | ✓ | ✓ | ✗ |

## Roadmap

- [ ] Core vector field network architecture
- [ ] Forward/reverse diffusion processes
- [ ] Training pipeline with score matching
- [ ] ShapeNet data loading and preprocessing
- [ ] Evaluation metrics (Chamfer, EMD, Coverage)
- [ ] Conditional generation (class-conditioned, text-conditioned)
- [ ] Implicit surface extraction from learned vector field
- [ ] Efficiency optimizations (adaptive ODE solvers, importance sampling)

## References

1. Ho, Jain, Abbeel. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
2. Song et al. "Score-Based Generative Modeling through SDEs." ICLR 2021.
3. Luo & Hu. "Diffusion Probabilistic Models for 3D Point Cloud Generation." CVPR 2021.
4. Cai et al. "Learning Gradient Fields for Shape Generation." ECCV 2020.
5. Yang et al. "PointFlow: 3D Point Cloud Generation with CNFs." ICCV 2019.
6. Huang et al. "PointInfinity: Resolution-Invariant Point Diffusion Models." ICCV 2023.
7. Lipman et al. "Flow Matching for Generative Modeling." NeurIPS 2022.

## License

MIT

## Citation

```bibtex
@misc{manifoldagnostic2024,
  title={Vector Field Diffusion Models for Point Cloud Generation on Manifolds},
  year={2024},
  url={https://github.com/colpark/ManifoldAgnostic}
}
```
