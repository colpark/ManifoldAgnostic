# ManifoldAgnostic

A neural field-based diffusion model for 3D point cloud generation that learns continuous vector fields over arbitrary-dimensional manifolds (1D curves, 2D surfaces, 3D solids) embedded in R³.

## Core Innovation: Learning the Vector Field Function

### The Key Distinction

**Traditional Flow Matching** learns velocity vectors at discrete sample points:
```
Input: Discrete points {x_i} from point cloud
Output: Discrete velocities {v_i} at those specific points
Limitation: Field only defined at sample locations, resolution-bound
```

**Our Approach** learns the vector field function itself as a continuous neural field:
```
Input: Any coordinate x ∈ ℝ³, time t ∈ [0,T]
Output: Velocity vector v ∈ ℝ³ at that continuous location
Model: v_θ: ℝ³ × [0,T] → ℝ³ (continuous over ALL space)
```

This paradigm shift enables:
- **Resolution-agnostic generation**: Query the field at any number of points
- **Compact representation**: Single neural field encodes the entire flow
- **Geometric information recovery**: Surface normals, SDFs, curvature from field gradients
- **Mesh-free generation**: No fixed topology constraints

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │     NEURAL FIELD v_θ(x, t)          │
                    │   Continuous over ℝ³ × [0,T]        │
                    └───────────────┬─────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
   Query at x₁              Query at x₂              Query at x_any
        │                           │                           │
        ▼                           ▼                           ▼
    v(x₁, t)                   v(x₂, t)                   v(x_any, t)
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
                         Integrate ODE: dx/dt = v_θ(x, t)
                         From noise → manifold
```

### Architecture (PixNerd-style)

The architecture directly follows PixNerd (Wang et al.), adapting it from 2D images to 3D point clouds:

1. **Global DiT Blocks**: Extract shape context with 3D spatial awareness
   ```
   points {p_i} → PointEmbedder → DiT Blocks (RoPE + AdaLN + SwiGLU) → shape context s
   ```
   - **3D RoPE**: Rotary position embeddings adapted for 3D coordinates
   - **AdaLN**: Adaptive LayerNorm modulates attention/FFN based on timestep
   - **SwiGLU**: Gated feedforward networks for expressiveness

2. **NerfBlocks (HyperNetwork)**: Shape context generates the neural field
   ```
   s → weight generators → MLP weights (with F.normalize for stability)
   (x, t) → Fourier encoding → batched MLP(weights) → v(x, t)
   ```
   - **Weight Normalization**: Critical for stable hyper-network training
   - **Per-point Processing**: Each point processed independently with shared weights

### Key PixNerd Components Adapted for 3D

| Component | PixNerd (2D) | ManifoldAgnostic (3D) |
|-----------|--------------|----------------------|
| Position Encoding | 2D DCT | 3D Fourier |
| RoPE | 2D (x,y) | 3D (x,y,z) |
| Patchify | unfold(image) | Direct points |
| Output | RGB (3ch) | Velocity (3D) |
| NerfBlock | Per-patch MLP | Per-point MLP |

The key insight from PixNerd's NerfBlock: the **hyper-network generates normalized MLP weights** that define a continuous function, enabling evaluation at any coordinate.

## Mathematical Framework

### Forward Process (Noise Addition)
```
dx = -½β(t)x dt + √β(t) dw
```
Points diffuse from manifold M to isotropic Gaussian.

### Reverse Process (Neural Field ODE)
```
dx/dt = v_θ(x, t)
```
The learned neural field guides points from noise back to the manifold.

### Training Objective (Flow Matching)
```
L(θ) = E_{t,x₀,ε}[||v_θ(x_t, t) - v_target(x_t, t)||²]

where:
  x_t = α(t)x₀ + σ(t)ε           (noised sample)
  v_target = α'(t)x₀ + σ'(t)ε    (target velocity)
```

### Manifold Convergence (Lemma 1)

Near t=0, the learned field approximates:
```
v_θ(x, 0) ≈ -r/σ² · n(x̄)
```
where r is distance to manifold and n is the surface normal. The field **points toward the manifold** with strength proportional to distance.

## Why Neural Fields Beat Sample-Based Methods

| Aspect | Traditional Flow Matching | Neural Field Approach |
|--------|--------------------------|----------------------|
| Representation | Velocities at N points | Continuous function over ℝ³ |
| Resolution | Fixed at training time | Arbitrary at inference |
| Memory | O(N) per shape | O(1) - single network |
| Geometry | Implicit in samples | Explicit via ∇v_θ |
| Interpolation | Between samples | Native continuity |

### Geometric Information from the Field

Since v_θ is differentiable everywhere:
```python
# Surface normals (at t→0, field points perpendicular to manifold)
normal = v_θ(x, 0) / ||v_θ(x, 0)||

# Distance-like quantity (field magnitude ~ distance to surface)
pseudo_distance = ||v_θ(x, 0)||

# Curvature information
curvature = ∇²v_θ(x, 0)
```

## Supported Manifold Types

- **1D Manifolds**: Curves, helices, knots
- **2D Manifolds**: Surfaces, spheres, tori, meshes
- **3D Manifolds**: Solid volumes, occupancy fields
- **Mixed**: Shapes with varying local dimensionality

## Project Structure

```
ManifoldAgnostic/
├── README.md
├── requirements.txt             # Dependencies (torch, numpy, etc.)
├── data/
│   ├── __init__.py
│   └── toy_data.py              # Toy dataset (13 shapes across 1D/2D/3D)
├── notebooks/
│   ├── visualize_data.ipynb     # Data visualization
│   └── train_and_test.ipynb     # Training notebook
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── neural_field.py      # PixNerd-style model:
│   │                            #   - RMSNorm, SwiGLU, TimestepEmbedder
│   │                            #   - RoPEAttention3D, DiTBlock3D
│   │                            #   - NerfEmbedder3D, NerfBlock
│   │                            #   - NeuralFieldDiffusion (main model)
│   ├── diffusion/
│   │   ├── __init__.py
│   │   └── flow_matching.py     # FlowMatchingLoss, FlowMatchingSampler
│   └── utils/
│       └── __init__.py
└── experiments/
    ├── train_toy.py             # CLI training script
    └── outputs/                 # Training outputs (samples, checkpoints)
```

## Roadmap

- [x] Theoretical framework and README
- [ ] Toy point cloud datasets (curves, surfaces, volumes)
- [ ] Core neural field architecture
- [ ] Flow matching training loop
- [ ] ODE sampling with arbitrary resolution
- [ ] Evaluation metrics (Chamfer, EMD, Coverage)
- [ ] Geometric analysis (normal extraction, SDF recovery)
- [ ] ShapeNet experiments
- [ ] Conditional generation

## References

1. Ho, Jain, Abbeel. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
2. Song et al. "Score-Based Generative Modeling through SDEs." ICLR 2021.
3. Luo & Hu. "Diffusion Probabilistic Models for 3D Point Cloud Generation." CVPR 2021.
4. Cai et al. "Learning Gradient Fields for Shape Generation." ECCV 2020.
5. Lipman et al. "Flow Matching for Generative Modeling." NeurIPS 2022.
6. Wang et al. "PixNerd: Pixel Neural Field Diffusion." arXiv 2025.

## Acknowledgments

Architecture inspired by [PixNerd](https://github.com/user/PixNerd) neural field diffusion framework.

## License

MIT

## Citation

```bibtex
@misc{manifoldagnostic2024,
  title={Neural Field Diffusion for Manifold-Agnostic Point Cloud Generation},
  year={2024},
  url={https://github.com/colpark/ManifoldAgnostic}
}
```
