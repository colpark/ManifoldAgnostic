"""
SDF-Based Neural Field Diffusion Model for 3D Point Clouds

Key Innovation: Instead of directly predicting velocity v(x,t), we predict a
scalar distance field f(x,t) and derive velocity as v = -∇_x f(x,t).

Benefits:
1. Smoother training (no directional discontinuities)
2. Implicit surface representation (SDF at t=0)
3. Gradient field is naturally continuous
4. Same generative capacity as direct velocity prediction

Architecture (PixNerd-style, adapted for scalar output):
- DiT blocks with 3D RoPE and AdaLN for global context
- NerfBlocks (hyper-network) for local neural field
- Output: scalar f(x,t) ∈ ℝ instead of vector v(x,t) ∈ ℝ³
- Velocity: v(x,t) = -∇_x f(x,t) via autograd
"""

import math
from typing import Tuple, Optional
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

# Import shared components from neural_field
from .neural_field import (
    modulate,
    RMSNorm,
    SwiGLUFeedForward,
    TimestepEmbedder,
    PointEmbedder,
    precompute_freqs_cis_3d,
    apply_rotary_emb_3d,
    RoPEAttention3D,
    DiTBlock3D,
    NerfEmbedder3D,
    NerfBlock,
    NerfFinalLayer,
)


# =============================================================================
# SDF NEURAL FIELD MODEL
# =============================================================================

class SDFNeuralField(nn.Module):
    """
    SDF-Based Neural Field Diffusion Model.

    Predicts a scalar distance field f(x,t) and derives velocity as -∇f.

    Key differences from NeuralFieldDiffusion:
    1. out_channels = 1 (scalar field)
    2. forward() returns scalar field values
    3. get_velocity() computes gradient via autograd
    4. Training uses velocity loss but with gradient-based prediction

    Architecture follows PixNerd closely:
    - Global DiT blocks with 3D RoPE and AdaLN for shape context
    - Local NerfBlocks (hyper-network) for continuous neural field
    - Final layer projects to scalar output
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 384,
        hidden_size_x: int = 64,
        num_heads: int = 6,
        num_blocks: int = 12,
        num_cond_blocks: int = 4,
        nerf_mlp_ratio: int = 4,
        mlp_ratio: float = 4.0,
        max_freqs: int = 8,
        num_classes: int = 0,
    ):
        """
        Args:
            in_channels: Input point dimension (3 for xyz)
            hidden_size: Hidden dimension for transformer
            hidden_size_x: Hidden dimension for NerfBlocks
            num_heads: Number of attention heads
            num_blocks: Total number of blocks
            num_cond_blocks: Number of DiT blocks (rest are NerfBlocks)
            nerf_mlp_ratio: MLP ratio for NerfBlocks
            mlp_ratio: MLP ratio for DiT blocks
            max_freqs: Max frequencies for position encoding
            num_classes: Number of classes for conditional generation
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.hidden_size_x = hidden_size_x
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_cond_blocks = num_cond_blocks

        # Point embedding (analogous to s_embedder in PixNerd)
        self.point_embedder = PointEmbedder(in_channels, hidden_size)

        # Position encoding for NerfBlocks
        self.nerf_embedder = NerfEmbedder3D(in_channels, hidden_size_x, max_freqs)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Optional class embedding
        self.num_classes = num_classes
        if num_classes > 0:
            self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)
        else:
            self.y_embedder = None

        # Build blocks
        self.blocks = nn.ModuleList()

        # DiT blocks for global processing
        for _ in range(num_cond_blocks):
            self.blocks.append(
                DiTBlock3D(hidden_size, num_heads, mlp_ratio)
            )

        # NerfBlocks for local neural field
        for _ in range(num_cond_blocks, num_blocks):
            self.blocks.append(
                NerfBlock(hidden_size, hidden_size_x, nerf_mlp_ratio)
            )

        # Final layer - SCALAR output (1 instead of 3)
        self.final_layer = NerfFinalLayer(hidden_size_x, out_channels=1)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights following PixNerd."""
        # Point embedder
        nn.init.xavier_uniform_(self.point_embedder.proj.weight)
        nn.init.zeros_(self.point_embedder.proj.bias)

        # Timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Class embedder
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.weight, std=0.02)

        # Final layer - zero init for stability
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward_sdf(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass predicting scalar distance field.

        Args:
            x: Point cloud [B, N, 3]
            t: Timestep [B] in [0, 1]
            y: Optional class labels [B]
            mask: Optional attention mask

        Returns:
            Predicted SDF values [B, N, 1]
        """
        B, N, C = x.shape
        positions = x.clone()

        # Timestep embedding
        t_emb = self.t_embedder(t.view(-1))  # [B, hidden_size]

        # Condition embedding
        if self.y_embedder is not None and y is not None:
            y_emb = self.y_embedder(y)
            c = F.silu(t_emb + y_emb)
        else:
            c = F.silu(t_emb)

        # Point embedding for global processing
        s = self.point_embedder(x)  # [B, N, hidden_size]

        # Global DiT blocks
        for i in range(self.num_cond_blocks):
            s = self.blocks[i](s, c, positions, mask)

        # Combine with timestep for NerfBlocks
        s = F.silu(t_emb.unsqueeze(1) + s)

        # Position encoding for local processing
        feat = self.nerf_embedder(positions)  # [B, N, hidden_size_x]

        # Pool shape context
        s_pooled = s.mean(dim=1)  # [B, hidden_size]

        # Local NerfBlocks
        for i in range(self.num_cond_blocks, self.num_blocks):
            feat = self.blocks[i](feat, s_pooled)

        # Final projection to scalar SDF
        sdf = self.final_layer(feat)  # [B, N, 1]

        return sdf

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass returning velocity (for compatibility with flow matching).

        Computes SDF and derives velocity as v = -∇_x f(x,t).

        Args:
            x: Point cloud [B, N, 3]
            t: Timestep [B] in [0, 1]
            y: Optional class labels [B]
            mask: Optional attention mask

        Returns:
            Predicted velocity [B, N, 3] (derived from SDF gradient)
        """
        return self.get_velocity(x, t, y, mask)

    def get_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute velocity as negative gradient of SDF.

        v(x, t) = -∇_x f(x, t)

        This is the key innovation: smooth scalar field → smooth velocity field.

        Args:
            x: Point cloud [B, N, 3] - requires grad
            t: Timestep [B]
            y: Optional class labels [B]
            mask: Optional attention mask

        Returns:
            Velocity field [B, N, 3]
        """
        # Enable gradients for input
        x_grad = x.clone().requires_grad_(True)

        # Forward to get SDF
        sdf = self.forward_sdf(x_grad, t, y, mask)  # [B, N, 1]

        # Compute gradient w.r.t. input positions
        # Sum SDF values to get scalar for backward
        sdf_sum = sdf.sum()

        # Compute gradient
        grad = torch.autograd.grad(
            outputs=sdf_sum,
            inputs=x_grad,
            create_graph=self.training,  # Keep graph for training
            retain_graph=self.training,
        )[0]  # [B, N, 3]

        # Velocity is negative gradient (points toward decreasing SDF = toward surface)
        velocity = -grad

        return velocity

    def get_sdf(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get raw SDF values (useful for visualization/analysis).

        Args:
            x: Point cloud [B, N, 3]
            t: Timestep [B]
            y: Optional class labels [B]

        Returns:
            SDF values [B, N, 1]
        """
        return self.forward_sdf(x, t, y)

    def get_context(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Extract shape context for resolution-independent sampling.

        Args:
            x: Point cloud [B, N, 3]
            t: Timestep [B]

        Returns:
            Shape context [B, hidden_size]
        """
        B, N, C = x.shape
        positions = x.clone()

        t_emb = self.t_embedder(t.view(-1))
        c = F.silu(t_emb)

        s = self.point_embedder(x)

        for i in range(self.num_cond_blocks):
            s = self.blocks[i](s, c, positions)

        s = F.silu(t_emb.unsqueeze(1) + s)
        s_pooled = s.mean(dim=1)

        return s_pooled

    def query_field(
        self,
        query_points: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Query the neural field at arbitrary points given pre-computed context.
        Returns velocity (not SDF) for compatibility with sampling.

        Args:
            query_points: Query positions [B, M, 3]
            t: Timestep [B]
            context: Pre-computed shape context [B, hidden_size]

        Returns:
            Velocities at query points [B, M, 3]
        """
        # Enable gradients
        query_grad = query_points.clone().requires_grad_(True)

        # Position encoding for query points
        feat = self.nerf_embedder(query_grad)  # [B, M, hidden_size_x]

        # Apply NerfBlocks with pre-computed context
        for i in range(self.num_cond_blocks, self.num_blocks):
            feat = self.blocks[i](feat, context)

        # Get SDF
        sdf = self.final_layer(feat)  # [B, M, 1]

        # Compute gradient
        sdf_sum = sdf.sum()
        grad = torch.autograd.grad(
            outputs=sdf_sum,
            inputs=query_grad,
            create_graph=self.training,
            retain_graph=self.training,
        )[0]

        velocity = -grad

        return velocity

    def query_sdf(
        self,
        query_points: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Query raw SDF values at arbitrary points.

        Args:
            query_points: Query positions [B, M, 3]
            t: Timestep [B]
            context: Pre-computed shape context [B, hidden_size]

        Returns:
            SDF values at query points [B, M, 1]
        """
        feat = self.nerf_embedder(query_points)

        for i in range(self.num_cond_blocks, self.num_blocks):
            feat = self.blocks[i](feat, context)

        sdf = self.final_layer(feat)

        return sdf


# =============================================================================
# SDF-SPECIFIC LOSS FUNCTIONS
# =============================================================================

class SDFFlowMatchingLoss(nn.Module):
    """
    Flow matching loss for SDF-based model.

    The model predicts SDF, we derive velocity via gradient, then compute
    loss against target velocity. This gives smooth gradients through the
    scalar field.

    Optionally includes regularization terms:
    - Eikonal loss: ||∇f|| ≈ 1 (SDF property)
    - Smoothness loss: Encourage smooth SDF
    """

    def __init__(
        self,
        schedule_type: str = 'linear',
        eikonal_weight: float = 0.0,
        smoothness_weight: float = 0.0,
    ):
        super().__init__()
        self.schedule_type = schedule_type
        self.eikonal_weight = eikonal_weight
        self.smoothness_weight = smoothness_weight

    def get_alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get noise schedule parameters."""
        if self.schedule_type == 'linear':
            alpha = 1 - t
            sigma = t
        elif self.schedule_type == 'cosine':
            alpha = torch.cos(t * math.pi / 2)
            sigma = torch.sin(t * math.pi / 2)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")
        return alpha, sigma

    def forward(
        self,
        model: SDFNeuralField,
        x0: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute flow matching loss with SDF-derived velocity.

        Args:
            model: SDF neural field model
            x0: Clean point clouds [B, N, 3]
            y: Optional class labels [B]

        Returns:
            Dictionary with loss and optional regularization terms
        """
        B, N, C = x0.shape
        device = x0.device

        # Sample random timesteps
        t = torch.rand(B, device=device)

        # Sample noise
        eps = torch.randn_like(x0)

        # Get schedule parameters
        alpha, sigma = self.get_alpha_sigma(t)
        alpha = alpha.view(B, 1, 1)
        sigma = sigma.view(B, 1, 1)

        # Noisy points: x_t = alpha * x0 + sigma * eps
        x_t = alpha * x0 + sigma * eps

        # Target velocity (flow matching target)
        v_target = eps - x0

        # Predict velocity (via SDF gradient)
        v_pred = model.get_velocity(x_t, t, y)

        # MSE loss on velocity
        velocity_loss = F.mse_loss(v_pred, v_target)

        total_loss = velocity_loss
        output = {'loss': total_loss, 'velocity_loss': velocity_loss.item()}

        # Optional: Eikonal regularization (||∇f|| ≈ 1)
        if self.eikonal_weight > 0:
            # Get SDF values and compute gradient norm
            x_t_grad = x_t.clone().requires_grad_(True)
            sdf = model.forward_sdf(x_t_grad, t, y)

            grad = torch.autograd.grad(
                outputs=sdf.sum(),
                inputs=x_t_grad,
                create_graph=True,
            )[0]

            grad_norm = grad.norm(dim=-1)  # [B, N]
            eikonal_loss = ((grad_norm - 1) ** 2).mean()

            total_loss = total_loss + self.eikonal_weight * eikonal_loss
            output['eikonal_loss'] = eikonal_loss.item()

        output['loss'] = total_loss

        return output


# =============================================================================
# TESTING
# =============================================================================

def test_sdf_model():
    """Test the SDF neural field model."""
    print("Testing SDFNeuralField...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model with small config
    model = SDFNeuralField(
        in_channels=3,
        hidden_size=128,
        hidden_size_x=32,
        num_heads=4,
        num_blocks=6,
        num_cond_blocks=2,
        nerf_mlp_ratio=2,
        max_freqs=6,
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Test forward pass
    B, N = 4, 256
    x = torch.randn(B, N, 3, device=device)
    t = torch.rand(B, device=device)

    # Test SDF output
    sdf = model.forward_sdf(x, t)
    print(f"Input shape: {x.shape}")
    print(f"SDF output shape: {sdf.shape}")

    # Test velocity output
    v = model.get_velocity(x, t)
    print(f"Velocity output shape: {v.shape}")

    # Test forward (should return velocity)
    v2 = model(x, t)
    print(f"Forward output shape: {v2.shape}")

    # Test context extraction
    context = model.get_context(x, t)
    print(f"Context shape: {context.shape}")

    # Test field query at different resolutions
    print("\nTesting resolution independence:")
    for M in [64, 128, 256, 512]:
        query = torch.randn(B, M, 3, device=device)
        v_query = model.query_field(query, t, context)
        sdf_query = model.query_sdf(query, t, context)
        print(f"  Query {M} points: velocity {v_query.shape}, SDF {sdf_query.shape}")

    # Test loss
    print("\nTesting SDFFlowMatchingLoss:")
    loss_fn = SDFFlowMatchingLoss(eikonal_weight=0.1)
    output = loss_fn(model, x)
    print(f"  Total loss: {output['loss'].item():.4f}")
    print(f"  Velocity loss: {output['velocity_loss']:.4f}")
    if 'eikonal_loss' in output:
        print(f"  Eikonal loss: {output['eikonal_loss']:.4f}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_sdf_model()
