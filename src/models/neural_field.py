"""
Neural Field Diffusion Model for 3D Point Clouds

Architecture closely follows PixNerd (Wang et al.):
- RMSNorm for normalization
- SwiGLU feedforward networks
- RoPE (Rotary Position Embedding) for 3D
- AdaLN modulation for condition injection
- DiT-style transformer blocks for global processing
- NerfBlock hyper-network for local neural field

Key adaptation from 2D images to 3D point clouds:
- Each point is treated as a token (like a patch in PixNerd)
- 3D RoPE instead of 2D RoPE
- 3D Fourier position encoding instead of 2D DCT
- Output is velocity v(x,t) ∈ ℝ³ instead of RGB
"""

import math
from typing import Tuple, Optional
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention


# =============================================================================
# UTILITY FUNCTIONS (from PixNerd)
# =============================================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


# =============================================================================
# NORMALIZATION (from PixNerd)
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (from LLaMA).
    More efficient than LayerNorm, used throughout PixNerd.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


# =============================================================================
# FEEDFORWARD (SwiGLU from PixNerd)
# =============================================================================

class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU feedforward network (from LLaMA/PixNerd).
    FFN(x) = W2(SiLU(W1(x)) * W3(x))
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # Following PixNerd: hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# =============================================================================
# EMBEDDINGS (adapted from PixNerd)
# =============================================================================

class TimestepEmbedder(nn.Module):
    """
    Sinusoidal timestep embedding (from PixNerd).
    Maps scalar timestep t to high-dimensional embedding.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10.0) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class PointEmbedder(nn.Module):
    """
    Embed 3D point coordinates into hidden dimension.
    Analogous to PixNerd's patch embedding (s_embedder).
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# =============================================================================
# 3D ROTARY POSITION EMBEDDING (adapted from PixNerd's 2D RoPE)
# =============================================================================

def precompute_freqs_cis_3d(dim: int, max_points: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute 3D rotary position embeddings.

    Adapts PixNerd's 2D RoPE to 3D by using xyz coordinates directly.
    Each point's position is encoded using its 3D coordinates.

    Args:
        dim: Head dimension (must be divisible by 6 for xyz pairs)
        max_points: Maximum number of points (for precomputation)
        theta: Base frequency

    Returns:
        Complex tensor for rotary embeddings
    """
    # For 3D, we need dim divisible by 6 (x, y, z each get dim/3, and complex needs pairs)
    assert dim % 6 == 0, f"Head dim {dim} must be divisible by 6 for 3D RoPE"

    dim_per_axis = dim // 3  # Each axis gets 1/3 of dimensions
    half_dim = dim_per_axis // 2  # Each axis uses complex pairs

    freqs = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))

    return freqs


def apply_rotary_emb_3d(
    xq: torch.Tensor,
    xk: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 3D rotary embeddings to query and key tensors.

    Args:
        xq: Query tensor [B, N, H, D]
        xk: Key tensor [B, N, H, D]
        positions: 3D positions [B, N, 3]
        freqs: Precomputed frequencies

    Returns:
        Rotated query and key tensors
    """
    B, N, H, D = xq.shape
    device = xq.device
    dtype = xq.dtype

    freqs = freqs.to(device)
    dim_per_axis = D // 3
    half_dim = dim_per_axis // 2

    # Split positions into x, y, z
    x_pos = positions[..., 0:1]  # [B, N, 1]
    y_pos = positions[..., 1:2]  # [B, N, 1]
    z_pos = positions[..., 2:3]  # [B, N, 1]

    # Compute angles for each axis
    x_angles = x_pos * freqs[None, None, :]  # [B, N, half_dim]
    y_angles = y_pos * freqs[None, None, :]
    z_angles = z_pos * freqs[None, None, :]

    # Create complex rotations
    x_cis = torch.polar(torch.ones_like(x_angles), x_angles)  # [B, N, half_dim]
    y_cis = torch.polar(torch.ones_like(y_angles), y_angles)
    z_cis = torch.polar(torch.ones_like(z_angles), z_angles)

    # Concatenate all rotations
    freqs_cis = torch.cat([x_cis, y_cis, z_cis], dim=-1)  # [B, N, D//2]
    freqs_cis = freqs_cis[:, :, None, :]  # [B, N, 1, D//2] for broadcasting over heads

    # Apply to queries and keys
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# =============================================================================
# ATTENTION WITH 3D ROPE (adapted from PixNerd's RAttention)
# =============================================================================

class RoPEAttention3D(nn.Module):
    """
    Multi-head attention with 3D Rotary Position Embeddings.
    Adapted from PixNerd's RAttention for 3D point clouds.

    Supports double backward (needed for SDF gradient computation).
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Ensure head_dim is divisible by 6 for 3D RoPE
        # If not, we'll pad internally
        self.rope_dim = (self.head_dim // 6) * 6

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Precompute RoPE frequencies
        if self.rope_dim > 0:
            self.register_buffer(
                'rope_freqs',
                precompute_freqs_cis_3d(self.rope_dim, max_points=4096)
            )
        else:
            self.rope_freqs = None

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features [B, N, C]
            positions: 3D positions [B, N, 3]
            mask: Optional attention mask
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, N, H, D]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply 3D RoPE if available
        if self.rope_freqs is not None and self.rope_dim > 0:
            # Only apply to first rope_dim dimensions
            q_rope = q[..., :self.rope_dim]
            k_rope = k[..., :self.rope_dim]
            q_rope, k_rope = apply_rotary_emb_3d(q_rope, k_rope, positions, self.rope_freqs)
            q = torch.cat([q_rope, q[..., self.rope_dim:]], dim=-1)
            k = torch.cat([k_rope, k[..., self.rope_dim:]], dim=-1)

        # Reshape for attention: [B, H, N, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Manual attention (supports double backward, needed for SDF gradients)
        # scaled_dot_product_attention doesn't support second-order derivatives
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# =============================================================================
# DiT BLOCK (from PixNerd's FlattenDiTBlock)
# =============================================================================

class DiTBlock3D(nn.Module):
    """
    DiT block with AdaLN modulation for 3D point clouds.
    Directly adapted from PixNerd's FlattenDiTBlock.

    Structure:
    1. AdaLN-modulated self-attention with 3D RoPE
    2. AdaLN-modulated SwiGLU feedforward
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()

        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RoPEAttention3D(hidden_size, num_heads=num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)

        # AdaLN modulation: 6 params (shift, scale, gate) × 2 (attn, mlp)
        self.adaLN_modulation = nn.Linear(hidden_size, 6 * hidden_size, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                positions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features [B, N, C]
            c: Condition embedding [B, 1, C] or [B, C]
            positions: 3D positions [B, N, 3]
            mask: Optional attention mask
        """
        # Get modulation parameters
        if c.dim() == 2:
            c = c.unsqueeze(1)

        mod = self.adaLN_modulation(c)  # [B, 1, 6*C]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Attention with modulation
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), positions, mask)

        # FFN with modulation
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


# =============================================================================
# NERF EMBEDDER (adapted from PixNerd for 3D)
# =============================================================================

class NerfEmbedder3D(nn.Module):
    """
    3D position encoding for neural field queries.
    Adapted from PixNerd's NerfEmbedder (2D DCT) to 3D Fourier features.

    Instead of 2D DCT: cos(x*fx*π) * cos(y*fy*π) * (1+fx*fy)^-1
    We use 3D Fourier: sin/cos of x*freq, y*freq, z*freq with learned weighting.
    """
    def __init__(self, in_channels: int, hidden_size: int, max_freqs: int = 8):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size = hidden_size

        # 3D Fourier features: for each freq, we have sin and cos for x, y, z
        # Total: max_freqs * 2 (sin/cos) * 3 (xyz) = max_freqs * 6
        fourier_dim = max_freqs * 6

        self.embedder = nn.Sequential(
            nn.Linear(in_channels + fourier_dim, hidden_size, bias=True),
        )

        # Precompute frequencies
        freqs = torch.arange(1, max_freqs + 1, dtype=torch.float32) * math.pi
        self.register_buffer('freqs', freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 3D coordinates [B, N, 3]

        Returns:
            Position-encoded features [B, N, hidden_size]
        """
        B, N, _ = x.shape

        # Compute Fourier features
        # x: [B, N, 3], freqs: [max_freqs]
        # x_scaled: [B, N, 3, max_freqs]
        x_scaled = x.unsqueeze(-1) * self.freqs[None, None, None, :]

        # Sin and cos for each coordinate
        fourier_sin = torch.sin(x_scaled).reshape(B, N, -1)  # [B, N, 3*max_freqs]
        fourier_cos = torch.cos(x_scaled).reshape(B, N, -1)  # [B, N, 3*max_freqs]

        # Concatenate raw coordinates and Fourier features
        features = torch.cat([x, fourier_sin, fourier_cos], dim=-1)  # [B, N, 3 + 6*max_freqs]

        return self.embedder(features)


# =============================================================================
# NERF BLOCK (from PixNerd - hyper-network for neural field)
# =============================================================================

class NerfBlock(nn.Module):
    """
    HyperNetwork block that generates MLP weights from shape context.
    Directly from PixNerd with weight normalization.

    This is the core of the neural field: the generated MLP defines
    a continuous function over 3D space.

    Structure:
    1. Shape context s generates MLP weights
    2. Weights are normalized (critical for stability)
    3. MLP is applied to position-encoded features
    """
    def __init__(self, hidden_size_s: int, hidden_size_x: int, mlp_ratio: int = 4):
        super().__init__()

        self.hidden_size_x = hidden_size_x
        self.mlp_ratio = mlp_ratio

        # Weight generator: generates both fc1 and fc2 weights
        # fc1: hidden_size_x -> hidden_size_x * mlp_ratio
        # fc2: hidden_size_x * mlp_ratio -> hidden_size_x
        self.param_generator = nn.Linear(
            hidden_size_s,
            2 * hidden_size_x * hidden_size_x * mlp_ratio,
            bias=True
        )

        self.norm = RMSNorm(hidden_size_x, eps=1e-6)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Position-encoded features [B*N, P, hidden_size_x] or [B, N, hidden_size_x]
            s: Shape context [B*N, hidden_size_s] or [B, hidden_size_s]

        Returns:
            Transformed features, same shape as input
        """
        if x.dim() == 3 and s.dim() == 2:
            # Standard case: x is [B, N, C], s is [B, C_s]
            # We need to expand s to match x
            B, N, C = x.shape
            s = s.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
            x = x.reshape(B * N, 1, C)
            needs_reshape = True
        else:
            needs_reshape = False
            B_times_N = x.shape[0]

        batch_size = x.shape[0]

        # Generate MLP weights
        mlp_params = self.param_generator(s)  # [batch, 2*C*C*ratio]
        fc1_param, fc2_param = mlp_params.chunk(2, dim=-1)

        # Reshape weights
        fc1_param = fc1_param.view(batch_size, self.hidden_size_x, self.hidden_size_x * self.mlp_ratio)
        fc2_param = fc2_param.view(batch_size, self.hidden_size_x * self.mlp_ratio, self.hidden_size_x)

        # CRITICAL: Normalize weights (from PixNerd)
        # This is essential for stable training with hyper-networks
        fc1_param = F.normalize(fc1_param, dim=-2)
        fc2_param = F.normalize(fc2_param, dim=-2)

        # Apply MLP with residual
        res_x = x
        x = self.norm(x)
        x = torch.bmm(x, fc1_param)  # [batch, seq, C*ratio]
        x = F.silu(x)
        x = torch.bmm(x, fc2_param)  # [batch, seq, C]
        x = x + res_x

        if needs_reshape:
            x = x.reshape(B, N, -1)

        return x


# =============================================================================
# FINAL LAYER (from PixNerd)
# =============================================================================

class NerfFinalLayer(nn.Module):
    """
    Final layer for neural field output.
    From PixNerd: RMSNorm + Linear projection to output channels.
    """
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear(x)
        return x


# =============================================================================
# MAIN MODEL (adapted from PixNerd's PixNerDiT)
# =============================================================================

class NeuralFieldDiffusion(nn.Module):
    """
    Neural Field Diffusion Model for 3D Point Clouds.

    Architecture follows PixNerd closely:
    1. Point embedding (like patch embedding)
    2. Timestep + condition embedding
    3. Global DiT blocks with 3D RoPE and AdaLN
    4. Local NerfBlocks (hyper-network) for neural field
    5. Final layer projects to velocity output

    The key insight from PixNerd: global transformer provides shape context,
    which is used by the hyper-network to generate a continuous neural field.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
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
            out_channels: Output dimension (3 for velocity)
            hidden_size: Hidden dimension for transformer
            hidden_size_x: Hidden dimension for NerfBlocks
            num_heads: Number of attention heads
            num_blocks: Total number of blocks
            num_cond_blocks: Number of DiT blocks (rest are NerfBlocks)
            nerf_mlp_ratio: MLP ratio for NerfBlocks
            mlp_ratio: MLP ratio for DiT blocks
            max_freqs: Max frequencies for position encoding
            num_classes: Number of classes for conditional generation (0 = unconditional)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
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
            self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)  # +1 for unconditional
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

        # Final layer
        self.final_layer = NerfFinalLayer(hidden_size_x, out_channels)

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

        # Final layer - zero init (from PixNerd)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass predicting velocity field.

        Args:
            x: Point cloud [B, N, 3]
            t: Timestep [B] in [0, 1]
            y: Optional class labels [B]
            mask: Optional attention mask

        Returns:
            Predicted velocity [B, N, 3]
        """
        B, N, C = x.shape
        positions = x.clone()  # Store positions for RoPE and NerfBlocks

        # Timestep embedding
        t_emb = self.t_embedder(t.view(-1))  # [B, hidden_size]

        # Condition embedding
        if self.y_embedder is not None and y is not None:
            y_emb = self.y_embedder(y)  # [B, hidden_size]
            c = F.silu(t_emb + y_emb)
        else:
            c = F.silu(t_emb)

        # Point embedding for global processing
        s = self.point_embedder(x)  # [B, N, hidden_size]

        # Global DiT blocks
        for i in range(self.num_cond_blocks):
            s = self.blocks[i](s, c, positions, mask)

        # Combine with timestep for NerfBlocks (like PixNerd)
        s = F.silu(t_emb.unsqueeze(1) + s)  # [B, N, hidden_size]

        # Position encoding for local processing
        x = self.nerf_embedder(positions)  # [B, N, hidden_size_x]

        # Local NerfBlocks
        # Pool shape context for each point
        s_pooled = s.mean(dim=1)  # [B, hidden_size] - global shape context

        for i in range(self.num_cond_blocks, self.num_blocks):
            x = self.blocks[i](x, s_pooled)

        # Final projection to velocity
        x = self.final_layer(x)  # [B, N, out_channels]

        return x

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

        This enables resolution-independent generation: compute context once,
        then query at any number of points.

        Args:
            query_points: Query positions [B, M, 3]
            t: Timestep [B]
            context: Pre-computed shape context [B, hidden_size]

        Returns:
            Velocities at query points [B, M, 3]
        """
        # Position encoding for query points
        x = self.nerf_embedder(query_points)  # [B, M, hidden_size_x]

        # Apply NerfBlocks with pre-computed context
        for i in range(self.num_cond_blocks, self.num_blocks):
            x = self.blocks[i](x, context)

        # Final projection
        x = self.final_layer(x)

        return x


# =============================================================================
# TESTING
# =============================================================================

def test_model():
    """Test the neural field model."""
    print("Testing NeuralFieldDiffusion (PixNerd-style)...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model with PixNerd-style config
    model = NeuralFieldDiffusion(
        in_channels=3,
        out_channels=3,
        hidden_size=384,
        hidden_size_x=64,
        num_heads=6,
        num_blocks=12,
        num_cond_blocks=4,
        nerf_mlp_ratio=4,
        max_freqs=8,
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Test forward pass
    B, N = 4, 256
    x = torch.randn(B, N, 3, device=device)
    t = torch.rand(B, device=device)

    v = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {v.shape}")

    # Test context extraction
    context = model.get_context(x, t)
    print(f"Context shape: {context.shape}")

    # Test field query at different resolutions
    for M in [64, 128, 256, 512, 1024]:
        query = torch.randn(B, M, 3, device=device)
        v_query = model.query_field(query, t, context)
        print(f"Query {M} points: {v_query.shape}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_model()
