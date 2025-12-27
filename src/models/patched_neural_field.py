"""
Patched Neural Field Diffusion for 3D Point Clouds

Architecture based on PixNerd (Neural Radiance Field Diffusion) adapted for point clouds.

Key architectural principles from PixNerd:
1. Two-stage processing:
   - Stage 1: Transformer blocks on patch tokens (global context)
   - Stage 2: NerfBlocks on local features (per-patch neural fields)
2. Time re-injection after transformer: s = silu(t + s)
3. Local position encoding within patches (not global positions)
4. HyperNetwork generates per-patch MLP weights from context

Adaptation for 3D point clouds:
- Points sorted by space-filling curve for locality preservation
- Patches are groups of spatially nearby points
- 3D local coordinates within patches replace 2D DCT encoding
- Unsort at end to restore original point order
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from functools import lru_cache


# =============================================================================
# SPACE-FILLING CURVE UTILITIES
# =============================================================================

def morton_encode_3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                     bits: int = 10) -> torch.Tensor:
    """
    Encode 3D coordinates into Morton (Z-order) code.

    Morton code interleaves bits of x, y, z coordinates to create a 1D ordering
    that preserves spatial locality.

    Args:
        x, y, z: Quantized integer coordinates [B, N], values in [0, 2^bits)
        bits: Number of bits per coordinate (max 10 for 32-bit output)

    Returns:
        Morton codes [B, N] as int64
    """
    def spread_bits(v: torch.Tensor) -> torch.Tensor:
        """Spread bits for 3D interleaving using magic numbers."""
        v = v.long() & 0x3FF  # 10 bits max
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)


def sort_by_space_filling_curve(
    points: torch.Tensor,
    bits: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sort points by Morton (Z-order) space-filling curve.

    This ordering ensures that spatially nearby points are also nearby
    in the 1D ordering, enabling effective patchification.

    Args:
        points: Point coordinates [B, N, 3]
        bits: Quantization resolution (10 bits = 1024 levels)

    Returns:
        sorted_points: Points in Morton order [B, N, 3]
        sort_indices: Indices for unsorting [B, N]
    """
    B, N, _ = points.shape
    device = points.device

    # Normalize to [0, 1] per batch for consistent quantization
    mins = points.min(dim=1, keepdim=True)[0]
    maxs = points.max(dim=1, keepdim=True)[0]
    range_val = (maxs - mins).clamp(min=1e-6)
    normalized = (points - mins) / range_val

    # Quantize to integer grid
    max_val = (1 << bits) - 1
    quantized = (normalized * max_val).long().clamp(0, max_val)

    # Compute Morton codes
    codes = morton_encode_3d(
        quantized[..., 0],
        quantized[..., 1],
        quantized[..., 2],
        bits
    )

    # Sort by codes
    sort_indices = codes.argsort(dim=1)
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
    sorted_points = points[batch_indices, sort_indices]

    return sorted_points, sort_indices


def unsort_points(
    sorted_tensor: torch.Tensor,
    sort_indices: torch.Tensor
) -> torch.Tensor:
    """
    Restore original point order after processing.

    Args:
        sorted_tensor: Tensor in sorted order [B, N, C]
        sort_indices: Indices from sort_by_space_filling_curve [B, N]

    Returns:
        Tensor in original order [B, N, C]
    """
    B, N, C = sorted_tensor.shape
    device = sorted_tensor.device

    # Create inverse permutation
    inverse_indices = sort_indices.argsort(dim=1)
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)

    return sorted_tensor[batch_indices, inverse_indices]


# =============================================================================
# PATCHIFICATION
# =============================================================================

def patchify_points(
    points: torch.Tensor,
    patch_size: int = 16
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Group consecutive points into patches.

    Points should already be sorted by space-filling curve to ensure
    each patch contains spatially nearby points.

    Args:
        points: Sorted points [B, N, 3]
        patch_size: Number of points per patch

    Returns:
        patches: Grouped points [B, num_patches, patch_size, 3]
        patch_centers: Centroid of each patch [B, num_patches, 3]
        original_n: Original point count before padding
    """
    B, N, C = points.shape
    original_n = N

    # Pad if necessary (repeat last point)
    if N % patch_size != 0:
        pad_size = patch_size - (N % patch_size)
        padding = points[:, -1:, :].expand(-1, pad_size, -1)
        points = torch.cat([points, padding], dim=1)
        N = points.shape[1]

    num_patches = N // patch_size
    patches = points.reshape(B, num_patches, patch_size, C)
    patch_centers = patches.mean(dim=2)

    return patches, patch_centers, original_n


def unpatchify_points(patches: torch.Tensor) -> torch.Tensor:
    """Flatten patches back to point sequence."""
    B, P, K, C = patches.shape
    return patches.reshape(B, P * K, C)


def compute_local_coordinates(
    patches: torch.Tensor,
    patch_centers: torch.Tensor
) -> torch.Tensor:
    """
    Compute normalized local coordinates within each patch.

    Points are normalized to roughly [-1, 1] range relative to patch center,
    providing translation-invariant local features.

    Args:
        patches: [B, P, K, 3] point positions
        patch_centers: [B, P, 3] patch centroids

    Returns:
        local_coords: [B, P, K, 3] normalized local coordinates
    """
    # Offset from center
    local = patches - patch_centers.unsqueeze(2)

    # Normalize by patch extent (max distance from center)
    extent = local.abs().max(dim=2, keepdim=True)[0].clamp(min=1e-6)
    local_normalized = local / extent

    return local_normalized


# =============================================================================
# CORE COMPONENTS
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * norm * self.weight).to(dtype)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU Feed-Forward Network (Shazeer 2020)."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or int(2 * dim * 4 / 3)
        # Round to multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding with MLP projection."""

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
        """Create sinusoidal timestep embeddings following PixNerd."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) *
            torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.unsqueeze(-1).float() * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# =============================================================================
# POSITION ENCODING
# =============================================================================

class LocalPositionEmbedder(nn.Module):
    """
    3D local position embedding for points within a patch.

    Analogous to PixNerd's DCT encoding for 2D patches, but adapted for 3D:
    - Uses Fourier features on normalized local coordinates
    - Captures relative positions within patch bounds
    """

    def __init__(self, hidden_size: int, max_freqs: int = 8):
        super().__init__()
        self.max_freqs = max_freqs
        # 3D coords + sin/cos for each freq for each dim
        input_dim = 3 + 6 * max_freqs  # xyz + (sin+cos) * 3dims * max_freqs
        self.proj = nn.Linear(input_dim, hidden_size, bias=True)

        # Frequency bands (similar to NeRF but simpler)
        freqs = torch.linspace(0, max_freqs - 1, max_freqs)
        self.register_buffer('freqs', freqs * math.pi)

    def forward(self, local_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_coords: Normalized local coordinates [*, 3] in roughly [-1, 1]

        Returns:
            Embedded features [*, hidden_size]
        """
        shape = local_coords.shape[:-1]
        x = local_coords.reshape(-1, 3)

        # Fourier features
        x_scaled = x.unsqueeze(-1) * self.freqs  # [N, 3, max_freqs]
        fourier = torch.cat([
            x,
            torch.sin(x_scaled).reshape(-1, 3 * self.max_freqs),
            torch.cos(x_scaled).reshape(-1, 3 * self.max_freqs),
        ], dim=-1)

        embedded = self.proj(fourier)
        return embedded.reshape(*shape, -1)


class PatchEmbedder(nn.Module):
    """Embed raw patch coordinates to hidden dimension."""

    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_size, bias=True)
        self.norm = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


# =============================================================================
# TRANSFORMER BLOCKS (Stage 1: Global Context)
# =============================================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


class Attention(nn.Module):
    """
    Multi-head self-attention with QK normalization.

    Uses manual attention computation for gradient compatibility
    (required for SDF training with second-order derivatives).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # QKV projection and reshape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # QK normalization (following PixNerd)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Manual attention (for double backward compatibility)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with AdaLN-Zero conditioning.

    Following PixNerd's FlattenDiTBlock architecture:
    - RMSNorm before attention and MLP
    - AdaLN modulation with 6 parameters (shift, scale, gate for each)
    - SwiGLU feed-forward network
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=False, qk_norm=True)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFeedForward(hidden_size, int(hidden_size * mlp_ratio))

        # AdaLN modulation (following PixNerd: just Linear, no activation before)
        self.adaLN_modulation = nn.Linear(hidden_size, 6 * hidden_size, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch tokens [B, P, hidden_size]
            c: Conditioning [B, hidden_size] or [B, 1, hidden_size]
        """
        # Ensure c has sequence dimension for broadcasting
        if c.dim() == 2:
            c = c.unsqueeze(1)

        # Get modulation parameters
        mod = self.adaLN_modulation(c)  # [B, 1, 6*hidden_size]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Attention block with modulation
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        # MLP block with modulation
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


# =============================================================================
# NERF BLOCKS (Stage 2: Per-Patch Neural Fields)
# =============================================================================

class NerfBlock(nn.Module):
    """
    HyperNetwork block that generates MLP weights from context.

    Following PixNerd's NerfBlock:
    - Generates fc1 and fc2 weights from context vector
    - Weight normalization on input dimension
    - Residual connection

    Key insight: Each patch gets its own dynamically-generated MLP,
    allowing different spatial regions to have different representations.
    """

    def __init__(
        self,
        hidden_size_s: int,  # Context dimension
        hidden_size_x: int,  # Feature dimension
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.hidden_size_x = hidden_size_x
        self.mlp_ratio = mlp_ratio

        # Weight generator: context -> MLP parameters
        self.param_generator = nn.Linear(
            hidden_size_s,
            2 * hidden_size_x * hidden_size_x * mlp_ratio,
            bias=True
        )
        self.norm = RMSNorm(hidden_size_x)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Local features [B*P, K, hidden_size_x] (flattened batch and patches)
            s: Context [B*P, hidden_size_s]

        Returns:
            Transformed features [B*P, K, hidden_size_x]
        """
        batch_size, num_x, hidden_size_x = x.shape

        # Generate MLP weights from context
        mlp_params = self.param_generator(s)  # [B*P, 2*C*C*ratio]
        fc1_param, fc2_param = mlp_params.chunk(2, dim=-1)

        # Reshape to weight matrices
        fc1_param = fc1_param.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc2_param = fc2_param.view(batch_size, hidden_size_x * self.mlp_ratio, hidden_size_x)

        # Weight normalization (following PixNerd)
        fc1_param = F.normalize(fc1_param, dim=-2)
        fc2_param = F.normalize(fc2_param, dim=-2)

        # Apply generated MLP with residual
        res_x = x
        x = self.norm(x)
        x = torch.bmm(x, fc1_param)  # [B*P, K, C*ratio]
        x = F.silu(x)
        x = torch.bmm(x, fc2_param)  # [B*P, K, C]

        return x + res_x


class NerfFinalLayer(nn.Module):
    """Final layer for NerfBlocks (RMSNorm + Linear, zero-initialized)."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        # Zero initialization (following PixNerd)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


# =============================================================================
# BLENDING (For inference at arbitrary resolutions)
# =============================================================================

class PatchBlender(nn.Module):
    """
    Blend outputs from multiple patch neural fields.

    For query points not in the training set, we evaluate through all
    patch NFs and blend based on distance to patch centers.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        patch_outputs: torch.Tensor,
        query_points: torch.Tensor,
        patch_centers: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            patch_outputs: [B, P, M, C] output from each patch's NF
            query_points: [B, M, 3] query positions
            patch_centers: [B, P, 3] patch centers

        Returns:
            Blended output [B, M, C]
        """
        # Compute distances: [B, M, P]
        diff = query_points.unsqueeze(2) - patch_centers.unsqueeze(1)  # [B, M, P, 3]
        distances = torch.norm(diff, dim=-1)  # [B, M, P]

        # Softmax weights (closer = higher weight)
        weights = F.softmax(-distances / self.temperature, dim=-1)  # [B, M, P]

        # Weighted sum
        patch_outputs = patch_outputs.permute(0, 2, 1, 3)  # [B, M, P, C]
        blended = (weights.unsqueeze(-1) * patch_outputs).sum(dim=2)  # [B, M, C]

        return blended


# =============================================================================
# MAIN MODEL
# =============================================================================

class PatchedNeuralFieldDiffusion(nn.Module):
    """
    Neural Field Diffusion for Point Clouds with Patched Architecture.

    Architecture follows PixNerd with adaptations for 3D point clouds:

    Stage 1 (Global Context):
        - Points sorted by Morton curve and grouped into patches
        - Patch tokens processed by DiT blocks with attention
        - Captures global structure and inter-patch relationships

    Stage 2 (Local Processing):
        - Time re-injected: s = silu(t + s)
        - Each patch generates its own MLP via HyperNetwork
        - NerfBlocks process local features within patches
        - Captures fine local details with per-patch specialization

    This two-stage design allows:
        - Different spatial regions to have different representations
        - Global coherence through attention on patch tokens
        - Local detail through per-patch neural fields
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_size: int = 256,
        hidden_size_x: int = 64,
        num_heads: int = 8,
        num_cond_blocks: int = 4,  # DiT blocks (Stage 1)
        num_nerf_blocks: int = 4,  # NerfBlocks (Stage 2)
        nerf_mlp_ratio: int = 4,
        mlp_ratio: float = 4.0,
        max_freqs: int = 8,
        patch_size: int = 16,
        blend_temperature: float = 0.1,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.hidden_size_x = hidden_size_x
        self.num_cond_blocks = num_cond_blocks
        self.num_nerf_blocks = num_nerf_blocks

        # === Embedders ===
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Patch embedding: raw coords -> hidden_size for transformer
        self.s_embedder = PatchEmbedder(in_channels * patch_size, hidden_size)

        # Local position embedding for NerfBlocks
        self.x_embedder = LocalPositionEmbedder(hidden_size_x, max_freqs)

        # === Stage 1: DiT Blocks (Global Context) ===
        self.dit_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(num_cond_blocks)
        ])

        # === Stage 2: NerfBlocks (Per-Patch Neural Fields) ===
        self.nerf_blocks = nn.ModuleList([
            NerfBlock(hidden_size, hidden_size_x, nerf_mlp_ratio)
            for _ in range(num_nerf_blocks)
        ])

        # === Output ===
        self.final_layer = NerfFinalLayer(hidden_size_x, out_channels)

        # === Blending (for inference) ===
        self.blender = PatchBlender(temperature=blend_temperature)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights following PixNerd."""
        # Patch embedder
        nn.init.xavier_uniform_(self.s_embedder.proj.weight)
        nn.init.zeros_(self.s_embedder.proj.bias)

        # Timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Local position embedder
        nn.init.xavier_uniform_(self.x_embedder.proj.weight)
        nn.init.zeros_(self.x_embedder.proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            x: Point cloud [B, N, 3]
            t: Timestep [B]

        Returns:
            Velocity field [B, N, 3]
        """
        B, N_orig, C = x.shape

        # === Preprocessing ===
        # 1. Sort by space-filling curve for locality
        sorted_x, sort_indices = sort_by_space_filling_curve(x)

        # 2. Patchify
        patches, patch_centers, original_n = patchify_points(sorted_x, self.patch_size)
        # patches: [B, P, K, 3], patch_centers: [B, P, 3]
        B, P, K, _ = patches.shape

        # === Stage 1: Global Context via Transformer ===
        # 3. Time embedding
        t_emb = self.t_embedder(t)  # [B, hidden_size]
        c = F.silu(t_emb).unsqueeze(1)  # [B, 1, hidden_size] for broadcasting

        # 4. Embed patches to tokens (flatten patch points)
        patches_flat = patches.reshape(B, P, K * C)  # [B, P, K*3]
        s = self.s_embedder(patches_flat)  # [B, P, hidden_size]

        # 5. DiT blocks (attention over patches)
        for block in self.dit_blocks:
            s = block(s, c)

        # 6. Time re-injection (critical for PixNerd architecture!)
        s = F.silu(t_emb.unsqueeze(1) + s)  # [B, P, hidden_size]

        # === Stage 2: Per-Patch Neural Fields ===
        # 7. Compute local coordinates within each patch
        local_coords = compute_local_coordinates(patches, patch_centers)  # [B, P, K, 3]

        # 8. Flatten batch and patches for NerfBlock processing
        # This matches PixNerd's approach: [B*P, K, hidden_size_x]
        local_coords_flat = local_coords.reshape(B * P, K, C)
        s_flat = s.reshape(B * P, self.hidden_size)

        # 9. Embed local coordinates
        x_features = self.x_embedder(local_coords_flat)  # [B*P, K, hidden_size_x]

        # 10. Apply NerfBlocks
        for block in self.nerf_blocks:
            x_features = block(x_features, s_flat)

        # 11. Final projection
        output = self.final_layer(x_features)  # [B*P, K, out_channels]

        # === Postprocessing ===
        # 12. Reshape back
        output = output.reshape(B, P, K, -1)  # [B, P, K, 3]
        output = unpatchify_points(output)  # [B, P*K, 3]

        # 13. Truncate padding
        output = output[:, :original_n, :]  # [B, N_orig, 3]

        # 14. Unsort to original order
        output = unsort_points(output, sort_indices)  # [B, N_orig, 3]

        return output

    def get_context(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract patch context for querying at arbitrary positions.

        Args:
            x: Reference point cloud [B, N, 3]
            t: Timestep [B]

        Returns:
            s: Patch context [B, P, hidden_size]
            patch_centers: [B, P, 3]
        """
        B, N, C = x.shape

        # Sort and patchify
        sorted_x, _ = sort_by_space_filling_curve(x)
        patches, patch_centers, _ = patchify_points(sorted_x, self.patch_size)
        P, K = patches.shape[1], patches.shape[2]

        # Time embedding
        t_emb = self.t_embedder(t)
        c = F.silu(t_emb).unsqueeze(1)

        # Patch embedding
        patches_flat = patches.reshape(B, P, K * C)
        s = self.s_embedder(patches_flat)

        # DiT blocks
        for block in self.dit_blocks:
            s = block(s, c)

        # Time re-injection
        s = F.silu(t_emb.unsqueeze(1) + s)

        return s, patch_centers

    def query_field(
        self,
        query_points: torch.Tensor,
        t: torch.Tensor,
        patch_context: torch.Tensor,
        patch_centers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Query the neural field at arbitrary points.

        Uses distance-weighted blending of outputs from all patch NFs.

        Args:
            query_points: [B, M, 3] query positions
            t: [B] timestep (unused but kept for API consistency)
            patch_context: [B, P, hidden_size] from get_context
            patch_centers: [B, P, 3] from get_context

        Returns:
            Velocities [B, M, 3]
        """
        B, M, C = query_points.shape
        P = patch_context.shape[1]

        # Compute local coordinates relative to each patch center
        # query_points: [B, M, 3] -> [B, P, M, 3] relative to each center
        query_expanded = query_points.unsqueeze(1).expand(-1, P, -1, -1)  # [B, P, M, 3]
        centers_expanded = patch_centers.unsqueeze(2).expand(-1, -1, M, -1)  # [B, P, M, 3]

        local_coords = query_expanded - centers_expanded  # [B, P, M, 3]

        # Normalize (approximate, using center-based extent)
        extent = (query_expanded - centers_expanded).abs().max(dim=2, keepdim=True)[0].clamp(min=1e-6)
        local_coords = local_coords / extent.mean(dim=-1, keepdim=True).clamp(min=0.1)

        # Flatten for NerfBlock processing
        local_coords_flat = local_coords.reshape(B * P, M, C)
        s_flat = patch_context.reshape(B * P, self.hidden_size)

        # Embed and process
        x_features = self.x_embedder(local_coords_flat)  # [B*P, M, hidden_size_x]

        for block in self.nerf_blocks:
            x_features = block(x_features, s_flat)

        output = self.final_layer(x_features)  # [B*P, M, out_channels]

        # Reshape and blend
        output = output.reshape(B, P, M, -1)  # [B, P, M, 3]
        blended = self.blender(output, query_points, patch_centers)  # [B, M, 3]

        return blended


# =============================================================================
# TESTING
# =============================================================================

def test_patched_model():
    """Comprehensive test of the patched neural field model."""
    print("=" * 60)
    print("Testing PatchedNeuralFieldDiffusion")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = PatchedNeuralFieldDiffusion(
        hidden_size=256,
        hidden_size_x=64,
        num_heads=8,
        num_cond_blocks=4,
        num_nerf_blocks=4,
        patch_size=16,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Test forward pass
    print("\n--- Forward Pass Test ---")
    B, N = 4, 512
    x = torch.randn(B, N, 3, device=device)
    t = torch.rand(B, device=device)

    model.train()
    v = model(x, t)
    print(f"Input: {x.shape}, Output: {v.shape}")
    assert v.shape == x.shape, f"Shape mismatch: {v.shape} != {x.shape}"
    print("Forward pass: PASSED")

    # Test gradient flow
    print("\n--- Gradient Flow Test ---")
    loss = v.mean()
    loss.backward()

    total_grad_norm = 0
    zero_grad_layers = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norm = p.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            if grad_norm < 1e-10:
                zero_grad_layers.append(name)
    total_grad_norm = total_grad_norm ** 0.5

    print(f"Total gradient norm: {total_grad_norm:.4f}")
    if zero_grad_layers:
        print(f"Warning: Zero gradients in: {zero_grad_layers[:5]}...")
    else:
        print("All parameters receiving gradients: PASSED")

    # Test context extraction
    print("\n--- Context Extraction Test ---")
    model.eval()
    with torch.no_grad():
        ctx, centers = model.get_context(x, t)
    print(f"Context: {ctx.shape}, Centers: {centers.shape}")
    assert ctx.shape[0] == B and ctx.shape[2] == model.hidden_size
    print("Context extraction: PASSED")

    # Test field query
    print("\n--- Field Query Test ---")
    M = 1024
    query = torch.randn(B, M, 3, device=device)
    with torch.no_grad():
        v_query = model.query_field(query, t, ctx, centers)
    print(f"Query {M} points: {v_query.shape}")
    assert v_query.shape == (B, M, 3)
    print("Field query: PASSED")

    # Test sort/unsort
    print("\n--- Sort/Unsort Test ---")
    sorted_x, indices = sort_by_space_filling_curve(x)
    restored = unsort_points(sorted_x, indices)
    error = (x - restored).abs().max().item()
    print(f"Sort/unsort reconstruction error: {error:.2e}")
    assert error < 1e-6, f"Sort/unsort error too large: {error}"
    print("Sort/unsort: PASSED")

    # Test with different input sizes
    print("\n--- Variable Size Test ---")
    for N_test in [256, 513, 1024]:  # Including non-divisible size
        x_test = torch.randn(2, N_test, 3, device=device)
        t_test = torch.rand(2, device=device)
        with torch.no_grad():
            v_test = model(x_test, t_test)
        assert v_test.shape == x_test.shape, f"Failed for N={N_test}"
        print(f"  N={N_test}: PASSED")

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_patched_model()
