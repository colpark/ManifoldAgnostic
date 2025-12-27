"""
Patched Neural Field with Space-Filling Curve Serialization

Key innovations:
1. Serialize points using space-filling curve (Morton/Hilbert) for locality preservation
2. Group points into patches (e.g., 16 points per patch)
3. Each patch gets its own context → its own neural field MLP
4. Query points blend multiple patch NFs weighted by distance to patch centers

This solves the global pooling collapse problem by preserving local structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# =============================================================================
# SPACE-FILLING CURVES
# =============================================================================

def morton_encode_3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                     bits: int = 10) -> torch.Tensor:
    """
    Encode 3D coordinates into Morton (Z-order) code.
    Morton code interleaves bits of x, y, z coordinates.

    Args:
        x, y, z: Quantized coordinates [B, N] in range [0, 2^bits)
        bits: Number of bits per coordinate

    Returns:
        Morton codes [B, N]
    """
    def spread_bits(v):
        # Spread bits for 3D Morton code
        v = v & 0x3FF  # 10 bits max
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)


def hilbert_encode_3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                      order: int = 5) -> torch.Tensor:
    """
    Encode 3D coordinates into Hilbert curve index.
    Better locality preservation than Morton code but more complex.

    For simplicity, we use Morton code in practice (similar locality for our use case).
    """
    # Hilbert curve is complex to implement efficiently on GPU
    # Fall back to Morton which has similar locality properties
    return morton_encode_3d(x, y, z, bits=order)


def sort_by_space_filling_curve(points: torch.Tensor,
                                 curve_type: str = 'morton',
                                 bits: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sort points by space-filling curve to preserve locality.

    Args:
        points: Point coordinates [B, N, 3]
        curve_type: 'morton' or 'hilbert'
        bits: Quantization bits (determines resolution)

    Returns:
        sorted_points: Reordered points [B, N, 3]
        sort_indices: Indices to reconstruct original order [B, N]
    """
    B, N, _ = points.shape
    device = points.device

    # Normalize to [0, 1] per batch
    mins = points.min(dim=1, keepdim=True)[0]
    maxs = points.max(dim=1, keepdim=True)[0]
    range_val = (maxs - mins).clamp(min=1e-6)
    normalized = (points - mins) / range_val

    # Quantize to integer grid
    max_val = (1 << bits) - 1
    quantized = (normalized * max_val).long().clamp(0, max_val)

    # Compute space-filling curve codes
    x, y, z = quantized[..., 0], quantized[..., 1], quantized[..., 2]

    if curve_type == 'morton':
        codes = morton_encode_3d(x, y, z, bits)
    else:
        codes = hilbert_encode_3d(x, y, z, order=bits)

    # Sort by codes
    sort_indices = codes.argsort(dim=1)

    # Gather sorted points
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
    sorted_points = points[batch_indices, sort_indices]

    return sorted_points, sort_indices


def unsort_points(sorted_tensor: torch.Tensor,
                  sort_indices: torch.Tensor) -> torch.Tensor:
    """
    Restore original point order after processing.

    Args:
        sorted_tensor: Tensor in sorted order [B, N, C]
        sort_indices: Indices from sort_by_space_filling_curve [B, N]

    Returns:
        Original order tensor [B, N, C]
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

def patchify_points(points: torch.Tensor, patch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Group consecutive points into patches.
    Points should already be sorted by space-filling curve.
    If N is not divisible by patch_size, pads with repeated last points.

    Args:
        points: Sorted points [B, N, 3]
        patch_size: Points per patch

    Returns:
        patches: Grouped points [B, num_patches, patch_size, 3]
        patch_centers: Center of each patch [B, num_patches, 3]
        original_n: Original number of points before padding
    """
    B, N, C = points.shape
    original_n = N

    # Pad if necessary
    if N % patch_size != 0:
        pad_size = patch_size - (N % patch_size)
        # Pad by repeating the last point
        padding = points[:, -1:, :].expand(-1, pad_size, -1)
        points = torch.cat([points, padding], dim=1)
        N = points.shape[1]

    num_patches = N // patch_size
    patches = points.reshape(B, num_patches, patch_size, C)
    patch_centers = patches.mean(dim=2)  # [B, num_patches, 3]

    return patches, patch_centers, original_n


def unpatchify_points(patches: torch.Tensor) -> torch.Tensor:
    """
    Flatten patches back to points.

    Args:
        patches: [B, num_patches, patch_size, C]

    Returns:
        points: [B, N, C]
    """
    B, num_patches, patch_size, C = patches.shape
    return patches.reshape(B, num_patches * patch_size, C)


# =============================================================================
# COMPONENTS (reuse from neural_field.py)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        hidden_dim = (hidden_dim + 63) // 64 * 64

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class FourierEmbedder(nn.Module):
    """3D Fourier position encoding."""
    def __init__(self, hidden_size: int, max_freqs: int = 8):
        super().__init__()
        self.max_freqs = max_freqs
        input_dim = 3 + 6 * max_freqs
        self.embedder = nn.Linear(input_dim, hidden_size)

        freqs = 2.0 ** torch.linspace(0, max_freqs - 1, max_freqs)
        self.register_buffer('freqs', freqs * math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        x_flat = x.reshape(-1, 3)

        x_scaled = x_flat.unsqueeze(-1) * self.freqs
        fourier = torch.cat([
            x_flat,
            torch.sin(x_scaled).reshape(-1, 3 * self.max_freqs),
            torch.cos(x_scaled).reshape(-1, 3 * self.max_freqs),
        ], dim=-1)

        embedded = self.embedder(fourier)
        return embedded.reshape(*shape, -1)


# =============================================================================
# PATCH-LEVEL TRANSFORMER
# =============================================================================

class PatchAttention(nn.Module):
    """
    Multi-head attention for patch tokens.
    Each patch is a token representing a local region of the point cloud.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Manual attention for gradient compatibility
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.proj(x)


class PatchTransformerBlock(nn.Module):
    """
    Transformer block for patch tokens with AdaLN conditioning.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = PatchAttention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFeedForward(hidden_size, int(hidden_size * mlp_ratio))

        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c: [B, hidden_size] condition
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Attention with modulation
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm)

        # MLP with modulation
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


# =============================================================================
# PER-PATCH NEURAL FIELD
# =============================================================================

class PatchNerfBlock(nn.Module):
    """
    HyperNetwork that generates a separate MLP for each patch.

    Unlike the original NerfBlock which uses global context for all points,
    this generates DIFFERENT MLPs for each patch based on local context.
    """
    def __init__(self, hidden_size_s: int, hidden_size_x: int, mlp_ratio: int = 4):
        super().__init__()
        self.hidden_size_x = hidden_size_x
        self.mlp_ratio = mlp_ratio

        # Weight generator per patch
        self.param_generator = nn.Linear(
            hidden_size_s,
            2 * hidden_size_x * hidden_size_x * mlp_ratio,
            bias=True
        )
        self.norm = RMSNorm(hidden_size_x)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Point features [B, num_patches, patch_size, hidden_size_x]
            s: Per-patch context [B, num_patches, hidden_size_s]

        Returns:
            Transformed features [B, num_patches, patch_size, hidden_size_x]
        """
        B, P, K, C = x.shape  # P = num_patches, K = patch_size

        # Generate MLP params for each patch
        mlp_params = self.param_generator(s)  # [B, P, 2*C*C*ratio]
        fc1_param, fc2_param = mlp_params.chunk(2, dim=-1)

        # Reshape: [B, P, C, C*ratio] and [B, P, C*ratio, C]
        fc1_param = fc1_param.view(B, P, self.hidden_size_x, self.hidden_size_x * self.mlp_ratio)
        fc2_param = fc2_param.view(B, P, self.hidden_size_x * self.mlp_ratio, self.hidden_size_x)

        # Normalize weights
        fc1_param = F.normalize(fc1_param, dim=-2)
        fc2_param = F.normalize(fc2_param, dim=-2)

        # Apply MLP per patch
        res_x = x
        x = self.norm(x)

        # Batched matrix multiply: [B, P, K, C] @ [B, P, C, C*ratio] -> [B, P, K, C*ratio]
        x = torch.einsum('bpkc,bpcd->bpkd', x, fc1_param)
        x = F.silu(x)
        x = torch.einsum('bpkd,bpdc->bpkc', x, fc2_param)

        return x + res_x


# =============================================================================
# DISTANCE-WEIGHTED BLENDING
# =============================================================================

class PatchBlender(nn.Module):
    """
    Blend outputs from multiple patch neural fields based on distance.

    For a query point, computes weighted average of all patch NF outputs,
    where weight is inversely proportional to distance to patch center.
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def compute_weights(self, query_points: torch.Tensor,
                        patch_centers: torch.Tensor) -> torch.Tensor:
        """
        Compute blending weights based on distance to patch centers.

        Args:
            query_points: [B, M, 3] query positions
            patch_centers: [B, P, 3] patch center positions

        Returns:
            weights: [B, M, P] normalized weights
        """
        # Compute distances: [B, M, P]
        # query_points: [B, M, 1, 3], patch_centers: [B, 1, P, 3]
        diff = query_points.unsqueeze(2) - patch_centers.unsqueeze(1)
        distances = torch.norm(diff, dim=-1)  # [B, M, P]

        # Convert to weights (closer = higher weight)
        # Use negative distance with softmax for smooth weighting
        weights = F.softmax(-distances / self.temperature, dim=-1)

        return weights

    def forward(self, patch_outputs: torch.Tensor,
                query_points: torch.Tensor,
                patch_centers: torch.Tensor) -> torch.Tensor:
        """
        Blend patch outputs for query points.

        Args:
            patch_outputs: [B, P, M, C] output from each patch's NF for query points
            query_points: [B, M, 3] query positions
            patch_centers: [B, P, 3] patch centers

        Returns:
            blended: [B, M, C] weighted average output
        """
        weights = self.compute_weights(query_points, patch_centers)  # [B, M, P]

        # Weighted sum: [B, M, P, 1] * [B, P, M, C] -> sum over P
        # Rearrange patch_outputs to [B, M, P, C]
        patch_outputs = patch_outputs.permute(0, 2, 1, 3)  # [B, M, P, C]

        blended = (weights.unsqueeze(-1) * patch_outputs).sum(dim=2)  # [B, M, C]

        return blended


# =============================================================================
# MAIN MODEL
# =============================================================================

class PatchedNeuralFieldDiffusion(nn.Module):
    """
    Neural Field Diffusion with Patched Architecture.

    Key differences from original:
    1. Points are sorted by space-filling curve and grouped into patches
    2. Transformer operates on patch tokens (not individual points)
    3. Each patch generates its own neural field MLP
    4. Query points blend outputs from nearby patches

    This preserves local structure and allows different parts to have
    different representations.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_size: int = 256,
        hidden_size_x: int = 64,
        num_heads: int = 8,
        num_blocks: int = 8,
        num_nerf_blocks: int = 4,
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
        self.num_blocks = num_blocks
        self.num_nerf_blocks = num_nerf_blocks

        # Embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.point_embedder = FourierEmbedder(hidden_size_x, max_freqs)

        # Patch embedding: aggregate points within patch to single token
        self.patch_aggregator = nn.Sequential(
            nn.Linear(hidden_size_x, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Transformer blocks for patch tokens
        self.transformer_blocks = nn.ModuleList([
            PatchTransformerBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(num_blocks)
        ])

        # NerfBlocks for per-patch neural fields
        self.nerf_embedder = FourierEmbedder(hidden_size_x, max_freqs)
        self.nerf_blocks = nn.ModuleList([
            PatchNerfBlock(hidden_size, hidden_size_x, nerf_mlp_ratio)
            for _ in range(num_nerf_blocks)
        ])

        # Final layer
        self.final_norm = RMSNorm(hidden_size_x)
        self.final_layer = nn.Linear(hidden_size_x, out_channels)
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

        # Blender for combining patch outputs
        self.blender = PatchBlender(temperature=blend_temperature)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            x: Point cloud [B, N, 3]
            t: Timestep [B]

        Returns:
            Velocity field [B, N, 3]
        """
        B, N_orig, _ = x.shape

        # 1. Sort by space-filling curve
        sorted_x, sort_indices = sort_by_space_filling_curve(x)

        # 2. Patchify (handles padding if N not divisible by patch_size)
        patches, patch_centers, original_n = patchify_points(sorted_x, self.patch_size)
        # patches: [B, P, K, 3], patch_centers: [B, P, 3]
        P = patches.shape[1]

        # 3. Embed points within patches
        point_features = self.point_embedder(patches)  # [B, P, K, hidden_size_x]

        # 4. Aggregate to patch tokens
        patch_tokens = point_features.mean(dim=2)  # [B, P, hidden_size_x]
        patch_tokens = self.patch_aggregator(patch_tokens)  # [B, P, hidden_size]

        # 5. Time embedding
        t_emb = self.t_embedder(t)  # [B, hidden_size]
        c = F.silu(t_emb)

        # 6. Transformer on patch tokens
        for block in self.transformer_blocks:
            patch_tokens = block(patch_tokens, c)

        # 7. Per-patch neural field processing
        # Each patch context generates its own NF
        patch_context = patch_tokens  # [B, P, hidden_size]

        # Embed query points using RELATIVE position (offset from patch center)
        # This gives more local variation within each patch
        relative_pos = patches - patch_centers.unsqueeze(2)  # [B, P, K, 3]
        query_features = self.nerf_embedder(relative_pos)  # [B, P, K, hidden_size_x]

        # Apply NerfBlocks
        for nerf_block in self.nerf_blocks:
            query_features = nerf_block(query_features, patch_context)

        # 8. Final projection
        output = self.final_norm(query_features)
        output = self.final_layer(output)  # [B, P, K, 3]

        # 9. Unpatchify
        output = unpatchify_points(output)  # [B, N_padded, 3]

        # 10. Truncate to original size (remove padding)
        output = output[:, :original_n, :]  # [B, N_orig, 3]

        # 11. Unsort to original order
        output = unsort_points(output, sort_indices)  # [B, N_orig, 3]

        return output

    def query_field(self, query_points: torch.Tensor, t: torch.Tensor,
                    patch_context: torch.Tensor, patch_centers: torch.Tensor) -> torch.Tensor:
        """
        Query the neural field at arbitrary points.
        Uses distance-weighted blending of patch neural fields.

        Args:
            query_points: [B, M, 3] query positions
            t: [B] timestep
            patch_context: [B, P, hidden_size] pre-computed patch features
            patch_centers: [B, P, 3] patch center positions

        Returns:
            Velocities [B, M, 3]
        """
        B, M, _ = query_points.shape
        P = patch_context.shape[1]

        # Compute relative positions to EACH patch center
        # query_points: [B, M, 3], patch_centers: [B, P, 3]
        # Result: [B, P, M, 3] - relative position of each query to each patch
        relative_pos = query_points.unsqueeze(1) - patch_centers.unsqueeze(2)  # [B, P, M, 3]

        # Embed using relative positions (different for each patch!)
        query_features = self.nerf_embedder(relative_pos)  # [B, P, M, hidden_size_x]

        # Apply each patch's NerfBlocks
        for nerf_block in self.nerf_blocks:
            # Need to reshape for PatchNerfBlock
            # It expects [B, P, K, C] where K is patch_size, but we have M query points
            # Treat M as the "patch_size" dimension
            query_features = nerf_block(query_features, patch_context)

        # Final projection: [B, P, M, 3]
        output = self.final_norm(query_features)
        output = self.final_layer(output)

        # Blend outputs based on distance to patch centers
        blended = self.blender(output, query_points, patch_centers)  # [B, M, 3]

        return blended

    def get_context(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract patch context for later querying.

        Args:
            x: Reference point cloud [B, N, 3]
            t: Timestep [B]

        Returns:
            patch_context: [B, P, hidden_size]
            patch_centers: [B, P, 3]
        """
        B, N, _ = x.shape

        # Sort and patchify
        sorted_x, _ = sort_by_space_filling_curve(x)
        patches, patch_centers, _ = patchify_points(sorted_x, self.patch_size)

        # Embed and aggregate
        point_features = self.point_embedder(patches)
        patch_tokens = point_features.mean(dim=2)
        patch_tokens = self.patch_aggregator(patch_tokens)

        # Time embedding and transformer
        t_emb = self.t_embedder(t)
        c = F.silu(t_emb)

        for block in self.transformer_blocks:
            patch_tokens = block(patch_tokens, c)

        return patch_tokens, patch_centers


# =============================================================================
# TESTING
# =============================================================================

def test_patched_model():
    """Test the patched neural field model."""
    print("Testing PatchedNeuralFieldDiffusion...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = PatchedNeuralFieldDiffusion(
        hidden_size=256,
        hidden_size_x=64,
        num_heads=8,
        num_blocks=6,
        num_nerf_blocks=3,
        patch_size=16,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Test forward
    B, N = 4, 512  # N must be divisible by patch_size
    x = torch.randn(B, N, 3, device=device)
    t = torch.rand(B, device=device)

    v = model(x, t)
    print(f"Forward - Input: {x.shape}, Output: {v.shape}")

    # Test context extraction
    ctx, centers = model.get_context(x, t)
    print(f"Context: {ctx.shape}, Centers: {centers.shape}")

    # Test field query
    M = 1024
    query = torch.randn(B, M, 3, device=device)
    v_query = model.query_field(query, t, ctx, centers)
    print(f"Query {M} points: {v_query.shape}")

    # Test space-filling curve
    sorted_x, indices = sort_by_space_filling_curve(x)
    restored = unsort_points(sorted_x, indices)
    error = (x - restored).abs().max()
    print(f"Sort/unsort error: {error:.6f}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_patched_model()
