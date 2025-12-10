"""
Neural Field Module for Point Cloud Diffusion

The core innovation: learns a continuous vector field v_θ: ℝ³ × [0,T] → ℝ³
that can be queried at ANY spatial location, not just training points.

Architecture:
1. Global Encoder: points → shape context s
2. HyperNetwork: s → MLP weights
3. Neural Field: (x, t, weights) → velocity v(x, t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class FourierPositionEncoder(nn.Module):
    """
    3D Fourier position encoding for continuous spatial coordinates.

    Maps (x, y, z) ∈ ℝ³ to high-dimensional feature space using
    sinusoidal functions at multiple frequencies.

    This enables the neural field to learn high-frequency details
    while maintaining smoothness.
    """

    def __init__(self, d_input: int = 3, n_frequencies: int = 10,
                 include_input: bool = True):
        """
        Args:
            d_input: Input dimension (3 for 3D points)
            n_frequencies: Number of frequency bands
            include_input: Whether to include raw input in output
        """
        super().__init__()
        self.d_input = d_input
        self.n_frequencies = n_frequencies
        self.include_input = include_input

        # Frequency bands: 2^0, 2^1, ..., 2^(n_frequencies-1)
        freqs = 2.0 ** torch.arange(n_frequencies)
        self.register_buffer('freqs', freqs)

        # Output dimension
        self.d_output = d_input * n_frequencies * 2  # sin + cos
        if include_input:
            self.d_output += d_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input coordinates [..., d_input]

        Returns:
            Encoded features [..., d_output]
        """
        # x: [..., 3]
        # freqs: [n_frequencies]

        # Compute x * freq for all frequencies
        # [..., 3, 1] * [n_frequencies] -> [..., 3, n_frequencies]
        x_freq = x.unsqueeze(-1) * self.freqs * math.pi

        # Flatten and apply sin/cos
        x_freq = x_freq.reshape(*x.shape[:-1], -1)  # [..., 3*n_frequencies]

        encoded = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)

        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        return encoded


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal embedding for diffusion timestep.

    Maps scalar t ∈ [0, 1] to high-dimensional feature space.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps [...] or [..., 1]

        Returns:
            Time embeddings [..., dim]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[-1] != 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb


class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder for extracting global shape features.

    Uses shared MLPs followed by max pooling to get a permutation-invariant
    global feature vector that summarizes the entire point cloud shape.
    """

    def __init__(self, d_input: int = 3, d_hidden: int = 128, d_output: int = 256):
        """
        Args:
            d_input: Input point dimension (3 for xyz)
            d_hidden: Hidden layer dimension
            d_output: Output global feature dimension
        """
        super().__init__()

        # Shared MLP layers (applied per-point)
        self.mlp1 = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2),
            nn.LayerNorm(d_hidden * 2),
            nn.GELU(),
            nn.Linear(d_hidden * 2, d_output),
            nn.LayerNorm(d_output),
        )

        self.d_output = d_output

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: Point cloud [B, N, 3]

        Returns:
            Global shape features [B, d_output]
        """
        # Per-point features
        x = self.mlp1(points)  # [B, N, d_hidden]
        x = self.mlp2(x)       # [B, N, d_output]

        # Global max pooling (permutation invariant)
        global_feat = x.max(dim=1)[0]  # [B, d_output]

        return global_feat


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for global shape features.

    Uses self-attention to capture point relationships, then pools
    to get a global shape descriptor.

    More expressive than PointNet but O(N²) complexity.
    """

    def __init__(self, d_input: int = 3, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, d_output: int = 256):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_output),
            nn.LayerNorm(d_output),
        )

        self.d_output = d_output

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: Point cloud [B, N, 3]

        Returns:
            Global shape features [B, d_output]
        """
        x = self.input_proj(points)  # [B, N, d_model]
        x = self.transformer(x)       # [B, N, d_model]

        # Global pooling (mean + max)
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1)[0]
        global_feat = self.output_proj(x_mean + x_max)  # [B, d_output]

        return global_feat


class HyperNetwork(nn.Module):
    """
    HyperNetwork that generates MLP weights from shape context.

    This is the key to neural field diffusion:
    - Takes global shape features s
    - Outputs weights for the neural field MLP
    - The generated MLP defines a continuous function over ℝ³

    Inspired by PixNerd's NerfBlock architecture.
    """

    def __init__(self, d_context: int = 256, d_time: int = 64,
                 field_hidden: int = 128, field_layers: int = 3):
        """
        Args:
            d_context: Dimension of shape context
            d_time: Dimension of time embedding
            field_hidden: Hidden dimension of generated MLP
            field_layers: Number of layers in generated MLP
        """
        super().__init__()

        self.field_hidden = field_hidden
        self.field_layers = field_layers

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(d_time)

        # Combine context and time
        self.context_proj = nn.Sequential(
            nn.Linear(d_context + d_time, d_context),
            nn.LayerNorm(d_context),
            nn.GELU(),
            nn.Linear(d_context, d_context),
            nn.LayerNorm(d_context),
            nn.GELU(),
        )

        # Calculate total parameters needed for field MLP
        # Layer sizes: [d_pos_enc, hidden, hidden, ..., 3]
        # We'll set d_pos_enc at runtime based on position encoder
        self.d_pos_enc = None  # Set during first forward

        # Weight generators for each layer
        # Will be initialized lazily
        self.weight_generators = None
        self.bias_generators = None
        self.d_context = d_context

    def _init_generators(self, d_pos_enc: int, device):
        """Initialize weight generators based on position encoding dim."""
        self.d_pos_enc = d_pos_enc

        # Layer dimensions
        dims = [d_pos_enc] + [self.field_hidden] * (self.field_layers - 1) + [3]

        self.weight_generators = nn.ModuleList()
        self.bias_generators = nn.ModuleList()

        for i in range(len(dims) - 1):
            d_in, d_out = dims[i], dims[i + 1]
            # Each generator outputs flattened weight matrix
            self.weight_generators.append(
                nn.Linear(self.d_context, d_in * d_out)
            )
            self.bias_generators.append(
                nn.Linear(self.d_context, d_out)
            )

        self.weight_generators = self.weight_generators.to(device)
        self.bias_generators = self.bias_generators.to(device)

        # Store dimensions for later
        self.layer_dims = dims

    def forward(self, context: torch.Tensor, t: torch.Tensor,
                d_pos_enc: int) -> Tuple[list, list]:
        """
        Generate MLP weights from shape context and time.

        Args:
            context: Shape context [B, d_context]
            t: Diffusion timestep [B] or scalar
            d_pos_enc: Dimension of position encoding

        Returns:
            weights: List of weight matrices for each layer
            biases: List of bias vectors for each layer
        """
        B = context.shape[0]
        device = context.device

        # Initialize generators if needed
        if self.weight_generators is None or self.d_pos_enc != d_pos_enc:
            self._init_generators(d_pos_enc, device)

        # Time embedding
        if t.dim() == 0:
            t = t.expand(B)
        t_emb = self.time_embed(t)  # [B, d_time]

        # Combine context and time
        combined = torch.cat([context, t_emb], dim=-1)  # [B, d_context + d_time]
        combined = self.context_proj(combined)  # [B, d_context]

        # Generate weights and biases
        weights = []
        biases = []

        for i, (wg, bg) in enumerate(zip(self.weight_generators, self.bias_generators)):
            d_in, d_out = self.layer_dims[i], self.layer_dims[i + 1]

            # Generate and reshape weights
            w = wg(combined)  # [B, d_in * d_out]
            w = w.view(B, d_out, d_in)  # [B, d_out, d_in]

            # Normalize weights for stability (from PixNerd)
            w = F.normalize(w, dim=-1)

            # Generate biases
            b = bg(combined)  # [B, d_out]

            weights.append(w)
            biases.append(b)

        return weights, biases


class NeuralFieldMLP(nn.Module):
    """
    The neural field MLP that maps position encodings to velocities.

    Uses dynamically generated weights from HyperNetwork.
    This allows the same architecture to represent different shapes.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, weights: list, biases: list) -> torch.Tensor:
        """
        Apply the neural field MLP with given weights.

        Args:
            x: Position encodings [B, N, d_pos_enc]
            weights: List of weight matrices [B, d_out, d_in]
            biases: List of bias vectors [B, d_out]

        Returns:
            Velocities [B, N, 3]
        """
        # x: [B, N, d_in]

        for i, (w, b) in enumerate(zip(weights, biases)):
            # Batched matrix multiply: [B, N, d_in] @ [B, d_in, d_out] -> [B, N, d_out]
            x = torch.bmm(x, w.transpose(-1, -2)) + b.unsqueeze(1)

            # Apply activation (except last layer)
            if i < len(weights) - 1:
                x = F.gelu(x)

        return x  # [B, N, 3]


class NeuralFieldDiffusion(nn.Module):
    """
    Complete Neural Field Diffusion Model.

    Combines:
    1. Point cloud encoder (PointNet or Transformer)
    2. Position encoder (Fourier features)
    3. HyperNetwork (generates field weights)
    4. Neural field MLP (maps positions to velocities)

    The model learns v_θ: ℝ³ × [0,T] → ℝ³, a continuous vector field
    that can be evaluated at ANY spatial location.
    """

    def __init__(self,
                 encoder_type: str = 'pointnet',
                 d_hidden: int = 128,
                 d_context: int = 256,
                 n_frequencies: int = 10,
                 field_hidden: int = 128,
                 field_layers: int = 3,
                 transformer_layers: int = 2,
                 transformer_heads: int = 4):
        """
        Args:
            encoder_type: 'pointnet' or 'transformer'
            d_hidden: Hidden dimension for encoder
            d_context: Dimension of shape context
            n_frequencies: Number of Fourier frequency bands
            field_hidden: Hidden dimension of neural field MLP
            field_layers: Number of layers in neural field MLP
            transformer_layers: Number of transformer layers (if using transformer)
            transformer_heads: Number of attention heads (if using transformer)
        """
        super().__init__()

        # Position encoder
        self.pos_encoder = FourierPositionEncoder(
            d_input=3,
            n_frequencies=n_frequencies,
            include_input=True
        )

        # Point cloud encoder
        if encoder_type == 'pointnet':
            self.encoder = PointNetEncoder(
                d_input=3,
                d_hidden=d_hidden,
                d_output=d_context
            )
        elif encoder_type == 'transformer':
            self.encoder = TransformerEncoder(
                d_input=3,
                d_model=d_hidden,
                n_heads=transformer_heads,
                n_layers=transformer_layers,
                d_output=d_context
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # HyperNetwork
        self.hyper_net = HyperNetwork(
            d_context=d_context,
            d_time=64,
            field_hidden=field_hidden,
            field_layers=field_layers
        )

        # Neural field MLP
        self.field_mlp = NeuralFieldMLP()

        # Store config
        self.encoder_type = encoder_type
        self.d_context = d_context

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict velocity field at given points and time.

        Args:
            x_t: Noised point positions [B, N, 3]
            t: Diffusion timestep [B] or scalar in [0, 1]
            condition: Optional conditioning (e.g., class label)

        Returns:
            Predicted velocities [B, N, 3]
        """
        B, N, _ = x_t.shape

        # 1. Encode shape context from noised points
        context = self.encoder(x_t)  # [B, d_context]

        # 2. Generate neural field weights
        d_pos_enc = self.pos_encoder.d_output
        weights, biases = self.hyper_net(context, t, d_pos_enc)

        # 3. Encode query positions
        pos_enc = self.pos_encoder(x_t)  # [B, N, d_pos_enc]

        # 4. Apply neural field to get velocities
        velocities = self.field_mlp(pos_enc, weights, biases)  # [B, N, 3]

        return velocities

    def query_field(self, query_points: torch.Tensor, t: torch.Tensor,
                    context: torch.Tensor) -> torch.Tensor:
        """
        Query the neural field at arbitrary points given pre-computed context.

        This is the key capability: once we have shape context, we can
        query the velocity field at ANY spatial location.

        Args:
            query_points: Query positions [B, M, 3] (M can differ from training N)
            t: Diffusion timestep
            context: Pre-computed shape context [B, d_context]

        Returns:
            Velocities at query points [B, M, 3]
        """
        # Generate field weights from context
        d_pos_enc = self.pos_encoder.d_output
        weights, biases = self.hyper_net(context, t, d_pos_enc)

        # Encode query positions
        pos_enc = self.pos_encoder(query_points)

        # Apply field
        velocities = self.field_mlp(pos_enc, weights, biases)

        return velocities

    def get_context(self, points: torch.Tensor) -> torch.Tensor:
        """
        Extract shape context from a point cloud.

        Args:
            points: Point cloud [B, N, 3]

        Returns:
            Shape context [B, d_context]
        """
        return self.encoder(points)


def test_model():
    """Test the neural field model."""
    print("Testing NeuralFieldDiffusion...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = NeuralFieldDiffusion(
        encoder_type='pointnet',
        d_hidden=64,
        d_context=128,
        n_frequencies=6,
        field_hidden=64,
        field_layers=3
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Test forward pass
    B, N = 4, 256
    x_t = torch.randn(B, N, 3, device=device)
    t = torch.rand(B, device=device)

    v = model(x_t, t)
    print(f"Input shape: {x_t.shape}")
    print(f"Output shape: {v.shape}")

    # Test query at different resolution
    M = 1024
    context = model.get_context(x_t)
    query_points = torch.randn(B, M, 3, device=device)
    v_query = model.query_field(query_points, t, context)
    print(f"Query at {M} points: {v_query.shape}")

    print("All tests passed!")


if __name__ == "__main__":
    test_model()
