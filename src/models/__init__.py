"""Neural field models for point cloud generation (PixNerd-style architecture)."""

from .neural_field import (
    # Main model
    NeuralFieldDiffusion,
    # Core components from PixNerd
    RMSNorm,
    SwiGLUFeedForward,
    TimestepEmbedder,
    PointEmbedder,
    RoPEAttention3D,
    DiTBlock3D,
    NerfEmbedder3D,
    NerfBlock,
    NerfFinalLayer,
    # Utility functions
    modulate,
    precompute_freqs_cis_3d,
    apply_rotary_emb_3d,
)

__all__ = [
    # Main model
    'NeuralFieldDiffusion',
    # Core components
    'RMSNorm',
    'SwiGLUFeedForward',
    'TimestepEmbedder',
    'PointEmbedder',
    'RoPEAttention3D',
    'DiTBlock3D',
    'NerfEmbedder3D',
    'NerfBlock',
    'NerfFinalLayer',
    # Utilities
    'modulate',
    'precompute_freqs_cis_3d',
    'apply_rotary_emb_3d',
]
