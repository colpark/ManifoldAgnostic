"""Neural field models for point cloud generation (PixNerd-style architecture)."""

from .neural_field import (
    # Main model (velocity-based)
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

from .sdf_field import (
    # SDF-based model (scalar field, gradient-derived velocity)
    SDFNeuralField,
    SDFFlowMatchingLoss,
)

__all__ = [
    # Main models
    'NeuralFieldDiffusion',  # Velocity-based
    'SDFNeuralField',        # SDF-based (smoother training)
    # SDF-specific loss
    'SDFFlowMatchingLoss',
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
