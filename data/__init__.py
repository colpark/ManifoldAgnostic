"""Toy point cloud data generation for ManifoldAgnostic."""

from .toy_data import (
    PointCloud,
    ManifoldDim,
    # 1D Generators
    generate_circle,
    generate_helix,
    generate_trefoil_knot,
    generate_figure8,
    # 2D Generators
    generate_sphere,
    generate_torus,
    generate_plane,
    generate_cylinder,
    generate_mobius_strip,
    # 3D Generators
    generate_cube_volume,
    generate_ball_volume,
    generate_ellipsoid_volume,
    generate_shell,
    # Multi-Object Scenes
    generate_multi_sphere,
    generate_multi_sphere_cube,
    generate_multi_sphere_ring,
    generate_multi_sphere_random,
    # Utilities
    get_all_generators,
    get_shapes_by_dimension,
    generate_dataset,
    compute_statistics,
)

__all__ = [
    'PointCloud',
    'ManifoldDim',
    'generate_circle',
    'generate_helix',
    'generate_trefoil_knot',
    'generate_figure8',
    'generate_sphere',
    'generate_torus',
    'generate_plane',
    'generate_cylinder',
    'generate_mobius_strip',
    'generate_cube_volume',
    'generate_ball_volume',
    'generate_ellipsoid_volume',
    'generate_shell',
    'generate_multi_sphere',
    'generate_multi_sphere_cube',
    'generate_multi_sphere_ring',
    'generate_multi_sphere_random',
    'get_all_generators',
    'get_shapes_by_dimension',
    'generate_dataset',
    'compute_statistics',
]
