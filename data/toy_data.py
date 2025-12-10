"""
Toy Point Cloud Dataset Generation

Generates point clouds on manifolds of different dimensions:
- 1D: Curves (circle, helix, trefoil knot, figure-8)
- 2D: Surfaces (sphere, torus, plane, cylinder, Möbius strip)
- 3D: Volumes (cube, ball, ellipsoid)

Each generator supports arbitrary point counts, demonstrating
the resolution-agnostic nature of manifold representations.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
from enum import Enum


class ManifoldDim(Enum):
    """Manifold dimension classification."""
    CURVE_1D = 1
    SURFACE_2D = 2
    VOLUME_3D = 3


@dataclass
class PointCloud:
    """Point cloud with metadata."""
    points: np.ndarray  # (N, 3) array of 3D points
    normals: Optional[np.ndarray] = None  # (N, 3) surface normals if applicable
    name: str = ""
    manifold_dim: ManifoldDim = ManifoldDim.SURFACE_2D

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    @property
    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """Bounding box (min, max)."""
        return self.points.min(axis=0), self.points.max(axis=0)

    @property
    def center(self) -> np.ndarray:
        """Centroid of point cloud."""
        return self.points.mean(axis=0)

    @property
    def scale(self) -> float:
        """Maximum extent from center."""
        centered = self.points - self.center
        return np.max(np.linalg.norm(centered, axis=1))

    def normalize(self, target_scale: float = 1.0) -> 'PointCloud':
        """Normalize to unit scale centered at origin."""
        centered = self.points - self.center
        scaled = centered / self.scale * target_scale
        normals = self.normals  # Normals are direction vectors, don't scale
        return PointCloud(scaled, normals, self.name, self.manifold_dim)

    def add_noise(self, std: float = 0.01) -> 'PointCloud':
        """Add Gaussian noise to points."""
        noisy = self.points + np.random.randn(*self.points.shape) * std
        return PointCloud(noisy, self.normals, self.name, self.manifold_dim)

    def subsample(self, n_points: int) -> 'PointCloud':
        """Randomly subsample to n_points."""
        if n_points >= self.num_points:
            return self
        indices = np.random.choice(self.num_points, n_points, replace=False)
        normals = self.normals[indices] if self.normals is not None else None
        return PointCloud(self.points[indices], normals, self.name, self.manifold_dim)

    def random_rotate(self) -> 'PointCloud':
        """Apply random 3D rotation."""
        # Random rotation using QR decomposition of random matrix
        random_matrix = np.random.randn(3, 3)
        q, _ = np.linalg.qr(random_matrix)
        # Ensure proper rotation (det = 1, not -1)
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        rotated = self.points @ q.T
        normals = self.normals @ q.T if self.normals is not None else None
        return PointCloud(rotated, normals, self.name, self.manifold_dim)

    def random_scale(self, scale_range: Tuple[float, float] = (0.5, 1.5)) -> 'PointCloud':
        """Apply random uniform scaling."""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        scaled = self.points * scale
        return PointCloud(scaled, self.normals, self.name, self.manifold_dim)

    def random_anisotropic_scale(self, scale_range: Tuple[float, float] = (0.5, 1.5)) -> 'PointCloud':
        """Apply random per-axis scaling (creates shape variation)."""
        scales = np.random.uniform(scale_range[0], scale_range[1], size=3)
        scaled = self.points * scales
        # Normals need inverse transpose of scale matrix
        if self.normals is not None:
            normals = self.normals / scales
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        else:
            normals = None
        return PointCloud(scaled, normals, self.name, self.manifold_dim)

    def random_transform(self,
                        rotate: bool = True,
                        scale_range: Optional[Tuple[float, float]] = (0.7, 1.3),
                        anisotropic: bool = True) -> 'PointCloud':
        """Apply random transformations for data augmentation."""
        pc = self
        if rotate:
            pc = pc.random_rotate()
        if scale_range is not None:
            if anisotropic:
                pc = pc.random_anisotropic_scale(scale_range)
            else:
                pc = pc.random_scale(scale_range)
        return pc


# =============================================================================
# 1D MANIFOLDS (Curves)
# =============================================================================

def generate_circle(n_points: int = 1024, radius: float = 1.0,
                    z_offset: float = 0.0) -> PointCloud:
    """
    Generate points on a circle in the xy-plane.

    Args:
        n_points: Number of points to sample
        radius: Circle radius
        z_offset: Z-coordinate offset

    Returns:
        PointCloud on 1D circle manifold
    """
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.full_like(x, z_offset)

    points = np.stack([x, y, z], axis=1)

    # Normals point radially outward (in xy-plane)
    normals = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)

    return PointCloud(points, normals, "circle", ManifoldDim.CURVE_1D)


def generate_helix(n_points: int = 1024, radius: float = 1.0,
                   pitch: float = 0.5, turns: float = 3.0) -> PointCloud:
    """
    Generate points on a helix.

    Args:
        n_points: Number of points to sample
        radius: Helix radius
        pitch: Vertical distance per turn
        turns: Number of complete turns

    Returns:
        PointCloud on 1D helix manifold
    """
    t = np.linspace(0, 2 * np.pi * turns, n_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = pitch * t / (2 * np.pi)

    points = np.stack([x, y, z], axis=1)

    # Tangent vectors (for reference, not true normals for 1D)
    dx = -radius * np.sin(t)
    dy = radius * np.cos(t)
    dz = np.full_like(t, pitch / (2 * np.pi))
    tangent = np.stack([dx, dy, dz], axis=1)
    tangent = tangent / np.linalg.norm(tangent, axis=1, keepdims=True)

    return PointCloud(points, tangent, "helix", ManifoldDim.CURVE_1D)


def generate_trefoil_knot(n_points: int = 1024, scale: float = 1.0) -> PointCloud:
    """
    Generate points on a trefoil knot.

    Parametric equations:
        x = sin(t) + 2*sin(2t)
        y = cos(t) - 2*cos(2t)
        z = -sin(3t)

    Args:
        n_points: Number of points to sample
        scale: Overall scale factor

    Returns:
        PointCloud on 1D trefoil knot manifold
    """
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = (np.sin(t) + 2 * np.sin(2 * t)) * scale / 3
    y = (np.cos(t) - 2 * np.cos(2 * t)) * scale / 3
    z = -np.sin(3 * t) * scale / 3

    points = np.stack([x, y, z], axis=1)

    return PointCloud(points, None, "trefoil_knot", ManifoldDim.CURVE_1D)


def generate_figure8(n_points: int = 1024, scale: float = 1.0) -> PointCloud:
    """
    Generate points on a figure-8 (lemniscate) curve in 3D.

    Args:
        n_points: Number of points to sample
        scale: Overall scale factor

    Returns:
        PointCloud on 1D figure-8 manifold
    """
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = scale * np.sin(t)
    y = scale * np.sin(t) * np.cos(t)
    z = scale * np.sin(2 * t) * 0.3  # Slight 3D variation

    points = np.stack([x, y, z], axis=1)

    return PointCloud(points, None, "figure8", ManifoldDim.CURVE_1D)


# =============================================================================
# 2D MANIFOLDS (Surfaces)
# =============================================================================

def generate_sphere(n_points: int = 1024, radius: float = 1.0,
                    method: str = "fibonacci") -> PointCloud:
    """
    Generate points on a sphere surface.

    Args:
        n_points: Number of points to sample
        radius: Sphere radius
        method: Sampling method ("fibonacci" for uniform, "random" for random)

    Returns:
        PointCloud on 2D sphere manifold
    """
    if method == "fibonacci":
        # Fibonacci spiral for uniform distribution
        indices = np.arange(n_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_points)
        theta = np.pi * (1 + 5 ** 0.5) * indices

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
    else:
        # Random sampling
        u = np.random.rand(n_points)
        v = np.random.rand(n_points)
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

    points = np.stack([x, y, z], axis=1)

    # Normals point radially outward
    normals = points / radius

    return PointCloud(points, normals, "sphere", ManifoldDim.SURFACE_2D)


def generate_torus(n_points: int = 1024, R: float = 1.0, r: float = 0.3) -> PointCloud:
    """
    Generate points on a torus surface.

    Args:
        n_points: Number of points to sample
        R: Major radius (center of tube to center of torus)
        r: Minor radius (tube radius)

    Returns:
        PointCloud on 2D torus manifold
    """
    # Approximate uniform sampling
    n_major = int(np.sqrt(n_points * R / r))
    n_minor = n_points // n_major

    u = np.linspace(0, 2 * np.pi, n_major, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n_minor, endpoint=False)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()

    # Torus parametric equations
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    points = np.stack([x, y, z], axis=1)

    # Surface normals
    nx = np.cos(v) * np.cos(u)
    ny = np.cos(v) * np.sin(u)
    nz = np.sin(v)
    normals = np.stack([nx, ny, nz], axis=1)

    # Subsample to exact n_points
    if len(points) > n_points:
        indices = np.random.choice(len(points), n_points, replace=False)
        points = points[indices]
        normals = normals[indices]

    return PointCloud(points, normals, "torus", ManifoldDim.SURFACE_2D)


def generate_plane(n_points: int = 1024, size: float = 2.0,
                   normal_dir: Tuple[float, float, float] = (0, 0, 1)) -> PointCloud:
    """
    Generate points on a flat plane.

    Args:
        n_points: Number of points to sample
        size: Side length of square plane
        normal_dir: Normal direction of plane

    Returns:
        PointCloud on 2D plane manifold
    """
    # Sample uniformly on unit square
    side = int(np.sqrt(n_points))
    u = np.linspace(-size/2, size/2, side)
    v = np.linspace(-size/2, size/2, side)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()

    # Default: xy-plane
    x = u
    y = v
    z = np.zeros_like(u)

    points = np.stack([x, y, z], axis=1)

    # Rotate to desired normal direction
    normal_dir = np.array(normal_dir)
    normal_dir = normal_dir / np.linalg.norm(normal_dir)

    if not np.allclose(normal_dir, [0, 0, 1]):
        # Rotation from (0,0,1) to normal_dir
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, normal_dir)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        angle = np.arccos(np.clip(np.dot(z_axis, normal_dir), -1, 1))

        # Rodrigues rotation
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        points = points @ R.T

    normals = np.tile(normal_dir, (len(points), 1))

    # Subsample to exact n_points
    if len(points) > n_points:
        indices = np.random.choice(len(points), n_points, replace=False)
        points = points[indices]
        normals = normals[indices]

    return PointCloud(points, normals, "plane", ManifoldDim.SURFACE_2D)


def generate_cylinder(n_points: int = 1024, radius: float = 0.5,
                      height: float = 2.0, caps: bool = False) -> PointCloud:
    """
    Generate points on a cylinder surface.

    Args:
        n_points: Number of points to sample
        radius: Cylinder radius
        height: Cylinder height
        caps: Whether to include top and bottom caps

    Returns:
        PointCloud on 2D cylinder manifold
    """
    if caps:
        # Distribute points between side and caps based on area
        side_area = 2 * np.pi * radius * height
        cap_area = 2 * np.pi * radius ** 2
        total_area = side_area + cap_area
        n_side = int(n_points * side_area / total_area)
        n_caps = n_points - n_side
    else:
        n_side = n_points
        n_caps = 0

    # Side surface
    n_theta = int(np.sqrt(n_side * 2 * np.pi * radius / height))
    n_z = n_side // n_theta

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z_vals = np.linspace(-height/2, height/2, n_z)
    theta, z = np.meshgrid(theta, z_vals)
    theta = theta.flatten()
    z = z.flatten()

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    side_points = np.stack([x, y, z], axis=1)
    side_normals = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)

    if caps and n_caps > 0:
        # Top and bottom caps (disks)
        n_per_cap = n_caps // 2
        cap_points = []
        cap_normals = []

        for z_pos, nz in [(height/2, 1), (-height/2, -1)]:
            r = np.sqrt(np.random.rand(n_per_cap)) * radius
            theta = np.random.rand(n_per_cap) * 2 * np.pi
            cap_x = r * np.cos(theta)
            cap_y = r * np.sin(theta)
            cap_z = np.full(n_per_cap, z_pos)
            cap_points.append(np.stack([cap_x, cap_y, cap_z], axis=1))
            cap_normals.append(np.tile([0, 0, nz], (n_per_cap, 1)))

        points = np.vstack([side_points] + cap_points)
        normals = np.vstack([side_normals] + cap_normals)
    else:
        points = side_points
        normals = side_normals

    # Subsample to exact n_points
    if len(points) > n_points:
        indices = np.random.choice(len(points), n_points, replace=False)
        points = points[indices]
        normals = normals[indices]

    return PointCloud(points, normals, "cylinder", ManifoldDim.SURFACE_2D)


def generate_mobius_strip(n_points: int = 1024, R: float = 1.0,
                          width: float = 0.4) -> PointCloud:
    """
    Generate points on a Möbius strip.

    Args:
        n_points: Number of points to sample
        R: Major radius
        width: Strip half-width

    Returns:
        PointCloud on 2D Möbius manifold (non-orientable!)
    """
    n_u = int(np.sqrt(n_points * 4))
    n_v = n_points // n_u

    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    v = np.linspace(-width, width, n_v)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()

    # Möbius parametric equations
    x = (R + v * np.cos(u / 2)) * np.cos(u)
    y = (R + v * np.cos(u / 2)) * np.sin(u)
    z = v * np.sin(u / 2)

    points = np.stack([x, y, z], axis=1)

    # Subsample to exact n_points
    if len(points) > n_points:
        indices = np.random.choice(len(points), n_points, replace=False)
        points = points[indices]

    return PointCloud(points, None, "mobius_strip", ManifoldDim.SURFACE_2D)


# =============================================================================
# 3D MANIFOLDS (Volumes)
# =============================================================================

def generate_cube_volume(n_points: int = 1024, size: float = 2.0) -> PointCloud:
    """
    Generate points uniformly inside a cube.

    Args:
        n_points: Number of points to sample
        size: Side length of cube

    Returns:
        PointCloud in 3D cube volume
    """
    points = (np.random.rand(n_points, 3) - 0.5) * size

    return PointCloud(points, None, "cube_volume", ManifoldDim.VOLUME_3D)


def generate_ball_volume(n_points: int = 1024, radius: float = 1.0) -> PointCloud:
    """
    Generate points uniformly inside a ball (solid sphere).

    Args:
        n_points: Number of points to sample
        radius: Ball radius

    Returns:
        PointCloud in 3D ball volume
    """
    # Rejection sampling for uniform distribution in ball
    points = []
    while len(points) < n_points:
        candidates = (np.random.rand(n_points * 2, 3) - 0.5) * 2 * radius
        distances = np.linalg.norm(candidates, axis=1)
        inside = candidates[distances <= radius]
        points.append(inside)

    points = np.vstack(points)[:n_points]

    return PointCloud(points, None, "ball_volume", ManifoldDim.VOLUME_3D)


def generate_ellipsoid_volume(n_points: int = 1024,
                               a: float = 1.0, b: float = 0.7, c: float = 0.5) -> PointCloud:
    """
    Generate points uniformly inside an ellipsoid.

    Args:
        n_points: Number of points to sample
        a, b, c: Semi-axes lengths

    Returns:
        PointCloud in 3D ellipsoid volume
    """
    # Generate uniform in unit ball, then scale
    ball = generate_ball_volume(n_points, radius=1.0)
    points = ball.points * np.array([a, b, c])

    return PointCloud(points, None, "ellipsoid_volume", ManifoldDim.VOLUME_3D)


def generate_shell(n_points: int = 1024, r_inner: float = 0.5,
                   r_outer: float = 1.0) -> PointCloud:
    """
    Generate points uniformly in a spherical shell.

    Args:
        n_points: Number of points to sample
        r_inner: Inner radius
        r_outer: Outer radius

    Returns:
        PointCloud in 3D shell volume
    """
    # Sample radius with correct volume weighting (r^2)
    u = np.random.rand(n_points)
    r = (r_inner**3 + u * (r_outer**3 - r_inner**3)) ** (1/3)

    # Sample direction uniformly on sphere
    theta = np.random.rand(n_points) * 2 * np.pi
    phi = np.arccos(2 * np.random.rand(n_points) - 1)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    points = np.stack([x, y, z], axis=1)

    return PointCloud(points, None, "shell", ManifoldDim.VOLUME_3D)


# =============================================================================
# DATASET COLLECTION
# =============================================================================

def get_all_generators() -> Dict[str, callable]:
    """Return dictionary of all point cloud generators."""
    return {
        # 1D Curves
        "circle": generate_circle,
        "helix": generate_helix,
        "trefoil_knot": generate_trefoil_knot,
        "figure8": generate_figure8,
        # 2D Surfaces
        "sphere": generate_sphere,
        "torus": generate_torus,
        "plane": generate_plane,
        "cylinder": generate_cylinder,
        "mobius_strip": generate_mobius_strip,
        # 3D Volumes
        "cube_volume": generate_cube_volume,
        "ball_volume": generate_ball_volume,
        "ellipsoid_volume": generate_ellipsoid_volume,
        "shell": generate_shell,
    }


def generate_dataset(shapes: List[str] = None, n_points: int = 1024,
                     n_samples_per_shape: int = 1,
                     normalize: bool = True,
                     noise_std: float = 0.0) -> List[PointCloud]:
    """
    Generate a dataset of point clouds.

    Args:
        shapes: List of shape names (None = all shapes)
        n_points: Points per cloud
        n_samples_per_shape: Number of samples per shape type
        normalize: Whether to normalize to unit scale
        noise_std: Gaussian noise standard deviation

    Returns:
        List of PointCloud objects
    """
    generators = get_all_generators()

    if shapes is None:
        shapes = list(generators.keys())

    dataset = []
    for shape_name in shapes:
        if shape_name not in generators:
            raise ValueError(f"Unknown shape: {shape_name}")

        for _ in range(n_samples_per_shape):
            pc = generators[shape_name](n_points=n_points)

            if normalize:
                pc = pc.normalize()

            if noise_std > 0:
                pc = pc.add_noise(noise_std)

            dataset.append(pc)

    return dataset


def get_shapes_by_dimension(dim: int) -> List[str]:
    """Get shape names filtered by manifold dimension."""
    generators = get_all_generators()
    result = []
    for name, gen in generators.items():
        pc = gen(n_points=10)  # Small sample to check dimension
        if pc.manifold_dim.value == dim:
            result.append(name)
    return result


# =============================================================================
# STATISTICS
# =============================================================================

def compute_statistics(pc: PointCloud) -> Dict:
    """Compute statistics for a point cloud."""
    points = pc.points

    # Basic stats
    stats = {
        "name": pc.name,
        "manifold_dim": pc.manifold_dim.value,
        "num_points": pc.num_points,
        "center": pc.center.tolist(),
        "scale": float(pc.scale),
    }

    # Bounding box
    bbox_min, bbox_max = pc.bbox
    stats["bbox_min"] = bbox_min.tolist()
    stats["bbox_max"] = bbox_max.tolist()
    stats["bbox_extent"] = (bbox_max - bbox_min).tolist()

    # Point distribution
    dists_from_center = np.linalg.norm(points - pc.center, axis=1)
    stats["mean_dist_from_center"] = float(np.mean(dists_from_center))
    stats["std_dist_from_center"] = float(np.std(dists_from_center))
    stats["max_dist_from_center"] = float(np.max(dists_from_center))

    # Per-axis statistics
    for i, axis in enumerate(['x', 'y', 'z']):
        stats[f"{axis}_mean"] = float(np.mean(points[:, i]))
        stats[f"{axis}_std"] = float(np.std(points[:, i]))
        stats[f"{axis}_min"] = float(np.min(points[:, i]))
        stats[f"{axis}_max"] = float(np.max(points[:, i]))

    # Nearest neighbor distances (sample for efficiency)
    if pc.num_points > 1:
        sample_size = min(500, pc.num_points)
        sample_indices = np.random.choice(pc.num_points, sample_size, replace=False)
        sample_points = points[sample_indices]

        nn_dists = []
        for i, p in enumerate(sample_points):
            dists = np.linalg.norm(sample_points - p, axis=1)
            dists[i] = np.inf  # Exclude self
            nn_dists.append(np.min(dists))

        stats["mean_nn_distance"] = float(np.mean(nn_dists))
        stats["std_nn_distance"] = float(np.std(nn_dists))

    return stats


if __name__ == "__main__":
    # Quick test
    print("Testing toy data generation...")

    for name, gen in get_all_generators().items():
        pc = gen(n_points=1000)
        pc = pc.normalize()
        stats = compute_statistics(pc)
        print(f"{name}: {pc.num_points} points, dim={pc.manifold_dim.value}, "
              f"scale={stats['scale']:.3f}")

    print("\nAll generators working!")
