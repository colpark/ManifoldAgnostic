"""
Flow Matching for Neural Field Diffusion

Implements the flow matching training objective:
    L(θ) = E_{t,x₀,ε}[||v_θ(x_t, t) - v_target||²]

where:
    x_t = α(t)x₀ + σ(t)ε        (linear interpolation)
    v_target = α'(t)x₀ + σ'(t)ε (target velocity)

We use the simple linear schedule:
    α(t) = 1 - t
    σ(t) = t

So:
    x_t = (1-t)x₀ + t*ε
    v_target = -x₀ + ε = ε - x₀

This means the velocity field should point from data to noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class FlowMatchingSchedule:
    """
    Defines the interpolation schedule for flow matching.

    Linear schedule (default):
        α(t) = 1 - t  (data weight decreases)
        σ(t) = t      (noise weight increases)

    At t=0: x_t = x₀ (pure data)
    At t=1: x_t = ε  (pure noise)
    """

    def __init__(self, schedule_type: str = 'linear'):
        self.schedule_type = schedule_type

    def get_coefficients(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                          torch.Tensor, torch.Tensor]:
        """
        Get interpolation coefficients at time t.

        Args:
            t: Timesteps [B] or [B, 1] in [0, 1]

        Returns:
            alpha: Data coefficient [B, 1, 1]
            sigma: Noise coefficient [B, 1, 1]
            dalpha: d(alpha)/dt [B, 1, 1]
            dsigma: d(sigma)/dt [B, 1, 1]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        elif t.dim() == 2:
            t = t.unsqueeze(-1)  # [B, 1, 1]

        if self.schedule_type == 'linear':
            alpha = 1 - t
            sigma = t
            dalpha = -torch.ones_like(t)
            dsigma = torch.ones_like(t)

        elif self.schedule_type == 'cosine':
            # Cosine schedule for smoother interpolation
            alpha = torch.cos(t * math.pi / 2)
            sigma = torch.sin(t * math.pi / 2)
            dalpha = -math.pi / 2 * torch.sin(t * math.pi / 2)
            dsigma = math.pi / 2 * torch.cos(t * math.pi / 2)

        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")

        return alpha, sigma, dalpha, dsigma

    def interpolate(self, x0: torch.Tensor, noise: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        """
        Interpolate between data and noise at time t.

        Args:
            x0: Clean data [B, N, 3]
            noise: Noise samples [B, N, 3]
            t: Timesteps [B]

        Returns:
            x_t: Noised data [B, N, 3]
        """
        alpha, sigma, _, _ = self.get_coefficients(t)
        return alpha * x0 + sigma * noise

    def get_velocity_target(self, x0: torch.Tensor, noise: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        """
        Compute target velocity for flow matching.

        The target is: v_target = dα/dt * x₀ + dσ/dt * ε

        For linear schedule: v_target = -x₀ + ε = ε - x₀

        Args:
            x0: Clean data [B, N, 3]
            noise: Noise samples [B, N, 3]
            t: Timesteps [B]

        Returns:
            v_target: Target velocity [B, N, 3]
        """
        _, _, dalpha, dsigma = self.get_coefficients(t)
        return dalpha * x0 + dsigma * noise


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching training loss.

    L(θ) = E_{t,x₀,ε}[w(t) ||v_θ(x_t, t) - v_target||²]

    where w(t) is an optional time-dependent weighting.
    """

    def __init__(self, schedule_type: str = 'linear',
                 weighting: str = 'uniform',
                 reduction: str = 'mean'):
        """
        Args:
            schedule_type: Interpolation schedule ('linear' or 'cosine')
            weighting: Loss weighting ('uniform', 'snr', 'min_snr')
            reduction: Loss reduction ('mean', 'sum', 'none')
        """
        super().__init__()
        self.schedule = FlowMatchingSchedule(schedule_type)
        self.weighting = weighting
        self.reduction = reduction

    def get_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Compute time-dependent loss weight."""
        if self.weighting == 'uniform':
            return torch.ones_like(t)

        elif self.weighting == 'snr':
            # Signal-to-noise ratio weighting
            alpha, sigma, _, _ = self.schedule.get_coefficients(t)
            snr = (alpha / (sigma + 1e-8)) ** 2
            return snr.squeeze()

        elif self.weighting == 'min_snr':
            # Clamped SNR weighting (from Min-SNR paper)
            alpha, sigma, _, _ = self.schedule.get_coefficients(t)
            snr = (alpha / (sigma + 1e-8)) ** 2
            return torch.clamp(snr, max=5.0).squeeze()

        else:
            return torch.ones_like(t)

    def forward(self, model: nn.Module, x0: torch.Tensor,
                t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss.

        Args:
            model: Neural field model
            x0: Clean point cloud [B, N, 3]
            t: Optional timesteps [B] (sampled if None)

        Returns:
            Dictionary with 'loss' and other metrics
        """
        B, N, _ = x0.shape
        device = x0.device

        # Sample timesteps uniformly in [0, 1]
        if t is None:
            t = torch.rand(B, device=device)

        # Sample noise
        noise = torch.randn_like(x0)

        # Interpolate to get noised samples
        x_t = self.schedule.interpolate(x0, noise, t)

        # Get target velocity
        v_target = self.schedule.get_velocity_target(x0, noise, t)

        # Predict velocity with model
        v_pred = model(x_t, t)

        # Compute loss
        loss_per_point = F.mse_loss(v_pred, v_target, reduction='none')  # [B, N, 3]
        loss_per_sample = loss_per_point.mean(dim=[1, 2])  # [B]

        # Apply weighting
        weight = self.get_weight(t)  # [B]
        weighted_loss = weight * loss_per_sample

        # Reduce
        if self.reduction == 'mean':
            loss = weighted_loss.mean()
        elif self.reduction == 'sum':
            loss = weighted_loss.sum()
        else:
            loss = weighted_loss

        return {
            'loss': loss,
            'mse': loss_per_sample.mean(),
            't_mean': t.mean(),
        }


class FlowMatchingSampler:
    """
    ODE-based sampler for flow matching.

    Integrates the learned velocity field from t=1 (noise) to t=0 (data):
        dx/dt = v_θ(x, t)
        x(1) = noise
        x(0) = generated sample
    """

    def __init__(self, model: nn.Module, schedule_type: str = 'linear'):
        self.model = model
        self.schedule = FlowMatchingSchedule(schedule_type)

    @torch.no_grad()
    def sample_euler(self, noise: torch.Tensor, n_steps: int = 50,
                     return_trajectory: bool = False) -> torch.Tensor:
        """
        Sample using Euler method.

        Args:
            noise: Starting noise [B, N, 3]
            n_steps: Number of integration steps
            return_trajectory: Whether to return full trajectory

        Returns:
            Generated samples [B, N, 3]
            or trajectory [n_steps+1, B, N, 3] if return_trajectory=True
        """
        device = noise.device
        B = noise.shape[0]

        # Time steps from t=1 to t=0
        timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        dt = -1.0 / n_steps  # Negative because we go from 1 to 0

        x = noise.clone()
        trajectory = [x.clone()] if return_trajectory else None

        for i in range(n_steps):
            t = timesteps[i].expand(B)

            # Predict velocity
            v = self.model(x, t)

            # Euler step: x = x + v * dt
            x = x + v * dt

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        return x

    @torch.no_grad()
    def sample_heun(self, noise: torch.Tensor, n_steps: int = 50,
                    return_trajectory: bool = False) -> torch.Tensor:
        """
        Sample using Heun's method (2nd order).

        More accurate than Euler but requires 2 model evaluations per step.
        """
        device = noise.device
        B = noise.shape[0]

        timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        dt = -1.0 / n_steps

        x = noise.clone()
        trajectory = [x.clone()] if return_trajectory else None

        for i in range(n_steps):
            t = timesteps[i].expand(B)
            t_next = timesteps[i + 1].expand(B)

            # First evaluation
            v1 = self.model(x, t)

            # Euler prediction
            x_pred = x + v1 * dt

            # Second evaluation at predicted point
            v2 = self.model(x_pred, t_next)

            # Heun correction: average velocities
            x = x + 0.5 * (v1 + v2) * dt

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        return x

    @torch.no_grad()
    def sample_at_resolution(self, context: torch.Tensor, n_points: int,
                             n_steps: int = 50) -> torch.Tensor:
        """
        Sample at arbitrary resolution using pre-computed context.

        This demonstrates resolution independence:
        - Context is computed from one point cloud
        - Sampling can be done at any number of points

        Args:
            context: Shape context from encoder [B, d_context]
            n_points: Number of points to generate
            n_steps: Number of integration steps

        Returns:
            Generated samples [B, n_points, 3]
        """
        device = context.device
        B = context.shape[0]

        # Start from noise
        noise = torch.randn(B, n_points, 3, device=device)

        # Time steps
        timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        dt = -1.0 / n_steps

        x = noise.clone()

        for i in range(n_steps):
            t = timesteps[i].expand(B)

            # Query field at current positions
            v = self.model.query_field(x, t, context)

            # Euler step
            x = x + v * dt

        return x


def test_flow_matching():
    """Test flow matching components."""
    print("Testing FlowMatchingLoss...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Import model
    from src.models.neural_field import NeuralFieldDiffusion

    # Create model
    model = NeuralFieldDiffusion(
        encoder_type='pointnet',
        d_hidden=64,
        d_context=128,
        n_frequencies=6,
        field_hidden=64,
        field_layers=3
    ).to(device)

    # Create loss
    loss_fn = FlowMatchingLoss(schedule_type='linear')

    # Test data
    B, N = 4, 256
    x0 = torch.randn(B, N, 3, device=device)

    # Compute loss
    output = loss_fn(model, x0)
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"MSE: {output['mse'].item():.4f}")

    # Test sampler
    print("\nTesting FlowMatchingSampler...")
    sampler = FlowMatchingSampler(model)

    noise = torch.randn(2, 256, 3, device=device)
    samples = sampler.sample_euler(noise, n_steps=20)
    print(f"Sample shape: {samples.shape}")

    # Test resolution independence
    print("\nTesting resolution independence...")
    context = model.get_context(x0[:2])
    for n_pts in [128, 256, 512, 1024]:
        samples = sampler.sample_at_resolution(context, n_points=n_pts, n_steps=20)
        print(f"  Generated {n_pts} points: {samples.shape}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_flow_matching()
