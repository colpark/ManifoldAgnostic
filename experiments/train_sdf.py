"""
Training Script for SDF-Based Neural Field Diffusion

Instead of directly predicting velocity v(x,t), this model predicts a scalar
distance field f(x,t) and derives velocity as v = -∇_x f(x,t).

Benefits over direct velocity prediction:
1. Smoother training (no directional discontinuities)
2. Implicit surface representation at t=0
3. Naturally continuous gradient field
4. Same generative capacity with better stability

Usage:
    python experiments/train_sdf.py --shapes multi_sphere_ring --n_points 512 --epochs 500

Key differences from train_toy.py:
- Uses SDFNeuralField instead of NeuralFieldDiffusion
- Uses SDFFlowMatchingLoss (optionally with eikonal regularization)
- Model outputs scalar SDF, velocity computed via autograd
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import time
from typing import Optional, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data.toy_data import get_all_generators, PointCloud
from src.models.sdf_field import SDFNeuralField, SDFFlowMatchingLoss
from src.diffusion.flow_matching import FlowMatchingSampler


class PointCloudDataset(Dataset):
    """Dataset of point clouds for training."""

    def __init__(self, shapes: List[str] = ['sphere'], n_samples: int = 1000,
                 n_points: int = 256, noise_std: float = 0.001,
                 random_transform: bool = True,
                 scale_range: tuple = (0.7, 1.3)):
        self.shapes = shapes if isinstance(shapes, list) else [shapes]
        self.n_samples = n_samples
        self.n_points = n_points
        self.noise_std = noise_std
        self.random_transform = random_transform
        self.scale_range = scale_range

        all_generators = get_all_generators()
        self.generators = []
        for shape in self.shapes:
            if shape not in all_generators:
                raise ValueError(f"Unknown shape: {shape}. Available: {list(all_generators.keys())}")
            self.generators.append(all_generators[shape])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        generator = self.generators[idx % len(self.generators)]
        pc = generator(n_points=self.n_points)

        if self.random_transform:
            pc = pc.random_transform(
                rotate=True,
                scale_range=self.scale_range,
                anisotropic=True
            )

        pc = pc.normalize()
        if self.noise_std > 0:
            pc = pc.add_noise(self.noise_std)

        return torch.tensor(pc.points, dtype=torch.float32)


class SDFTrainer:
    """Training loop for SDF-based neural field diffusion."""

    def __init__(self, model: SDFNeuralField, device: torch.device,
                 lr: float = 1e-4, weight_decay: float = 0.0,
                 eikonal_weight: float = 0.0):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # SDF-specific loss with optional eikonal regularization
        self.loss_fn = SDFFlowMatchingLoss(
            schedule_type='linear',
            eikonal_weight=eikonal_weight
        )

        # Sampler works with SDF model because forward() returns velocity
        self.sampler = FlowMatchingSampler(model)

        self.train_losses = []
        self.velocity_losses = []
        self.eikonal_losses = []
        self.epoch_times = []

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_vel_loss = 0.0
        total_eik_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x0 = batch.to(self.device)

            self.optimizer.zero_grad()
            output = self.loss_fn(self.model, x0)
            loss = output['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_vel_loss += output['velocity_loss']
            if 'eikonal_loss' in output:
                total_eik_loss += output['eikonal_loss']
            n_batches += 1

        return {
            'loss': total_loss / n_batches,
            'velocity_loss': total_vel_loss / n_batches,
            'eikonal_loss': total_eik_loss / n_batches if total_eik_loss > 0 else 0
        }

    def train(self, dataloader: DataLoader, n_epochs: int,
              log_interval: int = 10, sample_interval: int = 50,
              save_dir: Optional[str] = None):
        """Full training loop."""

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        pbar = tqdm(range(n_epochs), desc="Training SDF")

        for epoch in pbar:
            start_time = time.time()
            metrics = self.train_epoch(dataloader)
            epoch_time = time.time() - start_time

            self.train_losses.append(metrics['loss'])
            self.velocity_losses.append(metrics['velocity_loss'])
            self.eikonal_losses.append(metrics['eikonal_loss'])
            self.epoch_times.append(epoch_time)

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'vel': f"{metrics['velocity_loss']:.4f}"
            })

            if (epoch + 1) % log_interval == 0:
                msg = f"Epoch {epoch+1}/{n_epochs} | Loss: {metrics['loss']:.4f} | Vel: {metrics['velocity_loss']:.4f}"
                if metrics['eikonal_loss'] > 0:
                    msg += f" | Eik: {metrics['eikonal_loss']:.4f}"
                tqdm.write(msg)

            if save_dir and (epoch + 1) % sample_interval == 0:
                self.visualize_samples(save_dir, epoch + 1, dataloader.dataset)
                self.visualize_sdf(save_dir, epoch + 1, dataloader.dataset)

        print(f"\nTraining complete!")
        print(f"Final loss: {self.train_losses[-1]:.4f}")
        print(f"Average epoch time: {np.mean(self.epoch_times):.2f}s")

        if save_dir:
            self.save_training_curves(save_dir)

    @torch.no_grad()
    def generate_samples(self, n_samples: int = 4, n_points: int = 256,
                         n_steps: int = 50) -> torch.Tensor:
        """Generate point cloud samples."""
        self.model.eval()
        noise = torch.randn(n_samples, n_points, 3, device=self.device)
        samples = self.sampler.sample_euler(noise, n_steps=n_steps)
        return samples

    @torch.no_grad()
    def visualize_samples(self, save_dir: str, epoch: int, dataset):
        """Generate and save sample visualizations."""
        self.model.eval()

        samples = self.generate_samples(n_samples=4, n_points=dataset.n_points)
        samples = samples.cpu().numpy()

        gt_samples = [dataset[i].numpy() for i in range(4)]

        fig = plt.figure(figsize=(16, 8))

        for i in range(4):
            ax = fig.add_subplot(2, 4, i + 1, projection='3d')
            ax.scatter(gt_samples[i][:, 0], gt_samples[i][:, 1], gt_samples[i][:, 2],
                       s=1, alpha=0.5, c='blue')
            ax.set_title(f'Ground Truth {i+1}')
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])

        for i in range(4):
            ax = fig.add_subplot(2, 4, i + 5, projection='3d')
            ax.scatter(samples[i, :, 0], samples[i, :, 1], samples[i, :, 2],
                       s=1, alpha=0.5, c='red')
            ax.set_title(f'Generated {i+1}')
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])

        plt.suptitle(f'SDF Model - Epoch {epoch}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch:04d}.png'), dpi=100)
        plt.close()

    @torch.no_grad()
    def visualize_sdf(self, save_dir: str, epoch: int, dataset):
        """Visualize the learned SDF at t=0."""
        self.model.eval()

        # Create grid in xy plane at z=0
        resolution = 50
        x = torch.linspace(-1.5, 1.5, resolution)
        y = torch.linspace(-1.5, 1.5, resolution)
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        # Query at multiple z slices
        z_slices = [-0.5, 0.0, 0.5]

        fig, axes = plt.subplots(1, len(z_slices), figsize=(15, 5))

        for idx, z_val in enumerate(z_slices):
            zz = torch.full_like(xx, z_val)
            points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
            points = points.unsqueeze(0).to(self.device)  # [1, res*res, 3]

            t = torch.zeros(1, device=self.device)
            sdf = self.model.get_sdf(points, t)  # [1, res*res, 1]
            sdf = sdf.squeeze().cpu().numpy().reshape(resolution, resolution)

            ax = axes[idx]
            im = ax.contourf(xx.numpy(), yy.numpy(), sdf, levels=20, cmap='RdBu')
            ax.contour(xx.numpy(), yy.numpy(), sdf, levels=[0], colors='black', linewidths=2)
            ax.set_title(f'SDF at z={z_val:.1f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)

        plt.suptitle(f'Learned SDF at t=0 - Epoch {epoch}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sdf_epoch_{epoch:04d}.png'), dpi=100)
        plt.close()

    def save_training_curves(self, save_dir: str):
        """Save training loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(self.train_losses, label='Total')
        axes[0].plot(self.velocity_losses, label='Velocity')
        if any(e > 0 for e in self.eikonal_losses):
            axes[0].plot(self.eikonal_losses, label='Eikonal')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Losses')
        axes[0].legend()
        axes[0].grid(True)

        # Skip first few epochs for better scale
        start = min(10, len(self.train_losses) // 10)
        axes[1].plot(self.train_losses[start:])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss (after warmup)')
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=100)
        plt.close()

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'velocity_losses': self.velocity_losses,
            'eikonal_losses': self.eikonal_losses,
        }, path)


def main():
    parser = argparse.ArgumentParser(description='Train SDF Neural Field Diffusion')

    # Data
    parser.add_argument('--shapes', type=str, nargs='+', default=['multi_sphere_ring'],
                        help='Shape(s) to train on')
    parser.add_argument('--n_points', type=int, default=512,
                        help='Points per cloud')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of training samples')
    parser.add_argument('--no_transform', action='store_true',
                        help='Disable random transforms')

    # Model (SMALL config for toy data)
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Transformer hidden dimension')
    parser.add_argument('--hidden_size_x', type=int, default=32,
                        help='NerfBlock hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_blocks', type=int, default=6,
                        help='Total number of blocks')
    parser.add_argument('--num_cond_blocks', type=int, default=2,
                        help='Number of DiT blocks')
    parser.add_argument('--nerf_mlp_ratio', type=int, default=2,
                        help='MLP ratio for NerfBlocks')
    parser.add_argument('--max_freqs', type=int, default=6,
                        help='Fourier frequency bands')

    # Training
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--eikonal_weight', type=float, default=0.0,
                        help='Eikonal loss weight (0 = disabled)')

    # Logging
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Logging interval')
    parser.add_argument('--sample_interval', type=int, default=100,
                        help='Sample generation interval')
    parser.add_argument('--save_dir', type=str, default='experiments/outputs',
                        help='Output directory')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create save directory
    shapes_str = '_'.join(args.shapes)
    save_dir = os.path.join(args.save_dir, f'sdf_{shapes_str}_{args.n_points}pts')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")

    # Create dataset
    print(f"\nCreating dataset: {args.shapes} with {args.n_points} points")
    dataset = PointCloudDataset(
        shapes=args.shapes,
        n_samples=args.n_samples,
        n_points=args.n_points,
        noise_std=0.001,
        random_transform=not args.no_transform,
        scale_range=(0.7, 1.3)
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create SDF model
    print(f"\nCreating SDF model (scalar output, gradient-based velocity)")
    model = SDFNeuralField(
        in_channels=3,
        hidden_size=args.hidden_size,
        hidden_size_x=args.hidden_size_x,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        num_cond_blocks=args.num_cond_blocks,
        nerf_mlp_ratio=args.nerf_mlp_ratio,
        max_freqs=args.max_freqs,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Eikonal regularization: {args.eikonal_weight}")

    # Create trainer
    trainer = SDFTrainer(
        model, device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eikonal_weight=args.eikonal_weight
    )

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    print("="*60)
    trainer.train(
        dataloader,
        n_epochs=args.epochs,
        log_interval=args.log_interval,
        sample_interval=args.sample_interval,
        save_dir=save_dir
    )

    # Save final checkpoint
    trainer.save_checkpoint(os.path.join(save_dir, 'final_checkpoint.pt'))
    print(f"\nSaved checkpoint to {save_dir}/final_checkpoint.pt")


if __name__ == "__main__":
    main()
