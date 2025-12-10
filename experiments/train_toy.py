"""
Training Script for Neural Field Diffusion on Toy Data

Trains the neural field model on synthetic point cloud data,
demonstrating the core algorithm on simple manifolds.

Usage:
    python experiments/train_toy.py --shape sphere --n_points 256 --epochs 500

Key parameters:
    - n_points: 256-512 recommended (each point is a token, O(N²) attention)
    - shape: sphere, torus, helix, etc.
    - encoder: pointnet (default) or transformer
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

from data.toy_data import (
    get_all_generators, generate_sphere, generate_torus,
    generate_helix, PointCloud
)
from src.models.neural_field import NeuralFieldDiffusion
from src.diffusion.flow_matching import FlowMatchingLoss, FlowMatchingSampler


class PointCloudDataset(Dataset):
    """
    Dataset of point clouds for training.

    Each sample is a normalized point cloud with optional noise augmentation.
    """

    def __init__(self, shape: str = 'sphere', n_samples: int = 1000,
                 n_points: int = 256, noise_std: float = 0.0,
                 resample_each_epoch: bool = True):
        """
        Args:
            shape: Shape type ('sphere', 'torus', 'helix', etc.)
            n_samples: Number of samples in dataset
            n_points: Points per sample
            noise_std: Gaussian noise std for augmentation
            resample_each_epoch: If True, resample points each access
        """
        self.shape = shape
        self.n_samples = n_samples
        self.n_points = n_points
        self.noise_std = noise_std
        self.resample = resample_each_epoch

        # Get generator
        generators = get_all_generators()
        if shape not in generators:
            raise ValueError(f"Unknown shape: {shape}. Available: {list(generators.keys())}")
        self.generator = generators[shape]

        # Pre-generate if not resampling
        if not resample_each_epoch:
            self.data = []
            for _ in range(n_samples):
                pc = self.generator(n_points=n_points).normalize()
                if noise_std > 0:
                    pc = pc.add_noise(noise_std)
                self.data.append(torch.tensor(pc.points, dtype=torch.float32))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.resample:
            pc = self.generator(n_points=self.n_points).normalize()
            if self.noise_std > 0:
                pc = pc.add_noise(self.noise_std)
            return torch.tensor(pc.points, dtype=torch.float32)
        else:
            return self.data[idx]


class Trainer:
    """Training loop for neural field diffusion."""

    def __init__(self, model: nn.Module, device: torch.device,
                 lr: float = 1e-4, weight_decay: float = 0.0):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.loss_fn = FlowMatchingLoss(schedule_type='linear')
        self.sampler = FlowMatchingSampler(model)

        self.train_losses = []
        self.epoch_times = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
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
            n_batches += 1

        return total_loss / n_batches

    def train(self, dataloader: DataLoader, n_epochs: int,
              log_interval: int = 10, sample_interval: int = 50,
              save_dir: Optional[str] = None):
        """Full training loop."""

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        pbar = tqdm(range(n_epochs), desc="Training")

        for epoch in pbar:
            start_time = time.time()
            loss = self.train_epoch(dataloader)
            epoch_time = time.time() - start_time

            self.train_losses.append(loss)
            self.epoch_times.append(epoch_time)

            pbar.set_postfix({'loss': f'{loss:.4f}'})

            # Logging
            if (epoch + 1) % log_interval == 0:
                tqdm.write(f"Epoch {epoch+1}/{n_epochs} | Loss: {loss:.4f} | Time: {epoch_time:.2f}s")

            # Generate samples
            if save_dir and (epoch + 1) % sample_interval == 0:
                self.visualize_samples(save_dir, epoch + 1, dataloader.dataset)

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

        # Generate samples
        samples = self.generate_samples(n_samples=4, n_points=dataset.n_points)
        samples = samples.cpu().numpy()

        # Get ground truth samples
        gt_samples = [dataset[i].numpy() for i in range(4)]

        # Plot
        fig = plt.figure(figsize=(16, 8))

        # Ground truth
        for i in range(4):
            ax = fig.add_subplot(2, 4, i + 1, projection='3d')
            ax.scatter(gt_samples[i][:, 0], gt_samples[i][:, 1], gt_samples[i][:, 2],
                       s=1, alpha=0.5, c='blue')
            ax.set_title(f'Ground Truth {i+1}')
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])

        # Generated
        for i in range(4):
            ax = fig.add_subplot(2, 4, i + 5, projection='3d')
            ax.scatter(samples[i, :, 0], samples[i, :, 1], samples[i, :, 2],
                       s=1, alpha=0.5, c='red')
            ax.set_title(f'Generated {i+1}')
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])

        plt.suptitle(f'Epoch {epoch}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch:04d}.png'), dpi=100)
        plt.close()

    def save_training_curves(self, save_dir: str):
        """Save training loss curve."""
        plt.figure(figsize=(10, 4))
        plt.plot(self.train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=100)
        plt.close()

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])


def test_resolution_independence(trainer: Trainer, dataset, save_dir: str):
    """
    Test generation at multiple resolutions.

    This demonstrates the key capability: same model generates
    point clouds at any number of points.
    """
    print("\n" + "="*60)
    print("Testing Resolution Independence")
    print("="*60)

    trainer.model.eval()

    # Get context from a sample
    sample = dataset[0].unsqueeze(0).to(trainer.device)
    context = trainer.model.get_context(sample)

    resolutions = [64, 128, 256, 512, 1024]

    fig = plt.figure(figsize=(20, 4))

    for i, n_pts in enumerate(resolutions):
        # Generate at this resolution
        samples = trainer.sampler.sample_at_resolution(
            context, n_points=n_pts, n_steps=50
        )
        samples = samples.cpu().numpy()[0]

        ax = fig.add_subplot(1, 5, i + 1, projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
                   s=max(1, 5 - i), alpha=0.5, c='green')
        ax.set_title(f'N = {n_pts}')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])

        print(f"Generated {n_pts:5d} points: shape={samples.shape}")

    plt.suptitle('Resolution Independence: Same Model, Different Point Counts', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'resolution_independence.png'), dpi=100)
    plt.close()

    print("Saved resolution independence visualization")


def visualize_generation_process(trainer: Trainer, save_dir: str, n_points: int = 256):
    """Visualize the generation process step by step."""
    print("\n" + "="*60)
    print("Visualizing Generation Process")
    print("="*60)

    trainer.model.eval()

    # Generate with trajectory
    noise = torch.randn(1, n_points, 3, device=trainer.device)
    trajectory = trainer.sampler.sample_euler(noise, n_steps=50, return_trajectory=True)
    trajectory = trajectory.cpu().numpy()[:, 0]  # [steps, N, 3]

    # Select time steps to visualize
    n_steps = trajectory.shape[0]
    step_indices = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]
    t_values = [1.0, 0.75, 0.5, 0.25, 0.0]

    fig = plt.figure(figsize=(20, 4))

    for i, (step_idx, t_val) in enumerate(zip(step_indices, t_values)):
        points = trajectory[step_idx]

        ax = fig.add_subplot(1, 5, i + 1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   s=2, alpha=0.5, c=plt.cm.coolwarm(1 - t_val))
        ax.set_title(f't = {t_val:.2f}')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])

    plt.suptitle('Generation Process: Noise → Manifold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'generation_process.png'), dpi=100)
    plt.close()

    print("Saved generation process visualization")


def main():
    parser = argparse.ArgumentParser(description='Train Neural Field Diffusion')

    # Data
    parser.add_argument('--shape', type=str, default='sphere',
                        help='Shape to train on')
    parser.add_argument('--n_points', type=int, default=256,
                        help='Points per cloud (256-512 recommended)')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of training samples')

    # Model (PixNerd-style architecture)
    parser.add_argument('--hidden_size', type=int, default=384,
                        help='Transformer hidden dimension')
    parser.add_argument('--hidden_size_x', type=int, default=64,
                        help='NerfBlock hidden dimension')
    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of attention heads')
    parser.add_argument('--num_blocks', type=int, default=12,
                        help='Total number of blocks')
    parser.add_argument('--num_cond_blocks', type=int, default=4,
                        help='Number of DiT blocks (rest are NerfBlocks)')
    parser.add_argument('--nerf_mlp_ratio', type=int, default=4,
                        help='MLP ratio for NerfBlocks')
    parser.add_argument('--max_freqs', type=int, default=8,
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
    save_dir = os.path.join(args.save_dir, f'{args.shape}_{args.n_points}pts')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")

    # Create dataset
    print(f"\nCreating dataset: {args.shape} with {args.n_points} points")
    dataset = PointCloudDataset(
        shape=args.shape,
        n_samples=args.n_samples,
        n_points=args.n_points,
        noise_std=0.001,  # Small noise for augmentation
        resample_each_epoch=True
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create model (PixNerd-style)
    print(f"\nCreating model: PixNerd-style DiT + NerfBlocks")
    model = NeuralFieldDiffusion(
        in_channels=3,
        out_channels=3,
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

    # Create trainer
    trainer = Trainer(model, device, lr=args.lr, weight_decay=args.weight_decay)

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
    checkpoint_path = os.path.join(save_dir, 'checkpoint_final.pt')
    trainer.save_checkpoint(checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    # Test resolution independence
    test_resolution_independence(trainer, dataset, save_dir)

    # Visualize generation process
    visualize_generation_process(trainer, save_dir, n_points=args.n_points)

    print("\n" + "="*60)
    print("Training complete! Check outputs in:", save_dir)
    print("="*60)


if __name__ == "__main__":
    main()
