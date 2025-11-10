"""
Training script for audio-to-gesture model.
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import json
from typing import Dict, Optional

from data_loader import GestureDataset, create_dataloader
from model import create_model, GestureLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str
):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    logger.info(f"Saved checkpoint: {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    filepath: str,
    device: torch.device
) -> int:
    """Load model checkpoint and return epoch number."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    logger.info(f"Loaded checkpoint from epoch {epoch}")
    return epoch


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: GestureLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_losses = {'total': 0, 'mse': 0, 'smoothness': 0, 'velocity': 0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (audio_features, motion_targets) in enumerate(pbar):
        # Move to device
        audio_features = audio_features.to(device)
        motion_targets = motion_targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(audio_features)

        # Compute loss
        loss, loss_dict = criterion(predictions, motion_targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Accumulate losses
        for key, value in loss_dict.items():
            total_losses[key] += value
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})

        # Log to tensorboard
        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            for key, value in loss_dict.items():
                writer.add_scalar(f'batch/{key}_loss', value, global_step)

    # Average losses
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}

    return avg_losses


def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: GestureLoss,
    device: torch.device
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    total_losses = {'total': 0, 'mse': 0, 'smoothness': 0, 'velocity': 0}
    num_batches = 0

    with torch.no_grad():
        for audio_features, motion_targets in tqdm(dataloader, desc="Validating"):
            # Move to device
            audio_features = audio_features.to(device)
            motion_targets = motion_targets.to(device)

            # Forward pass
            predictions = model(audio_features)

            # Compute loss
            loss, loss_dict = criterion(predictions, motion_targets)

            # Accumulate losses
            for key, value in loss_dict.items():
                total_losses[key] += value
            num_batches += 1

    # Average losses
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}

    return avg_losses


def train_model(
    dataset_path: str,
    config: Dict,
    output_dir: str,
    resume_from: Optional[str] = None,
    use_tensorboard: bool = True
):
    """
    Main training loop.

    Args:
        dataset_path: Path to dataset
        config: Configuration dictionary
        output_dir: Output directory for checkpoints and logs
        resume_from: Checkpoint path to resume from
        use_tensorboard: Whether to use tensorboard logging
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create dataset and dataloaders
    logger.info("Loading dataset...")

    # Get data config
    data_config = config.get('data', {})
    training_config = config.get('training', {})

    full_dataset = GestureDataset(
        dataset_path=dataset_path,
        sample_rate=data_config.get('audio_sr', 22050),
        n_mfcc=data_config.get('n_mfcc', 26),
        frame_length=data_config.get('frame_length', 2048),
        hop_length=data_config.get('hop_length', 512),
        fps=data_config.get('fps', 24)
    )

    # Split into train/val
    val_split = training_config.get('validation_split', 0.2)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size]
    )

    logger.info(f"Train size: {train_size}, Val size: {val_size}")

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=training_config.get('batch_size', 16),
        shuffle=True,
        num_workers=training_config.get('num_workers', 4)
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=training_config.get('batch_size', 16),
        shuffle=False,
        num_workers=training_config.get('num_workers', 4)
    )

    # Create model
    logger.info("Creating model...")
    model_config = config.get('model', {})

    model = create_model(
        model_type=model_config.get('type', 'lstm'),
        input_dim=model_config.get('input_dim', 29),  # MFCC + 3 features
        hidden_dim=model_config.get('hidden_dim', 256),
        output_dim=model_config.get('output_dim', 84),
        num_layers=model_config.get('num_layers', 3),
        dropout=model_config.get('dropout', 0.1)
    )

    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Create criterion
    loss_config = config.get('loss', {})
    criterion = GestureLoss(
        mse_weight=loss_config.get('mse_weight', 1.0),
        smoothness_weight=loss_config.get('smoothness_weight', 0.1),
        velocity_weight=loss_config.get('velocity_weight', 0.05)
    )

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.get('learning_rate', 0.001),
        weight_decay=training_config.get('weight_decay', 1e-5)
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Tensorboard
    writer = None
    if use_tensorboard:
        writer = SummaryWriter(output_dir / 'runs')

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from:
        start_epoch = load_checkpoint(model, optimizer, resume_from, device) + 1

    # Training loop
    num_epochs = training_config.get('num_epochs', 100)
    checkpoint_interval = training_config.get('checkpoint_interval', 10)
    best_val_loss = float('inf')

    logger.info("Starting training...")

    for epoch in range(start_epoch, num_epochs):
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        # Validate
        val_losses = validate(model, val_loader, criterion, device)

        # Log epoch results
        logger.info(f"Epoch {epoch}:")
        logger.info(f"  Train loss: {train_losses['total']:.6f}")
        logger.info(f"  Val loss: {val_losses['total']:.6f}")

        # Tensorboard logging
        if writer:
            writer.add_scalar('epoch/train_loss', train_losses['total'], epoch)
            writer.add_scalar('epoch/val_loss', val_losses['total'], epoch)
            writer.add_scalar('epoch/learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Learning rate scheduling
        scheduler.step(val_losses['total'])

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            save_checkpoint(model, optimizer, epoch, val_losses['total'], checkpoint_path)

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_model_path = output_dir / 'best_model.pt'
            save_checkpoint(model, optimizer, epoch, val_losses['total'], best_model_path)
            logger.info(f"New best model! Val loss: {best_val_loss:.6f}")

    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    save_checkpoint(model, optimizer, num_epochs - 1, val_losses['total'], final_model_path)

    logger.info("Training complete!")

    if writer:
        writer.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train audio-to-gesture model")

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset directory'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='training_config.yaml',
        help='Path to config file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )

    parser.add_argument(
        '--no-tensorboard',
        action='store_true',
        help='Disable tensorboard logging'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Train
    train_model(
        dataset_path=args.dataset,
        config=config,
        output_dir=args.output,
        resume_from=args.resume,
        use_tensorboard=not args.no_tensorboard
    )


if __name__ == '__main__':
    main()
