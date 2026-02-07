"""
Training script for the BranchingPredictor model (v2).
Multi-label link prediction with masked loss.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from model_v2 import BranchingPredictor, BranchingLoss, compute_metrics


class MazeDatasetV2(Dataset):
    """Dataset for maze navigation with branching targets."""

    def __init__(self, data_path: str):
        """
        Load dataset from npz file.

        Args:
            data_path: Path to the .npz file containing training data
        """
        data = np.load(data_path)
        self.input_sequences = torch.from_numpy(data['input_sequences']).long()
        self.sequence_lengths = torch.from_numpy(data['sequence_lengths']).long()
        self.targets = torch.from_numpy(data['targets']).float()

    def __len__(self) -> int:
        return len(self.input_sequences)

    def __getitem__(self, idx: int) -> tuple:
        return (
            self.input_sequences[idx],
            self.sequence_lengths[idx],
            self.targets[idx]
        )


def train_epoch(
    model: BranchingPredictor,
    dataloader: DataLoader,
    criterion: BranchingLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> dict:
    """Train for one epoch."""
    model.train()
    total_metrics = {
        'total_loss': 0.0, 'loss_link': 0.0, 'loss_dist': 0.0, 'loss_imp': 0.0,
        'link_accuracy': 0.0, 'link_precision': 0.0, 'link_recall': 0.0,
        'link_f1': 0.0, 'dist_mae': 0.0, 'imp_accuracy': 0.0,
    }
    num_batches = 0

    for input_seq, lengths, targets in dataloader:
        input_seq = input_seq.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        preds = model(input_seq, lengths)

        # Compute loss
        loss, loss_dict = criterion(preds, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        for k, v in loss_dict.items():
            total_metrics[k] += v

        batch_metrics = compute_metrics(preds, targets)
        for k, v in batch_metrics.items():
            total_metrics[k] += v

        num_batches += 1

    # Average metrics
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    return avg_metrics


def evaluate(
    model: BranchingPredictor,
    dataloader: DataLoader,
    criterion: BranchingLoss,
    device: torch.device
) -> dict:
    """Evaluate the model."""
    model.eval()
    total_metrics = {
        'total_loss': 0.0, 'loss_link': 0.0, 'loss_dist': 0.0, 'loss_imp': 0.0,
        'link_accuracy': 0.0, 'link_precision': 0.0, 'link_recall': 0.0,
        'link_f1': 0.0, 'dist_mae': 0.0, 'imp_accuracy': 0.0,
    }
    num_batches = 0

    with torch.no_grad():
        for input_seq, lengths, targets in dataloader:
            input_seq = input_seq.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)

            preds = model(input_seq, lengths)
            _, loss_dict = criterion(preds, targets)

            for k, v in loss_dict.items():
                total_metrics[k] += v

            batch_metrics = compute_metrics(preds, targets)
            for k, v in batch_metrics.items():
                total_metrics[k] += v

            num_batches += 1

    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    return avg_metrics


def plot_training_history(history: dict, save_path: str = None):
    """Plot training history."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Total Loss
    axes[0, 0].plot(history['train_total_loss'], label='Train')
    axes[0, 0].plot(history['val_total_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Link F1 Score
    axes[0, 1].plot(history['train_link_f1'], label='Train')
    axes[0, 1].plot(history['val_link_f1'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Link Prediction F1')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Link Accuracy
    axes[0, 2].plot(history['train_link_accuracy'], label='Train')
    axes[0, 2].plot(history['val_link_accuracy'], label='Validation')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_title('Link Prediction Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Individual Losses
    axes[1, 0].plot(history['train_loss_link'], label='Link (Train)')
    axes[1, 0].plot(history['val_loss_link'], label='Link (Val)')
    axes[1, 0].plot(history['train_loss_dist'], label='Dist (Train)', linestyle='--')
    axes[1, 0].plot(history['val_loss_dist'], label='Dist (Val)', linestyle='--')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Individual Losses')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Distance MAE
    axes[1, 1].plot(history['train_dist_mae'], label='Train')
    axes[1, 1].plot(history['val_dist_mae'], label='Validation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('Distance Prediction MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Importance Accuracy
    axes[1, 2].plot(history['train_imp_accuracy'], label='Train')
    axes[1, 2].plot(history['val_imp_accuracy'], label='Validation')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Importance (Corner) Accuracy')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.suptitle('Training History - Branching Predictor', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train BranchingPredictor model')
    parser.add_argument('--data-path', type=str, default='data/training_data_v2.npz',
                        help='Path to training data')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='checkpoints_v2',
                        help='Output directory for model checkpoints')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_path = Path(__file__).parent / args.data_path
    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        print("Please run data_generation_v2.py first to generate training data.")
        return

    dataset = MazeDatasetV2(data_path)
    print(f"Loaded dataset with {len(dataset)} training examples")

    # Split data
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train size: {train_size}, Validation size: {val_size}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Create model
    model = BranchingPredictor(
        num_objects=7,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_seq_len=20
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss and optimizer
    criterion = BranchingLoss(link_weight=1.0, dist_weight=0.5, imp_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training history
    history_keys = [
        'total_loss', 'loss_link', 'loss_dist', 'loss_imp',
        'link_accuracy', 'link_precision', 'link_recall', 'link_f1',
        'dist_mae', 'imp_accuracy'
    ]
    history = {f'train_{k}': [] for k in history_keys}
    history.update({f'val_{k}': [] for k in history_keys})

    # Output directory
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # Training loop
    best_val_f1 = 0.0
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Record history
        for k in history_keys:
            history[f'train_{k}'].append(train_metrics[k])
            history[f'val_{k}'].append(val_metrics[k])

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, "
                  f"Link F1: {train_metrics['link_f1']:.4f}, "
                  f"Dist MAE: {train_metrics['dist_mae']:.4f}, "
                  f"Imp Acc: {train_metrics['imp_accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, "
                  f"Link F1: {val_metrics['link_f1']:.4f}, "
                  f"Dist MAE: {val_metrics['dist_mae']:.4f}, "
                  f"Imp Acc: {val_metrics['imp_accuracy']:.4f}")

        # Save best model (by F1 score)
        if val_metrics['link_f1'] > best_val_f1:
            best_val_f1 = val_metrics['link_f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['link_f1'],
                'val_metrics': val_metrics,
            }, output_dir / 'best_model.pt')

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_metrics['link_f1'],
        'val_metrics': val_metrics,
    }, output_dir / 'final_model.pt')

    print(f"\n{'=' * 50}")
    print(f"Training complete!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Models saved to {output_dir}")

    # Plot training history
    plot_training_history(history, save_path=output_dir / 'training_history.png')


if __name__ == '__main__':
    main()
