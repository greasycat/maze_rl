"""
Training script for human-like limited data scenario.

This script trains the BranchingPredictor on limited data to simulate
how a human might learn spatial relationships from few experiences.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from model_v2 import BranchingPredictor, BranchingLoss, compute_metrics


class HumanMazeDataset(Dataset):
    """Dataset for human-like maze navigation."""

    def __init__(self, data_path: str):
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

        optimizer.zero_grad()
        preds = model(input_seq, lengths)
        loss, loss_dict = criterion(preds, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in loss_dict.items():
            total_metrics[k] += v

        batch_metrics = compute_metrics(preds, targets)
        for k, v in batch_metrics.items():
            total_metrics[k] += v

        num_batches += 1

    return {k: v / num_batches for k, v in total_metrics.items()}


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

    return {k: v / num_batches for k, v in total_metrics.items()}


def plot_training_history(history: dict, save_path: str = None):
    """Plot training history."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Total Loss
    axes[0, 0].plot(history['train_total_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_total_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Link F1 Score
    axes[0, 1].plot(history['train_link_f1'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_link_f1'], label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Link Prediction F1')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.05)

    # Link Accuracy
    axes[0, 2].plot(history['train_link_accuracy'], label='Train', linewidth=2)
    axes[0, 2].plot(history['val_link_accuracy'], label='Validation', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_title('Link Prediction Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(0, 1.05)

    # Individual Losses
    axes[1, 0].plot(history['train_loss_link'], label='Link (Train)', linewidth=2)
    axes[1, 0].plot(history['val_loss_link'], label='Link (Val)', linewidth=2)
    axes[1, 0].plot(history['train_loss_dist'], label='Dist (Train)', linestyle='--', linewidth=2)
    axes[1, 0].plot(history['val_loss_dist'], label='Dist (Val)', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Individual Losses')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Distance MAE
    axes[1, 1].plot(history['train_dist_mae'], label='Train', linewidth=2)
    axes[1, 1].plot(history['val_dist_mae'], label='Validation', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('Distance Prediction MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Importance Accuracy
    axes[1, 2].plot(history['train_imp_accuracy'], label='Train', linewidth=2)
    axes[1, 2].plot(history['val_imp_accuracy'], label='Validation', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Importance (Corner) Accuracy')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(0, 1.05)

    plt.suptitle('Training History - Human-like Limited Data', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train on human-like limited data')
    parser.add_argument('--data-path', type=str, default='data/training_data_human.npz',
                        help='Path to training data')
    parser.add_argument('--hidden-dim', type=int, default=32,
                        help='Hidden dimension (smaller for limited data)')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of transformer layers (fewer for limited data)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (smaller for limited data)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='checkpoints_human',
                        help='Output directory')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_path = Path(__file__).parent / args.data_path
    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        print("Please run data_generation_human.py first.")
        return

    dataset = HumanMazeDataset(data_path)
    print(f"\n{'=' * 50}")
    print("HUMAN-LIKE LIMITED DATA TRAINING")
    print(f"{'=' * 50}")
    print(f"Total training examples: {len(dataset)}")

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

    # Create smaller model for limited data
    model = BranchingPredictor(
        num_objects=7,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_seq_len=10
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  (Reduced from 107k in full model)")

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

    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # Training loop
    best_val_f1 = 0.0
    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Train F1':>10} {'Val F1':>10}")
    print("-" * 55)

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        for k in history_keys:
            history[f'train_{k}'].append(train_metrics[k])
            history[f'val_{k}'].append(val_metrics[k])

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch + 1:>6} {train_metrics['total_loss']:>12.4f} "
                  f"{val_metrics['total_loss']:>12.4f} "
                  f"{train_metrics['link_f1']:>10.4f} {val_metrics['link_f1']:>10.4f}")

        if val_metrics['link_f1'] > best_val_f1:
            best_val_f1 = val_metrics['link_f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['link_f1'],
                'val_metrics': val_metrics,
                'config': {
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                }
            }, output_dir / 'best_model.pt')

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_metrics': val_metrics,
    }, output_dir / 'final_model.pt')

    print(f"\n{'=' * 50}")
    print("Training complete!")
    print(f"{'=' * 50}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Final validation metrics:")
    print(f"  Link F1: {val_metrics['link_f1']:.4f}")
    print(f"  Link Accuracy: {val_metrics['link_accuracy']:.4f}")
    print(f"  Distance MAE: {val_metrics['dist_mae']:.4f}")
    print(f"  Importance Accuracy: {val_metrics['imp_accuracy']:.4f}")
    print(f"\nModels saved to {output_dir}")

    # Plot training history
    plot_training_history(history, save_path=output_dir / 'training_history.png')


if __name__ == '__main__':
    main()
