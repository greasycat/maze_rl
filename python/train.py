"""
Training script for the MultiTaskGNN model.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from model import MultiTaskGNN, MultiTaskLoss, compute_metrics


class MazeDataset(Dataset):
    """Dataset for maze navigation sequences."""

    def __init__(self, data_path: str):
        """
        Load dataset from npz file.

        Args:
            data_path: Path to the .npz file containing training data
        """
        data = np.load(data_path)
        self.obj_ids = torch.from_numpy(data['obj_ids']).long()
        self.flags = torch.from_numpy(data['flags']).long()
        self.dists = torch.from_numpy(data['dists']).float()

    def __len__(self) -> int:
        return len(self.obj_ids)

    def __getitem__(self, idx: int) -> tuple:
        return self.obj_ids[idx], self.flags[idx], self.dists[idx]


def create_shifted_targets(obj_ids: torch.Tensor, flags: torch.Tensor, dists: torch.Tensor):
    """
    Create shifted targets for next-step prediction.
    The model predicts the next landmark, flag, and distance given current context.
    """
    # Targets are the next step's values
    target_obj = obj_ids[:, 1:]
    target_flag = flags[:, 1:]
    target_dist = dists[:, 1:]

    # Inputs are all but the last step
    input_obj = obj_ids[:, :-1]
    input_flag = flags[:, :-1]
    input_dist = dists[:, :-1]

    return (input_obj, input_flag, input_dist), (target_obj, target_flag, target_dist)


def train_epoch(
    model: MultiTaskGNN,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_metrics = {'obj_accuracy': 0.0, 'flag_accuracy': 0.0, 'dist_mae': 0.0}
    num_batches = 0

    for obj_ids, flags, dists in dataloader:
        obj_ids = obj_ids.to(device)
        flags = flags.to(device)
        dists = dists.to(device)

        # Create inputs and targets for next-step prediction
        inputs, targets = create_shifted_targets(obj_ids, flags, dists)
        input_obj, input_flag, input_dist = inputs
        target_obj, target_flag, target_dist = targets

        # Forward pass
        optimizer.zero_grad()
        preds = model(input_obj, input_flag, input_dist)

        # Compute loss
        loss = criterion(preds, (target_obj, target_flag, target_dist))

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        batch_metrics = compute_metrics(preds, (target_obj, target_flag, target_dist))
        for k, v in batch_metrics.items():
            total_metrics[k] += v
        num_batches += 1

    # Average metrics
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    avg_metrics['loss'] = avg_loss

    return avg_metrics


def evaluate(
    model: MultiTaskGNN,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device
) -> dict:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_metrics = {'obj_accuracy': 0.0, 'flag_accuracy': 0.0, 'dist_mae': 0.0}
    num_batches = 0

    with torch.no_grad():
        for obj_ids, flags, dists in dataloader:
            obj_ids = obj_ids.to(device)
            flags = flags.to(device)
            dists = dists.to(device)

            inputs, targets = create_shifted_targets(obj_ids, flags, dists)
            input_obj, input_flag, input_dist = inputs
            target_obj, target_flag, target_dist = targets

            preds = model(input_obj, input_flag, input_dist)
            loss = criterion(preds, (target_obj, target_flag, target_dist))

            total_loss += loss.item()
            batch_metrics = compute_metrics(preds, (target_obj, target_flag, target_dist))
            for k, v in batch_metrics.items():
                total_metrics[k] += v
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    avg_metrics['loss'] = avg_loss

    return avg_metrics


def plot_training_history(history: dict, save_path: str = None):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Object Accuracy
    axes[0, 1].plot(history['train_obj_accuracy'], label='Train')
    axes[0, 1].plot(history['val_obj_accuracy'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Object Prediction Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Flag Accuracy
    axes[1, 0].plot(history['train_flag_accuracy'], label='Train')
    axes[1, 0].plot(history['val_flag_accuracy'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Flag (Corner) Prediction Accuracy')
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

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train MultiTaskGNN model')
    parser.add_argument('--data-path', type=str, default='data/training_data.npz',
                        help='Path to training data')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
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
        print("Please run data_generation.py first to generate training data.")
        return

    dataset = MazeDataset(data_path)
    print(f"Loaded dataset with {len(dataset)} sequences")

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
    model = MultiTaskGNN(
        num_objects=7,
        hidden_dim=args.hidden_dim,
        max_seq_len=100
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss and optimizer
    criterion = MultiTaskLoss(alpha=1.0, beta=1.0, gamma=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_obj_accuracy': [], 'val_obj_accuracy': [],
        'train_flag_accuracy': [], 'val_flag_accuracy': [],
        'train_dist_mae': [], 'val_dist_mae': [],
    }

    # Output directory
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_metrics['loss'])

        # Record history
        for k in ['loss', 'obj_accuracy', 'flag_accuracy', 'dist_mae']:
            history[f'train_{k}'].append(train_metrics[k])
            history[f'val_{k}'].append(val_metrics[k])

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Obj Acc: {train_metrics['obj_accuracy']:.4f}, "
                  f"Flag Acc: {train_metrics['flag_accuracy']:.4f}, "
                  f"Dist MAE: {train_metrics['dist_mae']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Obj Acc: {val_metrics['obj_accuracy']:.4f}, "
                  f"Flag Acc: {val_metrics['flag_accuracy']:.4f}, "
                  f"Dist MAE: {val_metrics['dist_mae']:.4f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_metrics': val_metrics,
            }, output_dir / 'best_model.pt')

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['loss'],
        'val_metrics': val_metrics,
    }, output_dir / 'final_model.pt')

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to {output_dir}")

    # Plot training history
    plot_training_history(history, save_path=output_dir / 'training_history.png')


if __name__ == '__main__':
    main()
