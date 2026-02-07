"""
Training script for SPARSE data scenario.
Tests learning with very limited and incomplete data coverage.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from model_v2 import BranchingPredictor, BranchingLoss, compute_metrics


class SparseMazeDataset(Dataset):
    def __init__(self, data_path: str):
        data = np.load(data_path)
        self.input_sequences = torch.from_numpy(data['input_sequences']).long()
        self.sequence_lengths = torch.from_numpy(data['sequence_lengths']).long()
        self.targets = torch.from_numpy(data['targets']).float()

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return (
            self.input_sequences[idx],
            self.sequence_lengths[idx],
            self.targets[idx]
        )


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_metrics = {
        'total_loss': 0.0, 'loss_link': 0.0, 'loss_dist': 0.0, 'loss_imp': 0.0,
        'link_accuracy': 0.0, 'link_f1': 0.0, 'dist_mae': 0.0, 'imp_accuracy': 0.0,
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
            if k in total_metrics:
                total_metrics[k] += v

        batch_metrics = compute_metrics(preds, targets)
        for k, v in batch_metrics.items():
            if k in total_metrics:
                total_metrics[k] += v

        num_batches += 1

    return {k: v / num_batches for k, v in total_metrics.items()}


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_metrics = {
        'total_loss': 0.0, 'loss_link': 0.0, 'loss_dist': 0.0, 'loss_imp': 0.0,
        'link_accuracy': 0.0, 'link_f1': 0.0, 'dist_mae': 0.0, 'imp_accuracy': 0.0,
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
                if k in total_metrics:
                    total_metrics[k] += v

            batch_metrics = compute_metrics(preds, targets)
            for k, v in batch_metrics.items():
                if k in total_metrics:
                    total_metrics[k] += v

            num_batches += 1

    return {k: v / num_batches for k, v in total_metrics.items()}


def main():
    parser = argparse.ArgumentParser(description='Train on sparse data')
    parser.add_argument('--data-path', type=str, default='data/training_data_sparse.npz')
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='checkpoints_sparse')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = Path(__file__).parent / args.data_path
    if not data_path.exists():
        print(f"Data not found. Run data_generation_sparse.py first.")
        return

    dataset = SparseMazeDataset(data_path)

    print("=" * 50)
    print("SPARSE DATA TRAINING")
    print("=" * 50)
    print(f"Total examples: {len(dataset)}")

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = BranchingPredictor(
        num_objects=7,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_seq_len=10
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = BranchingLoss(link_weight=1.0, dist_weight=0.5, imp_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)

    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    best_val_f1 = 0.0

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Train F1':>10} {'Val F1':>10}")
    print("-" * 55)

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_f1'].append(train_metrics['link_f1'])
        history['val_f1'].append(val_metrics['link_f1'])

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"{epoch + 1:>6} {train_metrics['total_loss']:>12.4f} "
                  f"{val_metrics['total_loss']:>12.4f} "
                  f"{train_metrics['link_f1']:>10.4f} {val_metrics['link_f1']:>10.4f}")

        if val_metrics['link_f1'] > best_val_f1:
            best_val_f1 = val_metrics['link_f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_metrics['link_f1'],
                'val_metrics': val_metrics,
                'config': {'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers}
            }, output_dir / 'best_model.pt')

    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_metrics': val_metrics,
        'config': {'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers}
    }, output_dir / 'final_model.pt')

    print(f"\n{'=' * 50}")
    print(f"Training complete!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Models saved to {output_dir}")

    # Plot history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_f1'], label='Train')
    axes[1].plot(history['val_f1'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1')
    axes[1].set_title('Link F1 Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.suptitle('Sparse Data Training', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150)
    print(f"Saved plot to {output_dir / 'training_history.png'}")
    plt.show()


if __name__ == '__main__':
    main()
