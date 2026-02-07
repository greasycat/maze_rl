"""
Visualize the learned adjacency matrix from the human-like limited data model.
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from model_v2 import BranchingPredictor


def load_model(checkpoint_path: str, device: torch.device) -> BranchingPredictor:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint.get('config', {'hidden_dim': 32, 'num_layers': 1})

    model = BranchingPredictor(
        num_objects=7,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        max_seq_len=10
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def get_learned_adjacency(model: BranchingPredictor, device: torch.device) -> dict:
    """Get learned adjacency by querying model for each landmark."""
    landmarks = list('ABCDEFG')
    num_landmarks = len(landmarks)

    adjacency = np.zeros((num_landmarks, num_landmarks))
    distances = np.zeros((num_landmarks, num_landmarks))
    importance = np.zeros((num_landmarks, num_landmarks))

    with torch.no_grad():
        for i in range(num_landmarks):
            input_ids = torch.tensor([[i]], dtype=torch.long, device=device)
            lengths = torch.tensor([1], dtype=torch.long, device=device)

            preds = model(input_ids, lengths)

            link_probs = torch.sigmoid(preds[0, :, 0]).cpu().numpy()
            dist_preds = preds[0, :, 1].cpu().numpy()
            imp_probs = torch.sigmoid(preds[0, :, 2]).cpu().numpy()

            adjacency[i, :] = link_probs
            distances[i, :] = dist_preds
            importance[i, :] = imp_probs

    return {
        'adjacency': adjacency,
        'distances': distances,
        'importance': importance,
    }


def plot_comparison(
    learned: dict,
    cutoff: float = 0.5,
    save_path: str = None
):
    """Plot learned vs ground truth adjacency."""
    landmarks = list('ABCDEFG')

    # Ground truth
    gt_connectivity = {
        'A': ['B', 'E'], 'B': ['A', 'G'], 'C': ['E', 'D'],
        'D': ['C', 'G'], 'E': ['A', 'C', 'F'], 'F': ['E', 'G'],
        'G': ['B', 'D', 'F'],
    }

    gt_adj = np.zeros((7, 7))
    for i, lm in enumerate(landmarks):
        for neighbor in gt_connectivity[lm]:
            j = landmarks.index(neighbor)
            gt_adj[i, j] = 1.0

    positions = {
        'A': (0, 4), 'B': (4, 4), 'C': (0, 0), 'D': (4, 0),
        'E': (0, 2), 'F': (2, 2), 'G': (4, 2)
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    # --- Row 1: Adjacency matrices ---

    # Ground truth
    ax = axes[0, 0]
    im = ax.imshow(gt_adj, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(landmarks)
    ax.set_yticklabels(landmarks)
    ax.set_xlabel('To')
    ax.set_ylabel('From')
    ax.set_title('Ground Truth Adjacency')
    for i in range(7):
        for j in range(7):
            color = 'white' if gt_adj[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{gt_adj[i, j]:.0f}', ha='center', va='center',
                    color=color, fontsize=10, fontweight='bold')

    # Learned adjacency
    ax = axes[0, 1]
    im = ax.imshow(learned['adjacency'], cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(landmarks)
    ax.set_yticklabels(landmarks)
    ax.set_xlabel('To')
    ax.set_ylabel('From')
    ax.set_title('Learned Link Probabilities\n(Human-like Limited Data)')
    plt.colorbar(im, ax=ax, label='Probability')
    for i in range(7):
        for j in range(7):
            val = learned['adjacency'][i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=9)

    # Difference
    ax = axes[0, 2]
    diff = np.abs(learned['adjacency'] - gt_adj)
    im = ax.imshow(diff, cmap='Reds', vmin=0, vmax=1)
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(landmarks)
    ax.set_yticklabels(landmarks)
    ax.set_xlabel('To')
    ax.set_ylabel('From')
    ax.set_title(f'Absolute Error (MAE={diff.mean():.4f})')
    plt.colorbar(im, ax=ax, label='|Error|')
    for i in range(7):
        for j in range(7):
            val = diff[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=9)

    # --- Row 2: Graphs ---

    node_colors = ['#ff6b6b' if lm in 'ABCD' else '#4ecdc4' for lm in landmarks]

    # Ground truth graph
    ax = axes[1, 0]
    G_gt = nx.DiGraph()
    G_gt.add_nodes_from(landmarks)
    for i, lm in enumerate(landmarks):
        for neighbor in gt_connectivity[lm]:
            G_gt.add_edge(lm, neighbor)

    nx.draw(G_gt, pos=positions, ax=ax, with_labels=True,
            node_color=node_colors, node_size=800, font_size=12,
            font_weight='bold', edge_color='gray', width=2,
            arrows=True, arrowsize=15, connectionstyle='arc3,rad=0.1')
    ax.set_title('Ground Truth Graph')

    # Learned graph
    ax = axes[1, 1]
    G_learned = nx.DiGraph()
    G_learned.add_nodes_from(landmarks)

    edges = []
    edge_weights = []
    for i, from_lm in enumerate(landmarks):
        for j, to_lm in enumerate(landmarks):
            prob = learned['adjacency'][i, j]
            if prob > cutoff:
                edges.append((from_lm, to_lm))
                edge_weights.append(prob)
                G_learned.add_edge(from_lm, to_lm, weight=prob)

    nx.draw_networkx_nodes(G_learned, pos=positions, ax=ax,
                           node_color=node_colors, node_size=800)
    nx.draw_networkx_labels(G_learned, pos=positions, ax=ax,
                            font_size=12, font_weight='bold')

    if edges:
        edge_colors_plot = plt.cm.Blues(np.array(edge_weights))
        for (u, v), color, w in zip(edges, edge_colors_plot, edge_weights):
            nx.draw_networkx_edges(
                G_learned, pos=positions, ax=ax,
                edgelist=[(u, v)],
                edge_color=[color],
                width=1 + w * 3,
                arrows=True, arrowsize=15,
                connectionstyle='arc3,rad=0.1'
            )

    ax.set_title(f'Learned Graph (cutoff={cutoff})')

    # Analysis: Missing and extra edges
    ax = axes[1, 2]
    ax.axis('off')

    # Compute statistics
    pred_links = learned['adjacency'] > cutoff
    gt_links = gt_adj > 0.5

    true_pos = (pred_links & gt_links).sum()
    false_pos = (pred_links & ~gt_links).sum()
    false_neg = (~pred_links & gt_links).sum()
    true_neg = (~pred_links & ~gt_links).sum()

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Find missing and extra edges
    missing_edges = []
    extra_edges = []
    for i, from_lm in enumerate(landmarks):
        for j, to_lm in enumerate(landmarks):
            if gt_adj[i, j] > 0.5 and learned['adjacency'][i, j] <= cutoff:
                missing_edges.append(f"{from_lm}→{to_lm}")
            if gt_adj[i, j] < 0.5 and learned['adjacency'][i, j] > cutoff:
                extra_edges.append(f"{from_lm}→{to_lm}")

    stats_text = f"""
    Link Prediction Analysis
    ========================

    Metrics (cutoff={cutoff}):
    • Precision: {precision:.3f}
    • Recall: {recall:.3f}
    • F1 Score: {f1:.3f}

    Confusion Matrix:
    • True Positives: {true_pos:.0f}
    • False Positives: {false_pos:.0f}
    • False Negatives: {false_neg:.0f}
    • True Negatives: {true_neg:.0f}

    Missing Edges ({len(missing_edges)}):
    {', '.join(missing_edges) if missing_edges else 'None'}

    Extra Edges ({len(extra_edges)}):
    {', '.join(extra_edges) if extra_edges else 'None'}
    """

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Legend
    corner_patch = mpatches.Patch(color='#ff6b6b', label='Corner (A,B,C,D)')
    middle_patch = mpatches.Patch(color='#4ecdc4', label='Middle (E,F,G)')
    fig.legend(handles=[corner_patch, middle_patch], loc='lower center', ncol=2)

    plt.suptitle('Human-like Limited Data - Learned Graph Structure',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'missing_edges': missing_edges,
        'extra_edges': extra_edges,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_path = Path(__file__).parent
    checkpoint_path = base_path / 'checkpoints_human' / 'best_model.pt'

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train_human.py")
        return

    print("Loading model...")
    model = load_model(checkpoint_path, device)

    print("Computing learned adjacency matrix...")
    learned = get_learned_adjacency(model, device)

    # Print learned adjacency
    landmarks = list('ABCDEFG')
    print("\nLearned Link Probabilities:")
    print("     " + "  ".join(f"{lm:>6}" for lm in landmarks))
    for i, lm in enumerate(landmarks):
        row = "  ".join(f"{learned['adjacency'][i, j]:6.3f}" for j in range(7))
        print(f"{lm}:   {row}")

    # Plot and analyze
    stats = plot_comparison(
        learned,
        cutoff=0.5,
        save_path=base_path / 'checkpoints_human' / 'learned_adjacency_human.png'
    )

    print(f"\nSummary:")
    print(f"  F1 Score: {stats['f1']:.3f}")
    print(f"  Missing edges: {len(stats['missing_edges'])}")
    print(f"  Extra edges: {len(stats['extra_edges'])}")


if __name__ == '__main__':
    main()
