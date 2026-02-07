"""
Visualize the learned adjacency matrix from the BranchingPredictor model (v2).
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
    model = BranchingPredictor(
        num_objects=7, hidden_dim=64, num_layers=2, max_seq_len=20
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def get_learned_adjacency(model: BranchingPredictor, device: torch.device) -> dict:
    """
    Get the learned adjacency matrix by querying the model for each landmark.

    Returns:
        dict with:
            - adjacency: [7, 7] link probabilities
            - distances: [7, 7] predicted distances
            - importance: [7, 7] corner probabilities
    """
    landmarks = list('ABCDEFG')
    num_landmarks = len(landmarks)

    adjacency = np.zeros((num_landmarks, num_landmarks))
    distances = np.zeros((num_landmarks, num_landmarks))
    importance = np.zeros((num_landmarks, num_landmarks))

    with torch.no_grad():
        for i in range(num_landmarks):
            # Create input: single landmark
            input_ids = torch.tensor([[i]], dtype=torch.long, device=device)
            lengths = torch.tensor([1], dtype=torch.long, device=device)

            # Get predictions
            preds = model(input_ids, lengths)  # [1, 7, 3]

            # Extract probabilities
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


def plot_adjacency_comparison(
    learned: dict,
    cutoff: float = 0.1,
    save_path: str = None
):
    """Plot learned adjacency matrix compared to ground truth."""
    landmarks = list('ABCDEFG')

    # Ground truth connectivity and distances
    gt_connectivity = {
        'A': ['B', 'E'],
        'B': ['A', 'G'],
        'C': ['E', 'D'],
        'D': ['C', 'G'],
        'E': ['A', 'C', 'F'],
        'F': ['E', 'G'],
        'G': ['B', 'D', 'F'],
    }

    gt_distances = {
        ('A', 'B'): 4, ('A', 'E'): 2,
        ('B', 'A'): 4, ('B', 'G'): 2,
        ('C', 'E'): 2, ('C', 'D'): 4,
        ('D', 'C'): 4, ('D', 'G'): 2,
        ('E', 'A'): 2, ('E', 'C'): 2, ('E', 'F'): 2,
        ('F', 'E'): 2, ('F', 'G'): 2,
        ('G', 'B'): 2, ('G', 'D'): 2, ('G', 'F'): 2,
    }

    # Build ground truth matrices
    gt_adj = np.zeros((7, 7))
    gt_dist = np.zeros((7, 7))
    for i, lm in enumerate(landmarks):
        for neighbor in gt_connectivity[lm]:
            j = landmarks.index(neighbor)
            gt_adj[i, j] = 1.0
            gt_dist[i, j] = gt_distances[(lm, neighbor)]

    # Node positions (for graph visualization)
    positions = {
        'A': (0, 4), 'B': (4, 4),
        'C': (0, 0), 'D': (4, 0),
        'E': (0, 2), 'F': (2, 2), 'G': (4, 2)
    }

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    # --- Row 1: Adjacency matrices ---

    # Ground truth adjacency
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
    ax.set_title('Learned Link Probabilities')
    plt.colorbar(im, ax=ax, label='Probability')
    for i in range(7):
        for j in range(7):
            val = learned['adjacency'][i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=9)

    # Difference (error)
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

    # --- Row 2: Graph visualizations ---

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
        edge_colors = plt.cm.Blues(np.array(edge_weights))
        for (u, v), color, w in zip(edges, edge_colors, edge_weights):
            nx.draw_networkx_edges(
                G_learned, pos=positions, ax=ax,
                edgelist=[(u, v)],
                edge_color=[color],
                width=1 + w * 3,
                arrows=True, arrowsize=15,
                connectionstyle='arc3,rad=0.1'
            )

    ax.set_title(f'Learned Graph (cutoff={cutoff})')

    # Learned distances
    ax = axes[1, 2]
    # Only show distances where links exist
    dist_display = learned['distances'].copy()
    dist_display[learned['adjacency'] < 0.5] = np.nan

    im = ax.imshow(dist_display, cmap='YlOrRd', vmin=0, vmax=5)
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(landmarks)
    ax.set_yticklabels(landmarks)
    ax.set_xlabel('To')
    ax.set_ylabel('From')
    ax.set_title('Learned Distances (where link > 0.5)')
    plt.colorbar(im, ax=ax, label='Distance')

    for i in range(7):
        for j in range(7):
            if learned['adjacency'][i, j] > 0.5:
                pred_d = learned['distances'][i, j]
                true_d = gt_dist[i, j]
                ax.text(j, i, f'{pred_d:.1f}\n({true_d:.0f})', ha='center', va='center',
                        color='black', fontsize=8)

    # Add legend
    corner_patch = mpatches.Patch(color='#ff6b6b', label='Corner (A,B,C,D)')
    middle_patch = mpatches.Patch(color='#4ecdc4', label='Middle (E,F,G)')
    fig.legend(handles=[corner_patch, middle_patch], loc='lower center', ncol=2)

    plt.suptitle('Branching Predictor - Learned Graph Structure', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_path = Path(__file__).parent
    checkpoint_path = base_path / 'checkpoints_v2' / 'best_model.pt'

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train_v2.py")
        return

    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path, device)

    # Get learned adjacency
    print("Computing learned adjacency matrix...")
    learned = get_learned_adjacency(model, device)

    # Print learned adjacency
    landmarks = list('ABCDEFG')
    print("\nLearned Link Probabilities:")
    print("     " + "  ".join(f"{lm:>6}" for lm in landmarks))
    for i, lm in enumerate(landmarks):
        row = "  ".join(f"{learned['adjacency'][i, j]:6.3f}" for j in range(7))
        print(f"{lm}:   {row}")

    print("\nLearned Distances (where link prob > 0.5):")
    print("     " + "  ".join(f"{lm:>6}" for lm in landmarks))
    for i, lm in enumerate(landmarks):
        row_vals = []
        for j in range(7):
            if learned['adjacency'][i, j] > 0.5:
                row_vals.append(f"{learned['distances'][i, j]:6.2f}")
            else:
                row_vals.append("     -")
        print(f"{lm}:   {'  '.join(row_vals)}")

    # Plot
    plot_adjacency_comparison(
        learned,
        cutoff=0.1,
        save_path=base_path / 'checkpoints_v2' / 'learned_adjacency_v2.png'
    )


if __name__ == '__main__':
    main()
