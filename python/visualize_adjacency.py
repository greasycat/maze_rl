"""
Visualize the learned adjacency matrix from the trained model.
"""

import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from model import MultiTaskGNN


def load_model(checkpoint_path: str, device: torch.device) -> MultiTaskGNN:
    """Load trained model from checkpoint."""
    model = MultiTaskGNN(num_objects=7, hidden_dim=64, max_seq_len=100).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def get_average_adjacency(
    model: MultiTaskGNN,
    data_path: str,
    device: torch.device,
    num_samples: int = 100
) -> np.ndarray:
    """Get average adjacency matrix over multiple samples."""
    data = np.load(data_path)
    obj_ids = torch.from_numpy(data['obj_ids']).long().to(device)
    flags = torch.from_numpy(data['flags']).long().to(device)
    dists = torch.from_numpy(data['dists']).float().to(device)

    # Use subset of data
    num_samples = min(num_samples, len(obj_ids))
    obj_ids = obj_ids[:num_samples]
    flags = flags[:num_samples]
    dists = dists[:num_samples]

    with torch.no_grad():
        _, _, _, adjacency = model(obj_ids, flags, dists)

    # Average over batch and return
    # adjacency shape: [batch, 1, seq_len, seq_len] or [batch, seq_len, seq_len]
    avg_adj = adjacency.mean(dim=0).cpu().numpy()
    if avg_adj.ndim == 3:
        avg_adj = avg_adj[0]
    return avg_adj


def get_landmark_adjacency(
    model: MultiTaskGNN,
    data_path: str,
    device: torch.device,
    num_samples: int = 200
) -> np.ndarray:
    """
    Compute adjacency between landmarks by aggregating attention weights.
    For each pair of landmarks (i, j), compute average attention when
    landmark j attends to landmark i.
    """
    data = np.load(data_path)
    obj_ids = torch.from_numpy(data['obj_ids']).long().to(device)
    flags = torch.from_numpy(data['flags']).long().to(device)
    dists = torch.from_numpy(data['dists']).float().to(device)

    num_samples = min(num_samples, len(obj_ids))

    # Accumulate attention weights between landmarks
    landmark_attention = np.zeros((7, 7))
    landmark_counts = np.zeros((7, 7))

    with torch.no_grad():
        for i in range(num_samples):
            _, _, _, adjacency = model(
                obj_ids[i:i+1], flags[i:i+1], dists[i:i+1]
            )
            # adjacency shape: [batch, 1, seq_len, seq_len] or [batch, seq_len, seq_len]
            adj = adjacency[0].cpu().numpy()
            if adj.ndim == 3:
                adj = adj[0]  # Remove extra dimension: [seq_len, seq_len]
            seq_obj = obj_ids[i].cpu().numpy()

            # For each position pair, accumulate attention
            seq_len = len(seq_obj)
            for t in range(seq_len):
                for s in range(t + 1):  # Causal: only attend to past
                    from_landmark = int(seq_obj[s])
                    to_landmark = int(seq_obj[t])
                    attention_weight = float(adj[t, s])
                    landmark_attention[to_landmark, from_landmark] += attention_weight
                    landmark_counts[to_landmark, from_landmark] += 1

    # Average
    mask = landmark_counts > 0
    landmark_attention[mask] /= landmark_counts[mask]

    return landmark_attention


def plot_adjacency_graph(
    adjacency: np.ndarray,
    cutoff: float = 0.1,
    save_path: str = None,
    title: str = "Learned Adjacency Graph"
):
    """Plot adjacency matrix as a graph with network layout."""
    landmarks = list('ABCDEFG')

    # Ground truth positions (for reference layout)
    positions = {
        'A': (0, 4), 'B': (4, 4),
        'C': (0, 0), 'D': (4, 0),
        'E': (0, 2), 'F': (2, 2), 'G': (4, 2)
    }

    # Ground truth connectivity
    gt_edges = [
        ('A', 'B'), ('A', 'E'),
        ('B', 'G'),
        ('C', 'D'), ('C', 'E'),
        ('D', 'G'),
        ('E', 'F'),
        ('F', 'G'),
    ]

    # Create graph from learned adjacency
    G = nx.DiGraph()
    G.add_nodes_from(landmarks)

    # Add edges above cutoff
    for i, from_lm in enumerate(landmarks):
        for j, to_lm in enumerate(landmarks):
            weight = adjacency[j, i]  # attention from i to j
            if weight > cutoff and i != j:
                G.add_edge(from_lm, to_lm, weight=weight)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Adjacency matrix heatmap
    ax1 = axes[0]
    im = ax1.imshow(adjacency, cmap='viridis', vmin=0, vmax=1)
    ax1.set_xticks(range(7))
    ax1.set_yticks(range(7))
    ax1.set_xticklabels(landmarks)
    ax1.set_yticklabels(landmarks)
    ax1.set_xlabel('From Landmark')
    ax1.set_ylabel('To Landmark')
    ax1.set_title('Learned Adjacency Matrix')
    plt.colorbar(im, ax=ax1, label='Attention Weight')

    # Add values in cells
    for i in range(7):
        for j in range(7):
            val = adjacency[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=8)

    # Plot 2: Ground truth graph
    ax2 = axes[1]
    GT = nx.Graph()
    GT.add_nodes_from(landmarks)
    GT.add_edges_from(gt_edges)

    # Node colors
    node_colors = ['#ff6b6b' if lm in 'ABCD' else '#4ecdc4' for lm in landmarks]

    nx.draw(GT, pos=positions, ax=ax2, with_labels=True,
            node_color=node_colors, node_size=800, font_size=14,
            font_weight='bold', edge_color='gray', width=2)
    ax2.set_title('Ground Truth Connectivity')

    # Plot 3: Learned graph
    ax3 = axes[2]

    # Get edge weights for coloring
    edges = list(G.edges(data=True))
    if edges:
        edge_weights = [d['weight'] for _, _, d in edges]
        edge_colors = plt.cm.Reds(np.array(edge_weights))

        # Draw nodes
        nx.draw_networkx_nodes(G, pos=positions, ax=ax3,
                              node_color=node_colors, node_size=800)
        nx.draw_networkx_labels(G, pos=positions, ax=ax3,
                               font_size=14, font_weight='bold')

        # Draw edges with weights
        for (u, v, d), color in zip(edges, edge_colors):
            nx.draw_networkx_edges(
                G, pos=positions, ax=ax3,
                edgelist=[(u, v)],
                edge_color=[color],
                width=2 + d['weight'] * 3,
                alpha=0.7,
                arrows=True,
                arrowsize=20,
                connectionstyle='arc3,rad=0.1'
            )

        # Add edge labels
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos=positions, ax=ax3,
                                     edge_labels=edge_labels, font_size=8)
    else:
        nx.draw(G, pos=positions, ax=ax3, with_labels=True,
                node_color=node_colors, node_size=800, font_size=14,
                font_weight='bold')

    ax3.set_title(f'Learned Graph (cutoff={cutoff})')

    # Add legend
    corner_patch = mpatches.Patch(color='#ff6b6b', label='Corner (A,B,C,D)')
    middle_patch = mpatches.Patch(color='#4ecdc4', label='Middle (E,F,G)')
    ax3.legend(handles=[corner_patch, middle_patch], loc='upper left')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved graph to {save_path}")

    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_path = Path(__file__).parent
    checkpoint_path = base_path / 'checkpoints' / 'best_model.pt'
    data_path = base_path / 'data' / 'training_data.npz'

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return

    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path, device)

    # Compute landmark-level adjacency
    print("Computing landmark adjacency matrix...")
    landmark_adj = get_landmark_adjacency(model, data_path, device, num_samples=500)

    print("\nLearned Landmark Adjacency Matrix:")
    landmarks = list('ABCDEFG')
    print("     " + "  ".join(f"{lm:>5}" for lm in landmarks))
    for i, lm in enumerate(landmarks):
        row = "  ".join(f"{landmark_adj[i, j]:5.2f}" for j in range(7))
        print(f"{lm}:   {row}")

    # Plot
    plot_adjacency_graph(
        landmark_adj,
        cutoff=0.1,
        save_path=base_path / 'checkpoints' / 'learned_adjacency_graph.png',
        title='Learned Spatial Adjacency from Navigation Sequences'
    )


if __name__ == '__main__':
    main()
