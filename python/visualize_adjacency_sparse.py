"""
Visualize the learned adjacency from the SPARSE data model.
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from model_v2 import BranchingPredictor


def load_model(checkpoint_path: str, device: torch.device) -> BranchingPredictor:
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
    landmarks = list('ABCDEFG')
    num_landmarks = len(landmarks)

    adjacency = np.zeros((num_landmarks, num_landmarks))
    distances = np.zeros((num_landmarks, num_landmarks))

    with torch.no_grad():
        for i in range(num_landmarks):
            input_ids = torch.tensor([[i]], dtype=torch.long, device=device)
            lengths = torch.tensor([1], dtype=torch.long, device=device)
            preds = model(input_ids, lengths)

            adjacency[i, :] = torch.sigmoid(preds[0, :, 0]).cpu().numpy()
            distances[i, :] = preds[0, :, 1].cpu().numpy()

    return {'adjacency': adjacency, 'distances': distances}


def plot_comparison(learned: dict, cutoff: float = 0.5, save_path: str = None):
    landmarks = list('ABCDEFG')

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
    ax.set_title('Learned Link Probabilities\n(Sparse Data - 29 examples)')
    plt.colorbar(im, ax=ax, label='Probability')
    for i in range(7):
        for j in range(7):
            val = learned['adjacency'][i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=9)

    # Highlight F→G (unseen edge)
    f_idx, g_idx = landmarks.index('F'), landmarks.index('G')
    rect = plt.Rectangle((g_idx - 0.5, f_idx - 0.5), 1, 1,
                         fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(rect)
    ax.annotate('Unseen\nedge', xy=(g_idx, f_idx), xytext=(g_idx + 1.5, f_idx),
                fontsize=9, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

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

    # Ground truth graph
    ax = axes[1, 0]
    node_colors = ['#ff6b6b' if lm in 'ABCD' else '#4ecdc4' for lm in landmarks]

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
            # Highlight missing F→G differently
            if u == 'F' and v == 'G':
                ec = 'red'
                style = 'dashed'
            else:
                ec = color
                style = 'solid'

            nx.draw_networkx_edges(
                G_learned, pos=positions, ax=ax,
                edgelist=[(u, v)],
                edge_color=[ec] if style == 'dashed' else [color],
                width=1 + w * 3,
                style=style,
                arrows=True, arrowsize=15,
                connectionstyle='arc3,rad=0.1'
            )

    ax.set_title(f'Learned Graph (cutoff={cutoff})')

    # Analysis
    ax = axes[1, 2]
    ax.axis('off')

    pred_links = learned['adjacency'] > cutoff
    gt_links = gt_adj > 0.5

    true_pos = (pred_links & gt_links).sum()
    false_pos = (pred_links & ~gt_links).sum()
    false_neg = (~pred_links & gt_links).sum()

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    missing = []
    extra = []
    for i, from_lm in enumerate(landmarks):
        for j, to_lm in enumerate(landmarks):
            if gt_adj[i, j] > 0.5 and learned['adjacency'][i, j] <= cutoff:
                missing.append(f"{from_lm}→{to_lm}")
            if gt_adj[i, j] < 0.5 and learned['adjacency'][i, j] > cutoff:
                extra.append(f"{from_lm}→{to_lm}")

    # F→G analysis
    fg_prob = learned['adjacency'][landmarks.index('F'), landmarks.index('G')]

    stats_text = f"""
    Sparse Data Analysis
    ====================

    Training Data:
    • Only 29 examples (vs 7000 full)
    • Edge F→G never seen in training

    Link Prediction (cutoff={cutoff}):
    • Precision: {precision:.3f}
    • Recall: {recall:.3f}
    • F1 Score: {f1:.3f}

    Missing Edges ({len(missing)}):
    {', '.join(missing) if missing else 'None'}

    Extra Edges ({len(extra)}):
    {', '.join(extra) if extra else 'None'}

    Unseen Edge F→G:
    • Predicted probability: {fg_prob:.3f}
    • Status: {'RECOVERED!' if fg_prob > 0.5 else 'NOT learned'}
    """

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    corner_patch = mpatches.Patch(color='#ff6b6b', label='Corner')
    middle_patch = mpatches.Patch(color='#4ecdc4', label='Middle')
    fig.legend(handles=[corner_patch, middle_patch], loc='lower center', ncol=2)

    plt.suptitle('Sparse Data (20 sequences) - Can the model generalize?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return {'f1': f1, 'missing': missing, 'extra': extra, 'fg_prob': fg_prob}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_path = Path(__file__).parent
    checkpoint_path = base_path / 'checkpoints_sparse' / 'best_model.pt'

    if not checkpoint_path.exists():
        print("Train model first with train_sparse.py")
        return

    print("Loading sparse model...")
    model = load_model(checkpoint_path, device)

    print("Computing adjacency...")
    learned = get_learned_adjacency(model, device)

    landmarks = list('ABCDEFG')
    print("\nLearned Link Probabilities:")
    print("     " + "  ".join(f"{lm:>6}" for lm in landmarks))
    for i, lm in enumerate(landmarks):
        row = "  ".join(f"{learned['adjacency'][i, j]:6.3f}" for j in range(7))
        print(f"{lm}:   {row}")

    stats = plot_comparison(
        learned,
        cutoff=0.5,
        save_path=base_path / 'checkpoints_sparse' / 'learned_adjacency_sparse.png'
    )

    print(f"\nF→G (unseen edge) probability: {stats['fg_prob']:.3f}")
    print(f"Model {'CAN' if stats['fg_prob'] > 0.5 else 'CANNOT'} generalize to unseen edges")


if __name__ == '__main__':
    main()
