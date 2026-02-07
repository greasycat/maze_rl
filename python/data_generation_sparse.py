"""
Data generation with VERY sparse human-like data.

Simulates a person who only navigated the maze a handful of times,
resulting in incomplete coverage of the graph.
"""

import random
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from data_generation_human import (
    GridSpace, HumanAgent, prepare_training_data,
    pad_sequences, plot_sequence_stats
)


def generate_sparse_dataset(
    num_sequences: int = 20,
    min_steps: int = 2,
    max_steps: int = 3,
    seed: int = None
) -> tuple[list[list[str]], GridSpace]:
    """Generate very sparse training data."""
    if seed is not None:
        random.seed(seed)

    grid = GridSpace()
    agent = HumanAgent(grid)

    sequences = []
    for i in range(num_sequences):
        agent_seed = seed + i if seed is not None else None
        if agent_seed is not None:
            random.seed(agent_seed)

        sequence = agent.generate_sequence(min_steps, max_steps)
        sequences.append(sequence)

    return sequences, grid


def analyze_coverage(sequences: list[list[str]], grid: GridSpace) -> dict:
    """Analyze which edges are covered in the training data."""
    # Count edge occurrences
    edge_counts = {}
    for lm in grid.landmarks:
        for neighbor in grid.get_neighbors(lm):
            edge_counts[(lm, neighbor)] = 0

    for seq in sequences:
        for i in range(len(seq) - 1):
            from_lm = seq[i]
            to_lm = seq[i + 1]
            if (from_lm, to_lm) in edge_counts:
                edge_counts[(from_lm, to_lm)] += 1

    # Count landmark visits
    landmark_counts = {lm: 0 for lm in grid.landmarks}
    for seq in sequences:
        for lm in seq:
            landmark_counts[lm] += 1

    covered_edges = sum(1 for c in edge_counts.values() if c > 0)
    total_edges = len(edge_counts)

    return {
        'edge_counts': edge_counts,
        'landmark_counts': landmark_counts,
        'covered_edges': covered_edges,
        'total_edges': total_edges,
        'coverage_ratio': covered_edges / total_edges,
    }


def plot_coverage(coverage: dict, save_path: str = None):
    """Visualize edge coverage."""
    landmarks = list('ABCDEFG')
    positions = {
        'A': (0, 4), 'B': (4, 4), 'C': (0, 0), 'D': (4, 0),
        'E': (0, 2), 'F': (2, 2), 'G': (4, 2)
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Edge coverage heatmap
    ax = axes[0]
    edge_matrix = np.zeros((7, 7))
    for (from_lm, to_lm), count in coverage['edge_counts'].items():
        i = landmarks.index(from_lm)
        j = landmarks.index(to_lm)
        edge_matrix[i, j] = count

    max_count = max(coverage['edge_counts'].values()) if coverage['edge_counts'] else 1
    im = ax.imshow(edge_matrix, cmap='YlOrRd', vmin=0, vmax=max(max_count, 1))
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(landmarks)
    ax.set_yticklabels(landmarks)
    ax.set_xlabel('To')
    ax.set_ylabel('From')
    ax.set_title(f'Edge Visit Counts\n({coverage["covered_edges"]}/{coverage["total_edges"]} edges covered)')
    plt.colorbar(im, ax=ax, label='Count')

    for i in range(7):
        for j in range(7):
            val = edge_matrix[i, j]
            if val > 0:
                ax.text(j, i, f'{int(val)}', ha='center', va='center',
                        color='white' if val > max_count / 2 else 'black',
                        fontsize=10, fontweight='bold')

    # Graph visualization with edge thickness
    ax = axes[1]
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(landmarks)

    node_colors = ['#ff6b6b' if lm in 'ABCD' else '#4ecdc4' for lm in landmarks]

    # Add all ground truth edges
    gt_connectivity = {
        'A': ['B', 'E'], 'B': ['A', 'G'], 'C': ['E', 'D'],
        'D': ['C', 'G'], 'E': ['A', 'C', 'F'], 'F': ['E', 'G'],
        'G': ['B', 'D', 'F'],
    }

    for lm, neighbors in gt_connectivity.items():
        for neighbor in neighbors:
            count = coverage['edge_counts'].get((lm, neighbor), 0)
            G.add_edge(lm, neighbor, weight=count)

    nx.draw_networkx_nodes(G, pos=positions, ax=ax,
                           node_color=node_colors, node_size=800)
    nx.draw_networkx_labels(G, pos=positions, ax=ax,
                            font_size=12, font_weight='bold')

    # Draw edges with varying colors based on coverage
    for (u, v, d) in G.edges(data=True):
        count = d['weight']
        if count == 0:
            color = 'lightgray'
            style = 'dashed'
            width = 1
        else:
            color = plt.cm.Reds(min(count / max_count, 1.0))
            style = 'solid'
            width = 1 + count

        nx.draw_networkx_edges(
            G, pos=positions, ax=ax,
            edgelist=[(u, v)],
            edge_color=[color],
            width=width,
            style=style,
            arrows=True, arrowsize=15,
            connectionstyle='arc3,rad=0.1'
        )

    ax.set_title(f'Edge Coverage (gray=unseen)\nCoverage: {coverage["coverage_ratio"]*100:.1f}%')

    plt.suptitle('Sparse Data - Training Coverage Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved coverage plot to {save_path}")

    plt.show()

    return fig


def save_dataset(
    sequences: list[list[str]],
    grid: GridSpace,
    output_dir: Path
):
    """Save sparse dataset."""
    output_dir.mkdir(exist_ok=True)

    data = prepare_training_data(sequences, grid)
    padded_inputs, lengths = pad_sequences(data['input_sequences'])
    targets = np.array(data['targets'], dtype=np.float32)

    np.savez(
        output_dir / 'training_data_sparse.npz',
        input_sequences=padded_inputs,
        sequence_lengths=lengths,
        targets=targets,
    )
    print(f"Saved: {output_dir / 'training_data_sparse.npz'}")
    print(f"  Shape: {padded_inputs.shape[0]} examples")

    with open(output_dir / 'sequences_sparse.json', 'w') as f:
        json.dump(sequences, f, indent=2)


def main():
    output_dir = Path(__file__).parent / 'data'

    # Very sparse parameters
    NUM_SEQUENCES = 20   # Only 20 navigation experiences
    MIN_STEPS = 2
    MAX_STEPS = 3        # Very short sequences

    print("=" * 50)
    print("Generating SPARSE training data")
    print("=" * 50)
    print(f"  Number of sequences: {NUM_SEQUENCES}")
    print(f"  Sequence length: {MIN_STEPS}-{MAX_STEPS} steps")
    print()

    sequences, grid = generate_sparse_dataset(
        num_sequences=NUM_SEQUENCES,
        min_steps=MIN_STEPS,
        max_steps=MAX_STEPS,
        seed=42
    )

    # Analyze coverage
    coverage = analyze_coverage(sequences, grid)
    print(f"\nCoverage Analysis:")
    print(f"  Edges covered: {coverage['covered_edges']}/{coverage['total_edges']}")
    print(f"  Coverage ratio: {coverage['coverage_ratio']*100:.1f}%")

    print("\nUncovered edges:")
    for (from_lm, to_lm), count in coverage['edge_counts'].items():
        if count == 0:
            print(f"  {from_lm} → {to_lm}")

    # Save dataset
    save_dataset(sequences, grid, output_dir)

    # Plot coverage
    plot_coverage(coverage, save_path=output_dir / 'coverage_sparse.png')

    # Print sequences
    print("\nAll sequences:")
    for i, seq in enumerate(sequences):
        print(f"  {i + 1}: {' -> '.join(seq)}")

    # Comparison table
    print("\n" + "=" * 60)
    print("Dataset Comparison:")
    print("=" * 60)
    print(f"  {'Parameter':<25} {'Sparse':<12} {'Human':<12} {'Full':<12}")
    print(f"  {'-' * 60}")
    print(f"  {'Sequences':<25} {NUM_SEQUENCES:<12} {100:<12} {1000:<12}")
    print(f"  {'Steps per sequence':<25} {f'{MIN_STEPS}-{MAX_STEPS}':<12} {'2-4':<12} {'8':<12}")

    data = prepare_training_data(sequences, grid)
    print(f"  {'Training examples':<25} {len(data['input_sequences']):<12} {195:<12} {7000:<12}")
    cov_pct = f"{coverage['coverage_ratio']*100:.0f}%"
    print(f"  {'Edge coverage':<25} {cov_pct:<12} {'100%':<12} {'100%':<12}")


if __name__ == '__main__':
    main()
