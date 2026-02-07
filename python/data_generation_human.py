"""
Data generation script simulating human-like navigation behavior.

Key differences from full dataset:
- Limited number of experiences (humans don't navigate thousands of times)
- Shorter sequences (working memory ~5-7 items)
- Variable sequence lengths (humans don't always complete full paths)
"""

import random
import json
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


@dataclass
class Landmark:
    name: str
    row: int
    col: int
    is_corner: bool = False


@dataclass
class GridSpace:
    """Defines the 5x5 grid space with landmarks and walls."""

    size: int = 5
    num_landmarks: int = 7
    landmarks: dict = field(default_factory=dict)
    walls: set = field(default_factory=set)
    connectivity: dict = field(default_factory=dict)
    landmark_to_idx: dict = field(default_factory=dict)
    idx_to_landmark: dict = field(default_factory=dict)

    def __post_init__(self):
        self.landmarks = {
            'A': Landmark('A', 0, 0, is_corner=True),
            'B': Landmark('B', 0, 4, is_corner=True),
            'C': Landmark('C', 4, 0, is_corner=True),
            'D': Landmark('D', 4, 4, is_corner=True),
            'E': Landmark('E', 2, 0, is_corner=False),
            'F': Landmark('F', 2, 2, is_corner=False),
            'G': Landmark('G', 2, 4, is_corner=False),
        }

        self.walls = {
            (1, 1), (1, 2), (1, 3),
            (3, 1), (3, 2), (3, 3),
        }

        self.connectivity = {
            'A': ['B', 'E'],
            'B': ['A', 'G'],
            'C': ['E', 'D'],
            'D': ['C', 'G'],
            'E': ['A', 'C', 'F'],
            'F': ['E', 'G'],
            'G': ['B', 'D', 'F'],
        }

        self.landmark_to_idx = {name: i for i, name in enumerate('ABCDEFG')}
        self.idx_to_landmark = {i: name for name, i in self.landmark_to_idx.items()}

    def get_distance(self, from_landmark: str, to_landmark: str) -> int:
        lm1 = self.landmarks[from_landmark]
        lm2 = self.landmarks[to_landmark]
        return abs(lm1.row - lm2.row) + abs(lm1.col - lm2.col)

    def get_neighbors(self, landmark: str) -> list[str]:
        return self.connectivity.get(landmark, [])

    def get_neighbor_target(self, current_landmark: str) -> np.ndarray:
        target = np.zeros((self.num_landmarks, 3), dtype=np.float32)
        neighbors = self.get_neighbors(current_landmark)

        for neighbor in neighbors:
            idx = self.landmark_to_idx[neighbor]
            distance = self.get_distance(current_landmark, neighbor)
            is_corner = 1.0 if self.landmarks[neighbor].is_corner else 0.0

            target[idx, 0] = 1.0
            target[idx, 1] = distance
            target[idx, 2] = is_corner

        return target


class HumanAgent:
    """
    Agent that simulates human-like navigation behavior.

    Characteristics:
    - Variable sequence lengths (not always completing full paths)
    - Tendency to revisit familiar locations
    - Occasional "mistakes" or suboptimal choices
    """

    def __init__(self, grid: GridSpace, seed: int = None):
        self.grid = grid
        self.current_position: str = None
        self.visited_set: set = set()
        self.history: list[str] = []

        if seed is not None:
            random.seed(seed)

    def reset(self):
        self.current_position = None
        self.visited_set = set()
        self.history = []

    def start_random(self) -> str:
        landmarks = list(self.grid.landmarks.keys())
        self.current_position = random.choice(landmarks)
        self.history.append(self.current_position)
        return self.current_position

    def move(self) -> str:
        neighbors = self.grid.get_neighbors(self.current_position)
        unvisited = [n for n in neighbors if n not in self.visited_set]

        if unvisited:
            next_pos = random.choice(unvisited)
        else:
            self.visited_set = set()
            next_pos = random.choice(neighbors)

        self.visited_set.add(self.current_position)
        self.current_position = next_pos
        self.history.append(next_pos)

        return next_pos

    def generate_sequence(self, min_steps: int = 2, max_steps: int = 4) -> list[str]:
        """Generate a sequence with variable length (human-like)."""
        self.reset()
        self.start_random()

        # Variable number of steps (simulating incomplete explorations)
        num_steps = random.randint(min_steps, max_steps)

        for _ in range(num_steps - 1):
            self.move()

        return self.history


def generate_human_dataset(
    num_sequences: int = 100,
    min_steps: int = 2,
    max_steps: int = 4,
    seed: int = None
) -> tuple[list[list[str]], GridSpace]:
    """
    Generate human-like training data.

    Args:
        num_sequences: Number of navigation experiences (default: 100, simulating
                      limited human experience)
        min_steps: Minimum sequence length (default: 2)
        max_steps: Maximum sequence length (default: 4, simulating working memory limits)
        seed: Random seed for reproducibility
    """
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


def prepare_training_data(
    sequences: list[list[str]],
    grid: GridSpace
) -> dict:
    """Prepare training data in branching predictor format."""
    input_sequences = []
    targets = []

    for sequence in sequences:
        for t in range(len(sequence) - 1):
            input_seq = [grid.landmark_to_idx[lm] for lm in sequence[:t + 1]]
            input_sequences.append(input_seq)

            current_landmark = sequence[t]
            target = grid.get_neighbor_target(current_landmark)
            targets.append(target)

    return {
        'input_sequences': input_sequences,
        'targets': targets,
    }


def pad_sequences(sequences: list[list[int]], max_len: int = None) -> tuple[np.ndarray, np.ndarray]:
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    padded = np.zeros((len(sequences), max_len), dtype=np.int64)
    lengths = np.array([len(seq) for seq in sequences], dtype=np.int64)

    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq

    return padded, lengths


def save_dataset(
    sequences: list[list[str]],
    grid: GridSpace,
    output_dir: Path,
    prefix: str = "human"
):
    """Save dataset to files."""
    output_dir.mkdir(exist_ok=True)

    data = prepare_training_data(sequences, grid)
    padded_inputs, lengths = pad_sequences(data['input_sequences'])
    targets = np.array(data['targets'], dtype=np.float32)

    np.savez(
        output_dir / f'training_data_{prefix}.npz',
        input_sequences=padded_inputs,
        sequence_lengths=lengths,
        targets=targets,
    )
    print(f"Saved numpy arrays to {output_dir / f'training_data_{prefix}.npz'}")
    print(f"  input_sequences shape: {padded_inputs.shape}")
    print(f"  sequence_lengths shape: {lengths.shape}")
    print(f"  targets shape: {targets.shape}")

    with open(output_dir / f'sequences_{prefix}.json', 'w') as f:
        json.dump(sequences, f, indent=2)
    print(f"Saved sequences to {output_dir / f'sequences_{prefix}.json'}")

    # Save config
    config = {
        'num_sequences': len(sequences),
        'total_examples': len(data['input_sequences']),
        'max_seq_len': int(padded_inputs.shape[1]),
        'description': 'Human-like limited training data'
    }
    with open(output_dir / f'config_{prefix}.json', 'w') as f:
        json.dump(config, f, indent=2)


def plot_sequence_stats(sequences: list[list[str]], save_path: str = None):
    """Plot statistics about the generated sequences."""
    lengths = [len(seq) for seq in sequences]

    # Count landmark visits
    landmark_counts = {lm: 0 for lm in 'ABCDEFG'}
    for seq in sequences:
        for lm in seq:
            landmark_counts[lm] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Sequence length distribution
    axes[0].hist(lengths, bins=range(min(lengths), max(lengths) + 2),
                 edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Sequence Length Distribution\n(n={len(sequences)}, mean={np.mean(lengths):.1f})')

    # Landmark visit frequency
    landmarks = list(landmark_counts.keys())
    counts = list(landmark_counts.values())
    colors = ['#ff6b6b' if lm in 'ABCD' else '#4ecdc4' for lm in landmarks]
    axes[1].bar(landmarks, counts, color=colors, edgecolor='black')
    axes[1].set_xlabel('Landmark')
    axes[1].set_ylabel('Visit Count')
    axes[1].set_title('Landmark Visit Frequency')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved stats plot to {save_path}")

    plt.show()


def main():
    output_dir = Path(__file__).parent / 'data'

    # Human-like parameters
    NUM_SEQUENCES = 100  # Limited experiences (vs 1000 in full dataset)
    MIN_STEPS = 2        # Minimum path length
    MAX_STEPS = 4        # Maximum path length (working memory limit)

    print("=" * 50)
    print("Generating HUMAN-LIKE training data")
    print("=" * 50)
    print(f"  Number of sequences: {NUM_SEQUENCES}")
    print(f"  Sequence length: {MIN_STEPS}-{MAX_STEPS} steps")
    print()

    sequences, grid = generate_human_dataset(
        num_sequences=NUM_SEQUENCES,
        min_steps=MIN_STEPS,
        max_steps=MAX_STEPS,
        seed=42
    )

    save_dataset(sequences, grid, output_dir, prefix="human")

    # Plot statistics
    print("\nPlotting sequence statistics...")
    plot_sequence_stats(
        sequences,
        save_path=output_dir / 'stats_human.png'
    )

    # Print sample data
    print("\nSample sequences:")
    for i, seq in enumerate(sequences[:5]):
        print(f"  {i + 1}: {' -> '.join(seq)}")

    # Compare with full dataset
    print("\n" + "=" * 50)
    print("Comparison with full dataset:")
    print("=" * 50)
    print(f"  {'Parameter':<25} {'Human-like':<15} {'Full (v2)':<15}")
    print(f"  {'-' * 55}")
    print(f"  {'Sequences':<25} {NUM_SEQUENCES:<15} {1000:<15}")
    print(f"  {'Steps per sequence':<25} {f'{MIN_STEPS}-{MAX_STEPS}':<15} {'8':<15}")

    data = prepare_training_data(sequences, grid)
    print(f"  {'Training examples':<25} {len(data['input_sequences']):<15} {7000:<15}")


if __name__ == '__main__':
    main()
