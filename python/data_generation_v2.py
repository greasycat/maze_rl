"""
Data generation script for maze RL training data (v2).
Generates sequences with multi-label targets for branching prediction.

Target format: [M_objects, 3] where:
  - Channel 0: Link existence (0 or 1)
  - Channel 1: Distance (real value, 0 if no link)
  - Channel 2: Importance/Corner flag (0 or 1, 0 if no link)
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
        # Define landmarks: 4 corners + 3 middle row
        self.landmarks = {
            'A': Landmark('A', 0, 0, is_corner=True),
            'B': Landmark('B', 0, 4, is_corner=True),
            'C': Landmark('C', 4, 0, is_corner=True),
            'D': Landmark('D', 4, 4, is_corner=True),
            'E': Landmark('E', 2, 0, is_corner=False),
            'F': Landmark('F', 2, 2, is_corner=False),
            'G': Landmark('G', 2, 4, is_corner=False),
        }

        # Define walls (rows 1 and 3, columns 1-3)
        self.walls = {
            (1, 1), (1, 2), (1, 3),
            (3, 1), (3, 2), (3, 3),
        }

        # Define connectivity graph
        self.connectivity = {
            'A': ['B', 'E'],
            'B': ['A', 'G'],
            'C': ['E', 'D'],
            'D': ['C', 'G'],
            'E': ['A', 'C', 'F'],
            'F': ['E', 'G'],
            'G': ['B', 'D', 'F'],
        }

        # Create index mappings
        self.landmark_to_idx = {name: i for i, name in enumerate('ABCDEFG')}
        self.idx_to_landmark = {i: name for name, i in self.landmark_to_idx.items()}

    def get_distance(self, from_landmark: str, to_landmark: str) -> int:
        """Calculate Manhattan distance between two landmarks."""
        lm1 = self.landmarks[from_landmark]
        lm2 = self.landmarks[to_landmark]
        return abs(lm1.row - lm2.row) + abs(lm1.col - lm2.col)

    def get_neighbors(self, landmark: str) -> list[str]:
        """Get connected neighbors of a landmark."""
        return self.connectivity.get(landmark, [])

    def get_neighbor_target(self, current_landmark: str) -> np.ndarray:
        """
        Get dense target vector for all possible neighbors from current landmark.

        Returns:
            target: [num_landmarks, 3] array where:
                - [:, 0] = link exists (0 or 1)
                - [:, 1] = distance (0 if no link)
                - [:, 2] = is_corner (0 if no link)
        """
        target = np.zeros((self.num_landmarks, 3), dtype=np.float32)
        neighbors = self.get_neighbors(current_landmark)

        for neighbor in neighbors:
            idx = self.landmark_to_idx[neighbor]
            distance = self.get_distance(current_landmark, neighbor)
            is_corner = 1.0 if self.landmarks[neighbor].is_corner else 0.0

            target[idx, 0] = 1.0  # Link exists
            target[idx, 1] = distance
            target[idx, 2] = is_corner

        return target


class Agent:
    """Agent that navigates through the grid space."""

    def __init__(self, grid: GridSpace, seed: int = None):
        self.grid = grid
        self.current_position: str = None
        self.visited_set: set = set()
        self.history: list[str] = []  # Sequence of visited landmarks

        if seed is not None:
            random.seed(seed)

    def reset(self):
        """Reset agent state for a new sequence."""
        self.current_position = None
        self.visited_set = set()
        self.history = []

    def start_random(self) -> str:
        """Pick a random starting landmark."""
        landmarks = list(self.grid.landmarks.keys())
        self.current_position = random.choice(landmarks)
        self.history.append(self.current_position)
        return self.current_position

    def move(self) -> str:
        """Move to the next landmark according to the visiting rules."""
        neighbors = self.grid.get_neighbors(self.current_position)

        # Filter out recently visited neighbors if possible
        unvisited = [n for n in neighbors if n not in self.visited_set]

        if unvisited:
            next_pos = random.choice(unvisited)
        else:
            # All neighbors visited, reset visited set and choose any neighbor
            self.visited_set = set()
            next_pos = random.choice(neighbors)

        # Add current position to visited set
        self.visited_set.add(self.current_position)
        self.current_position = next_pos
        self.history.append(next_pos)

        return next_pos

    def generate_sequence(self, num_steps: int = 8) -> list[str]:
        """Generate a full sequence of visits."""
        self.reset()
        self.start_random()

        for _ in range(num_steps - 1):
            self.move()

        return self.history


def generate_dataset(
    num_sequences: int = 1000,
    steps_per_sequence: int = 8,
    seed: int = None
) -> tuple[list[list[str]], GridSpace]:
    """Generate multiple sequences of training data."""
    if seed is not None:
        random.seed(seed)

    grid = GridSpace()
    agent = Agent(grid)

    sequences = []
    for i in range(num_sequences):
        agent_seed = seed + i if seed is not None else None
        if agent_seed is not None:
            random.seed(agent_seed)

        sequence = agent.generate_sequence(steps_per_sequence)
        sequences.append(sequence)

    return sequences, grid


def prepare_training_data(
    sequences: list[list[str]],
    grid: GridSpace
) -> dict:
    """
    Prepare training data in the format required by the branching predictor.

    For each position in a sequence, the input is the history up to that point,
    and the target is the dense neighbor vector from the current position.

    Returns:
        dict with:
            - input_sequences: List of input landmark index sequences
            - targets: List of [num_landmarks, 3] target arrays
            - sequence_lengths: Length of each input sequence
    """
    input_sequences = []
    targets = []

    for sequence in sequences:
        # For each position (except the last), create a training example
        for t in range(len(sequence) - 1):
            # Input: history up to position t (inclusive)
            input_seq = [grid.landmark_to_idx[lm] for lm in sequence[:t + 1]]
            input_sequences.append(input_seq)

            # Target: neighbors of current position
            current_landmark = sequence[t]
            target = grid.get_neighbor_target(current_landmark)
            targets.append(target)

    return {
        'input_sequences': input_sequences,
        'targets': targets,
    }


def pad_sequences(sequences: list[list[int]], max_len: int = None) -> tuple[np.ndarray, np.ndarray]:
    """Pad sequences to the same length."""
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
    output_dir: Path
):
    """Save dataset to files."""
    output_dir.mkdir(exist_ok=True)

    # Prepare training data
    data = prepare_training_data(sequences, grid)

    # Pad input sequences
    padded_inputs, lengths = pad_sequences(data['input_sequences'])
    targets = np.array(data['targets'], dtype=np.float32)

    # Save numpy arrays
    np.savez(
        output_dir / 'training_data_v2.npz',
        input_sequences=padded_inputs,
        sequence_lengths=lengths,
        targets=targets,
    )
    print(f"Saved numpy arrays to {output_dir / 'training_data_v2.npz'}")
    print(f"  input_sequences shape: {padded_inputs.shape}")
    print(f"  sequence_lengths shape: {lengths.shape}")
    print(f"  targets shape: {targets.shape}")

    # Save raw sequences as JSON
    with open(output_dir / 'sequences_v2.json', 'w') as f:
        json.dump(sequences, f, indent=2)
    print(f"Saved sequences to {output_dir / 'sequences_v2.json'}")

    # Save landmark mapping
    with open(output_dir / 'landmark_mapping.json', 'w') as f:
        json.dump(grid.landmark_to_idx, f, indent=2)

    # Save connectivity for reference
    with open(output_dir / 'connectivity.json', 'w') as f:
        json.dump(grid.connectivity, f, indent=2)


def plot_grid_with_sequence(
    grid: GridSpace,
    sequence: list[str],
    ax: plt.Axes = None,
    title: str = ""
):
    """Plot the grid space with a sequence path."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Draw grid
    for i in range(grid.size + 1):
        ax.axhline(y=i, color='lightgray', linewidth=0.5)
        ax.axvline(x=i, color='lightgray', linewidth=0.5)

    # Draw walls
    for row, col in grid.walls:
        rect = mpatches.Rectangle(
            (col, grid.size - 1 - row), 1, 1,
            facecolor='gray', edgecolor='black', alpha=0.7
        )
        ax.add_patch(rect)

    # Draw landmarks
    colors = {'A': 'red', 'B': 'blue', 'C': 'green', 'D': 'orange',
              'E': 'purple', 'F': 'cyan', 'G': 'magenta'}

    for name, lm in grid.landmarks.items():
        x = lm.col + 0.5
        y = grid.size - 1 - lm.row + 0.5

        marker = 's' if lm.is_corner else 'o'
        ax.plot(x, y, marker=marker, markersize=20, color=colors[name],
                markeredgecolor='black', markeredgewidth=2)
        ax.text(x, y, name, ha='center', va='center', fontsize=12,
                fontweight='bold', color='white')

    # Draw sequence path
    if sequence:
        path_x = []
        path_y = []
        for lm_name in sequence:
            lm = grid.landmarks[lm_name]
            path_x.append(lm.col + 0.5)
            path_y.append(grid.size - 1 - lm.row + 0.5)

        # Draw path with arrows
        for i in range(len(path_x) - 1):
            ax.annotate(
                '', xy=(path_x[i + 1], path_y[i + 1]),
                xytext=(path_x[i], path_y[i]),
                arrowprops=dict(
                    arrowstyle='->', color='black',
                    lw=2, connectionstyle='arc3,rad=0.1'
                )
            )

        # Add step numbers
        for i, (x, y) in enumerate(zip(path_x, path_y)):
            ax.text(x + 0.3, y + 0.3, str(i + 1), fontsize=10,
                    color='darkblue', fontweight='bold')

    ax.set_xlim(0, grid.size)
    ax.set_ylim(0, grid.size)
    ax.set_aspect('equal')
    ax.set_title(title)

    return ax


def plot_sequences(
    sequences: list[list[str]],
    grid: GridSpace,
    num_to_plot: int = 10,
    save_path: str = None
):
    """Plot multiple sequences."""
    num_to_plot = min(num_to_plot, len(sequences))
    cols = 5
    rows = (num_to_plot + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if num_to_plot > 1 else [axes]

    for i in range(num_to_plot):
        sequence = sequences[i]
        path_str = ' -> '.join(sequence)
        plot_grid_with_sequence(
            grid, sequence, ax=axes[i],
            title=f"Sequence {i + 1}\n{path_str}"
        )

    for i in range(num_to_plot, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def main():
    """Main function to generate and save training data."""
    output_dir = Path(__file__).parent / 'data'

    # Generate dataset
    print("Generating training data (v2 - branching format)...")
    sequences, grid = generate_dataset(
        num_sequences=1000,
        steps_per_sequence=8,
        seed=42
    )

    # Save dataset
    save_dataset(sequences, grid, output_dir)

    # Plot first 10 sequences
    print("\nPlotting sequences...")
    plot_sequences(
        sequences,
        grid,
        num_to_plot=10,
        save_path=output_dir / 'sample_sequences_v2.png'
    )

    # Print sample data
    print("\nSample training examples:")
    data = prepare_training_data(sequences[:2], grid)

    for i in range(min(5, len(data['input_sequences']))):
        input_seq = data['input_sequences'][i]
        target = data['targets'][i]
        input_landmarks = [grid.idx_to_landmark[idx] for idx in input_seq]

        print(f"\nExample {i + 1}:")
        print(f"  Input (history): {' -> '.join(input_landmarks)}")
        print(f"  Current position: {input_landmarks[-1]}")
        print(f"  Target neighbors:")
        for j in range(grid.num_landmarks):
            if target[j, 0] > 0:
                lm = grid.idx_to_landmark[j]
                print(f"    {lm}: link=1, dist={target[j, 1]:.0f}, corner={target[j, 2]:.0f}")


if __name__ == '__main__':
    main()
