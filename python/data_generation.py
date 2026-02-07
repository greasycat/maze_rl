"""
Data generation script for maze RL training data.
Generates sequences of agent navigation through a 5x5 grid with 7 landmarks.
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
class VisitRecord:
    landmark: str
    distance: int
    is_corner: bool


@dataclass
class GridSpace:
    """Defines the 5x5 grid space with landmarks and walls."""

    size: int = 5
    landmarks: dict = field(default_factory=dict)
    walls: set = field(default_factory=set)
    connectivity: dict = field(default_factory=dict)

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

    def get_distance(self, from_landmark: str, to_landmark: str) -> int:
        """Calculate Manhattan distance between two landmarks."""
        lm1 = self.landmarks[from_landmark]
        lm2 = self.landmarks[to_landmark]
        return abs(lm1.row - lm2.row) + abs(lm1.col - lm2.col)

    def get_neighbors(self, landmark: str) -> list[str]:
        """Get connected neighbors of a landmark."""
        return self.connectivity.get(landmark, [])


class Agent:
    """Agent that navigates through the grid space."""

    def __init__(self, grid: GridSpace, seed: int = None):
        self.grid = grid
        self.current_position: str = None
        self.visited_set: set = set()
        self.sequence: list[VisitRecord] = []

        if seed is not None:
            random.seed(seed)

    def reset(self):
        """Reset agent state for a new sequence."""
        self.current_position = None
        self.visited_set = set()
        self.sequence = []

    def start_random(self) -> str:
        """Pick a random starting landmark."""
        landmarks = list(self.grid.landmarks.keys())
        self.current_position = random.choice(landmarks)
        lm = self.grid.landmarks[self.current_position]
        # First visit has distance 0
        self.sequence.append(VisitRecord(
            landmark=self.current_position,
            distance=0,
            is_corner=lm.is_corner
        ))
        return self.current_position

    def move(self) -> str:
        """Move to the next landmark according to the visiting rules."""
        neighbors = self.grid.get_neighbors(self.current_position)

        # Filter out recently visited neighbors if possible
        unvisited = [n for n in neighbors if n not in self.visited_set]

        if unvisited:
            # Choose from unvisited neighbors
            next_pos = random.choice(unvisited)
        else:
            # All neighbors visited, reset visited set and choose any neighbor
            self.visited_set = set()
            next_pos = random.choice(neighbors)

        # Add current position to visited set
        self.visited_set.add(self.current_position)

        # Calculate distance
        distance = self.grid.get_distance(self.current_position, next_pos)

        # Record the visit
        lm = self.grid.landmarks[next_pos]
        self.sequence.append(VisitRecord(
            landmark=next_pos,
            distance=distance,
            is_corner=lm.is_corner
        ))

        self.current_position = next_pos
        return next_pos

    def generate_sequence(self, num_steps: int = 8) -> list[VisitRecord]:
        """Generate a full sequence of visits."""
        self.reset()
        self.start_random()

        for _ in range(num_steps - 1):
            self.move()

        return self.sequence


def generate_dataset(
    num_sequences: int = 1000,
    steps_per_sequence: int = 8,
    seed: int = None
) -> list[list[VisitRecord]]:
    """Generate multiple sequences of training data."""
    if seed is not None:
        random.seed(seed)

    grid = GridSpace()
    agent = Agent(grid)

    dataset = []
    for i in range(num_sequences):
        # Use different seed for each sequence for reproducibility
        agent_seed = seed + i if seed is not None else None
        if agent_seed is not None:
            random.seed(agent_seed)

        sequence = agent.generate_sequence(steps_per_sequence)
        dataset.append(sequence)

    return dataset


def save_dataset(dataset: list[list[VisitRecord]], filepath: str):
    """Save dataset to a text file in a parseable format."""
    path = Path(filepath)

    # Convert to serializable format
    data = []
    for seq in dataset:
        seq_data = [
            {
                'landmark': r.landmark,
                'distance': r.distance,
                'is_corner': r.is_corner
            }
            for r in seq
        ]
        data.append(seq_data)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(dataset)} sequences to {filepath}")


def load_dataset(filepath: str) -> list[list[VisitRecord]]:
    """Load dataset from a text file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    dataset = []
    for seq_data in data:
        sequence = [
            VisitRecord(
                landmark=r['landmark'],
                distance=r['distance'],
                is_corner=r['is_corner']
            )
            for r in seq_data
        ]
        dataset.append(sequence)

    return dataset


def plot_grid_with_sequence(
    grid: GridSpace,
    sequence: list[VisitRecord],
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
        for record in sequence:
            lm = grid.landmarks[record.landmark]
            path_x.append(lm.col + 0.5)
            path_y.append(grid.size - 1 - lm.row + 0.5)

        # Draw path with arrows
        for i in range(len(path_x) - 1):
            ax.annotate(
                '', xy=(path_x[i+1], path_y[i+1]),
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

    # Add legend
    corner_patch = mpatches.Patch(color='gray', label='Corner landmarks (squares)')
    middle_patch = mpatches.Patch(color='lightblue', label='Middle landmarks (circles)')
    wall_patch = mpatches.Patch(color='gray', alpha=0.7, label='Walls')

    return ax


def plot_sequences(
    dataset: list[list[VisitRecord]],
    num_to_plot: int = 10,
    save_path: str = None
):
    """Plot multiple sequences from the dataset."""
    grid = GridSpace()

    num_to_plot = min(num_to_plot, len(dataset))
    cols = 5
    rows = (num_to_plot + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if num_to_plot > 1 else [axes]

    for i in range(num_to_plot):
        sequence = dataset[i]
        path_str = ' -> '.join([r.landmark for r in sequence])
        plot_grid_with_sequence(
            grid, sequence, ax=axes[i],
            title=f"Sequence {i + 1}\n{path_str}"
        )

    # Hide unused subplots
    for i in range(num_to_plot, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def convert_to_tensors(dataset: list[list[VisitRecord]]) -> dict:
    """Convert dataset to tensor-ready format for model training."""
    landmark_to_idx = {name: i for i, name in enumerate('ABCDEFG')}

    obj_ids = []
    flags = []
    dists = []

    for sequence in dataset:
        seq_obj = [landmark_to_idx[r.landmark] for r in sequence]
        seq_flag = [1 if r.is_corner else 0 for r in sequence]
        seq_dist = [r.distance for r in sequence]

        obj_ids.append(seq_obj)
        flags.append(seq_flag)
        dists.append(seq_dist)

    return {
        'obj_ids': np.array(obj_ids, dtype=np.int64),
        'flags': np.array(flags, dtype=np.int64),
        'dists': np.array(dists, dtype=np.float32),
        'landmark_to_idx': landmark_to_idx,
    }


def main():
    """Main function to generate and save training data."""
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)

    # Generate dataset
    print("Generating training data...")
    dataset = generate_dataset(
        num_sequences=1000,
        steps_per_sequence=8,
        seed=42
    )

    # Save to JSON file
    save_dataset(dataset, output_dir / 'training_data.json')

    # Convert to numpy arrays for training
    tensors = convert_to_tensors(dataset)
    np.savez(
        output_dir / 'training_data.npz',
        obj_ids=tensors['obj_ids'],
        flags=tensors['flags'],
        dists=tensors['dists'],
    )
    print(f"Saved numpy arrays to {output_dir / 'training_data.npz'}")

    # Save landmark mapping
    with open(output_dir / 'landmark_mapping.json', 'w') as f:
        json.dump(tensors['landmark_to_idx'], f, indent=2)
    print(f"Saved landmark mapping to {output_dir / 'landmark_mapping.json'}")

    # Plot first 10 sequences
    print("Plotting sequences...")
    plot_sequences(
        dataset,
        num_to_plot=10,
        save_path=output_dir / 'sample_sequences.png'
    )

    # Print sample data
    print("\nSample sequence (first 3):")
    for i, seq in enumerate(dataset[:3]):
        print(f"  Sequence {i + 1}:")
        for j, record in enumerate(seq):
            print(f"    Step {j + 1}: {record.landmark}, dist={record.distance}, corner={record.is_corner}")

    print("\nData shapes:")
    print(f"  obj_ids: {tensors['obj_ids'].shape}")
    print(f"  flags: {tensors['flags'].shape}")
    print(f"  dists: {tensors['dists'].shape}")


if __name__ == '__main__':
    main()
