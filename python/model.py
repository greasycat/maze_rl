"""
Multi-task GNN model for learning spatial configuration of objects in a grid space.
Based on the architecture specified in PLAN.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskGNN(nn.Module):
    def __init__(self, num_objects: int = 7, hidden_dim: int = 64, max_seq_len: int = 100):
        super().__init__()
        # --- Shared Encoder ---
        self.obj_embedding = nn.Embedding(num_objects, hidden_dim)
        self.flag_embedding = nn.Linear(1, hidden_dim)
        self.dist_embedding = nn.Linear(1, hidden_dim)

        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

        self.gnn_update = nn.Linear(hidden_dim, hidden_dim)

        # --- Multi-Head Decoders ---
        self.head_object = nn.Linear(hidden_dim, num_objects)  # Classification
        self.head_flag = nn.Linear(hidden_dim, 1)              # Binary Class
        self.head_dist = nn.Linear(hidden_dim, 1)              # Regression

        # Causal Mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, obj_ids: torch.Tensor, flags: torch.Tensor, dists: torch.Tensor):
        """
        Forward pass through the model.

        Args:
            obj_ids: [Batch, Seq] - Landmark indices
            flags: [Batch, Seq] - Boolean flags for corners
            dists: [Batch, Seq] - Distances from previous landmark

        Returns:
            pred_obj: [Batch, Seq, num_objects] - Object predictions
            pred_flag: [Batch, Seq, 1] - Flag predictions
            pred_dist: [Batch, Seq, 1] - Distance predictions
            adjacency: [Batch, Seq, Seq] - Learned adjacency matrix
        """
        B, L = obj_ids.shape

        # 1. Embed Inputs
        x = self.obj_embedding(obj_ids) + \
            self.flag_embedding(flags.unsqueeze(-1).float()) + \
            self.dist_embedding(dists.unsqueeze(-1))

        # 2. Structure Learning (Self-Attention)
        Q, K = self.W_Q(x), self.W_K(x)
        scores = (Q @ K.transpose(-2, -1)) * self.scale

        # Apply Causal Mask (prevent seeing future)
        mask = self.mask[:, :, :L, :L]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        adjacency = F.softmax(scores, dim=-1)

        # 3. GNN Propagation
        context = adjacency @ x
        h = F.relu(self.gnn_update(context))

        # 4. Multi-Task Prediction
        pred_obj = self.head_object(h)             # Logits for CrossEntropy
        pred_flag = self.head_flag(h)              # Logits for BCEWithLogits
        pred_dist = F.softplus(self.head_dist(h))  # Positive distance

        return pred_obj, pred_flag, pred_dist, adjacency


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.loss_obj = nn.CrossEntropyLoss()      # For Object Label
        self.loss_flag = nn.BCEWithLogitsLoss()    # For Importance (Boolean)
        self.loss_dist = nn.MSELoss()              # For Distance

        self.alpha = alpha  # Weight for Object
        self.beta = beta    # Weight for Flag
        self.gamma = gamma  # Weight for Distance

    def forward(self, preds: tuple, targets: tuple) -> torch.Tensor:
        """
        Compute multi-task loss.

        Args:
            preds: (pred_obj, pred_flag, pred_dist, adjacency)
            targets: (target_obj, target_flag, target_dist)

        Returns:
            total_loss: Weighted sum of individual losses
        """
        # Unpack predictions
        p_obj, p_flag, p_dist, _ = preds
        t_obj, t_flag, t_dist = targets

        # Calculate individual losses
        # Reshape [Batch, Seq, Class] -> [Batch*Seq, Class]
        l_obj = self.loss_obj(p_obj.reshape(-1, p_obj.size(-1)), t_obj.reshape(-1))

        l_flag = self.loss_flag(p_flag.reshape(-1), t_flag.float().reshape(-1))

        l_dist = self.loss_dist(p_dist.reshape(-1), t_dist.float().reshape(-1))

        # Weighted Sum
        total_loss = (self.alpha * l_obj) + (self.beta * l_flag) + (self.gamma * l_dist)

        return total_loss


def compute_metrics(preds: tuple, targets: tuple) -> dict:
    """Compute metrics for evaluation."""
    p_obj, p_flag, p_dist, _ = preds
    t_obj, t_flag, t_dist = targets

    # Object accuracy
    pred_obj_class = p_obj.argmax(dim=-1)
    obj_accuracy = (pred_obj_class == t_obj).float().mean().item()

    # Flag accuracy (binary classification)
    pred_flag_class = (torch.sigmoid(p_flag.squeeze(-1)) > 0.5).long()
    flag_accuracy = (pred_flag_class == t_flag).float().mean().item()

    # Distance MAE
    dist_mae = (p_dist.squeeze(-1) - t_dist).abs().mean().item()

    return {
        'obj_accuracy': obj_accuracy,
        'flag_accuracy': flag_accuracy,
        'dist_mae': dist_mae,
    }
