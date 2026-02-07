"""
Branching Predictor model for multi-label link prediction.
Based on the updated architecture in PLAN.md.

Instead of predicting a single next node, predicts all possible neighbors
simultaneously with their properties (link existence, distance, importance).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchingPredictor(nn.Module):
    """
    GNN-based model that predicts all possible neighbors from current position.

    Output shape: [Batch, Num_Objects, 3]
        - Channel 0: Link existence logits (use Sigmoid)
        - Channel 1: Distance (regression)
        - Channel 2: Importance/Corner logits (use Sigmoid)
    """

    def __init__(
        self,
        num_objects: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        max_seq_len: int = 100
    ):
        super().__init__()
        self.num_objects = num_objects
        self.hidden_dim = hidden_dim

        # Input embedding
        self.obj_embedding = nn.Embedding(num_objects, hidden_dim)

        # Position encoding
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer encoder for sequence processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: Projects to (Num_Objects * 3)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objects * 3)
        )

        # Causal mask for autoregressive processing
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [Batch, Seq_Len] - Landmark indices for history
            lengths: [Batch] - Actual sequence lengths (for masking padding)

        Returns:
            prediction: [Batch, Num_Objects, 3]
                - [:, :, 0] = Link logits
                - [:, :, 1] = Distance prediction
                - [:, :, 2] = Importance logits
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Embed inputs
        x = self.obj_embedding(input_ids)

        # Add positional encoding
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(positions)

        # Create attention mask (causal + padding)
        causal_mask = self.causal_mask[:L, :L]

        # Create padding mask if lengths provided
        if lengths is not None:
            padding_mask = torch.arange(L, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            padding_mask = None

        # Encode sequence
        h = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        # Get representation of the last valid position
        if lengths is not None:
            # Gather last valid position for each sequence
            last_idx = (lengths - 1).clamp(min=0)
            batch_idx = torch.arange(B, device=device)
            h_last = h[batch_idx, last_idx]
        else:
            h_last = h[:, -1]  # [Batch, Hidden_Dim]

        # Project to output
        raw_output = self.output_head(h_last)  # [Batch, Num_Objects * 3]

        # Reshape to [Batch, Num_Objects, 3]
        prediction = raw_output.view(B, self.num_objects, 3)

        return prediction


class BranchingLoss(nn.Module):
    """
    Masked loss function for branching prediction.

    Only computes distance/importance loss for objects where links exist.
    """

    def __init__(self, link_weight: float = 1.0, dist_weight: float = 1.0, imp_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()  # For Link & Importance
        self.mse = nn.MSELoss(reduction='none')  # For Distance (allows masking)

        self.link_weight = link_weight
        self.dist_weight = dist_weight
        self.imp_weight = imp_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Compute masked loss.

        Args:
            preds: [Batch, M, 3] - Model predictions
            targets: [Batch, M, 3] - Ground truth
                - Channel 0: Link existence (0 or 1)
                - Channel 1: Distance (real value)
                - Channel 2: Importance/Corner (0 or 1)

        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary of individual losses
        """
        # --- 1. Link Prediction Loss (on all objects) ---
        pred_link_logits = preds[:, :, 0]
        gt_link = targets[:, :, 0]

        loss_link = self.bce(pred_link_logits, gt_link)

        # --- Create Mask (only care about distance/imp if link exists) ---
        mask = gt_link  # 1 if link exists, 0 if not

        # --- 2. Distance Regression Loss (masked) ---
        pred_dist = preds[:, :, 1]
        gt_dist = targets[:, :, 1]

        raw_dist_loss = self.mse(pred_dist, gt_dist)
        # Apply mask: Sum errors only where link exists
        loss_dist = (raw_dist_loss * mask).sum() / (mask.sum() + 1e-6)

        # --- 3. Importance Classification Loss (masked) ---
        pred_imp_logits = preds[:, :, 2]
        gt_imp = targets[:, :, 2]

        raw_imp_loss = F.binary_cross_entropy_with_logits(
            pred_imp_logits, gt_imp, reduction='none'
        )
        loss_imp = (raw_imp_loss * mask).sum() / (mask.sum() + 1e-6)

        # Weighted total loss
        total_loss = (
            self.link_weight * loss_link +
            self.dist_weight * loss_dist +
            self.imp_weight * loss_imp
        )

        loss_dict = {
            'loss_link': loss_link.item(),
            'loss_dist': loss_dist.item(),
            'loss_imp': loss_imp.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_dict


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Compute evaluation metrics.

    Args:
        preds: [Batch, M, 3]
        targets: [Batch, M, 3]

    Returns:
        dict with metrics
    """
    with torch.no_grad():
        # Link prediction metrics
        pred_link = torch.sigmoid(preds[:, :, 0]) > 0.5
        gt_link = targets[:, :, 0] > 0.5

        # Accuracy
        link_accuracy = (pred_link == gt_link).float().mean().item()

        # Precision, Recall, F1 for link prediction
        tp = (pred_link & gt_link).sum().float()
        fp = (pred_link & ~gt_link).sum().float()
        fn = (~pred_link & gt_link).sum().float()

        precision = (tp / (tp + fp + 1e-6)).item()
        recall = (tp / (tp + fn + 1e-6)).item()
        f1 = (2 * precision * recall / (precision + recall + 1e-6))

        # Distance MAE (only where links exist)
        mask = targets[:, :, 0] > 0.5
        if mask.sum() > 0:
            pred_dist = preds[:, :, 1]
            gt_dist = targets[:, :, 1]
            dist_mae = ((pred_dist - gt_dist).abs() * mask).sum() / mask.sum()
            dist_mae = dist_mae.item()
        else:
            dist_mae = 0.0

        # Importance accuracy (only where links exist)
        if mask.sum() > 0:
            pred_imp = torch.sigmoid(preds[:, :, 2]) > 0.5
            gt_imp = targets[:, :, 2] > 0.5
            imp_correct = ((pred_imp == gt_imp) & mask).sum().float()
            imp_accuracy = (imp_correct / mask.sum()).item()
        else:
            imp_accuracy = 0.0

    return {
        'link_accuracy': link_accuracy,
        'link_precision': precision,
        'link_recall': recall,
        'link_f1': f1,
        'dist_mae': dist_mae,
        'imp_accuracy': imp_accuracy,
    }
