"""Multi-task tabular neural network — shared trunk with classification + regression heads.

Architecture:
    Numeric features  -> BatchNorm -> Dense(128)
    Categorical feats -> Embedding layers -> Concat
    Shared trunk      -> Dense(256) -> Dropout -> Dense(128) -> Dropout -> Dense(64)
    Classification    -> Dense(num_classes, Softmax) -> CrossEntropy / Focal Loss
    Regression        -> Dense(1, Linear) -> MSE Loss
    Combined loss     = alpha * CE + (1 - alpha) * MSE
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as func


class FocalLoss(nn.Module):
    """Focal Loss — down-weights easy examples, improves minority class learning."""

    def __init__(self, alpha: list[float] | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha) if alpha else None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = func.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha[targets]
            focal_loss = at * focal_loss

        return focal_loss.mean()


class MultiTaskTabNet(nn.Module):
    """Multi-task network for tabular data with entity embeddings."""

    def __init__(
        self,
        n_numeric: int,
        categorical_dims: list[tuple[int, int]],  # [(cardinality, embed_dim), ...]
        num_classes: int = 4,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]

        # Numeric branch
        self.bn_numeric = nn.BatchNorm1d(n_numeric)
        self.numeric_fc = nn.Linear(n_numeric, hidden_dims[0] // 2)

        # Categorical embedding layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim)
            for cardinality, embed_dim in categorical_dims
        ])
        total_embed_dim = sum(dim for _, dim in categorical_dims)

        # Embedding projection
        self.embed_fc = nn.Linear(total_embed_dim, hidden_dims[0] // 2)

        # Shared trunk
        trunk_layers: list[nn.Module] = []
        in_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            trunk_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.trunk = nn.Sequential(*trunk_layers)

        # Classification head
        self.cls_head = nn.Linear(in_dim, num_classes)

        # Regression head
        self.reg_head = nn.Linear(in_dim, 1)

    def forward(
        self,
        x_numeric: torch.Tensor,
        x_categorical: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            (cls_logits, reg_output) — classification logits and regression value.
        """
        # Numeric path
        x_num = self.bn_numeric(x_numeric)
        x_num = func.relu(self.numeric_fc(x_num))

        # Embedding path
        embeds = [emb(x_cat) for emb, x_cat in zip(self.embeddings, x_categorical, strict=False)]
        x_embed = torch.cat(embeds, dim=1)
        x_embed = func.relu(self.embed_fc(x_embed))

        # Merge + trunk
        x = torch.cat([x_num, x_embed], dim=1)
        x = self.trunk(x)

        # Heads
        cls_logits = self.cls_head(x)
        reg_output = self.reg_head(x).squeeze(-1)

        return cls_logits, reg_output


class MultiTaskLoss(nn.Module):
    """Combined loss: alpha * classification_loss + (1-alpha) * regression_loss."""

    def __init__(
        self,
        alpha: float = 0.6,
        focal_gamma: float = 2.0,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.cls_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.reg_loss = nn.MSELoss()

    def forward(
        self,
        cls_logits: torch.Tensor,
        reg_pred: torch.Tensor,
        cls_target: torch.Tensor,
        reg_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (total_loss, cls_loss, reg_loss)."""
        l_cls = self.cls_loss(cls_logits, cls_target)
        l_reg = self.reg_loss(reg_pred, reg_target)
        total = self.alpha * l_cls + (1 - self.alpha) * l_reg
        return total, l_cls, l_reg
