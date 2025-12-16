# viewformer_cem_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """
    Simple PointNet-style encoder:
      - input: (N, 3) point cloud (already normalized)
      - output: (F,) global feature
    """

    def __init__(self, in_dim=3, hidden_dims=(64, 128, 256), out_dim=256):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        self.mlp = nn.Sequential(*layers)
        self.fc_out = nn.Linear(last, out_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: (N,3) or (B,N,3). Here we use (N,3) for single model.
        """
        if points.dim() == 3:
            # (B,N,3) -> treat as batch
            B, N, _ = points.shape
            x = self.mlp(points)          # (B,N,H)
            x = x.max(dim=1).values       # (B,H)
            x = self.fc_out(x)            # (B,F)
            return x
        elif points.dim() == 2:
            # (N,3)
            x = self.mlp(points)          # (N,H)
            x = x.max(dim=0).values       # (H,)
            x = self.fc_out(x)            # (F,)
            return x
        else:
            raise ValueError(f"points must be (N,3) or (B,N,3), got {points.shape}")


class ViewFormerPlanner(nn.Module):
    """
    View planner for single-shot CEM training.

    - Input: point cloud (N,3), normalized (zero mean, unit-ish radius).
    - Output: per-view raw spherical angles for K views:
        per_view_dim = 2  (theta_raw, phi_raw)
      where:
        theta_raw -> theta ∈ [0, 2π)
        phi_raw   -> phi   ∈ [0, π]
    """

    def __init__(
        self,
        num_views: int = 6,
        per_view_dim: int = 2,
        pc_feat_dim: int = 256,
        view_token_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_views = num_views
        self.per_view_dim = per_view_dim

        self.encoder = PointNetEncoder(out_dim=pc_feat_dim)

        # K learnable view tokens
        self.view_tokens = nn.Parameter(
            torch.randn(num_views, view_token_dim) * 0.01
        )

        # Per-view MLP: [global_feat, view_token] -> raw_action (2 dim)
        self.view_mlp = nn.Sequential(
            nn.Linear(pc_feat_dim + view_token_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, per_view_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: (N,3) normalized point cloud.
        returns: raw_actions_mean: (num_views, per_view_dim)
        """
        if points.dim() != 2:
            raise ValueError("Expected points shape (N,3)")

        global_feat = self.encoder(points)      # (F,)
        global_feat = global_feat.unsqueeze(0)  # (1,F)
        global_feat = global_feat.expand(self.num_views, -1)  # (K,F)

        tokens = self.view_tokens              # (K,T)
        x = torch.cat([global_feat, tokens], dim=-1)  # (K, F+T)
        raw_actions = self.view_mlp(x)         # (K, per_view_dim)

        return raw_actions
