# viewset_policy.py
#
# PointNet style policy network for viewpoint set selection.
#
# Inputs:
#   - pc: normalized point cloud (N,3) or (B,N,3)
#
# Outputs:
#   - mean directions for K candidate viewpoints on the unit sphere
#   - gating logits for each candidate (to decide which views to use)
#   - optional scalar state value (critic) when return_value=True

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewSetPolicy(nn.Module):
    def __init__(self, num_views=4, pc_feat_dim=128, hidden_dim=256):
        super(ViewSetPolicy, self).__init__()
        self.num_views = num_views
        self.pc_feat_dim = pc_feat_dim
        self.hidden_dim = hidden_dim

        # Per point feature extractor
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, pc_feat_dim),
            nn.ReLU(inplace=True),
        )

        # Global feature extractor after symmetric pooling
        self.global_mlp = nn.Sequential(
            nn.Linear(pc_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Heads: directions, gating logits, and value
        self.dir_head = nn.Linear(hidden_dim, num_views * 3)
        self.gate_head = nn.Linear(hidden_dim, num_views)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Light initialization
        nn.init.xavier_uniform_(self.dir_head.weight)
        nn.init.zeros_(self.dir_head.bias)
        nn.init.xavier_uniform_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, pc, return_value=False):
        """
        pc: (N,3) or (B,N,3) tensor of normalized points.

        If return_value is False (default):
            returns (dirs_norm, gate_logits)

        If return_value is True:
            returns (dirs_norm, gate_logits, value)
        """
        if pc.dim() == 2:
            # Single point cloud, shape (N,3)
            x = self.point_mlp(pc)          # (N, pc_feat_dim)
            x = torch.max(x, dim=0)[0]      # (pc_feat_dim,)
            x = self.global_mlp(x)          # (hidden_dim,)
            x = x.unsqueeze(0)              # (1, hidden_dim)
        elif pc.dim() == 3:
            # Batch of point clouds, shape (B,N,3)
            B, N, _ = pc.shape
            x = pc.view(B * N, 3)
            x = self.point_mlp(x)
            x = x.view(B, N, self.pc_feat_dim)
            x, _ = torch.max(x, dim=1)      # (B, pc_feat_dim)
            x = self.global_mlp(x)          # (B, hidden_dim)
        else:
            raise ValueError("pc must have shape (N,3) or (B,N,3), got {}".format(pc.shape))

        B = x.shape[0]

        # Directions: reshape and normalize to unit sphere
        dirs_raw = self.dir_head(x)                 # (B, K*3)
        dirs_raw = dirs_raw.view(B, self.num_views, 3)
        dirs_norm = dirs_raw / (dirs_raw.norm(dim=-1, keepdim=True) + 1e-8)

        # Gating logits
        gate_logits = self.gate_head(x)             # (B, K)

        # State value
        value = self.value_head(x).view(B)          # (B,)

        if pc.dim() == 2:
            dirs_norm = dirs_norm[0]                # (K,3)
            gate_logits = gate_logits[0]            # (K,)
            value = value[0]                        # scalar

        if return_value:
            return dirs_norm, gate_logits, value
        else:
            return dirs_norm, gate_logits
