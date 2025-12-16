#!/usr/bin/env python3
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class TransformerBlock(nn.Module):
    """
    Simple Transformer block with self-attention + MLP.
    Works on sequences of shape (B, N, D).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # (B,N,D)
        x = x + self.dropout(attn_out)

        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        x = x + self.dropout(mlp_out)

        return x


class ViewFormerEncoder(nn.Module):
    """
    Encoder over surface voxel point cloud.

    Input:  (B, N, F_in)  (e.g. F_in=3 for xyz, later you can extend to normals etc.)
    Output: point_feats: (B, N, D)
            global_feat: (B, D)
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # Per-point MLP to lift to hidden_dim
        self.mlp1 = nn.Linear(in_dim, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, in_dim)
        returns:
            point_feats: (B, N, hidden_dim)
            global_feat: (B, hidden_dim)
        """
        # per-point lifting
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)  # (B,N,D)

        # transformer over point tokens
        for blk in self.blocks:
            x = blk(x)

        # global pooling (max)
        global_feat = x.max(dim=1).values  # (B,D)

        return x, global_feat


class ViewFormerPlanner(nn.Module):
    """
    ViewFormer-based single-shot planner.

    - Encoder: ViewFormerEncoder over surface voxels.
    - Decoder: K learnable view tokens cross-attend to point features,
               then per-view MLP generates SE(3)+gate parameters.

    Action parameterization:
        per_view_dim = 8:
          [ dir_x, dir_y, dir_z, rad_raw,
            yaw_raw, pitch_raw, roll_raw,
            gate_raw ]
    """

    def __init__(
        self,
        num_views: int = 16,
        per_view_dim: int = 8,
        point_feat_dim: int = 256,
        view_token_dim: int = 256,
        view_emb_dim: int = 256,
        num_point_layers: int = 2,
        num_point_heads: int = 4,
        num_view_layers: int = 1,
        num_view_heads: int = 4,
        init_log_std: float = -0.5,
        in_point_dim: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_views = num_views
        self.per_view_dim = per_view_dim
        self.action_dim = num_views * per_view_dim

        # Encoder over point cloud
        self.encoder = ViewFormerEncoder(
            in_dim=in_point_dim,
            hidden_dim=point_feat_dim,
            num_layers=num_point_layers,
            num_heads=num_point_heads,
            dropout=dropout,
        )

        # K learnable view tokens (queries)
        self.view_tokens = nn.Parameter(
            torch.randn(num_views, view_token_dim) * 0.01
        )

        # Project point features to same dim as view tokens if needed
        if view_token_dim != point_feat_dim:
            self.point_proj = nn.Linear(point_feat_dim, view_token_dim)
        else:
            self.point_proj = nn.Identity()

        # Cross-attention: views (queries) attend to point features
        self.view_cross_attn = nn.MultiheadAttention(
            embed_dim=view_token_dim,
            num_heads=num_view_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.view_norm1 = nn.LayerNorm(view_token_dim)
        self.view_norm2 = nn.LayerNorm(view_token_dim)
        self.view_mlp = nn.Sequential(
            nn.Linear(view_token_dim, view_token_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(view_token_dim * 4, view_token_dim),
        )
        self.view_dropout = nn.Dropout(dropout)

        # Actor head: per-view MLP
        self.actor_mlp1 = nn.Linear(view_token_dim + point_feat_dim, 256)
        self.actor_mlp2 = nn.Linear(256, 256)
        self.actor_out = nn.Linear(256, per_view_dim)

        # Critic head: global value from encoder global feature
        self.critic_mlp1 = nn.Linear(point_feat_dim, 256)
        self.critic_mlp2 = nn.Linear(256, 256)
        self.critic_out = nn.Linear(256, 1)

        # Global log-std for all action dimensions
        self.log_std = nn.Parameter(
            torch.ones(1, self.action_dim) * init_log_std
        )

    # -----------------------------
    # Core passes
    # -----------------------------
    def encode_points(self, points: torch.Tensor):
        """
        points: (B, N, in_point_dim)
        returns:
            point_feats: (B, N, Dp)
            global_feat: (B, Dp)
        """
        point_feats, global_feat = self.encoder(points)
        return point_feats, global_feat

    def view_token_decoder(
        self, point_feats: torch.Tensor, global_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        point_feats: (B, N, Dp)
        global_feat: (B, Dp)
        returns:
            per_view_mu: (B, K, per_view_dim)
        """
        B, N, Dp = point_feats.shape
        K = self.num_views
        Dt = self.view_tokens.shape[-1]

        # project points to token dim if needed
        points_t = self.point_proj(point_feats)  # (B,N,Dt)

        # expand view tokens for batch
        view_tok = self.view_tokens.unsqueeze(0).expand(B, -1, -1)  # (B,K,Dt)

        # cross-attention: views as queries, points as key/value
        view_norm = self.view_norm1(view_tok)
        points_norm = self.view_norm1(points_t)  # reuse norm1 for simplicity

        view_attn, _ = self.view_cross_attn(
            query=view_norm,
            key=points_norm,
            value=points_norm,
        )
        view_tok = view_tok + self.view_dropout(view_attn)

        # MLP on view tokens
        view_norm2 = self.view_norm2(view_tok)
        mlp_out = self.view_mlp(view_norm2)
        view_tok = view_tok + self.view_dropout(mlp_out)  # (B,K,Dt)

        # concatenate global shape feature
        g_exp = global_feat.unsqueeze(1).expand(-1, K, -1)  # (B,K,Dp)
        x = torch.cat([view_tok, g_exp], dim=-1)            # (B,K,Dt+Dp)

        x = F.relu(self.actor_mlp1(x))
        x = F.relu(self.actor_mlp2(x))
        per_view_mu = self.actor_out(x)                     # (B,K,Dv)

        return per_view_mu

    def forward(self, points: torch.Tensor):
        """
        points: (B, N, in_point_dim)
        returns:
            mu: (B, action_dim)
            value: (B,)
        """
        point_feats, global_feat = self.encode_points(points)
        per_view_mu = self.view_token_decoder(point_feats, global_feat)
        B, K, Dv = per_view_mu.shape
        mu = per_view_mu.reshape(B, K * Dv)  # (B, A)

        # critic
        x = F.relu(self.critic_mlp1(global_feat))
        x = F.relu(self.critic_mlp2(x))
        value = self.critic_out(x).squeeze(-1)  # (B,)

        return mu, value

    # -----------------------------
    # Policy interface
    # -----------------------------
    def sample_actions(self, points: torch.Tensor):
        """
        Sample actions from Gaussian policy.

        points: (B,N,F)
        returns:
            raw_actions: (B, action_dim)
            log_probs:   (B,)
            values:      (B,)
        """
        mu, values = self.forward(points)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        raw_actions = dist.rsample()
        log_probs = dist.log_prob(raw_actions).sum(dim=-1)
        return raw_actions, log_probs, values

    def deterministic_actions(self, points: torch.Tensor) -> torch.Tensor:
        """
        Deterministic policy (use mean of Gaussian).
        """
        mu, _ = self.forward(points)
        return mu

    # -----------------------------
    # SE(3) decoding
    # -----------------------------
    def decode_raw_actions(
        self,
        raw_actions: torch.Tensor,
        r_min: float,
        r_max: float,
        center: torch.Tensor,
    ):
        """
        Convert raw action vector to world-space poses.

        raw_actions: (B, action_dim)
        center:      (B, 3)
        returns:
            positions: (B, K, 3)
            ypr:       (B, K, 3)   yaw, pitch, roll in radians
            gates:     (B, K)      in (0,1)
        """
        B = raw_actions.shape[0]
        per = self.per_view_dim
        K = self.num_views

        x = raw_actions.view(B, K, per)

        dir_raw = x[..., 0:3]    # (B,K,3)
        rad_raw = x[..., 3:4]    # (B,K,1)
        ypr_raw = x[..., 4:7]    # (B,K,3)
        gate_raw = x[..., 7]     # (B,K,)

        # direction vectors
        dir_norm = torch.norm(dir_raw, dim=-1, keepdim=True).clamp(min=1e-6)
        dirs = dir_raw / dir_norm  # unit

        # radius in [r_min, r_max]
        radii_norm = torch.sigmoid(rad_raw)
        radii = r_min + radii_norm * (r_max - r_min)

        # positions: center + r * dir
        center_exp = center.unsqueeze(1)         # (B,1,3)
        positions = center_exp + radii * dirs   # (B,K,3)

        # yaw/pitch/roll ∈ (-pi, pi)
        ypr = math.pi * torch.tanh(ypr_raw)      # (B,K,3)

        # gates in (0,1)
        gates = torch.sigmoid(gate_raw)          # (B,K)

        return positions, ypr, gates
