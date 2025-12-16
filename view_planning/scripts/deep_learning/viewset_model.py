# viewset_model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """
    Very small PointNet-like encoder:
      - per-point MLP: 3 -> 64 -> 128 -> 256
      - global max pooling
      - final FC to feat_dim
    """

    def __init__(self, feat_dim=256):
        super(PointNetEncoder, self).__init__()
        self.feat_dim = feat_dim
        self.mlp1 = nn.Linear(3, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, 256)
        self.fc = nn.Linear(256, feat_dim)

    def forward(self, pc):
        """
        Args:
            pc: (B,N,3) tensor
        Returns:
            feat: (B,feat_dim)
        """
        x = F.relu(self.mlp1(pc))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        x = torch.max(x, dim=1)[0]  # (B,256)
        x = F.relu(self.fc(x))      # (B,feat_dim)
        return x


class ViewSetAngularPolicy(nn.Module):
    """
    Policy: given a point cloud, output a distribution over K viewpoint angles.

    Latent action per view i: a_i = (a_theta, a_phi) in R^2, with
      a ~ N(mu, diag(sigma^2))
    Then we map to angles:
      theta = sigmoid(a_theta) * pi      in (0, pi)
      phi   = sigmoid(a_phi) * 2pi - pi  in (-pi, pi)

    View direction:
      dir = [ sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta) ]
    """

    def __init__(self, num_views=4, feat_dim=256, hidden_dim=128):
        super(ViewSetAngularPolicy, self).__init__()
        self.num_views = num_views
        self.encoder = PointNetEncoder(feat_dim=feat_dim)
        self.fc = nn.Linear(feat_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, num_views * 2)  # (theta, phi) per view

        # log-std shared across all views, separate per angle dim (theta, phi)
        self.log_std = nn.Parameter(torch.zeros(2))  # (2,)

    def forward(self, pc):
        """
        Deterministic forward: returns mean directions (no sampling).

        Args:
            pc: (N,3) or (1,N,3) tensor

        Returns:
            dirs: (num_views,3) tensor of unit vectors
        """
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)  # (1,N,3)
        feat = self.encoder(pc)  # (1,feat_dim)
        h = F.relu(self.fc(feat))  # (1,hidden_dim)
        mu = self.mu_head(h).view(self.num_views, 2)  # (K,2)

        # Map mu to angles deterministically
        theta_raw = mu[:, 0]
        phi_raw = mu[:, 1]

        theta = torch.sigmoid(theta_raw) * math.pi
        phi = torch.sigmoid(phi_raw) * (2.0 * math.pi) - math.pi

        dirs = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=-1)  # (K,3)

        return dirs

    def sample(self, pc):
        """
        Stochastic sampling + log_prob + entropy for REINFORCE.

        Args:
            pc: (N,3) or (1,N,3) tensor

        Returns:
            dirs: (K,3) unit vectors (torch)
            log_prob: scalar tensor
            entropy: scalar tensor
        """
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)
        feat = self.encoder(pc)
        h = F.relu(self.fc(feat))
        mu = self.mu_head(h).view(self.num_views, 2)  # (K,2)

        # Gaussian in latent angle space
        log_std = self.log_std.view(1, 2)            # (1,2) -> broadcast
        std = torch.exp(log_std)                     # (1,2)

        eps = torch.randn_like(mu)                   # (K,2)
        a = mu + eps * std                           # (K,2)

        # log_prob for Gaussian N(mu, std^2)
        # Using reparam: x = mu + std * eps => (x-mu)/std = eps
        log_probs_per_dim = -0.5 * (
            eps * eps + 2.0 * log_std + math.log(2.0 * math.pi)
        )  # broadcast (K,2)
        log_prob = log_probs_per_dim.sum()           # scalar

        # Entropy of the Gaussian (per dim), multiplied by #views
        entropy_per_dim = 0.5 + 0.5 * math.log(2.0 * math.pi) + self.log_std  # (2,)
        entropy = (entropy_per_dim.sum() * float(self.num_views))

        # Map a -> angles -> dirs
        theta_raw = a[:, 0]
        phi_raw = a[:, 1]

        theta = torch.sigmoid(theta_raw) * math.pi
        phi = torch.sigmoid(phi_raw) * (2.0 * math.pi) - math.pi

        dirs = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=-1)

        return dirs, log_prob, entropy
