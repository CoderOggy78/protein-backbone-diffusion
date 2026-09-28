"""
Equivariant Graph Neural Network (EGNN) for Protein Denoising.
Pure PyTorch implementation of rotation-equivariant, translation-invariant
noise prediction on C-alpha coordinates.

Mathematical Guarantees:
- Pairwise distances d_ij are SE(3)-invariant.
- Scalar node features h_i remain invariant across all layers.
- Coordinate updates Delta x_i transform equivariantly: Delta x_i -> R @ Delta x_i.
- Predicted noise epsilon_hat satisfies:
    epsilon_hat(R * X + t) = R * epsilon_hat(X)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from src.geometry import rbf_expansion

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal diffusion timestep embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000) * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / half_dim
        freqs = torch.exp(exponent)
        args = timesteps.unsqueeze(-1).float() * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

class EGNNLayer(nn.Module):
    """
    Equivariant Graph Convolutional Layer.
    Updates:
      m_ij = Phi_e(h_i, h_j, d_ij^2, e_ij)
      x_i^(l+1) = x_i^(l) + sum_j (x_i - x_j) * Phi_x(m_ij) / (deg_i + eps)
      h_i^(l+1) = Phi_h(h_i, sum_j m_ij)
    """
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        adj_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, N, hidden_dim]
        x: [B, N, 3]
        edge_attr: [B, N, N, edge_dim]
        adj_mask: [B, N, N] binary mask (1 if edge is active and non-padded, 0 otherwise)
        """
        B, N, _ = x.shape

        diff_x = x.unsqueeze(2) - x.unsqueeze(1)
        dist_sq = torch.sum(diff_x ** 2, dim=-1, keepdim=True)

        h_i = h.unsqueeze(2).expand(B, N, N, self.hidden_dim)
        h_j = h.unsqueeze(1).expand(B, N, N, self.hidden_dim)
        edge_input = torch.cat([h_i, h_j, dist_sq, edge_attr], dim=-1)

        m_ij = self.edge_mlp(edge_input)
        m_ij = m_ij * adj_mask.unsqueeze(-1)

        coord_weights = self.coord_mlp(m_ij)
        coord_weights = coord_weights * adj_mask.unsqueeze(-1)

        deg = torch.sum(adj_mask, dim=-1, keepdim=True).unsqueeze(-1) + 1e-6
        delta_x = torch.sum(diff_x * coord_weights, dim=2) / deg.squeeze(-1)
        x_next = x + delta_x

        m_agg = torch.sum(m_ij, dim=2)
        h_next = h + self.node_mlp(torch.cat([h, m_agg], dim=-1))
        h_next = self.node_norm(h_next)

        return h_next, x_next

class EquivariantProteinDenoiser(nn.Module):
    """
    Complete EGNN-based Denoiser for Protein Backbone C-alpha Coordinates.
    Predicts coordinate noise epsilon_hat in R^{B x N x 3}.
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_rbf: int = 16,
        cutoff_radius: float = 12.0,
        k_neighbors: int = 16,
        k_seq_neighbors: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff_radius = cutoff_radius
        self.k_neighbors = k_neighbors
        self.k_seq_neighbors = k_seq_neighbors

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.node_embed = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_dim = num_rbf + 1 + 1 + 2

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, self.edge_dim) for _ in range(num_layers)
        ])

        self.final_coord_scale = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

    def _build_graph(
        self,
        coords: torch.Tensor,
        residue_mask: torch.Tensor,
        seq_pos: torch.Tensor,
        observed_mask: torch.Tensor,
        generated_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds static sequence edges and dynamic spatial k-NN edges from NOISY coordinates.
        CRITICAL NO-LEAKAGE RULE:
        Edges are strictly computed from CURRENT NOISY coordinates x_t, NEVER from ground-truth x_0.
        """
        B, N, _ = coords.shape
        device = coords.device

        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)

        valid_pair = (residue_mask.unsqueeze(2) * residue_mask.unsqueeze(1)) > 0.5
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        dist_masked = dist.clone()
        dist_masked[~valid_pair | eye] = 1e6

        k = min(self.k_neighbors, N - 1)
        _, topk_idx = torch.topk(dist_masked, k=k, dim=-1, largest=False)
        spatial_adj = torch.zeros((B, N, N), device=device, dtype=torch.bool)
        spatial_adj.scatter_(2, topk_idx, True)

        seq_diff = torch.abs(seq_pos.unsqueeze(2) - seq_pos.unsqueeze(1))
        seq_adj = (seq_diff <= self.k_seq_neighbors) & (~eye) & valid_pair

        adj_mask = (spatial_adj | seq_adj) & valid_pair
        adj_mask = adj_mask.float()

        rbf_feat = rbf_expansion(dist, num_rbf=self.num_rbf, cutoff=self.cutoff_radius)
        seq_feat = (seq_diff.float() / 50.0).clamp(0.0, 1.0).unsqueeze(-1)
        covalent_feat = (seq_diff == 1).float().unsqueeze(-1)
        mask_pair = torch.stack([
            observed_mask.unsqueeze(2).expand(B, N, N),
            generated_mask.unsqueeze(2).expand(B, N, N)
        ], dim=-1)

        edge_attr = torch.cat([rbf_feat, seq_feat, covalent_feat, mask_pair], dim=-1)
        return edge_attr, adj_mask

    def forward(
        self,
        coords_t: torch.Tensor,
        timesteps: torch.Tensor,
        residue_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        generated_mask: torch.Tensor,
        seq_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of equivariant denoiser.
        Returns:
          noise_pred: [B, N, 3] estimated noise vector epsilon_hat
        """
        B, N, _ = coords_t.shape
        device = coords_t.device

        t_embed = self.time_embed(timesteps).unsqueeze(1).expand(B, N, self.hidden_dim)

        half_dim = self.hidden_dim // 2
        exp = -math.log(10000) * torch.arange(half_dim, device=device, dtype=torch.float32) / half_dim
        seq_args = seq_pos.unsqueeze(-1).float() * exp.unsqueeze(0).unsqueeze(0)
        pos_embed = torch.cat([torch.sin(seq_args), torch.cos(seq_args)], dim=-1)

        masks = torch.stack([observed_mask, generated_mask], dim=-1)
        node_in = torch.cat([pos_embed + t_embed, masks], dim=-1)
        h = self.node_embed(node_in)

        edge_attr, adj_mask = self._build_graph(coords_t, residue_mask, seq_pos, observed_mask, generated_mask)

        x_curr = coords_t
        for layer in self.layers:
            h, x_curr = layer(h, x_curr, edge_attr, adj_mask)

        coord_scale = self.final_coord_scale(h)
        noise_pred = (x_curr - coords_t) * coord_scale

        noise_pred = noise_pred * residue_mask.unsqueeze(-1)

        return noise_pred
