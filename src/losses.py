"""
Geometry-Aware Loss Objectives for Protein Coordinate Denoising.
Combines:
- Noise prediction MSE over generated residues
- Consecutive C-alpha - C-alpha bond length regularization (ideal ~3.81 A)
- Anchor junction continuity at loop-context boundaries
- Soft steric clash repulsion for non-bonded pairs (< 3.50 A)
- Timestep-dependent gating for optimization stability
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from src.geometry import IDEAL_CA_CA_DISTANCE, CLASH_DISTANCE_THRESHOLD

class ProteinGeometryLoss(nn.Module):
    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.w_mse = weights.get("mse", 1.0)
        self.w_bond = weights.get("bond", 0.25)
        self.w_anchor = weights.get("anchor", 0.50)
        self.w_clash = weights.get("clash", 0.15)

    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
        x_0_pred: torch.Tensor,
        generated_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        residue_mask: torch.Tensor,
        alpha_bars: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        eps_pred, eps_true: [B, N, 3]
        x_0_pred: [B, N, 3] reconstructed clean coordinates
        generated_mask: [B, N] (1 for loop residues, 0 for context/pad)
        observed_mask: [B, N] (1 for visible context, 0 for loop/pad)
        residue_mask: [B, N] (1 for valid residues, 0 for pad)
        alpha_bars: [B] cumulative alphas for timestep gating
        """
        B, N, _ = eps_pred.shape
        gen_mask_3d = generated_mask.unsqueeze(-1)
        valid_gen_count = torch.sum(generated_mask) + 1e-6

        diff_eps = (eps_pred - eps_true) * gen_mask_3d
        loss_mse = torch.sum(diff_eps ** 2) / valid_gen_count

        bond_vecs = x_0_pred[:, 1:] - x_0_pred[:, :-1]
        bond_dists = torch.norm(bond_vecs + 1e-6, dim=-1)

        valid_bonds = (
            (residue_mask[:, 1:] > 0.5) &
            (residue_mask[:, :-1] > 0.5) &
            ((generated_mask[:, 1:] > 0.5) | (generated_mask[:, :-1] > 0.5))
        )
        bond_dev = torch.abs(bond_dists - IDEAL_CA_CA_DISTANCE)
        loss_bond = torch.sum((bond_dev ** 2) * valid_bonds.float()) / (torch.sum(valid_bonds.float()) + 1e-6)

        is_junction = (
            (observed_mask[:, :-1] > 0.5) & (generated_mask[:, 1:] > 0.5)
        ) | (
            (generated_mask[:, :-1] > 0.5) & (observed_mask[:, 1:] > 0.5)
        )
        loss_anchor = torch.sum((bond_dev ** 2) * is_junction.float()) / (torch.sum(is_junction.float()) + 1e-6)

        pair_diff = x_0_pred.unsqueeze(2) - x_0_pred.unsqueeze(1)
        pair_dists = torch.norm(pair_diff + 1e-6, dim=-1)

        idx = torch.arange(N, device=x_0_pred.device)
        seq_sep = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0)).unsqueeze(0)

        clash_candidates = (
            (seq_sep >= 3) &
            ((generated_mask.unsqueeze(2) > 0.5) | (generated_mask.unsqueeze(1) > 0.5)) &
            (residue_mask.unsqueeze(2) > 0.5) &
            (residue_mask.unsqueeze(1) > 0.5)
        )
        clash_violations = torch.clamp(CLASH_DISTANCE_THRESHOLD - pair_dists, min=0.0, max=3.0)
        loss_clash = torch.sum((clash_violations ** 2) * clash_candidates.float()) / (torch.sum(clash_candidates.float()) + 1e-6)

        gate = torch.mean(alpha_bars).detach()
        loss_total = (
            self.w_mse * loss_mse +
            gate * (
                self.w_bond * loss_bond +
                self.w_anchor * loss_anchor +
                self.w_clash * loss_clash
            )
        )

        metrics = {
            "loss_total": float(loss_total.item()),
            "loss_mse": float(loss_mse.item()),
            "loss_bond": float(loss_bond.item()),
            "loss_anchor": float(loss_anchor.item()),
            "loss_clash": float(loss_clash.item())
        }
        return loss_total, metrics
