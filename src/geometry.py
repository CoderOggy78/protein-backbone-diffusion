"""
Protein Geometry and Structural Biology Utilities.
Includes:
- Frame construction (Gram-Schmidt on N-Calpha-C)
- Torsions (phi, psi, omega) and C-alpha pseudo-torsions
- Kabsch alignment on context anchors
- Context-aligned RMSD, bond lengths, and clash metrics
- Radial Basis Function (RBF) distance expansions
- PDB export for C-alpha traces with explicit REMARK headers
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List

IDEAL_CA_CA_DISTANCE = 3.81
CLASH_DISTANCE_THRESHOLD = 3.50

def rbf_expansion(distances: torch.Tensor, num_rbf: int = 16, cutoff: float = 12.0) -> torch.Tensor:
    """
    Gaussian Radial Basis Function (RBF) expansion of pairwise distances.
    Input: [..., 1] or [...] distances in Angstroms
    Output: [..., num_rbf] feature vectors
    """
    if distances.dim() == 0:
        distances = distances.unsqueeze(0)
    if distances.shape[-1] != 1:
        distances = distances.unsqueeze(-1)
        
    mu = torch.linspace(0.0, cutoff, num_rbf, device=distances.device, dtype=distances.dtype)
    sigma = (cutoff / num_rbf)
    return torch.exp(-0.5 * ((distances - mu) / sigma) ** 2)

def compute_dihedral(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute dihedral angle between 4 points in radians [-pi, pi].
    Points shape: [..., 3]
    Formula: atan2( (b1 x b2) x (b2 x b3) . (b2 / ||b2||), (b1 x b2) . (b2 x b3) )
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    n1_norm = torch.norm(n1, dim=-1, keepdim=True) + eps
    n2_norm = torch.norm(n2, dim=-1, keepdim=True) + eps
    m1 = n1 / n1_norm
    m2 = n2 / n2_norm

    b2_norm = torch.norm(b2, dim=-1, keepdim=True) + eps
    u2 = b2 / b2_norm

    m3 = torch.cross(m1, u2, dim=-1)
    cos_angle = torch.sum(m1 * m2, dim=-1)
    sin_angle = torch.sum(m3 * m2, dim=-1)

    return torch.atan2(sin_angle, cos_angle)

def compute_backbone_torsions(n_xyz: torch.Tensor, ca_xyz: torch.Tensor, c_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute phi, psi, omega backbone dihedral angles from N, C-alpha, C coordinates.
    Tensors shape: [N_res, 3]
    Returns:
      phi: [N_res] (first residue is NaN)
      psi: [N_res] (last residue is NaN)
      omega: [N_res] (last residue is NaN)
    """
    n_res = ca_xyz.shape[0]
    phi = torch.full((n_res,), float('nan'), device=ca_xyz.device)
    psi = torch.full((n_res,), float('nan'), device=ca_xyz.device)
    omega = torch.full((n_res,), float('nan'), device=ca_xyz.device)

    if n_res >= 2:
        phi[1:] = compute_dihedral(c_xyz[:-1], n_xyz[1:], ca_xyz[1:], c_xyz[1:])
        psi[:-1] = compute_dihedral(n_xyz[:-1], ca_xyz[:-1], c_xyz[:-1], n_xyz[1:])
        omega[:-1] = compute_dihedral(ca_xyz[:-1], c_xyz[:-1], n_xyz[1:], ca_xyz[1:])

    return phi, psi, omega

def compute_ca_pseudo_dihedrals(ca_xyz: torch.Tensor) -> torch.Tensor:
    """
    Compute coarse-grained C-alpha pseudo-dihedral angles tau between consecutive C-alphas.
    ca_xyz: [N_res, 3]
    tau_i = Dihedral(CA_{i-1}, CA_i, CA_{i+1}, CA_{i+2})
    Returns: [N_res - 3] in radians
    """
    if ca_xyz.shape[0] < 4:
        return torch.empty(0, device=ca_xyz.device)
    return compute_dihedral(ca_xyz[:-3], ca_xyz[1:-2], ca_xyz[2:-1], ca_xyz[3:])

def construct_local_frames(n_xyz: torch.Tensor, ca_xyz: torch.Tensor, c_xyz: torch.Tensor, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct Gram-Schmidt orthonormal residue frames from N, CA, C coordinates.
    Origin t = CA
    e1 = (C - CA) / ||C - CA||
    v2 = N - CA
    u2 = v2 - (v2 . e1) * e1
    e2 = u2 / ||u2||
    e3 = e1 x e2
    Rotation matrix R = [e1, e2, e3] in SO(3)
    Returns:
      R: [N_res, 3, 3] rotation matrices
      t: [N_res, 3] translation origins (CA coordinates)
    """
    v1 = c_xyz - ca_xyz
    e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)

    v2 = n_xyz - ca_xyz
    dot_v2_e1 = torch.sum(v2 * e1, dim=-1, keepdim=True)
    u2 = v2 - dot_v2_e1 * e1
    e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)

    e3 = torch.cross(e1, e2, dim=-1)

    R = torch.stack([e1, e2, e3], dim=-1)
    t = ca_xyz
    return R, t

def kabsch_superposition(P: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute optimal rotation R and translation t aligning P onto Q via Kabsch algorithm.
    P, Q: [N, 3] coordinates
    Returns:
      R: [3, 3] rotation matrix such that (P - centroid_P) @ R + centroid_Q minimizes RMSD.
      t: [3] translation vector
    """
    device = P.device
    dtype = P.dtype

    if torch.max(torch.abs(P - Q)) < 1e-4:
        return torch.eye(3, device=device, dtype=dtype), torch.zeros(3, device=device, dtype=dtype)

    centroid_P = torch.mean(P, dim=0)
    centroid_Q = torch.mean(Q, dim=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = torch.mm(P_centered.t(), Q_centered).to(torch.float64)
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.t()

    d = torch.sign(torch.linalg.det(torch.mm(V, U.t())))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=device, dtype=torch.float64))

    R = torch.mm(torch.mm(V, D), U.t()).to(dtype)
    t = centroid_Q - torch.mv(R, centroid_P)
    return R, t

def context_aligned_loop_rmsd(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    observed_mask: torch.Tensor,
    generated_mask: torch.Tensor,
) -> float:
    """
    Calculate context-aligned loop RMSD.
    CRITICAL STRUCTURAL BIOLOGY RULE:
    Do NOT align the generated loop independently to the true loop, as that conceals
    gross displacement from anchor residues.
    Instead, align the entire structure using the FIXED OBSERVED CONTEXT,
    and then measure RMSD exclusively over the GENERATED LOOP coordinates.
    """
    obs_idx = torch.where(observed_mask > 0.5)[0]
    gen_idx = torch.where(generated_mask > 0.5)[0]

    if len(gen_idx) == 0:
        return 0.0

    if len(obs_idx) >= 3:
        P_ctx = pred_coords[obs_idx]
        Q_ctx = true_coords[obs_idx]
        R, t = kabsch_superposition(P_ctx, Q_ctx)
        pred_aligned = torch.mm(pred_coords, R.t()) + t
    else:
        pred_aligned = pred_coords

    diff = pred_aligned[gen_idx] - true_coords[gen_idx]
    rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1)) + 1e-8)
    return float(rmsd.item())

def evaluate_ca_geometry(coords: torch.Tensor, generated_mask: torch.Tensor) -> Dict[str, float]:
    """
    Evaluate physical plausibility metrics on a C-alpha trace:
    - Bond length deviation (|d_i,i+1 - 3.81| A)
    - Junction bond distances at loop boundaries
    - Steric clash rate: fraction of non-adjacent C-alpha pairs with distance < 3.5 A
    """
    n_res = coords.shape[0]
    gen_idx = torch.where(generated_mask > 0.5)[0].tolist()

    if n_res < 2:
        return {"mean_bond_dev": 0.0, "junction_dev": 0.0, "clash_fraction": 0.0}

    bond_vecs = coords[1:] - coords[:-1]
    bond_dists = torch.norm(bond_vecs, dim=-1)
    bond_devs = torch.abs(bond_dists - IDEAL_CA_CA_DISTANCE)

    mean_bond_dev = float(torch.mean(bond_devs).item())

    junction_devs = []
    for idx in gen_idx:
        if idx > 0 and (idx - 1) not in gen_idx:
            d = torch.norm(coords[idx] - coords[idx - 1]).item()
            junction_devs.append(abs(d - IDEAL_CA_CA_DISTANCE))
        if idx < n_res - 1 and (idx + 1) not in gen_idx:
            d = torch.norm(coords[idx + 1] - coords[idx]).item()
            junction_devs.append(abs(d - IDEAL_CA_CA_DISTANCE))

    mean_junction_dev = float(np.mean(junction_devs)) if junction_devs else 0.0

    dists = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
    clash_count = 0
    total_evaluated_pairs = 0

    for i in range(n_res):
        for j in range(i + 3, n_res):
            if (i in gen_idx) or (j in gen_idx):
                total_evaluated_pairs += 1
                if dists[i, j] < CLASH_DISTANCE_THRESHOLD:
                    clash_count += 1

    clash_fraction = (clash_count / total_evaluated_pairs) if total_evaluated_pairs > 0 else 0.0

    return {
        "mean_bond_dev": mean_bond_dev,
        "junction_dev": mean_junction_dev,
        "clash_fraction": clash_fraction
    }

def export_ca_to_pdb(
    coords: torch.Tensor,
    filepath: str,
    sequence: Optional[str] = None,
    chain_id: str = "A",
    b_factors: Optional[torch.Tensor] = None,
    remarks: Optional[List[str]] = None
):
    """
    Save coarse-grained C-alpha coordinates as a valid PDB file.
    Includes explicit REMARK tags documenting that this is a C-alpha trace,
    preventing confusion with all-atom crystallographic structures.
    """
    n_res = coords.shape[0]
    coords_np = coords.detach().cpu().numpy()
    b_factors_np = b_factors.detach().cpu().numpy() if b_factors is not None else np.zeros(n_res)

    lines = []
    lines.append("REMARK 250")
    lines.append("REMARK 250 COARSE-GRAINED C-ALPHA TRACE GENERATED VIA GEOMETRIC DIFFUSION")
    lines.append("REMARK 250 EXPERIMENTAL NOTE: ALL-ATOM TOPOLOGY AND SIDECHAINS ARE OMITTED.")
    if remarks:
        for r in remarks:
            lines.append(f"REMARK 250 {r}")

    amino_acids_3 = [
        "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
        "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"
    ]
    one_to_three = {
        "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
        "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
        "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
        "W": "TRP", "Y": "TYR"
    }

    for i in range(n_res):
        atom_num = i + 1
        res_num = i + 1
        res_name = "GLY"
        if sequence and i < len(sequence):
            res_name = one_to_three.get(sequence[i], "GLY")
        x, y, z = coords_np[i]
        b_fac = b_factors_np[i] if i < len(b_factors_np) else 0.0

        atom_line = (
            f"ATOM  {atom_num:5d}  CA  {res_name:3s} {chain_id:1s}{res_num:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_fac:6.2f}           C  "
        )
        lines.append(atom_line)

    lines.append("TER")
    lines.append("END")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
