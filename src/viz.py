"""
Scientific Visualization Utilities for Protein Geometry and Diffusion.
Generates:
1. Training and validation loss curves with auxiliary geometric components.
2. Comparative RMSD and geometry distribution boxplots against baselines.
3. Interactive 3D molecular visualization using py3Dmol:
   - Visible context rendered in slate gray.
   - Ground truth loop rendered in cyan.
   - Generated candidate loop(s) rendered in magenta/orange.
4. Multi-step denoising trajectory snapshot visualization.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional, Any

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

def plot_training_curves(history: Dict[str, list], output_path: str):
    """Plot multi-panel training loss curves and auxiliary loss components."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["step"], history["train_loss"], label="Train Loss (Total)", color="#1f77b4", lw=2.2)
    if "val_loss" in history and len(history["val_loss"]) > 0:
        axes[0].plot(history["step"], history["val_loss"], label="Validation Loss", color="#ff7f0e", lw=2.2, ls="--")
    axes[0].set_title("Equivariant Diffusion Training Dynamics", fontweight="bold")
    axes[0].set_xlabel("Optimization Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=True)

    if "loss_mse" in history and len(history["loss_mse"]) > 0:
        axes[1].plot(history["step"], history["loss_mse"], label="Denoising MSE", color="#2ca02c", lw=1.8)
        axes[1].plot(history["step"], history["loss_bond"], label="Cα-Cα Bond Penalty", color="#d62728", lw=1.8)
        axes[1].plot(history["step"], history["loss_anchor"], label="Anchor Junction Loss", color="#9467bd", lw=1.8)
        axes[1].plot(history["step"], history["loss_clash"], label="Clash Penalty", color="#8c564b", lw=1.8)
        axes[1].set_title("Geometric Auxiliary Loss Decomposition", fontweight="bold")
        axes[1].set_xlabel("Optimization Step")
        axes[1].set_ylabel("Component Loss")
        axes[1].legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {output_path}")

def plot_benchmark_comparison(results: List[Dict[str, Any]], output_path: str):
    """Bar/box plot comparing diffusion model against linear and random-walk baselines."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not results:
        return

    lin_rmsds = [r["lin_rmsd"] for r in results]
    rw_rmsds = [r["rw_rmsd"] for r in results]
    diff_med = [r["diff_median_rmsd"] for r in results]
    diff_best = [r["diff_best_rmsd"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Linear Baseline", "Random Walk", "Diffusion (Median)", "Diffusion (Best-of-K)"]
    means = [np.mean(lin_rmsds), np.mean(rw_rmsds), np.mean(diff_med), np.mean(diff_best)]
    stds = [np.std(lin_rmsds), np.std(rw_rmsds), np.std(diff_med), np.std(diff_best)]
    colors = ["#7f7f7f", "#bcbd22", "#1f77b4", "#2ca02c"]

    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax.set_ylabel("Context-Aligned Loop RMSD (Å)", fontweight="bold")
    ax.set_title("Loop Reconstruction Accuracy Comparison", fontweight="bold")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f} Å",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved benchmark comparison to {output_path}")

def render_interactive_py3dmol(
    context_coords: torch.Tensor,
    true_loop_coords: torch.Tensor,
    generated_loop_coords: torch.Tensor,
    width: int = 700,
    height: int = 500
):
    """
    Renders an interactive 3D molecular visualization using py3Dmol.
    - Context: Slate Gray trace
    - True Loop: Cyan stick/trace
    - Generated Loop: Magenta stick/trace
    """
    if not PY3DMOL_AVAILABLE:
        print("py3Dmol is not installed in the current environment.")
        return None

    view = py3Dmol.view(width=width, height=height)

    def coords_to_pdb_block(coords: np.ndarray, res_offset: int = 1) -> str:
        lines = []
        for i, (x, y, z) in enumerate(coords):
            lines.append(
                f"ATOM  {i+1:5d}  CA  GLY A{i+res_offset:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
            )
        return "\n".join(lines) + "\n"

    ctx_np = context_coords.detach().cpu().numpy()
    true_np = true_loop_coords.detach().cpu().numpy()
    gen_np = generated_loop_coords.detach().cpu().numpy()

    pdb_ctx = coords_to_pdb_block(ctx_np, res_offset=1)
    pdb_true = coords_to_pdb_block(true_np, res_offset=len(ctx_np) + 1)
    pdb_gen = coords_to_pdb_block(gen_np, res_offset=len(ctx_np) + len(true_np) + 1)

    view.addModel(pdb_ctx, "pdb")
    view.setStyle({"model": 0}, {"line": {"color": "#888888", "linewidth": 3}})

    view.addModel(pdb_true, "pdb")
    view.setStyle({"model": 1}, {"stick": {"color": "#00d2d3", "radius": 0.35}})

    view.addModel(pdb_gen, "pdb")
    view.setStyle({"model": 2}, {"stick": {"color": "#ff4757", "radius": 0.40}})

    view.zoomTo()
    return view
