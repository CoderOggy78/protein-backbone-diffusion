"""
End-to-End Pipeline Smoke Test Runner.
Executes the full pipeline:
- Configuration setup
- PDB download & curation
- Zero-leakage splitting
- Pre-flight checks
- Educational model training
- Benchmark evaluation
- Visualization generation
- Output verification
"""

import os
import torch
from torch.utils.data import DataLoader

from src.config import ProteinDiffusionConfig
from src.dataset import curate_structural_dataset, split_dataset_without_leakage, ProteinLoopInpaintingDataset
from src.models import EquivariantProteinDenoiser
from src.diffusion import ProteinDiffusionScheduler
from src.train import run_preflight_checks, train_protein_denoiser
from src.evaluate import evaluate_test_set
from src.viz import plot_training_curves, plot_benchmark_comparison

def main():
    print("Initializing Protein Geometry & Equivariant Diffusion Pipeline...")
    config = ProteinDiffusionConfig(
        dataset_limit=8,
        max_steps=40,
        batch_size=2,
        hidden_dim=64,
        num_layers=3,
        diffusion_steps=50,
        sampling_steps=15,
        evaluation_interval=20,
        checkpoint_interval=20,
        device="cpu"
    )
    config.save_json(os.path.join(config.output_dir, "config.json"))

    print("\n--- STAGE 1: Curating Structural Dataset ---")
    proteins, manifest_p, exclusion_p = curate_structural_dataset(config)
    print(f"Accepted {len(proteins)} chains. Manifest: {manifest_p}")

    assert len(proteins) > 0, "No protein chains were accepted!"

    print("\n--- STAGE 2: Leakage-Free Parent Protein Splitting ---")
    train_prots, val_prots, test_prots = split_dataset_without_leakage(proteins, config)
    print(f"Splits: Train={len(train_prots)}, Val={len(val_prots)}, Test={len(test_prots)}")

    train_ds = ProteinLoopInpaintingDataset(train_prots, config)
    val_ds = ProteinLoopInpaintingDataset(val_prots, config)
    test_ds = ProteinLoopInpaintingDataset(test_prots, config)
    print(f"Loop inpainting examples: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    print("\n--- STAGE 3: Equivariant Denoiser & Diffusion Scheduler ---")
    model = EquivariantProteinDenoiser(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_rbf=config.num_rbf,
        k_neighbors=config.k_neighbors,
        k_seq_neighbors=config.k_seq_neighbors
    )
    scheduler = ProteinDiffusionScheduler(
        num_timesteps=config.diffusion_steps,
        beta_schedule=config.beta_schedule,
        device=config.device
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture: EGNN with {config.num_layers} layers, {config.hidden_dim} hidden units")
    print(f"Total Trainable Parameters: {num_params:,}")

    sample_batch = next(iter(train_loader))
    run_preflight_checks(model, scheduler, sample_batch, config)

    print("\n--- STAGE 4: Model Training ---")
    model, history = train_protein_denoiser(model, scheduler, train_loader, val_loader, config)

    print("\n--- STAGE 5: Scientific Evaluation & Benchmark Comparison ---")
    results, summary = evaluate_test_set(model, scheduler, test_ds, config, num_samples_k=2)
    print("\nEvaluation Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n--- STAGE 6: Plot Generation ---")
    curve_path = os.path.join(config.output_dir, "figures", "training_curves.png")
    bench_path = os.path.join(config.output_dir, "figures", "benchmark_comparison.png")
    plot_training_curves(history, curve_path)
    plot_benchmark_comparison(results, bench_path)

    print("\nPIPELINE EXECUTION FINISHED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
