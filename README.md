# De Novo Protein Backbone and Loop Generation with Geometric Deep Learning

A comprehensive, mathematically rigorous research and educational framework for equivariant diffusion on protein backbones.

## Key Features
- **Track A (Educational Model from Scratch):** Pure PyTorch $SE(3)$-equivariant graph neural network (EGNN) denoiser trained on $C\alpha$ coordinates. Primary task: conditional reconstruction of missing loop segments (inpainting). Secondary task: unconditional $C\alpha$ trace sampling.
- **Track B (Pretrained Generation & Self-Consistency Pipeline):** Integration architecture for RFdiffusion, FrameDiPT, SE(3) diffusion, ProteinMPNN inverse-folding sequence design, and ESMFold structure prediction.
- **Leakage-Free Splitting:** Cluster-based partitioning strictly at the parent protein chain level before segment windowing.
- **Rigorous Verification:** 8-point pre-flight test suite including numerical $SE(3)$ equivariance verification ($R \in \mathrm{SO}(3), t \in \mathbb{R}^3$), zero-leakage checks, and context stability.
- **Evaluation & Baselines:** Comparison against linear anchor interpolation and constrained random-walk baselines with context-aligned RMSD, clash rate, bond deviation, and ensemble diversity.

## Directory Structure
```
protein_geometry_project/
├── README.md
├── config.json
├── environment.txt
├── data_manifest.csv
├── exclusion_log.csv
├── notebooks/
│   └── protein_geometry_diffusion.ipynb
├── splits/
│   ├── train_manifest.csv
│   ├── val_manifest.csv
│   └── test_manifest.csv
├── checkpoints/
│   ├── best_model.pt
│   └── latest_model.pt
├── generated/
│   └── ca_traces/
├── metrics/
│   └── evaluation_metrics.csv
├── figures/
│   ├── training_curves.png
│   └── benchmark_comparison.png
└── src/
    ├── config.py
    ├── dataset.py
    ├── geometry.py
    ├── models.py
    ├── diffusion.py
    ├── losses.py
    ├── train.py
    ├── evaluate.py
    ├── viz.py
    └── track_b_pretrained.py
```

## Running on Google Colab or Local GPU
Open `notebooks/protein_geometry_diffusion.ipynb` in Google Colab (select GPU: T4, V100, or A100).
Run cells sequentially.
