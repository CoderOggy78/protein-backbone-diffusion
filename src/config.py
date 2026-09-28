"""
Configuration Dataclass for Protein Backbone and Loop Diffusion.
Centralizes all hyperparameters, directory paths, loss weights, and feature flags.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json
import os
import torch

@dataclass
class ProteinDiffusionConfig:
    seed: int = 42
    output_dir: str = "./protein_geometry_project"
    cache_dir: str = "./protein_geometry_project/data/raw_pdb"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    smoke_test: bool = True

    dataset_limit: int = 24
    minimum_chain_length: int = 50
    maximum_chain_length: int = 150
    minimum_loop_length: int = 5
    maximum_loop_length: int = 15
    resolution_cutoff: float = 2.5
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    hidden_dim: int = 128
    num_layers: int = 4
    num_rbf: int = 16
    cutoff_radius: float = 12.0
    k_neighbors: int = 16
    k_seq_neighbors: int = 4

    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    max_steps: int = 120
    clip_grad_norm: float = 1.0
    checkpoint_interval: int = 30
    evaluation_interval: int = 30

    diffusion_steps: int = 100
    sampling_steps: int = 25
    beta_schedule: str = "linear"
    beta_start: float = 1e-3
    beta_end: float = 0.03
    num_samples: int = 4

    geometry_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "mse": 1.0,
        "bond": 0.25,
        "anchor": 0.50,
        "clash": 0.15,
    })

    enable_pretrained: bool = False
    enable_sequence_design: bool = False
    enable_refolding: bool = False
    enable_openmm: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ProteinDiffusionConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
