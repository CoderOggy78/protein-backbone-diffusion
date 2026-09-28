"""
Diffusion Process Scheduler for Protein C-alpha Coordinates.
Implements:
- Discrete forward noising schedules (Cosine and Linear)
- Conditional context-preserving forward diffusion
- DDPM (Denoising Diffusion Probabilistic Models) reverse sampler
- DDIM (Denoising Diffusion Implicit Models) accelerated sampler
- Unconditional C-alpha trace generation
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List

from src.geometry import IDEAL_CA_CA_DISTANCE

class ProteinDiffusionScheduler:
    def __init__(
        self,
        num_timesteps: int = 100,
        beta_schedule: str = "linear",
        beta_start: float = 1e-3,
        beta_end: float = 0.03,
        device: str = "cpu"
    ):
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.device = device

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device)
        elif beta_schedule == "cosine":
            steps = num_timesteps + 1
            s = 0.008
            x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32, device=device)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            alphas_cumprod = torch.clamp(alphas_cumprod, min=0.02)
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: str):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        For conditional inpainting:
        Keep observed context clean: x_t[observed] = x_0[observed]
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        B = x_start.shape[0]
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)

        x_noisy = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

        if observed_mask is not None:
            m_obs = observed_mask.unsqueeze(-1)
            x_noisy = m_obs * x_start + (1.0 - m_obs) * x_noisy

        return x_noisy, noise

    def predict_x_start_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct x_0 estimate: x_0 = (x_t - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)"""
        B = x_t.shape[0]
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
        x_0 = (x_t - sqrt_one_minus_alpha_bar * eps) / (torch.clamp(sqrt_alpha_bar, min=1e-3))
        return torch.clamp(x_0, -50.0, 50.0)

    @torch.no_grad()
    def p_sample_ddim(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_curr: int,
        t_prev: int,
        residue_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        generated_mask: torch.Tensor,
        seq_pos: torch.Tensor,
        clean_context: Optional[torch.Tensor] = None,
        eta: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Accelerated DDIM step from t_curr to t_prev.
        Guarantees that context coordinates remain exactly fixed.
        """
        B = x_t.shape[0]
        t_batch = torch.full((B,), t_curr, device=self.device, dtype=torch.long)

        eps_pred = model(x_t, t_batch, residue_mask, observed_mask, generated_mask, seq_pos)

        x_0_pred = self.predict_x_start_from_eps(x_t, t_batch, eps_pred)

        alpha_bar_curr = self.alphas_cumprod[t_curr]
        alpha_bar_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)

        sigma = eta * torch.sqrt(
            (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_curr) * (1.0 - alpha_bar_curr / alpha_bar_prev)
        )

        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0)) * eps_pred
        noise = torch.randn_like(x_t) if sigma > 0 else 0.0

        x_prev = torch.sqrt(alpha_bar_prev) * x_0_pred + dir_xt + sigma * noise

        if clean_context is not None:
            m_obs = observed_mask.unsqueeze(-1)
            x_prev = m_obs * clean_context + (1.0 - m_obs) * x_prev

        x_prev = x_prev * residue_mask.unsqueeze(-1)

        return x_prev, x_0_pred

    @torch.no_grad()
    def sample_loop(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        sampling_steps: int = 25,
        eta: float = 0.0,
        return_trajectory: bool = False
    ) -> Dict[str, Any]:
        """
        Sample loop candidates conditioned on fixed context.
        """
        model.eval()
        coords_orig = batch["coords"].to(self.device)
        residue_mask = batch["residue_mask"].to(self.device)
        observed_mask = batch["observed_mask"].to(self.device)
        generated_mask = batch["generated_mask"].to(self.device)
        seq_pos = batch["seq_pos"].to(self.device)

        B, N, _ = coords_orig.shape

        noise = torch.randn((B, N, 3), device=self.device)
        x_t = observed_mask.unsqueeze(-1) * coords_orig + generated_mask.unsqueeze(-1) * noise

        times = torch.linspace(self.num_timesteps - 1, 0, sampling_steps + 1, dtype=torch.long, device=self.device)

        trajectory = [x_t.clone()] if return_trajectory else []

        for i in range(sampling_steps):
            t_curr = int(times[i].item())
            t_prev = int(times[i + 1].item())

            x_t, _ = self.p_sample_ddim(
                model=model,
                x_t=x_t,
                t_curr=t_curr,
                t_prev=t_prev,
                residue_mask=residue_mask,
                observed_mask=observed_mask,
                generated_mask=generated_mask,
                seq_pos=seq_pos,
                clean_context=coords_orig,
                eta=eta
            )
            if return_trajectory:
                trajectory.append(x_t.clone())

        res = {"sampled_coords": x_t}
        if return_trajectory:
            res["trajectory"] = trajectory
        return res

    @torch.no_grad()
    def sample_unconditional(
        self,
        model: nn.Module,
        chain_length: int = 60,
        batch_size: int = 2,
        sampling_steps: int = 25,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Sample unconditional C-alpha traces from pure Gaussian noise.
        """
        model.eval()
        B, N = batch_size, chain_length

        residue_mask = torch.ones((B, N), device=self.device)
        observed_mask = torch.zeros((B, N), device=self.device)
        generated_mask = torch.ones((B, N), device=self.device)
        seq_pos = torch.arange(N, device=self.device).unsqueeze(0).expand(B, N)

        x_t = torch.randn((B, N, 3), device=self.device)
        x_t = x_t - torch.mean(x_t, dim=1, keepdim=True)

        times = torch.linspace(self.num_timesteps - 1, 0, sampling_steps + 1, dtype=torch.long, device=self.device)

        for i in range(sampling_steps):
            t_curr = int(times[i].item())
            t_prev = int(times[i + 1].item())

            x_t, _ = self.p_sample_ddim(
                model=model,
                x_t=x_t,
                t_curr=t_curr,
                t_prev=t_prev,
                residue_mask=residue_mask,
                observed_mask=observed_mask,
                generated_mask=generated_mask,
                seq_pos=seq_pos,
                clean_context=None,
                eta=eta
            )
            x_t = x_t - torch.mean(x_t, dim=1, keepdim=True)

        return x_t
