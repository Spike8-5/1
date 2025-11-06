import torch
import torch.nn.functional as F
import math

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule from Nichol & Dhariwal 2021.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999).float()

class DDPMHelper:
    def __init__(self, timesteps=1000, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        self.betas = cosine_beta_schedule(timesteps).to(device)  # [T]
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=device), self.alphas_cumprod[:-1]], dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ac * x0 + sqrt_om * noise, noise

    def p_losses(self, model, x0, t, cond_img):
        x_t, noise = self.q_sample(x0, t)
        eps_pred = model(x_t, t.float(), cond_img)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, model, shape, cond_img, steps=25):
        """
        DDPM sampling (teacher) or DDIM-like (student with fewer steps).
        """
        b, c, h, w = shape
        img = torch.randn(shape, device=self.device)
        ts = torch.linspace(self.timesteps-1, 0, steps, dtype=torch.long, device=self.device)

        for i, t in enumerate(ts):
            t_b = torch.full((b,), t, device=self.device, dtype=torch.float32)
            betas_t = self.betas[t]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]

            # predict noise
            eps = model(img, t_b, cond_img)

            # DDPM update
            model_mean = sqrt_recip_alphas_t * (img - betas_t / sqrt_one_minus_alphas_cumprod_t * eps)
            if t > 0:
                noise = torch.randn_like(img)
                posterior_var = self.posterior_variance[t]
                img = model_mean + torch.sqrt(posterior_var) * noise
            else:
                img = model_mean
        return img
