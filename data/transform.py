import math
import numbers
import numpy as np
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.fft as fft
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF


class TMS(torch.nn.Module): # updated radius
    def __init__(self,
                 p=0.5,
                 radius_range=(5, 15),
                 gamma_range=(0.6, 1.4),
                 freq_strength=0.5,       
                 inversion_p=0.4, 
                 spectral=True,
                 frequency=True,
                 inversion=True):
        super().__init__()
        self.p = p
        self.inversion_p = inversion_p
        self.radius_range = radius_range
        self.gamma_range = gamma_range
        self.freq_strength = freq_strength
        self.spectral = spectral
        self.frequency = frequency
        self.inversion = inversion
        
    def forward(self, img):
        if random.random() > self.p:
            return img
        is_pil = isinstance(img, Image.Image)
        img_tensor = TF.to_tensor(img) if is_pil else img

        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        device = img_tensor.device
        b, c, h, w = img_tensor.shape

        # ── Spectral compression ──────────────────────────────────────────────
        if self.spectral:
            mix_weights = torch.randn(b, 3, 1, 1, device=device).abs()
            mix_weights = mix_weights / (mix_weights.sum(dim=1, keepdim=True) + 1e-6)
            pseudo_ir = torch.sum(img_tensor * mix_weights, dim=1, keepdim=True)

        # ── Frequency enhancement ─────────────────────────────────────────────
        if self.frequency:
            f_shift = fft.fftshift(fft.fftn(pseudo_ir, dim=(-2, -1)))
            y_g, x_g = torch.meshgrid(
                torch.linspace(-1, 1, h, device=device),
                torch.linspace(-1, 1, w, device=device),
                indexing='ij'
            )
            r2 = x_g ** 2 + y_g ** 2  # (h, w)

            radius = torch.empty(b, 1, 1, 1, device=device).uniform_(*self.radius_range)
            sigma = radius / (min(h, w) / 2.0)       
            sigma2 = sigma ** 2 + 1e-8

            mask = 1.0 + self.freq_strength * (r2 / sigma2) * torch.exp(-r2 / (2 * sigma2))

            pseudo_ir = fft.ifftn(
                fft.ifftshift(f_shift * mask), dim=(-2, -1)
            ).real
            pseudo_ir = torch.clamp(pseudo_ir, 0.0, 1.0)

        if self.inversion and random.random() < self.inversion_p:
            pseudo_ir = 1.0 - pseudo_ir

        gamma = torch.empty(b, 1, 1, 1, device=device).uniform_(*self.gamma_range)
        pseudo_ir = torch.pow(torch.clamp(pseudo_ir, 0.0, 1.0) + 1e-6, gamma)

        mi = pseudo_ir.view(b, -1).min(1)[0].view(b, 1, 1, 1)
        ma = pseudo_ir.view(b, -1).max(1)[0].view(b, 1, 1, 1)
        dynamic_range = ma - mi
        pseudo_ir = torch.where(
            dynamic_range > 0.05,                            # only rescale if there is real range
            (pseudo_ir - mi) / (dynamic_range + 1e-6),
            pseudo_ir - mi                                   # near-flat image: just zero-base it
        )

        out_tensor = pseudo_ir.repeat(1, 3, 1, 1).squeeze(0)
        if is_pil:
            return TF.to_pil_image(out_tensor.cpu())
        return out_tensor

 
class NoTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img
