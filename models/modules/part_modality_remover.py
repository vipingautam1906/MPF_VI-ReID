import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.tools import set_seed

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None

class PartModalityRemover(nn.Module):
    def __init__(self, feat_dim, num_parts, eps=1e-5):
        super().__init__()
        self.num_parts = num_parts
        self.eps = eps

        self.linear_projection = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.LayerNorm(feat_dim)
            ) for _ in range(num_parts)
        ])

        self.modality_classifier = nn.ModuleList([
            nn.Linear(feat_dim, 2)
            for _ in range(num_parts)
        ])

        self.recon_heads = nn.ModuleList([
            nn.Linear(feat_dim, feat_dim)
            for _ in range(num_parts)
        ])

    
    def forward(self, part_feats, modality_labels=None, labels=None, grl_alpha=1.0):        
        B, P, D = part_feats.shape
        loss_PMR = 0.0
        purified_parts = []

        total_loss_cov   = 0.0
        total_loss_mod   = 0.0
        total_loss_recon = 0.0

        if self.training:
            if modality_labels is None or labels is None:
                raise ValueError(
                    "modality_labels and labels are required during training."
                )
            for i in range(P):
                pf   = part_feats[:, i]         # B × D
                pf_w = self.linear_projection[i](pf) # B × D

                loss_cov = self.Covariance_Supression(pf_w, labels, modality_labels)
                pf_w_rev   = GradientReversal.apply(pf_w, grl_alpha)
                mod_logits = self.modality_classifier[i](pf_w_rev)
                loss_mod   = F.cross_entropy(mod_logits, modality_labels)  
                pf_recon   = self.recon_heads[i](pf_w)
                loss_recon = F.mse_loss(pf_recon, pf.detach())
                total_loss_cov   += loss_cov
                total_loss_mod   += loss_mod
                total_loss_recon += loss_recon

                loss_PMR += loss_cov + 0.5 * loss_mod + 0.5 * loss_recon
                purified_parts.append(pf_w)

            purified_parts = torch.stack(purified_parts, dim=1)  # B × P × D

            breakdown = {
                "loss_cov":   (total_loss_cov   / P).detach(),
                "loss_mod":   (total_loss_mod   / P).detach(),
                "loss_recon": (total_loss_recon / P).detach(),
            }

            return purified_parts, loss_PMR / P, breakdown
        else:
            for i in range(P):
                pf   = part_feats[:, i]
                pf_w = self.linear_projection[i](pf)
                purified_parts.append(pf_w)
            purified_parts = torch.stack(purified_parts, dim=1)
            return purified_parts
        
    def Covariance_Supression(self, feats, labels, modality_labels):
        loss  = 0.0
        count = 0
        for pid in torch.unique(labels):
            mask = (labels == pid)
            if mask.sum() < 2:
                continue
            id_feats = feats[mask]
            id_mods  = modality_labels[mask].float()
            id_feats_c = id_feats - id_feats.mean(dim=0, keepdim=True)
            id_mods_c  = id_mods  - id_mods.mean()
            if id_mods_c.std() < 1e-6:
                continue
            cov   = (id_mods_c.unsqueeze(1) * id_feats_c).mean(dim=0)  # D
            loss += (cov ** 2).mean()
            count += 1
        return loss / max(count, 1)



