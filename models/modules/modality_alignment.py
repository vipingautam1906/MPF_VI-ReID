import torch
import torch.nn as nn
import torch.nn.functional as F

class IntermediateModalityAlignment(nn.Module):
    def __init__(self, feat_dim=1024, proj_dim=256, eps=1e-6):
        super().__init__()
        self.eps      = eps
        self.feat_dim = feat_dim   # store for assertion

        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, proj_dim), # 1024 -> 256
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x3, id_labels, mod_labels):
        assert x3.shape[1] == self.feat_dim, (
            f"IntermediateModalityAlignment expected {self.feat_dim} channels "
            f"(layer3 output), got {x3.shape[1]}. "
            f"Did you accidentally pass layer4 output (2048ch) instead?"
        )
        feats = self.proj(x3)
        feats = F.normalize(feats, dim=1)
        loss_center  = self._cross_modal_center_loss(feats, id_labels, mod_labels)
        loss_moments = self._moment_alignment_loss(feats, mod_labels)
        return loss_center + 0.5 * loss_moments
    

    def _cross_modal_center_loss(self, feats, id_labels, mod_labels):
        loss  = torch.tensor(0.0, device=feats.device)
        count = 0
        for pid in torch.unique(id_labels):
            id_mask  = id_labels == pid
            rgb_mask = id_mask & (mod_labels == 0) 
            ir_mask  = id_mask & (mod_labels == 1)
            if rgb_mask.sum() == 0 or ir_mask.sum() == 0:
                continue

            rgb_center = feats[rgb_mask].mean(0)    # proj_dim
            ir_center  = feats[ir_mask].mean(0)     # proj_dim

            loss  = loss + (1 - F.cosine_similarity(
                rgb_center.unsqueeze(0),
                ir_center.unsqueeze(0)
            ).squeeze())
            count += 1

        return loss / max(count, 1)

    def _moment_alignment_loss(self, feats, mod_labels):
        rgb_feats = feats[mod_labels == 0]
        ir_feats  = feats[mod_labels == 1]

        if rgb_feats.shape[0] < 2 or ir_feats.shape[0] < 2:
            return torch.tensor(0.0, device=feats.device)

        # First moment (mean)
        rgb_mean = rgb_feats.mean(0)
        ir_mean  = ir_feats.mean(0)
        loss_mean = F.mse_loss(rgb_mean, ir_mean)

        # Second moment (variance)
        rgb_var = rgb_feats.var(0)
        ir_var  = ir_feats.var(0)
        loss_var = F.mse_loss(rgb_var, ir_var)

        return loss_mean + loss_var
