import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicPrototypeGenerator(nn.Module):
    def __init__(self, feat_dim, num_classes, tau=0.05):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.tau = tau
        self.register_buffer("prototype_memory", torch.zeros(num_classes, feat_dim))

        self.attention = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=8,
            batch_first=True
        )

    def forward(self, x, labels=None, cam_ids=None):
        if self.training and labels is not None:
            with torch.no_grad():
                for i, label in enumerate(labels):
                    if torch.norm(self.prototype_memory[label]) < 1e-6:
                        self.prototype_memory[label] = x[i]
                    else:
                        self.prototype_memory[label] = (
                            0.9 * self.prototype_memory[label] + 0.1 * x[i]
                        )
                    self.prototype_memory[label] = F.normalize(
                        self.prototype_memory[label], dim=0
                    )

        query = x.unsqueeze(1)                        # [B, 1, D]
        keys = self.prototype_memory.unsqueeze(0)     # [1, C, D]
        keys = keys.expand(x.size(0), -1, -1)         # [B, C, D]
        refined_feat, _ = self.attention(query, keys, keys)
        refined_feat = refined_feat.squeeze(1)        # [B, D]
        refined_feat = x + refined_feat
        refined_feat = F.normalize(refined_feat, dim=1)
        logits = torch.matmul(
            refined_feat, self.prototype_memory.t()
        ) / self.tau                                   # [B, C]

        return logits
