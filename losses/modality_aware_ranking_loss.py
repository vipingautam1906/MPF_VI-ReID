import torch
import torch.nn as nn
import torchsort
from torch.nn import functional as F

class MARLoss(nn.Module):
    def __init__(
        self,
        margin=0.2, # [0.1, 0.2, 0.3, 0.5]
        regularization="l2",
        reg_strength=0.5,
        alpha=4.0,     # [2.0, 4.0, 6.0, 8.0, 10.0]     # 4 gives best
        eps=1e-12,
    ):
        super().__init__()
        self.margin = margin
        self.reg = regularization
        self.reg_strength = reg_strength
        self.alpha = alpha
        self.eps = eps

        # print(
        #     f"[Modality Aware Ranking Loss Hyperparameters] "
        #     f"margin={self.margin}, "
        #     f"regularization={self.reg}, "
        #     f"reg_strength={self.reg_strength}, "
        #     f"alpha={self.alpha}, "
        #     f"eps={self.eps}"
        # )

    def forward(self, embeddings, id_labels, m_labels):
        embeddings = F.normalize(embeddings, dim=1)
        
        modalities = torch.unique(m_labels)
        total_loss = 0.0
        valid_m_count = 0

        for m in modalities:
            q_mask = m_labels == m
            g_mask = m_labels != m

            if q_mask.sum() == 0 or g_mask.sum() == 0:
                continue

            q_emb = embeddings[q_mask]
            q_ids = id_labels[q_mask]

            g_emb = embeddings[g_mask]
            g_ids = id_labels[g_mask]

            dist_mat = (1 - torch.matmul(q_emb, g_emb.t())) / 2             # (1 - consin similarlity)

            ranks = torchsort.soft_rank(
                dist_mat,
                regularization=self.reg,
                regularization_strength=self.reg_strength
            )

            pos_mask = q_ids.unsqueeze(1) == g_ids.unsqueeze(0)
            neg_mask = ~pos_mask

            pos_count = pos_mask.sum(dim=1)
            neg_count = neg_mask.sum(dim=1)
            valid_q = (pos_count > 0) & (neg_count > 0)

            if valid_q.sum() == 0:
                continue

            dist_mat = dist_mat[valid_q]
            ranks = ranks[valid_q]
            pos_mask = pos_mask[valid_q]
            neg_mask = neg_mask[valid_q]

            pos_ranks = ranks.masked_fill(~pos_mask, 1e6) 
            best_pos_idx = pos_ranks.argmin(dim=1) 
            best_pos_dist = dist_mat.gather(
                1, best_pos_idx.unsqueeze(1)
            ).squeeze(1)

            neg_ranks = ranks.masked_fill(~neg_mask, 1e6)

            neg_weights = torch.exp(-self.alpha * neg_ranks)
            neg_weights = neg_weights * neg_mask.float()
            neg_weights = neg_weights / (neg_weights.sum(dim=1, keepdim=True) + self.eps)

            weighted_neg_dist = (dist_mat * neg_weights).sum(dim=1)

            loss = torch.log1p(
                torch.exp(best_pos_dist - weighted_neg_dist + self.margin)
            ).mean()

            total_loss += loss
            valid_m_count += 1

        if valid_m_count == 0:
            return torch.tensor(0.0, device=embeddings.device)

        return total_loss / valid_m_count
