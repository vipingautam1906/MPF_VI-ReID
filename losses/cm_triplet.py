import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalityTripletLoss(nn.Module):

    def __init__(self, margin=0.2, mutual_flag=False):
        super(CrossModalityTripletLoss, self).__init__()
        print('CM Triplet Loss')
        self.margin = margin
        self.mutual_flag = mutual_flag
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, id_labels, modality_labels):

        """
        embeddings      : [B, D]
        id_labels       : [B]
        modality_labels : [B]

        modality_labels:
            0 -> RGB
            1 -> IR
        """

        embeddings = F.normalize(embeddings, p=2, dim=1)
        batch_size = embeddings.size(0)
        dist = torch.pow(embeddings, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size)
        dist = dist + dist.t()
        dist.addmm_(embeddings, embeddings.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()
        id_mask = id_labels.expand(batch_size, batch_size).eq(
            id_labels.expand(batch_size, batch_size).t()
        )

        modality_mask = modality_labels.expand(batch_size, batch_size).ne(
            modality_labels.expand(batch_size, batch_size).t()
        )

        positive_mask = id_mask & modality_mask
        negative_mask = (~id_mask) & modality_mask
        hardest_positive_dist = []
        hardest_negative_dist = []

        for i in range(batch_size):
            positive_dist = dist[i][positive_mask[i]]
            negative_dist = dist[i][negative_mask[i]]

            if len(positive_dist) == 0 or len(negative_dist) == 0:
                continue

            hardest_positive_dist.append(
                positive_dist.max().unsqueeze(0)
            )

            hardest_negative_dist.append(
                negative_dist.min().unsqueeze(0)
            )

        if len(hardest_positive_dist) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        hardest_positive_dist = torch.cat(hardest_positive_dist)
        hardest_negative_dist = torch.cat(hardest_negative_dist)
        target = torch.ones_like(hardest_negative_dist)

        loss = self.ranking_loss(
            hardest_negative_dist,
            hardest_positive_dist,
            target
        )

        if self.mutual_flag:
            return loss, dist

        return loss