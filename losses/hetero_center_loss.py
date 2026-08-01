import torch
import torch.nn as nn


class HeteroCenterLoss(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
    "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
    (arxiv: https://arxiv.org/abs/2008.06223)

    Groups features into per-identity, per-modality centers using explicit
    label/modality masks instead of positional chunk(). 
    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, batch_size=128, margin=0.3):
        super(HeteroCenterLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        print("Hetero Center Loss")

    def forward(self, feats, labels, modality_labels):
        """
        Args:
        - feats: feature matrix with shape (batch_size, feat_dim)
        - labels: identity ground-truth labels, shape (batch_size,)
        - modality_labels: modality indicator, shape (batch_size,)
                            e.g. 0 = visible, 1 = IR
        """
        label_uni = labels.unique()
        modalities = modality_labels.unique()
        assert len(modalities) == 2, "Expected exactly 2 modalities in the batch"

        # Keep only identities that have samples in BOTH modalities this batch
        valid_labels = []
        for lbl in label_uni:
            has_both = all(
                ((labels == lbl) & (modality_labels == m)).sum() > 0
                for m in modalities
            )
            if has_both:
                valid_labels.append(lbl)

        if len(valid_labels) < 2:
            # Not enough identities with both modalities to form pos/neg pairs
            return torch.tensor(0.0, device=feats.device, requires_grad=True), 0

        valid_labels = torch.stack(valid_labels)

        targets = []
        centers = []
        for m in modalities:
            for lbl in valid_labels:
                mask = (labels == lbl) & (modality_labels == m)
                centers.append(feats[mask].mean(dim=0, keepdim=True))
                targets.append(lbl)

        inputs = torch.cat(centers)           # (num_valid*2, feat_dim)
        targets = torch.stack(targets)        # (num_valid*2,)

        n = inputs.size(0)

        # Compute pairwise distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist = dist - 2 * torch.mm(inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct