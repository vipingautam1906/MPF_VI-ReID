from typing import Tuple

import torch
from torch import nn, Tensor


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    """
    Circle Loss: A Unified Perspective of Pair Similarity Optimization (CVPR 2020)
    https://arxiv.org/abs/2002.10857

    """

    def __init__(self, m: float = 0.25, gamma: float = 16.0):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()
        print("Circle Loss")

    def forward(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor, int]:
        normed_feats = nn.functional.normalize(inputs, dim=-1)
        sp, sn = convert_label_to_similarity(normed_feats, targets)

        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        with torch.no_grad():
            if sn.numel() > 0 and sp.numel() > 0:
                correct = int(sn.max() < sp.min()) * sn.numel()
            else:
                correct = 0

        return loss, correct

    