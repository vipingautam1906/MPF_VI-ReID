import torch
from torch import nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2, mutual_flag = False):
        super(TripletLoss, self).__init__()
        print('Triplet Loss')
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin) 
        self.mutual = mutual_flag

    def forward(self, inputs, targets, _):
        inputs = F.normalize(inputs, dim=1)
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n) 
        dist = dist + dist.t() 
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)  
        dist = dist.clamp(min=1e-12).sqrt() 
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0)) 
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0)) 
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss

