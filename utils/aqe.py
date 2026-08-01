import torch
import torch.nn.functional as F

def aqe(q_feats, g_feats, k=5, alpha=3.0):
    q_norm = F.normalize(q_feats, dim=1)
    g_norm = F.normalize(g_feats, dim=1)
    dist = 1 - q_norm @ g_norm.T  # Q × G
    
    for i in range(len(q_feats)):
        topk_idx = dist[i].argsort()[:k]
        topk_feats = g_feats[topk_idx]
        weights = (1 - dist[i][topk_idx]) ** alpha
        q_feats[i] = F.normalize(
            q_feats[i] + (weights.unsqueeze(1) * topk_feats).sum(0), dim=0
        )
    return q_feats