import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import torch

class Gate(nn.Module):
    def __init__(self, in_dim, T=3.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        self.T = T

    def forward(self, x):
        return torch.sigmoid(self.fc(x) / self.T)
    
def stripe_pool(stripes):
    pooled = []
    for s in stripes:
        f = torch.mean(s, dim=(2, 3))   
        f = F.normalize(f, dim=1)       
        pooled.append(f)
    return pooled

def horizontal_pool(feat_map, num_stripes=4):
    B, C, H, W = feat_map.size()
    stripe_h = H // num_stripes
    stripes = []
    for i in range(num_stripes):
        start = i * stripe_h
        end = H if i == num_stripes - 1 else (i + 1) * stripe_h
        stripes.append(feat_map[:, :, start:end, :])
    
    return stripe_pool(stripes)


def log_model_parameters(model, input_size=(1, 3, 384, 192)):
    dummy_input = torch.randn(input_size).cuda()
    _, params = profile(
        model,
        inputs=(dummy_input,),
        verbose=False
    )
    print(f"Parameters: {params:,}")
    print(f"Parameters (M): {params / 1e6:.2f}M")

