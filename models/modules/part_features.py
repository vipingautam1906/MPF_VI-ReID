import torch
import torch.nn as nn
import torch.nn.functional as F

class Part_Features(nn.Module):
    def __init__(self, in_channels, num_parts=6, p=3):
        super().__init__()
        self.num_parts = num_parts
        self.p = p
        self.part_p = nn.Parameter(torch.ones(num_parts))

    def forward(self, x):
        x = x.clamp(min=1e-12)
        part_strips = torch.chunk(x, self.num_parts, dim=2)
        part_features = []
        for strip in part_strips:
            feat = torch.mean(strip.pow(self.p), dim=(2,3)).pow(1/self.p)
            part_features.append(feat)
        part_features = torch.stack(part_features, dim=1)  # B × P × C
        confidences = None
        return part_features, confidences
