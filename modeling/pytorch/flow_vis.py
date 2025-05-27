import torch
import torch.nn as nn

import sys

sys.path.append(".")

from modeling.Flow_based.ResFlow import ResFlow
from modeling.Flow_based.RealNVP import RealNVP
from modeling.Flow_based.FlowPP import FlowPP


class FlowVis(nn.Module):
    def __init__(self, nf_type, dims, cfg):
        super(FlowVis, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc1 = nn.Linear(128, dims[0])
        if nf_type == "ResFlow":
            self.flow = ResFlow(dims=dims, cfg=cfg)
        elif nf_type == "RealNVP":
            self.flow = RealNVP(dims=dims, cfg=cfg)
        elif nf_type == "FlowPP":
            self.flow = FlowPP(dims=dims, cfg=cfg)
        else:
            raise ValueError

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.flow(x)
        return x
