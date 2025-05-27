import sys
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet34_Weights, resnet34

sys.path.append(".")


class pretrained_MelResnet34_IDdiscriminator(nn.Module):
    def __init__(self, id_num, bottleneck, dropout=0.2):
        super().__init__()

        conv2d = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(3, 106), stride=(1, 1)),
            nn.BatchNorm2d(3),
            nn.ReLU6(True),
        )
        _resnet34 = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        self.encoder = nn.Sequential(
            conv2d,
            _resnet34,
            nn.ReLU(True),
            nn.Linear(1000, bottleneck),
        )

        self.discriminator = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(bottleneck, id_num),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        bn = self.encoder(x)
        output = self.discriminator(bn)
        return output, bn
