import torch.nn as nn
from modeling.mobilenet.CoordAttention import CoordAttention


class InvertedResidual(nn.Module):
    def __init__(self, input, output, kernel_size, stride, expand_ratio=6, use_res_connect=False):
        super().__init__()

        if len(kernel_size) != 2:
            raise ValueError("kernel should be two dimensions.")

        hidden_dim = int(round(input * expand_ratio))
        is_same = True
        if isinstance(stride, int):
            is_same = (stride == 1)
        else:
            for i in stride:
                if i != 1:
                    is_same = False
                    break
        self.use_res_connect = use_res_connect and is_same and (input == output)

        layers = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(input, hidden_dim, kernel_size=1, stride=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=hidden_dim,
                    padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                    padding_mode="replicate",
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(True),
                CoordAttention(hidden_dim, hidden_dim),
                nn.Conv2d(hidden_dim, output, kernel_size=1, stride=1),
                nn.ReLU6(True),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidualWithCoordAttention(nn.Module):
    def __init__(self, input, output, kernel_size, stride, expand_ratio=6, use_res_connect=False):
        super().__init__()

        if len(kernel_size) != 2:
            raise ValueError("kernel should be two dimensions.")
        if not isinstance(stride, int):
            raise ValueError("stride should be integer.")

        hidden_dim = int(round(input * expand_ratio))
        self.use_res_connect = use_res_connect and stride == 1 and (input == output)

        layers = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(input, hidden_dim, kernel_size=1, stride=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=hidden_dim,
                    padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                    padding_mode="replicate",
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(True),

                CoordAttention(hidden_dim, hidden_dim),

                nn.Conv2d(hidden_dim, output, kernel_size=1, stride=1),
                nn.ReLU6(True),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
