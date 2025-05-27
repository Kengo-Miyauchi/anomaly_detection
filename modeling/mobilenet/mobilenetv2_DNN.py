import sys
import torch.nn as nn

sys.path.append(".")
from modeling.mobilenet.InvertedResidual import InvertedResidual, InvertedResidualWithCoordAttention
from modeling.mobilenet.CoordAttention import CoordAttention


class MelMobileNetV2_DNN(nn.Module):
    def __init__(self, bottleneck, mid1):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), padding_mode="replicate")
        self.ir1 = InvertedResidualWithCoordAttention(16, 32, kernel_size=(1, 3), stride=1)
        self.ir2 = InvertedResidualWithCoordAttention(16, 32, kernel_size=(3, 3), stride=2)
        self.ir3 = InvertedResidualWithCoordAttention(32, 32, kernel_size=(1, 3), stride=1)
        self.ir4 = InvertedResidualWithCoordAttention(32, 32, kernel_size=(3, 3), stride=2)
        self.upsample1 = nn.Upsample((256, 16))
        self.ir5 = InvertedResidual(32, 32, kernel_size=(3, 3), stride=2)
        self.ir6 = InvertedResidual(32, 32, kernel_size=(3, 3), stride=1, use_res_connect=True)
        self.upsample2 = nn.Upsample((256, 16))
        self.conv2 = nn.Conv2d(32, bottleneck, kernel_size=(1, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.relu1 = nn.ReLU6(True)
        self.avg1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(bottleneck, mid1)
        self.relu2 = nn.ReLU(True)
        self.fc2 = nn.Linear(mid1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)

        x_up = self.ir1(x)
        x_up = self.ir3(x_up)

        x_down = self.ir2(x)
        x_down = self.ir4(x_down)
        x_down = self.upsample1(x_down)

        x = x_up + x_down

        x_up = x

        x_down = self.ir5(x)
        x_down = self.ir6(x_down)
        x_down = self.upsample2(x_down)

        x = x_up + x_down

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.avg1(x)
        bn = x.flatten(1)

        output = self.fc1(bn)
        output = self.relu2(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        return output, bn


class MelMobileNetV2_DNN2(nn.Module):
    def __init__(self, bottleneck, mid1):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(True)
        self.mobile_block1 = InvertedResidual(32, 32, kernel_size=(3, 3), stride=(2, 2))
        self.mobile_block2 = InvertedResidual(32, 48, kernel_size=(3, 3), stride=(2, 2))
        self.moible_block3 = InvertedResidual(48, 64, kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.dropout = nn.Dropout(0.3)
        self.ca_attention = CoordAttention(64, 64, groups=4)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, bottleneck, kernel_size=(1, 1), stride=(1, 1))
        self.bn4 = nn.BatchNorm2d(bottleneck)

    def forward(self, x):
        x = self.conv1(x)

        x_up = self.ir1(x)
        x_up = self.ir3(x_up)

        x_down = self.ir2(x)
        x_down = self.ir4(x_down)
        x_down = self.upsample1(x_down)

        x = x_up + x_down

        x_up = x

        x_down = self.ir5(x)
        x_down = self.ir6(x_down)
        x_down = self.upsample2(x_down)

        x = x_up + x_down

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.avg1(x)
        bn = x.flatten(1)

        output = self.fc1(bn)
        output = self.relu2(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        return output, bn


class MelMobileNetV2_DNN3(nn.Module):
    def __init__(self, bottleneck, mid1):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), padding_mode="replicate")
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), padding_mode="replicate")
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), padding_mode="replicate")
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), padding_mode="replicate")
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, bottleneck, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), padding_mode="replicate")
        self.bn5 = nn.BatchNorm2d(bottleneck)
        self.avg1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(bottleneck, mid1)
        self.fc2 = nn.Linear(mid1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.avg1(x)

        bn = x.flatten(1)

        output = self.fc1(bn)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.sigmoid(output)

        return output, bn
