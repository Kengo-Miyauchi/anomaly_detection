import torch
import torch.nn as nn
import torchvision
import sys
from distutils.version import LooseVersion

sys.path.append(".")
from modeling.mobilenet.InvertedResidual import InvertedResidual, InvertedResidualWithCoordAttention
from modeling.mobilenet.ArcMarginProduct import ArcMarginProduct


if LooseVersion(torchvision.__version__) >= LooseVersion("0.9.0"):
    from torchvision.models.mobilenetv2 import MobileNetV2, MobileNet_V2_Weights

    class MobileNetV2_IDdiscriminator(nn.Module):
        def __init__(self, inverted_residual_setting, id_num, bottleneck, dropout=0.2):
            super().__init__()

            conv2d = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=(2001, 55), stride=(209, 1)),
                nn.BatchNorm2d(3),
                nn.ReLU6(True),
            )
            mobilenet = MobileNetV2(inverted_residual_setting=inverted_residual_setting, dropout=dropout)

            self.encoder = nn.Sequential(
                conv2d,
                mobilenet,
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

    class MelMobileNetV2_IDdiscriminator(nn.Module):
        def __init__(self, inverted_residual_setting, id_num, bottleneck, dropout=0.2):
            super().__init__()

            conv2d = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=(3, 106), stride=(1, 1)),
                nn.BatchNorm2d(3),
                nn.ReLU6(True),
            )
            mobilenet = MobileNetV2(inverted_residual_setting=inverted_residual_setting, dropout=dropout)

            self.encoder = nn.Sequential(
                conv2d,
                mobilenet,
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

    class pretrainedMelMobileNetV2_IDdiscriminator(nn.Module):
        def __init__(self, inverted_residual_setting, id_num, bottleneck, dropout=0.2):
            super().__init__()

            conv2d = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=(3, 106), stride=(1, 1)),
                nn.BatchNorm2d(3),
                nn.ReLU6(True),
            )
            mobilenet = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", weights=MobileNet_V2_Weights.DEFAULT)

            self.encoder = nn.Sequential(
                conv2d,
                mobilenet,
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

    class MelMobileNetV2_IDdiscriminator2(nn.Module):
        def __init__(self, inverted_residual_setting, id_num, bottleneck, dropout=0.2):
            super().__init__()

            self.conv2d = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(3),
                nn.ReLU6(True),
            )
            self.mobilenet = MobileNetV2(inverted_residual_setting=inverted_residual_setting, dropout=dropout)
            self.fc1 = nn.Linear(1000, bottleneck)
            self.fc2 = nn.Linear(bottleneck, id_num)
            self.log_softmax = nn.LogSoftmax(dim=1)
            self.relu = nn.ReLU(True)

        def forward(self, x):
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
            x = self.conv2d(x)
            x = self.mobilenet(x)
            x = self.relu(x)
            bn = self.fc1(x)
            x = self.relu(bn)
            x = self.fc2(x)
            output = self.log_softmax(x)
            return output, bn

    class MelMobileNetV2_IDdiscriminator3(nn.Module):
        def __init__(self, inverted_residual_setting, id_num, bottleneck, dropout=0.2):
            super().__init__()

            self.mobilenet = MobileNetV2(inverted_residual_setting=inverted_residual_setting, dropout=dropout)
            self.fc1 = nn.Linear(1000, bottleneck)
            self.fc2 = nn.Linear(bottleneck, id_num)
            self.log_softmax = nn.LogSoftmax(dim=1)
            self.relu = nn.ReLU(True)

        def forward(self, x):
            x = self.mobilenet(x)
            x = self.relu(x)
            bn = self.fc1(x)
            x = self.relu(bn)
            x = self.fc2(x)
            output = self.log_softmax(x)
            return output, bn


class MelMobileNetV2_IDdiscriminator4(nn.Module):
    def __init__(self, bottleneck, id_num):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), padding_mode="replicate")
        self.ir1 = InvertedResidualWithCoordAttention(16, 32, kernel_size=(1, 3), stride=1)
        self.ir2 = InvertedResidualWithCoordAttention(16, 32, kernel_size=(3, 3), stride=2)
        self.ir3 = InvertedResidualWithCoordAttention(32, 32, kernel_size=(1, 3), stride=1)
        self.ir4 = InvertedResidualWithCoordAttention(32, 32, kernel_size=(3, 3), stride=2)
        self.upsample1 = nn.Upsample((256, 180))
        self.ir5 = InvertedResidual(32, 32, kernel_size=(3, 3), stride=2)
        self.ir6 = InvertedResidual(32, 32, kernel_size=(3, 3), stride=1, use_res_connect=True)
        self.upsample2 = nn.Upsample((256, 180))
        self.conv2 = nn.Conv2d(32, bottleneck, kernel_size=(1, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.relu1 = nn.ReLU6(True)
        self.avg1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(bottleneck, id_num)
        self.log_softmax = nn.LogSoftmax(dim=1)

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
        output = self.log_softmax(output)
        return output, bn


class MelMobileNetV2_IDdiscriminator5(nn.Module):
    def __init__(self, bottleneck, mid1, id_num):
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
        self.fc2 = nn.Linear(mid1, id_num)
        self.log_softmax = nn.LogSoftmax(dim=1)

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
        output = self.log_softmax(output)

        return output, bn


class MelMobileNetV2_IDdiscriminator6(nn.Module):
    def __init__(self, bottleneck, mid1, id_num, width):
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
        self.avg1 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(bottleneck, mid1)
        self.fc2 = nn.Linear(mid1, id_num)
        self.log_softmax = nn.LogSoftmax(dim=1)

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
        output = self.log_softmax(output)

        return output, bn


class MelMobileNetV2_IDdiscriminator6_1(nn.Module):
    def __init__(self, bottleneck, mid1, id_num, width):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 3), stride=(2, 2), padding=(1, 1), padding_mode="replicate")
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
        self.avg1 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(bottleneck, mid1)
        self.fc2 = nn.Linear(mid1, id_num)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, label):
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
        if label is None:
            return None, bn

        output = self.fc1(bn)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.log_softmax(output)

        return output, bn


class MelMobileNetV2_IDdiscriminator6_2(nn.Module):
    def __init__(self, bottleneck, mid1, id_num, width):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 3), stride=(2, 2), padding=(1, 1), padding_mode="replicate")
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
        self.avg1 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(bottleneck, mid1)
        self.arc_face = ArcMarginProduct(mid1, id_num, m=0.1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, label):
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
        if label is None:
            return None, bn

        output = self.fc1(bn)
        output = self.relu(output)
        output = self.arc_face(output, label)
        output = self.log_softmax(output)

        return output, bn


class MelMobileNetV2_IDdiscriminator7(nn.Module):
    def __init__(self, bottleneck, mid1, id_num, width):
        super().__init__()

        self.relu = nn.ReLU(True)
        self.ir1 = InvertedResidualWithCoordAttention(3, 16, kernel_size=(3, 3), stride=2)
        self.ir2 = InvertedResidualWithCoordAttention(16, 16, kernel_size=(3, 3), stride=2)
        self.ir3 = InvertedResidualWithCoordAttention(16, 32, kernel_size=(3, 3), stride=2)
        self.ir4 = InvertedResidualWithCoordAttention(32, 32, kernel_size=(3, 3), stride=2)
        self.ir5 = InvertedResidualWithCoordAttention(32, bottleneck, kernel_size=(3, 3), stride=2)
        self.avg1 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(bottleneck, mid1)
        self.fc2 = nn.Linear(mid1, id_num)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.ir1(x)
        x = self.ir2(x)
        x = self.ir3(x)
        x = self.ir4(x)
        x = self.ir5(x)
        x = self.avg1(x)

        bn = x.flatten(1)

        output = self.fc1(bn)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.log_softmax(output)

        return output, bn


class MelMobileNetV2_IDdiscriminator8(nn.Module):
    def __init__(self, bottleneck, mid1, id_num, width):
        super().__init__()

        self.relu = nn.ReLU(True)
        self.ir1 = InvertedResidualWithCoordAttention(1, 8, kernel_size=(3, 3), stride=2)
        self.ir2 = InvertedResidualWithCoordAttention(8, 64, kernel_size=(3, 3), stride=2)
        self.ir3 = InvertedResidualWithCoordAttention(64, 512, kernel_size=(3, 3), stride=2)
        self.ir4 = InvertedResidualWithCoordAttention(512, 64, kernel_size=(3, 3), stride=2)
        self.ir5 = InvertedResidualWithCoordAttention(64, bottleneck, kernel_size=(3, 3), stride=2)

        self.avg1 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(bottleneck, mid1)
        self.fc2 = nn.Linear(mid1, id_num)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, label):
        x = self.ir1(x)
        x = self.ir2(x)
        x = self.ir3(x)
        x = self.ir4(x)
        x = self.ir5(x)
        x = self.avg1(x)

        bn = x.flatten(1)

        if label is None:
            return None, bn

        output = self.fc1(bn)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.log_softmax(output)

        return output, bn


class MelMobileNetV2_IDdiscriminator8_1(nn.Module):
    def __init__(self, bottleneck, mid1, id_num, width):
        super().__init__()

        self.relu = nn.ReLU(True)
        self.ir1 = InvertedResidualWithCoordAttention(1, 8, kernel_size=(3, 3), stride=2)
        self.ir2 = InvertedResidualWithCoordAttention(8, 64, kernel_size=(3, 3), stride=2)
        self.ir3 = InvertedResidualWithCoordAttention(64, 512, kernel_size=(3, 3), stride=2)
        self.ir4 = InvertedResidualWithCoordAttention(512, 64, kernel_size=(3, 3), stride=2)
        self.ir5 = InvertedResidualWithCoordAttention(64, bottleneck, kernel_size=(3, 3), stride=2)

        self.avg1 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(bottleneck, mid1)
        self.arc_face = ArcMarginProduct(mid1, id_num)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, label):
        x = self.ir1(x)
        x = self.ir2(x)
        x = self.ir3(x)
        x = self.ir4(x)
        x = self.ir5(x)
        x = self.avg1(x)

        bn = x.flatten(1)

        if label is None:
            return None, bn

        output = self.fc1(bn)
        output = self.relu(output)
        output = self.arc_face(output, label)
        output = self.log_softmax(output)

        return output, bn


# Based on MobileFaceNet
class MelMobileNetV2_IDdiscriminator9(nn.Module):
    def __init__(self, bottleneck, mid1, id_num, width):
        super().__init__()

        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=2, padding=1, padding_mode="replicate")
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, padding_mode="replicate")
        self.bn2 = nn.BatchNorm2d(64)

        self.ir1 = InvertedResidual(64, 128, kernel_size=(3, 3), stride=2, expand_ratio=4)
        self.ir2 = InvertedResidual(128, 128, kernel_size=(3, 3), stride=1, expand_ratio=4)

        self.ir3 = InvertedResidual(128, 128, kernel_size=(3, 3), stride=2, expand_ratio=4)
        self.ir4 = InvertedResidual(128, 128, kernel_size=(3, 3), stride=1, expand_ratio=4)

        self.ir5 = InvertedResidual(128, 128, kernel_size=(3, 3), stride=2, expand_ratio=4)
        self.ir6 = InvertedResidual(128, 128, kernel_size=(3, 3), stride=1, expand_ratio=4)

        self.conv3 = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=1)
        self.bn3 = nn.BatchNorm2d(512)

        self.conv4 = nn.Conv2d(512, 512, kernel_size=(7, 4), stride=1, groups=512)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, bottleneck, kernel_size=(1, 1), stride=1)
        self.bn5 = nn.BatchNorm2d(bottleneck)

        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(bottleneck, mid1)
        self.fc2 = nn.Linear(mid1, id_num)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.ir1(x)
        x = self.ir2(x)
        x = self.ir3(x)
        x = self.ir4(x)
        x = self.ir5(x)
        x = self.ir6(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.bn5(x)

        bn = x.flatten(1)

        output = self.dropout(bn)

        output = self.fc1(output)
        output = self.log_softmax(output)
        return output, bn


# Based on MobileFaceNet
class MelMobileNetV2_IDdiscriminator10(nn.Module):
    def __init__(self, bottleneck, mid1, id_num, width):
        super().__init__()

        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=2, padding=1, padding_mode="replicate")
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=1, padding_mode="replicate")
        self.bn2 = nn.BatchNorm2d(16)

        self.ir1 = InvertedResidual(16, 32, kernel_size=(3, 3), stride=2, expand_ratio=4)

        self.ir3 = InvertedResidual(32, 32, kernel_size=(3, 3), stride=2, expand_ratio=4)

        self.ir5 = InvertedResidual(32, 64, kernel_size=(3, 3), stride=2, expand_ratio=4)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=(16, 1), stride=1, groups=64)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, bottleneck, kernel_size=(1, 1), stride=1)
        self.bn6 = nn.BatchNorm2d(bottleneck)

        self.fc1 = nn.Linear(bottleneck, id_num)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.ir1(x)
        x = self.ir3(x)
        x = self.ir5(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        x = self.conv6(x)
        x = self.bn6(x)

        bn = x.flatten(1)

        output = self.fc1(bn)
        output = self.log_softmax(output)
        return output, bn


class MelMobileNetV2_IDdiscriminator10_1(nn.Module):
    def __init__(self, bottleneck, mid1, id_num, width):
        super().__init__()

        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=2, padding=1, padding_mode="replicate")
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=1, padding=1, padding_mode="replicate")
        self.bn2 = nn.BatchNorm2d(8)

        self.ir1 = InvertedResidual(8, 16, kernel_size=(3, 3), stride=2, expand_ratio=6)

        self.ir3 = InvertedResidual(16, 16, kernel_size=(3, 3), stride=2, expand_ratio=6)

        self.ir5 = InvertedResidual(16, 16, kernel_size=(3, 3), stride=2, expand_ratio=6)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=1)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=(16, 1), stride=1, groups=16)
        self.bn4 = nn.BatchNorm2d(16)

        self.conv6 = nn.Conv2d(16, bottleneck, kernel_size=(1, 1), stride=1)
        self.bn6 = nn.BatchNorm2d(bottleneck)

        self.fc1 = nn.Linear(bottleneck, id_num)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, label):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)

        x = self.ir1(x)
        x = self.ir3(x)
        x = self.ir5(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        # x = self.bn4(x)

        x = self.conv6(x)
        # x = self.bn6(x)

        bn = x.flatten(1)

        output = self.fc1(bn)
        output = self.log_softmax(output)
        return output, bn


class MelMobileNetV2_IDdiscriminator10_2(nn.Module):
    def __init__(self, bottleneck, mid1, id_num, width):
        super().__init__()

        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=2, padding=1, padding_mode="replicate")
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=1, padding=1, padding_mode="replicate")
        self.bn2 = nn.BatchNorm2d(8)

        self.ir1 = InvertedResidual(8, 16, kernel_size=(3, 3), stride=2, expand_ratio=6)

        self.ir3 = InvertedResidual(16, 16, kernel_size=(3, 3), stride=2, expand_ratio=6)

        self.ir5 = InvertedResidual(16, 16, kernel_size=(3, 3), stride=2, expand_ratio=6)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=1)
        self.bn3 = nn.BatchNorm2d(16)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        # self.avg = nn.AvgPool2d((16, 1))

        self.conv6 = nn.Conv2d(16, bottleneck, kernel_size=(1, 1), stride=1)
        self.bn6 = nn.BatchNorm2d(bottleneck)

        self.fc1 = nn.Linear(bottleneck, id_num)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, label):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)

        x = self.ir1(x)
        x = self.ir3(x)
        x = self.ir5(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)

        x = self.avg(x)
        # x = self.bn4(x)

        x = self.conv6(x)
        # x = self.bn6(x)

        bn = x.flatten(1)

        output = self.fc1(bn)
        output = self.log_softmax(output)
        return output, bn


class MelMobileNetV2_IDdiscriminator10_3(nn.Module):
    def __init__(self, bottleneck, mid1, id_num, width, arc_m):
        super().__init__()

        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=2, padding=1, padding_mode="replicate")
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=1, padding=1, padding_mode="replicate")
        self.bn2 = nn.BatchNorm2d(8)

        self.ir1 = InvertedResidual(8, 16, kernel_size=(3, 3), stride=2, expand_ratio=6)

        self.ir3 = InvertedResidual(16, 16, kernel_size=(3, 3), stride=2, expand_ratio=6)

        self.ir5 = InvertedResidual(16, 16, kernel_size=(3, 3), stride=2, expand_ratio=6)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=1)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=(16, 1), stride=1, groups=16)
        self.bn4 = nn.BatchNorm2d(16)

        self.conv6 = nn.Conv2d(16, bottleneck, kernel_size=(1, 1), stride=1)
        self.bn6 = nn.BatchNorm2d(bottleneck)

        self.arcface = ArcMarginProduct(bottleneck, id_num, m=arc_m)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, label):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)

        x = self.ir1(x)
        x = self.ir3(x)
        x = self.ir5(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        # x = self.bn4(x)

        x = self.conv6(x)
        # x = self.bn6(x)

        bn = x.flatten(1)

        if label is None:
            return None, bn

        output = self.arcface(bn, label)
        output = self.log_softmax(output)
        return output, bn


class MelMobileNetV2_IDdiscriminator11(nn.Module):
    def __init__(self, bottleneck, mid1, id_num, width):
        super().__init__()

        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=2, padding=1, padding_mode="replicate")
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=1, padding=1, padding_mode="replicate")
        self.bn2 = nn.BatchNorm2d(8)

        self.ir1 = InvertedResidual(8, 16, kernel_size=(3, 3), stride=2, expand_ratio=6)

        self.ir3 = InvertedResidual(16, 16, kernel_size=(3, 3), stride=2, expand_ratio=6)

        self.ir5 = InvertedResidual(16, 16, kernel_size=(3, 3), stride=2, expand_ratio=6)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=1)
        self.bn3 = nn.BatchNorm2d(16)

        width = width // (2 ** 4)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=(width, width), stride=1, groups=16)
        self.bn4 = nn.BatchNorm2d(16)

        self.conv6 = nn.Conv2d(16, bottleneck, kernel_size=(1, 1), stride=1)
        self.bn6 = nn.BatchNorm2d(bottleneck)

        self.fc1 = nn.Linear(bottleneck, id_num)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, label):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)

        x = self.ir1(x)
        x = self.ir3(x)
        x = self.ir5(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        # x = self.bn4(x)

        x = self.conv6(x)
        # x = self.bn6(x)

        bn = x.flatten(1)

        output = self.fc1(bn)
        output = self.log_softmax(output)
        return output, bn
