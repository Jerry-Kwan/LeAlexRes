import torch
import torch.nn as nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    """Blocks with solid line."""
    def __init__(self, in_channels, out_channels, stride) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class DownBlock(nn.Module):
    """Blocks with dotted line."""
    def __init__(self, in_channels, out_channels, stride) -> None:
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # refer to pytorch resnet implementation
        self.extra = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
                                   nn.BatchNorm2d(out_channels))

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class MyResNet18(nn.Module):
    def __init__(self, num_classes) -> None:
        super(MyResNet18, self).__init__()  # call nn.Module's init
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(BasicBlock(64, 64, 1), BasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(DownBlock(64, 128, [2, 1]), BasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(DownBlock(128, 256, [2, 1]), BasicBlock(256, 256, 1))
        self.layer4 = nn.Sequential(DownBlock(256, 512, [2, 1]), BasicBlock(512, 512, 1))

        # refer to pytorch resnet implementation
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, num_classes)  # in_features, out_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = x.reshape(x.shape[0], -1)  # [0] is batch_size
        x = self.fc(x)

        return x
