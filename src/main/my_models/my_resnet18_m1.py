import torch
import torch.nn as nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    """Blocks with solid line."""
    def __init__(self, in_channels, out_channels, stride, size, alpha, beta, k) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.lrn1 = nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.lrn2 = nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.lrn1(out)
        out = self.conv2(out)
        return self.lrn2(F.relu(x + out))


class DownBlock(nn.Module):
    """Blocks with dotted line."""
    def __init__(self, in_channels, out_channels, stride, size, alpha, beta, k) -> None:
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.lrn1 = nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.lrn2 = nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)

        # refer to pytorch resnet implementation
        self.extra = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0)

    def forward(self, x):
        extra_x = self.extra(x)
        out = self.conv1(x)
        out = F.relu(out)
        out = self.lrn1(out)
        out = self.conv2(out)
        return self.lrn2(F.relu(extra_x + out))


class MyResNet18M1(nn.Module):
    """My modified ResNet18 (version 1, replace BN with LRN)."""
    def __init__(self, num_classes, size, alpha, beta, k) -> None:
        super(MyResNet18M1, self).__init__()  # call nn.Module's init
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.lrn1 = nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(BasicBlock(64, 64, 1, size, alpha, beta, k),
                                    BasicBlock(64, 64, 1, size, alpha, beta, k))
        self.layer2 = nn.Sequential(DownBlock(64, 128, [2, 1], size, alpha, beta, k),
                                    BasicBlock(128, 128, 1, size, alpha, beta, k))
        self.layer3 = nn.Sequential(DownBlock(128, 256, [2, 1], size, alpha, beta, k),
                                    BasicBlock(256, 256, 1, size, alpha, beta, k))
        self.layer4 = nn.Sequential(DownBlock(256, 512, [2, 1], size, alpha, beta, k),
                                    BasicBlock(512, 512, 1, size, alpha, beta, k))

        # refer to pytorch resnet implementation
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, num_classes)  # in_features, out_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrn1(F.relu(x))
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
