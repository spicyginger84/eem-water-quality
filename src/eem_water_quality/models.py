"""Original SimpleCNN, ResNet10, and ResNet18 fusion architectures."""

import torch
from torch import nn


class SimpleCNNFusion(nn.Module):
    def __init__(self, in_channels=1, n_tabular=0, dropout=0.3):
        super().__init__()
        self.n_tabular = n_tabular
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        if n_tabular > 0:
            self.tabular = nn.Sequential(
                nn.Linear(n_tabular, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU()
            )
            fusion_dim = 128 + 16
        else:
            self.tabular = None
            fusion_dim = 128
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, eem, tabular=None):
        x_img = self.cnn(eem)
        x_img = self.global_pool(x_img)
        x_img = torch.flatten(x_img, 1)
        if self.tabular is None:
            x = x_img
        else:
            x_tab = self.tabular(tabular)
            x = torch.cat([x_img, x_tab], dim=1)
        return self.regressor(x).squeeze(1)




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class ResNetEEMBackbone(nn.Module):
    def __init__(self, layers, in_channels=1):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, out_channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def get_resnet10_backbone(in_channels=1):
    return ResNetEEMBackbone(layers=[1, 1, 1, 1], in_channels=in_channels)


def get_resnet18_backbone(in_channels=1):
    return ResNetEEMBackbone(layers=[2, 2, 2, 2], in_channels=in_channels)


class ResNetFusion(nn.Module):
    def __init__(self, backbone, n_tabular=0, tab_hidden=16, fusion_hidden=128, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        if n_tabular > 0:
            self.tabular_branch = nn.Sequential(
                nn.Linear(n_tabular, tab_hidden),
                nn.ReLU(),
                nn.Linear(tab_hidden, tab_hidden),
                nn.ReLU(),
            )
            fusion_dim = 512 + tab_hidden
        else:
            self.tabular_branch = None
            fusion_dim = 512
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, eem, tabular=None):
        z_eem = self.backbone(eem)
        if self.tabular_branch is None:
            z = z_eem
        else:
            if tabular is None:
                raise ValueError("tabular input is required when n_tabular > 0")
            z_tab = self.tabular_branch(tabular)
            z = torch.cat([z_eem, z_tab], dim=1)
        return self.regressor(z).squeeze(1)


def get_resnet10_fusion(in_channels=1, n_tabular=0):
    return ResNetFusion(
        backbone=get_resnet10_backbone(in_channels=in_channels), n_tabular=n_tabular, dropout=0.3
    )


def get_resnet18_fusion(in_channels=1, n_tabular=0):
    return ResNetFusion(
        backbone=get_resnet18_backbone(in_channels=in_channels), n_tabular=n_tabular, dropout=0.3
    )
