"""Attention-enhanced multi-stage ResNet feature extractor.

Each ResNet residual stage is followed by a Squeeze-and-Excitation (SE) block.
The recalibrated output of each stage is reduced with a 1x1 convolution, pooled
to 3x3 and fused with the previous stages by element-wise addition. The fused
map is globally average-pooled into a single feature vector.
"""
import torch
from torch import nn
from torchvision import models

_WEIGHTS = {
    "resnet50": models.ResNet50_Weights,
    "resnet101": models.ResNet101_Weights,
    "resnet152": models.ResNet152_Weights,
}


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiStageFeatureResNet(nn.Module):
    """ResNet backbone + SE attention + progressive multi-stage fusion.

    stage = 1..4 selects how many residual stages are used:
        1 -> x1,  2 -> x1+x2,  3 -> x1+x2+x3,  4 -> x1+x2+x3+x4 (full model)
    """

    def __init__(self, base_model="resnet50", weights="IMAGENET1K_V1",
                 reduced_dim=64, stage=4):
        super().__init__()
        if stage not in (1, 2, 3, 4):
            raise ValueError("stage must be 1, 2, 3 or 4")
        self.stage = stage
        net = getattr(models, base_model)(weights=_WEIGHTS[base_model][weights])
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layers = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        channels = [256, 512, 1024, 2048]
        self.se = nn.ModuleList([SEBlock(c) for c in channels])
        self.reduce = nn.ModuleList([nn.Conv2d(c, reduced_dim, 1) for c in channels])
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stem(x)
        fused = None
        for i in range(self.stage):
            x = self.se[i](self.layers[i](x))
            feat = self.pool3(self.reduce[i](x))
            fused = feat if fused is None else fused + feat
        return torch.flatten(self.gap(fused), 1)
