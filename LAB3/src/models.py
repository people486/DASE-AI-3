"""Small CNNs designed for 32 x 32 RGB images; no external model weights."""

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Three convolution blocks and global average pooling (about 0.29M parameters)."""

    def __init__(self, num_classes=10):
        super().__init__()
        layers = []
        in_channels = 3
        for channels in (32, 64, 128):
            layers.extend([
                nn.Conv2d(in_channels, channels, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2),
            ])
            in_channels = channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, num_classes))

    @property
    def gradcam_layer(self):
        return self.features[-3]

    def forward(self, x):
        x = self.features(x)
        return self.classifier(self.pool(x).flatten(1))


class SqueezeExcitation(nn.Module):
    """Channel attention computed from global context."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(self.pool(x))


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)
        self.shortcut = (
            nn.Identity() if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.relu(out + residual)


class ImprovedCNN(nn.Module):
    """CIFAR-specific residual CNN with SE channel attention and BatchNorm."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=False),
        )
        self.features = nn.Sequential(
            ResidualSEBlock(32, 32), ResidualSEBlock(32, 32),
            ResidualSEBlock(32, 64, 2), ResidualSEBlock(64, 64),
            ResidualSEBlock(64, 128, 2), ResidualSEBlock(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, num_classes))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(self.stem(x))
        return self.classifier(self.pool(x).flatten(1))

    @property
    def gradcam_layer(self):
        # The SE gate's last 1x1 convolution has no spatial information.
        return self.features[-1].conv2


def build_model(name, num_classes=10):
    """Create one architecture; weights are initialized rather than pretrained."""
    constructors = {"baseline": BaselineCNN, "improved": ImprovedCNN}
    if name not in constructors:
        raise ValueError("Unknown model {!r}; choose {}".format(name, list(constructors)))
    return constructors[name](num_classes=num_classes)


if __name__ == "__main__":
    for model_name in ("baseline", "improved"):
        network = build_model(model_name)
        with torch.no_grad():
            prediction = network(torch.randn(2, 3, 32, 32))
        print(model_name, "parameters:", sum(p.numel() for p in network.parameters()),
              "output:", tuple(prediction.shape))
