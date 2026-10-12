"""CNN architectures for Bee Pollen (binary: pollen / non-pollen)."""
import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    """Custom CNN ~0.5M params. Input 3x128x128 -> 2 logits.
    Conv(3->32) - BN - ReLU - MaxPool
    Conv(32->64) - BN - ReLU - MaxPool
    Conv(64->128) - BN - ReLU - MaxPool
    Conv(128->256) - BN - ReLU - AdaptiveAvgPool
    FC(256->128) - Dropout - FC(128->2)
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
