import torch
import torch.nn as nn
import torch.nn.init as init

class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()

        # Layer 1: 3 channels to 64
        layers = [
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        ]

        # Layers 2-19: 64 to 64
        for _ in range(18):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))

        # Layer 20: 64 to 3 (Predicts the residual)
        layers.append(nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False))

        self.residual_layer = nn.Sequential(*layers)

        # Weight Initialization: He (Kaiming) normal
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        residual = self.residual_layer(x)
        return x + residual