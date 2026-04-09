import torch
import torch.nn as nn

class FastCNN_PINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.flatten = nn.Flatten()

        self.feature_size = 16 * 4 * 4

        self.k = nn.Parameter(torch.tensor(0.5))

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size + 1, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, t):
        f = self.cnn(x)
        f = self.flatten(f)
        x = torch.cat([f, t], dim=1)
        return self.fc(x)