import torch
import torch.nn as nn
from torchvision import models


class ResNetModel(nn.Module):
    """
    ResNet50 for Music Genre Classification

    Input:
        (Batch, 1, 128, 1292)

    Output:
        (Batch, num_classes)
    """

    def __init__(self, num_classes):
        super().__init__()

        # Load pretrained ResNet50
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        # Change first convolution layer
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Replace classification head
        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)