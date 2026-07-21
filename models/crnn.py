import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN)
    Input Shape:
        (B, 1, 128, 1292)

    Output:
        (B, num_classes)
    """

    def __init__(self, num_classes):
        super().__init__()

        # =====================================================
        # CNN Feature Extractor
        # =====================================================

        self.cnn = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # =====================================================
        # BiLSTM
        # =====================================================

        self.lstm = nn.LSTM(
            input_size=128 * 16,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # =====================================================
        # Classifier
        # =====================================================

        self.classifier = nn.Sequential(

            nn.Dropout(0.5),

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        # -------------------------------------------
        # CNN
        # -------------------------------------------

        x = self.cnn(x)

        # Shape:
        #
        # (B,128,16,161)
        #

        batch_size = x.size(0)

        # -------------------------------------------
        # Convert CNN output into sequence
        # -------------------------------------------

        x = x.permute(0, 3, 1, 2)

        # Shape:
        #
        # (B,161,128,16)
        #

        x = x.contiguous().view(
            batch_size,
            x.size(1),
            -1
        )

        # Shape:
        #
        # (B,161,2048)
        #

        # -------------------------------------------
        # BiLSTM
        # -------------------------------------------

        x, _ = self.lstm(x)

        # Use last time step

        x = x[:, -1, :]

        # Shape:
        #
        # (B,512)
        #

        # -------------------------------------------
        # Classifier
        # -------------------------------------------

        x = self.classifier(x)

        return x
