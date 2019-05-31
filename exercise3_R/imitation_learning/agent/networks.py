import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=3):
        super(CNN, self).__init__()

        ch = history_length

        # TODO : define layers of a convolutional neural network
        self.conv_block1 = nn.Sequential(  # 96*96 -> 48*48
            nn.Conv2d(ch, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(  # 48*48 -> 24*24
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block3 = nn.Sequential(  # 24*24 -> 12*12
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block4 = nn.Sequential(  # 12*12 -> 6*6
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # self.conv_block5 = nn.Sequential(  # 6*6   -> 3*3
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )
        self.fc_block = nn.Sequential(
            nn.Linear(6*6*64, 100),
            # nn.Linear(3*3*128, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes)
        )

    def forward(self, x):
        # TODO: compute forward pass
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        # x = self.conv_block5(x)
        x = self.fc_block(x.view(x.shape[0], -1))  # flatten tensor for linear input
        return x

