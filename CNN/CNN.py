import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) 

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)


        self.layers = nn.Sequential(
            nn.Linear(3*3*64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 10),
        )

        self.pool = nn.MaxPool2d(stride=2, kernel_size=2)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)  
        x = self.bn1(x)
        x = self.relu(x)    # x.shape -> 1 x 16 x 28 x 28 
        x = self.pool(x)    # x.shape -> 1 x 16 x 14 x 14 

        x = self.conv2(x)  
        x = self.bn2(x)
        x = self.relu(x)     # x.shape -> 1 x 32 x 14 x 14 
        x = self.pool(x)     # x.shape -> 1 x 32 x 07 x 07 

        x = self.conv3(x)  
        x = self.bn3(x)
        x = self.relu(x)    # x.shape -> 1 x 64 x 07 x 07 
        x = self.pool(x)    # x.shape -> 1 x 64 x 03 x 03 

        x = nn.Flatten()(x)
        return self.layers(x)
