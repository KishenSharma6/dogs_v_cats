import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels= 3,
                                out_channels= 10, 
                                kernel_size= 5,
                                padding= 2,
                                stride=1)

        
        self.pool1 = nn.MaxPool2d(kernel_size=3,
                                  stride=2)

        self.conv2 = nn.Conv2d(in_channels= 10,
                                out_channels= 20,
                                kernel_size=5,
                                padding=2, 
                                stride=1)

        self.pool2 = nn.MaxPool2d(kernel_size=3,

                                  stride=2)
        self.fc1 = nn.Linear(109520, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 2)

    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x