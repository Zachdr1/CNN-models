import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    ''' Implementation of LeNet-5 for 2 channel 32x32 images. '''
    def __init__(self):
        super(Net, self).__init__()
        # Input: 32x32 image
        # Layers: 3 5x5 convolutions,
        #         2 2x2, stride =2, max pooling 
        #         2 FC layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
