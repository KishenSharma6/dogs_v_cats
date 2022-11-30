import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

import numpy as np

def accuracy(preds, labels):
    preds = preds > 0.5
    return (labels == preds).sum().item() / labels.size(0)
def train(model, loader, epochs, criterion, optimizer):
    
    #move model to gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    epoch_loss = [0] * epochs
    epoch_acc = [0] * epochs
    for epoch in range(epochs):
        running_loss = 0.00
        correct = 0.00
        for i, data in enumerate(loader):

            #move data to gpu
            inputs, labels = data['image'], data['target']
            inputs, labels = inputs.to(device), labels.float().to(device)

            #zero gradients
            optimizer.zero_grad() 

            #model computations
            preds = model(inputs.type(torch.cuda.FloatTensor))
            loss = criterion(preds, labels.unsqueeze(1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += accuracy(preds, labels)

        epoch_loss[epoch] = (running_loss / i) #i is the num_batches in epoch
        epoch_acc[epoch] = (correct/len(loader.dataset)) * 100

        print("Loss for Epoch %s: %s\nAccuracy: %s" % (str(epoch + 1), str(round(epoch_loss[epoch],3)),str(round(epoch_acc[epoch],3))))

    print("Training complete!")

class BaseConvNet(nn.Module):
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

                                  stride=1)
        self.fc1 = nn.Linear(237620, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1) #Binary classification/ we just want 1 output

    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ConvNet1_0(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels= 3,
                               out_channels= 32,
                               kernel_size= 3,
                               stride= 1, 
                               padding= 1)

        self.conv2 = nn.Conv2d(in_channels= 32,
                               out_channels= 64,
                               kernel_size= 5,
                               stride=1, 
                               padding=2)

        self.pool1 = nn.MaxPool2d(kernel_size=5,
                                    stride=1)

        self.conv3 = nn.Conv2d(in_channels= 64,
                               out_channels= 64,
                               kernel_size=3,
                               stride=1, 
                               padding=1)

        self.conv4 = nn.Conv2d(in_channels= 64,
                               out_channels= 64,
                               kernel_size=3,
                               stride=1, 
                               padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,
                                    stride= 2)
        self.conv5 = nn.Conv2d(in_channels= 64,
                               out_channels= 64,
                               kernel_size=3,
                               stride=1, 
                               padding=1)

        self.fc1 = nn.Linear( 49561600 , 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))

        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x