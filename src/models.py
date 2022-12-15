import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)



import numpy as np

def accuracy(preds, labels):
    preds = preds > 0.5
    return (labels == preds).sum().item() / labels.size(0)

def plot_acc(acc_scores, ax):
    ax.plot(acc_scores, color = 'orange')
    ax.set_xlabel("epochs")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy during Training")

def plot_loss(loss_scores, ax):
    ax.plot(loss_scores, color = 'blue')
    ax.set_xlabel("epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss during Training")

def train(model, train_set, loader, epochs, criterion, optimizer):
    
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
        epoch_acc[epoch] = 100 * correct / i

        print("Loss for Epoch %s: %s\nAccuracy: %s" % (str(epoch + 1), str(round(epoch_loss[epoch],3)),str(epoch_acc[epoch])))

    print("Training complete!")
    return epoch_loss, epoch_acc

class BaseConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels= 3,
                                out_channels= 32, 
                                kernel_size= 5,
                                padding= 2,
                                stride=1)

        self.batchnorm1 = nn.BatchNorm2d(num_features=32)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2,
                                  stride=1)

        self.conv2 = nn.Conv2d(in_channels= 32,
                                out_channels= 32,
                                kernel_size=5,
                                padding=2, 
                                stride=1)

        self.batchnorm2 = nn.BatchNorm2d(num_features=32)

        self.pool2 = nn.MaxPool2d(kernel_size=2,
                                  stride=1)

        self.conv3 = nn.Conv2d(in_channels= 20,
                                out_channels= 40, 
                                kernel_size= 5,
                                padding= 2,
                                stride=1)

        self.batchnorm3 = nn.BatchNorm2d(num_features=40)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2,
                                  stride=1)

        self.fc1 = nn.Linear(508032, 512)
        self.fc2 = nn.Linear(512, 1) #Binary classification/ we just want 1 output

        self.dropout1 = torch.nn.Dropout(p= .5)
       

    
    def forward(self, x):
        x = self.pool1(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool2(F.relu(self.batchnorm2(self.conv2(x))))
        #x = self.pool2(F.relu(self.batchnorm3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x