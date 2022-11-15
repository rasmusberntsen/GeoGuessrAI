# Importing packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import resnet18
from torchvision.models import resnet18

# Import ImageDataset
from dataloader import ImageDataset

# Importing training loop
from TrainingLoop import Optimization

# Import countries data
data = ImageDataset()

# Split data into train and test
train_set, test_set, val_set = torch.utils.data.random_split(data, [0.64, 0.2, 0.16])

# Create loaders
batch_size = 2
trainloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True
)
testloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=True
)
valloader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=True
)

# Create a pytorch CNN that classify countries
num_countries = 13
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 4, 3, 1)
        self.conv3 = nn.Conv2d(4, 2, 3, 1)
        self.fc1 = nn.Linear(2 * 126 * 254, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_countries)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


# Test the network
trainiter = iter(trainloader)
images, labels = next(trainiter)
# Make images float
images = images.float().cuda()
# net = Net()
# Create a resnet18 with 13 outputs
net = resnet18(pretrained=True).cuda()
net.fc = nn.Linear(512, num_countries)
# Make the network use the GPU
net = net.cuda()
# Get the output
output = net(images)


# Defining parameters 
num_epochs = 10
lr = 0.001

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

# binary cross entropy loss
#borders = pd.read_csv("borderloss.csv", sep=";", header=None)
#border_loss = lambda y_pred, y_true: borders.iloc[y_pred, y_true]
#loss_fn = nn.CrossEntropyLoss + 0.5 * border_loss
loss_fn = nn.CrossEntropyLoss()
model = resnet18().cuda()
model.fc = nn.Linear(512, num_countries)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

optimization = Optimization(model, loss_fn, optimizer, device)
optimization.train(trainloader, valloader, num_epochs)
optimization.plot_losses()