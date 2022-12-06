# Importing packages
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
torch.manual_seed(0xB00B1E5)

# Import ImageDataset
from dataloader import ImageDataset

# Importing training loop
from TrainingLoop import Optimization

# Import countries data
data = ImageDataset()


# Split data into train and test
train_set, test_set, val_set = torch.utils.data.random_split(data, [0.8, 0.1, 0.1])

# grid_params = [{"batch_size": bs, "weight_decay": wd, "lr": lr, "drop_out": do, "model": model} for bs in [4, 2] for wd in [0, 0.1] for lr in [0.001, 0.01] for do in [True, False] for model in [18, 34]]
# 34_bs2_wd0_lr0.001_dropout-False	
grid_params = [{"batch_size": bs, "weight_decay": wd, "lr": lr, "drop_out": do, "model": model} for bs in [2] for wd in [0] for lr in [0.001] for do in [False] for model in [34]]
print(len(grid_params))
# Create a function that appends a dropout layer to the model after each ReLU layer
def append_dropout(model):
    names = list(model.named_children())
    for name, child in names:
        if isinstance(child, nn.ReLU):
            model.add_module(name + "_dropout", nn.Dropout(p=0.5))
        else:
            append_dropout(child)

def grid_search(params, train_set, val_set):
    for param in params:
        batch_size = param["batch_size"]
        weight_decay = param["weight_decay"]
        lr = param["lr"]
        model_type = param["model"]
        num_epochs = 100
        model_name = str(model_type) + "_bs" + str(batch_size) + "_wd" + str(weight_decay) + "_lr" + str(lr) + "_dropout-" + str(param["drop_out"]) + "_801010"
        try:
            os.mkdir(f'models/{model_name}')
        except FileExistsError:
            print("Folder already there - skipping")
        print(model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_type == 18:
            model = resnet18()
        else:
            model = resnet34()
        if param["drop_out"]:
            append_dropout(model)
        
        trainloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True
            )
        valloader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=True
            )
        
        loss_fn = nn.CrossEntropyLoss()
        model = model.to(device)

        model.fc = nn.Linear(512, 13)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
        optimization = Optimization(model, loss_fn, optimizer, device, model_name)
        print("Starting to train the model")
        optimization.train(trainloader, valloader, num_epochs)
        print("Done training!")
        optimization.plot_losses()
        optimization.plot_accuracy()
        models_folder = "models/"
        torch.save(model, models_folder + model_name + "/" + model_name)
        

grid_search(grid_params, train_set, val_set)