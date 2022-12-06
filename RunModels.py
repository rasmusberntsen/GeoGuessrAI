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
train_set, test_set, val_set = torch.utils.data.random_split(data, [0.6, 0.2, 0.2])

grid_params = [{"batch_size": bs, "weight_decay": wd, "lr": lr, "drop_out": do, "model": model} for bs in [4, 2] for wd in [0, 0.1] for lr in [0.001, 0.01] for do in [True, False] for model in [18, 34]][0:1]

# Creating result Dictionary
results = {}

def append_dropout(model):
    names = list(model.named_children())
    for name, child in names:
        if isinstance(child, nn.ReLU):
            model.add_module(name + "_dropout", nn.Dropout(p=0.5))
        else:
            append_dropout(child)


# for param in grid_params:
#     batch_size = param["batch_size"]
#     weight_decay = param["weight_decay"]
#     lr = param["lr"]
#     model_type = param["model"]
#     num_epochs = 100
#     model_name = str(model_type) + "_bs" + str(batch_size) + "_wd" + str(weight_decay) + "_lr" + str(lr) + "_dropout-" + str(param["drop_out"])

#     if model_type == 18:
#         model = resnet18()
#     else:
#         model = resnet34()
#     if param["drop_out"]:
#         append_dropout(model)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     loss_fn = nn.CrossEntropyLoss()
#     model = model.to(device)
#     model.fc = nn.Linear(512, 13)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     trainloader = torch.utils.data.DataLoader(
#         train_set,
#         batch_size=batch_size,
#         shuffle=True
#         )
#     valloader = torch.utils.data.DataLoader(
#         val_set,
#         batch_size=batch_size,
#         shuffle=True
#         )
#     testloader = torch.utils.data.DataLoader(
#         test_set,
#         batch_size=batch_size,
#         shuffle=False
#         )


#     # Load the state dict
#     print(model.load_state_dict(torch.load(f'models/{model_name}/best_params.pt')))

#     # Load the optimizer
#     optimization = Optimization(model, loss_fn, trainloader, device, model_name)
    
#     # Run evaluation
#     preds, vals, props = optimization.evaluate(testloader)

#     print(np.mean(np.array(preds) == np.array(vals)))
#     results[model_name] = np.mean(np.array(preds) == np.array(vals))





results = {'18_bs4_wd0_lr0.001_dropout-True': 0.4166666666666667,
 '34_bs4_wd0_lr0.001_dropout-True': 0.3553113553113553,
 '18_bs4_wd0_lr0.001_dropout-False': 0.4652014652014652,
 '34_bs4_wd0_lr0.001_dropout-False': 0.4267399267399267,
 '18_bs4_wd0_lr0.01_dropout-True': 0.44963369963369965,
 '34_bs4_wd0_lr0.01_dropout-True': 0.4139194139194139,
 '18_bs4_wd0_lr0.01_dropout-False': 0.36721611721611724,
 '34_bs4_wd0_lr0.01_dropout-False': 0.3882783882783883,
 '18_bs2_wd0_lr0.001_dropout-True': 0.5009157509157509,
 '34_bs2_wd0_lr0.001_dropout-True': 0.44871794871794873,
 '18_bs2_wd0_lr0.001_dropout-False': 0.4542124542124542,
 '34_bs2_wd0_lr0.001_dropout-False': 0.5274725274725275,
 '18_bs2_wd0_lr0.01_dropout-True': 0.4597069597069597,
 '34_bs2_wd0_lr0.01_dropout-True': 0.46794871794871795,
 '18_bs2_wd0_lr0.01_dropout-False': 0.5064102564102564,
 '34_bs2_wd0_lr0.01_dropout-False': 0.09615384615384616}


# Convert results to dataframe
results = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
results = results.nlargest(3, 'Accuracy').reset_index().rename(columns={'index': 'Model'})

# Load the best models

results.Model.values[1]

'34_bs2_wd0_lr0.001_dropout-False'
batch_size = 2
weight_decay = 0
lr = 0.001
model_type = 34
num_epochs = 100

if model_type == 18:
    model = resnet18()
else:
    model = resnet34()
# if param["drop_out"]:
#     append_dropout(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss()
model.fc = nn.Linear(512, 13)
model = model.to(device)

testloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False
    )
batch = next(iter(testloader))
x = batch[0].float()
x = x.to(device)
yhat = model(x)



# Visualize the model
from torchviz import make_dot

make_dot(yhat, params=dict(model.named_parameters()))

# Use netron to visualize the model
import netron
input_names = ["Image"]
output_names = ["Country"]
torch.onnx.export(model, x, "model.onnx", verbose=True, input_names=input_names, output_names=output_names)

netron.start("model.onnx")
