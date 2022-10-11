import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from subnet import SignetLinear, SignetConv2d
from utils import *

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
import matplotlib.pyplot as plt

net_type = 'subnet'
sub_type = 'softnet'
epochs = 100
lr_rate = 1e-2

# download datasets
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

#print(df.head())

x = df.drop(['target'], axis=1).values # 독립변인들의 value값만 추출
y = df['target'].values # 종속변인 추출

# define datasets
dataset = df
mappings = {
   "Iris-setosa": 0,
   "Iris-versicolor": 1,
   "Iris-virginica": 2
}
dataset["target"] = dataset["target"].apply(lambda x: mappings[x])
#print(dataset)

X = dataset.drop("target",axis=1).values
y = dataset["target"].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

if True:
    X_train += torch.randn_like(X_train)
    X_test += torch.randn_like(X_test)

def get_mask(model):
    task_mask = {}
    for name, module in model.named_modules():
        # For the time being we only care about the current task outputhead
        if 'last' in name:
            if name != 'last.' + str(task_id):
                continue

        if isinstance(module, SignetLinear) or isinstance(module, SignetConv2d):
            task_mask[name + '.weight'] = module.weight_mask.detach().clone().float()

            if getattr(module, 'bias') is not None:
                task_mask[name + '.bias'] = module.bias_mask.detach().clone().float()
            else:
                task_mask[name + '.bias'] = None
    return task_mask

# define models
class Model(nn.Module):
    def __init__(self, input_features=4, hidden_layer1=25, hidden_layer2=30, output_features=3):
        super().__init__()
        self.fc1 = nn.Linear(input_features,hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.out = nn.Linear(hidden_layer2, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

class Subnet(nn.Module):
    def __init__(self, input_features=4,
                 hidden_layer1=25, hidden_layer2=30, output_features=3,
                 sub_type='hardnet'):

        super().__init__()
        self.fc1 = SignetLinear(input_features, hidden_layer1, sub_type=sub_type)
        self.fc2 = SignetLinear(hidden_layer1, hidden_layer2, sub_type=sub_type)
        self.out = SignetLinear(hidden_layer2, output_features, sub_type=sub_type)

    def forward(self, x, mask=None):
        if mask is None:
            mask = {}
            mask['fc1.weight'] = None
            mask['fc2.weight'] = None
            mask['out.weight'] = None

        x = F.relu(self.fc1(x, mask['fc1.weight']))
        x = F.relu(self.fc2(x, mask['fc2.weight']))
        x = self.out(x, mask['out.weight'])

        return x


# train models
def train(model, X_train, y_train, criterion, optimizer, subnet=False):
    losses = []
    model_list = []
    mask_list = []
    for i in range(epochs):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss.item())
        if i % 10 == 0:
            print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
            model_list.append(model)

            if subnet:
                mask = get_mask(model)
                mask_list.append(mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses, model_list, mask_list

# test models
def test(model, X_test, y_test, mask=None):
    preds = []
    with torch.no_grad():
        for val in X_test:

            if mask is not None:
                y_hat = model.forward(val, mask)
            else:
                y_hat = model.forward(val)
            preds.append(y_hat.argmax().item())

    return preds


# --- dense network ---------
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

dense_losses, dense_model,_ = train(model, X_train, y_train, criterion, optimizer)
preds = test(model, X_test, y_test)

df = pd.DataFrame({'Y': y_test, 'YHat': preds})
df['dense_Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]

loss_dense = loss_landscape(model, criterion, X_test, y_test, 'dense')

del model

# --- hard network ---------
model = Subnet(sub_type='hardnet')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

hardnet_losses, hardnet_model, hardnet_mask_list = train(model, X_train, y_train, criterion, optimizer, subnet=True)
hardnet_mask = get_mask(model)
preds = test(model, X_test, y_test, hardnet_mask)

df['hardnet_Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]

loss_hardnet = loss_landscape(model, criterion, X_test, y_test, 'hardnet', hardnet_mask)

del model

# --- soft network ---------
model = Subnet(sub_type='softnet')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

softnet_losses, softnet_model, softnet_mask_list = train(model, X_train, y_train, criterion, optimizer, subnet=True)
softnet_mask = get_mask(model)
preds = test(model, X_test, y_test, softnet_mask)

df['softnet_Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]

loss_softnet = loss_landscape(model, criterion, X_test, y_test, 'softnet', softnet_mask)

print('test acc. of densenet:{}'.format(df['dense_Correct'].mean() * 100.0))
print('test acc. of hardnet:{}'.format(df['hardnet_Correct'].mean() * 100.0))
print('test acc. of softnet:{}'.format(df['softnet_Correct'].mean() * 100.0))

del model

plot_trisurf(loss_dense, loss_hardnet, loss_softnet, epoch=100)

epoch_list = [0,10,20,30,40,50,60,70,80,90]
for densenet, hardnet, softnet, epoch, hardnet_mask, softnet_mask in zip(dense_model, hardnet_model, softnet_model, epoch_list, hardnet_mask_list, softnet_mask_list):
    loss_dense = loss_landscape(densenet, criterion, X_test, y_test, 'dense')
    loss_hardnet = loss_landscape(hardnet, criterion, X_test, y_test, 'hardnet', hardnet_mask)
    loss_softnet = loss_landscape(softnet, criterion, X_test, y_test, 'softnet', softnet_mask)
    plot_trisurf(loss_dense, loss_hardnet, loss_softnet, epoch=epoch)

plot_losses(epochs, dense_losses, hardnet_losses, softnet_losses)









