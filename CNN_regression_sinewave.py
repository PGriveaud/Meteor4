#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:45:08 2020

@author: pipoune
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn import metrics
import matplotlib.pyplot as plt
plt.close('all')

##### Creating the data #####

def sine(x, A0, f0, x0, phi0):
    y = A0 * np.sin( 2 * np.pi * f0 * (x - x0) + phi0)
    return y 

N_samples = 1000
Freq1 = np.random.uniform(0.009, 1.0, N_samples)


A0 = 5.
phi0 = 0.2
x0 = 25

L = 1000
x = np.linspace(0,50,L)

dataset = []
for f0 in Freq1:
    y = sine(x, A0, f0, x0, phi0) + 1
    dataset.append(y)


dataset_size = len(dataset)

#Adding labels: 0 for sine and 1 for gaussian sine
labels = Freq1

df = pd.DataFrame(dataset)
df.insert(0, "label", labels) 

# Shuffling the data
df = df.sample(frac=1)

test_size = 199
df_test = df.iloc[:test_size,]
df_train = df.iloc[(test_size+1):,]

# Translating the data to pytorch tensors
x_train = torch.tensor(df_train[[x for x in range(1,L)]].values, dtype = torch.float)
x_test = torch.tensor(df_test[[x for x in range(1,L)]].values, dtype = torch.float)

print(x_train.type(), x_train.size())

y_train = torch.tensor(df_train['label'].values, dtype = torch.float)
y_test = torch.tensor(df_test['label'].values, dtype = torch.float)

# Adding 1 dimension to x tensor s.t. shape(x)= [2N, 1, L]
# Problem TO BE SOLVED: here channel = 1, cannnot change this value. 
x_train = x_train.unsqueeze(1)
x_test = x_test.unsqueeze(1)

print(x_train.type(), x_train.size())

train_set = TensorDataset(x_train, y_train)
trainloader = DataLoader(train_set, batch_size=100, shuffle=True)

test_set = TensorDataset(x_test, y_test)
testloader = DataLoader(test_set, batch_size=1, shuffle=True)


#%%

class Net1D(torch.nn.Module):
    def __init__(self, n0, n1, n2, kern):
        super(Net1D, self).__init__()
        self.conv1 = nn.Conv1d(n0,n1,kern)         
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(n1, n2, kern)

        #  /!\ only if NO padding, dilatation or stride 
        dim_x = math.floor((D_in - (kern - 1))/2)   
        dim_x = math.floor((dim_x - (kern - 1))/2)
        dim_x = dim_x*n2
        self.dim_x = dim_x

        self.fc1 = nn.Linear(dim_x, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.conv1(x)
        
        x = self.pool(F.relu(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.dim_x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


# D_in is input dimension
D_in =  L-1 #np.shape(dataset)[1]

# n0 is channel on input; n1 channel out conv1; n2; channel out conv2; kern is kernel size.
net = Net1D(1, 10, 20, 5)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)

num_epochs = 100

def train(net, trainloader):
    for epoch in range(num_epochs): 
        running_loss = 0
        for data in trainloader:
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
              
        if num_epochs >= 101:
            if epoch % 100 == 99:
                print('[Epoch %d] loss: %.3f' %
                              (epoch + 1, running_loss/len(trainloader)))
        else:
            print('[Epoch %d] loss: %.3f' %
                              (epoch + 1, running_loss/len(trainloader)))           
    print('Done Training')

train(net, trainloader)


#%%

y_true = []
y_pred = []
def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

            outputs = net.forward(inputs)

            print(f'Output = {outputs.item()}', f'True Param = {labels.item()}')
            a = outputs.numpy()
            a = a[0][0]
            y_pred.append(a)
           
            # print(labels)
            y_true.append(labels.item())
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on test signals: %0.3f %%' % (
        100 * correct / total))
    
    
test(net, testloader)


plt.figure()
plt.scatter(y_true, y_pred, s=2)
plt.plot([0, 1], [0, 1],ls="--")
plt.xlabel('True Parameter')
plt.ylabel('Predicted Parameter')
plt.show()