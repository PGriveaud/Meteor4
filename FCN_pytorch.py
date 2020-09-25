#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:23:22 2020

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

import matplotlib.pyplot as plt
plt.close('all')


##### Creating the data #####

def sine(x,f0):
    y = np.sin( 2 * np.pi * f0 * x)
    return y 

N_samples = 500
Freq1 = np.random.uniform(0.009, 1.0, N_samples)


L = 100
x = np.linspace(0,50,L)

N = 100

dataset = []
f = []
for f0 in Freq1:
    y = sine(x, f0) + 1
    dataset.append(y)
    f.append(f0)
    

df = pd.DataFrame(dataset)
df.insert(0, "frequency", f) 

test_size = 30
df_test = df.iloc[:test_size-1,]
df_train = df.iloc[(test_size):,]

# Translating the data to pytorch tensors
x_train = torch.tensor(df_train[[x for x in range(0,L)]].values, dtype = torch.float)
x_test = torch.tensor(df_test[[x for x in range(0,L)]].values, dtype = torch.float)

y_train = torch.tensor(df_train['frequency'].values, dtype = torch.float)
y_test = torch.tensor(df_test['frequency'].values, dtype = torch.float)

# Adding 1 dimension to x tensor s.t. shape(x)= [2N, 1, L]
# Problem TO BE SOLVED: here channel = 1, cannnot change this value. 
# x_train = x_train.unsqueeze(1)
# x_test = x_test.unsqueeze(1)


train_set = TensorDataset(x_train, y_train)
trainloader = DataLoader(train_set, batch_size=100, shuffle=True)

test_set = TensorDataset(x_test, y_test)
testloader = DataLoader(test_set)



#%%


##### Defining the CNN #####



class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.nonlin = torch.nn.ReLU()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.nonlin(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred


net = TwoLayerNet(D_in=100, H=5, D_out=1)

num_epochs = 1

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)

# run the main training loop
for epoch in range(num_epochs):
    running_loss = 0
    for data in trainloader:
        data, target = data[0], data[1]
        
        # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
        # data = data.view(-1, 100)
        
        optimizer.zero_grad()
        outputs = net.forward(data)

        loss = criterion(outputs, target)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        print('[Epoch %d] loss: %.3f' %
                          (epoch + 1, running_loss/len(trainloader)))
                
print('Done Training')



#%%
# # run a test loop
# test_loss = 0
# correct = 0
# for data, target in test_loader:
#     data, target = Variable(data, volatile=True), Variable(target)
#     data = data.view(-1, 28 * 28)
#     net_out = net(data)
#     # sum up batch loss
#     test_loss += criterion(net_out, target).data[0]
#     pred = net_out.data.max(1)[1]  # get the index of the max log-probability
#     correct += pred.eq(target.data).sum()

# test_loss /= len(test_loader.dataset)
# print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))








