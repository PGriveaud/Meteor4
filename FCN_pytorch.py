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

N_samples = 100
x = np.random.uniform(0, 0.5, (N_samples,2))
y = []
for i in range(len(x)):
    y1 = np.sum(x[i])
    y.append(y1)

N_tests = 100
x_t = np.random.uniform(0, 0.5, (N_samples,2))
y_t = []
for i in range(len(x_t)):
    y1_t = np.sum(x_t[i])
    y_t.append(y1_t)

# Translating the data to pytorch tensors
x_train = torch.tensor(x, dtype = torch.float)
x_test = torch.tensor(x_t, dtype = torch.float)

y_train = torch.tensor(y, dtype = torch.float)
y_test = torch.tensor(y_t, dtype = torch.float)


train_set = TensorDataset(x_train, y_train)
trainloader = DataLoader(train_set, batch_size=100)

test_set = TensorDataset(x_test, y_test)
testloader = DataLoader(test_set)



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


net = TwoLayerNet(D_in=2, H=5, D_out=1)

num_epochs = 1000

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)


losses = []
# run the main training loop
for epoch in range(num_epochs):
    running_loss = 0
    for data in trainloader:
        data, target = data[0], data[1]
        target = target.view(-1, 1)        
        
        optimizer.zero_grad()
        outputs = net.forward(data)
        
        loss = criterion(outputs, target)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        losses.append(running_loss)
        if epoch % 100 == 99:
            print('[Epoch %d] loss: %.3f' %
                              (epoch + 1, running_loss/len(trainloader)))
                
print('Done Training')



# run a test loop
correct = 0
running_test_loss = 0 
test_losses = []
net.eval()
for data in testloader:
    data, target = data[0], data[1]

    target = target.view(-1, 1)
    net_out = net.forward(data)
    
    # # sum up batch loss
    test_loss = criterion(net_out, target)
    
    running_test_loss += test_loss.item()
    
    test_losses.append(running_test_loss)
  


# plt.figure()
# plt.plot(losses, color='darkred')
# plt.plot(test_losses)
# plt.show()



