#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 18:32:42 2020

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

def sine(x):
    A0 = 1
    # f0 = 0.01
    phi0 = 0.2
    x0 = 0 
    
    y = A0 * np.sin( 2 * np.pi * f0 * (x - x0) + phi0)
    return y 

def sine_gaussian(x):
    B0 = 1
    x1 = 500
    tau0 = 300
    # f1 = 0.01
    phi1 = 0.2

    y = B0*np.exp(-(x - x1)**2 / tau0**2) * np.cos( 2 * np.pi * f1 * (x - x1) + phi1)
    return y 


L = 1000
x = np.arange(0, L, 1)

N = 100
freq_low = 0.01
freq_high = 1.0

freq = np.arange(0.009, 1.0, 0.01)
amplitude = range(1,101)

dataset = []
for f0 in freq:
    y = sine(x)
    dataset.append(y)
    
for A0 in amplitude:
    y = sine(x)
    dataset.append(y)
    
    
for f1 in freq:
    y = sine_gaussian(x)
    dataset.append(y)
    
for B0 in amplitude:
    y = sine_gaussian(x)
    dataset.append(y)
    
# plt.figure()
# plt.plot(x, dataset[0], label= 'Sine')
# plt.plot(x, dataset[99], label= 'Sine')

# # plt.plot(x, sine_gaussian(x), label = 'Gaussian sine')
# plt.xlabel('x')
# plt.ylabel('y(x)')
# plt.legend()
# plt.show()




#Adding labels: 0 for sine and 1 for gaussian sine
labels = np.append(np.zeros(N),np.ones(N))
labels = np.append(labels, labels)

df = pd.DataFrame(dataset)
df.insert(0, "label", labels) 

# Shuffling the data
df = df.sample(frac=1)

test_size = 50
df_test = df.iloc[:test_size,]
df_train = df.iloc[(test_size+1):,]

# Translating the data to pytorch tensors

x_train = torch.tensor(df_train[[x for x in range(1,L)]].values, dtype = torch.float)
x_test = torch.tensor(df_test[[x for x in range(1,L)]].values, dtype = torch.float)

y_train = torch.tensor(df_train['label'].values, dtype = torch.float)
y_test = torch.tensor(df_test['label'].values, dtype = torch.float)

# Adding 1 dimension to x tensor s.t. shape(x)= [2N, 1, L]
# Problem TO BE SOLVED: here channel = 1, cannnot change this value. 
x_train = x_train.unsqueeze(1)
x_test = x_test.unsqueeze(1)




train_set = TensorDataset(x_train, y_train)
trainloader = DataLoader(train_set)

test_set = TensorDataset(x_test, y_test)
testloader = DataLoader(test_set)



class Net1D(torch.nn.Module):
    def __init__(self):
        super(Net1D, self).__init__()
        self.conv1 = nn.Conv1d(n0,n1,kern)         
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(n1, n2, kern)

        #  /!\ only if NO padding, dilatation or stride 
        dim_x = math.floor((D_in - (kern - 1))/2)   
        dim_x = math.floor((dim_x - (kern - 1))/2)
        dim_x = dim_x*n2
        self.dim_x = dim_x

        self.fc1 = nn.Linear(dim_x, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.dim_x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


# D_in is input dimension
D_in =  L-1

# n0 is channel on input; n1 channel out conv1; 
# n2; channel out conv2; kern is kernel size.
n0, n1, n2, kern = 1, 6, 16, 5


net = Net1D()





device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Loss
criterion = torch.nn.MSELoss(reduction='sum')
# Optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)

num_epochs = 100

def train(net, trainloader):
    for epoch in range(num_epochs): # no. of epochs
        running_loss = 0
        for data in trainloader:
            # data pixels and labels to GPU if available
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            # set the parameter gradients to zero
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # propagate the loss backward
            loss.backward()
            # update the gradients
            optimizer.step()
 
            running_loss += loss.item()
        if num_epochs >= 500:
            if epoch % 100 == 99:
                print('[Epoch %d] loss: %.3f' %
                              (epoch + 1, running_loss/len(trainloader)))
        else:
            print('[Epoch %d] loss: %.3f' %
                              (epoch + 1, running_loss/len(trainloader)))
                
    print('Done Training')
    

train(net, trainloader)


    
def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on test signals: %0.3f %%' % (
        100 * correct / total))
    
test(net, testloader)

























