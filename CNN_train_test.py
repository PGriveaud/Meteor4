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
from sklearn import metrics
import matplotlib.pyplot as plt
plt.close('all')

##### Creating the data #####

def sine(x, A0, f0, x0, phi0):
    y = A0 * np.sin( 2 * np.pi * f0 * (x - x0) + phi0)
    return y 

def sine_gaussian(x, B0, f0, x0, phi0, tau0):
    y = B0 * np.exp(-(x - x0)**2 / tau0**2) * np.cos( 2 * np.pi * f0 * (x - x0) + phi0)
    return y 

N_samples = 10
Amp1 = np.random.uniform(0, 10, N_samples)
Amp2 = np.random.uniform(0, 10, N_samples)

Freq1 = np.random.uniform(0.009, 1.0, N_samples)
Freq2 = np.random.uniform(0.009, 1.0, N_samples)


Phase1 = np.random.uniform(0, 2*np.pi, N_samples)
Phase2 = np.random.uniform(0, 2*np.pi, N_samples)

Tau = np.random.uniform(0.1, 10, 10)


phi0 = 0.2
x0 = 25
tau0 = 10


L = 1000
# x = np.arange(0, L, 1)
x = np.linspace(0,50,L)

N = 100

dataset = []
for f0 in Freq1:
    for A0 in Amp1:
        for phi0 in Phase1:
            y = sine(x, A0, f0, x0, phi0) + 1
            dataset.append(y)
            
for f0 in Freq2:
    for A0 in Amp2:
        for phi0 in Phase2:
            y = sine_gaussian(x, A0, f0, x0, phi0, tau0) + 1
            dataset.append(y)

    
# plt.figure()
# plt.plot(x, dataset[0], label= 'Sine')
# # plt.plot(x, dataset[99], label= 'Sine')

# plt.plot(x, dataset[-1], label = 'Gaussian sine')
# plt.xlabel('x')
# plt.ylabel('y(x)')
# plt.legend()
# plt.show()

#%%

dataset_size = np.shape(dataset)[0]
N = int(dataset_size/2)

#Adding labels: 0 for sine and 1 for gaussian sine
labels = np.append(np.zeros(N, dtype=int),np.ones(N, dtype=int))
# labels = labels.astype(int)
# labels = np.append(labels, labels)

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

y_train = torch.tensor(df_train['label'].values, dtype = torch.int64)
y_test = torch.tensor(df_test['label'].values, dtype = torch.int64)

# Adding 1 dimension to x tensor s.t. shape(x)= [2N, 1, L]
# Problem TO BE SOLVED: here channel = 1, cannnot change this value. 
x_train = x_train.unsqueeze(1)
x_test = x_test.unsqueeze(1)


train_set = TensorDataset(x_train, y_train)
trainloader = DataLoader(train_set, batch_size=100, shuffle=True)

test_set = TensorDataset(x_test, y_test)
testloader = DataLoader(test_set)

#%%


##### Defining the CNN #####



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

        self.fc1 = nn.Linear(dim_x, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.dim_x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x
    


# D_in is input dimension
D_in =  L-1

# n0 is channel on input; n1 channel out conv1; 
# n2; channel out conv2; kern is kernel size.
n0, n1, n2, kern = 1, 6, 16, 5 


net = Net1D(1, 6, 16, 5)

#%%


##### Training & Testing  #####



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Loss
criterion = torch.nn.CrossEntropyLoss()
# Optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

num_epochs = 100

def train(net, trainloader):
    for epoch in range(num_epochs): # no. of epochs
        running_loss = 0
        for data in trainloader:
            
            # data pixels and labels to GPU if available
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            
            # set the parameter gradients to zero
            optimizer.zero_grad()
            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)
            
            # propagate the loss backward
            loss.backward()
            # update the gradients
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
y0_pred = []
y1_pred = []
def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            outputs = net(inputs)
            
            # print(outputs)
            a = outputs.numpy()
            y0_pred.append(a[0][0])
            y1_pred.append(a[0][1])
            y_true.append(labels.item())
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on test signals: %0.3f %%' % (
        100 * correct / total))
    
    
test(net, testloader)

#%%

roc_df = pd.DataFrame({'y_true':y_true,'y0_pred':y0_pred, 'y1_pred':y1_pred})


fpr, tpr, thresholds = metrics.roc_curve(roc_df['y_true'], roc_df['y1_pred'])


plt.figure()
plt.plot(fpr, tpr, color='darkred')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()






















