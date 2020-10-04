#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:31:25 2020

@author: pipoune
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import h5py



# =================================================================================
# ============================   Loading waveforms   ==============================
# =================================================================================

path = '/Users/pipoune/Desktop/M2/METEOR_4/Datasets/'

class HDF5Dataset(Dataset):

     def __init__(self, h5path):

         self.h5path = h5path


     def __getitem__(self, index):

         with h5py.File(self.h5path, 'r') as f:

             dataset_name = 'data_' + str(index)
             d = f[dataset_name]

             data = d[0:15426]

             mass = d.attrs['mass']
             beta = d.attrs['ra']/(2.0*np.pi)
             lambd = d.attrs['dec']/(2.0*np.pi)

             return data, mass, lambd, beta

     def __len__(self):

         with h5py.File(self.h5path, 'r') as f:
             return len(f.keys())
         

train_dataset = HDF5Dataset(path + 'waveforms_dataset_mass_croped.hdf')
test_dataset = HDF5Dataset(path + 'waveforms_dataset_mass_croped_testset.hdf')


trainloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=True)


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

        self.fc1 = nn.Linear(dim_x, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

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
D_in =  len(train_dataset[0][0])

# n0 is channel on input; n1 channel out conv1; 
# n2; channel out conv2; kern is kernel size.
net = Net1D(1, 6, 16, 5)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Loss
criterion = torch.nn.MSELoss()
# Optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

num_epochs = 10

loss_store = []

def train(net, trainloader):
    for epoch in range(num_epochs): # no. of epochs
        running_loss = 0
        for data in trainloader:
            # data pixels and labels to GPU if available
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            
            # Set the input dimensions and type to fit with network
            inputs = inputs.unsqueeze(1).type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # set the parameter gradients to zero
            optimizer.zero_grad()
            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)
            
            
            # propagate the loss backward
            loss.backward()
            # update the gradients
            optimizer.step()
 
            running_loss += loss.item()
            loss_store.append(running_loss/len(trainloader))
            
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
            inputs = inputs.unsqueeze(1).type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            
            outputs = net.forward(inputs)
            
            print(f'Output = {outputs}', f'True Mass = {labels.item()}')
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
#%%



#%%


plt.figure()
plt.scatter(y_true, y_pred, s=2)
plt.xlabel('True Mass')
plt.ylabel('Predicted Mass')
plt.show()
