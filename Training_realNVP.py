#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:32:34 2020

@author: pipoune
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import h5py
import matplotlib.pyplot as plt
import glob


class HDF5Dataset(Dataset):
    
     def __init__(self, h5path):
         self.h5path = h5path

     def __getitem__(self, index):
         with h5py.File(self.h5path, 'r') as f:

             dataset_name = 'data_' + str(index)
             d = f[dataset_name]

             data = d[:]

             ra = d.attrs['ra']/(2*np.pi) #Normalizing data
             dec = (d.attrs['dec'] + np.pi)/(2*np.pi)

             return data, ra, dec

     def __len__(self):
         with h5py.File(self.h5path, 'r') as f:
             return len(f.keys())

def combine_Datasets(path_dir):

     all_datasets = []

     for filepath in glob.iglob(path_dir):

         train_dataset = HDF5Dataset(filepath)
         all_datasets.append(train_dataset)


     final_dataset = ConcatDataset(all_datasets)

     return final_dataset


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()
        
        self.masks = mask
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(self.masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(self.masks))])
        
    def g(self, z, w):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            x_wf = torch.cat([x_, w],dim=1)
            s = self.s[i](x_wf)*(1 - self.mask[i])
            t = self.t[i](x_wf)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x, w):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            z_wf = torch.cat([z_, w],dim=1)
            s = self.s[i](z_wf) * (1-self.mask[i])
            t = self.t[i](z_wf) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self, x, w):
        z, logp = self.f(x, w)
        return self.prior.log_prob(z) + logp
        
    def sample(self, batchSize, w): 
        z = self.prior.sample((batchSize,1))
        logp = self.prior.log_prob(z)
        x = self.g(z.view(batchSize,-1), w)
        return x
    
    def train(self, inputs, epochs, learning_rate):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs): 
            running_loss = 0
            
            for i, (data, ra, dec) in enumerate(inputs):
                
                x_ra = ra.type(dtype)
                x_dec = dec.type(dtype)
                waveform = data.type(dtype)
                
                x = torch.cat([x_ra.view(-1,1), x_dec.view(-1,1)], dim=1)

                loss = - self.log_prob(x,waveform).mean()
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                
                running_loss += loss.item()
                
            if epoch + 1 % 20 == 0:
                print(epoch)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': flow.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, path_model_save + f'trained_model_{epoch + 1}.pth')

            print('[Epoch %d] loss: %.3f' %
                                      (epoch + 1, running_loss/len(inputs))) 
            
            



if __name__ == "__main__":
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
            dev = "cuda:0"
            dtype = torch.cuda.FloatTensor
    else:
            dev = "cpu"
            dtype = torch.FloatTensor
    
    print('device = ', dev)

    
    batch_size = 100
    num_epochs = 20
    
    path_model_save = '/Users/pipoune/Desktop/M2/METEOR_4/'

    path_datasets = '/Users/pipoune/Desktop/M2/METEOR_4/Datasets/Training/*'
    train_dataset = HDF5Dataset(path_datasets + 'waveforms_ra_dec_2500_1.hdf')
    # inputs = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    inputs = DataLoader(combine_Datasets(path_datasets), batch_size=batch_size, shuffle=True)
    
    Din = 6146 #len(train_dataset[0][0]) + 2
    nets = lambda: nn.Sequential(nn.Linear(Din,10), nn.LeakyReLU(), nn.Linear(10,10), nn.LeakyReLU(), nn.Linear(10,2), nn.Tanh())
    nett = lambda: nn.Sequential(nn.Linear(Din,10), nn.LeakyReLU(), nn.Linear(10,10), nn.LeakyReLU(), nn.Linear(10,2))
    masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
    prior = distributions.MultivariateNormal(torch.zeros(2).type(dtype), torch.eye(2).type(dtype))
    
    flow = RealNVP(nets, nett, masks, prior)
    flow.to(torch.device(dev))
    flow.train(inputs, num_epochs, 1e-4)


































