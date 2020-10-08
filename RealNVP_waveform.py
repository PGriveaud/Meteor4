#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:53:19 2020

@author: pipoune
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import h5py
from sklearn import datasets
import matplotlib.pyplot as plt
plt.close('all')


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()
        
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])
        
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


#%%

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

             data = d[:]

             mass = d.attrs['mass']
             ra = d.attrs['ra']/(2.0*np.pi)
             dec = d.attrs['dec']/(2.0*np.pi)

             return data, mass, ra, dec

     def __len__(self):
         with h5py.File(self.h5path, 'r') as f:
             return len(f.keys())
         

train_dataset = HDF5Dataset(path + 'waveforms_dataset_ra_dec_reduced.hdf')

trainloader = DataLoader(train_dataset, batch_size=100, shuffle=True)

#%%


# =================================================================================
# ============================ Training Maping function f =========================
# =================================================================================

nets = lambda: nn.Sequential(nn.Linear(6146,10), nn.LeakyReLU(), nn.Linear(10,10), nn.LeakyReLU(), nn.Linear(10,2), nn.Tanh())
nett = lambda: nn.Sequential(nn.Linear(6146,10), nn.LeakyReLU(), nn.Linear(10,10), nn.LeakyReLU(), nn.Linear(10,2))
masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
prior = distributions.MultivariateNormal(torch.zeros(2).type(torch.FloatTensor), torch.eye(2).type(torch.FloatTensor)) #torch.eye = ones()

flow = RealNVP(nets, nett, masks, prior)

optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4)
    
for t in range(100):    
    for i, (data, mass, ra, dec) in enumerate(trainloader):
        
        x_ra = ra.type(torch.FloatTensor)
        x_dec = dec.type(torch.FloatTensor)
        waveform = data.type(torch.FloatTensor)
        
        x = torch.cat([x_ra.view(-1,1), x_dec.view(-1,1)], dim=1)
        
        loss = -flow.log_prob(x,waveform).mean()
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    
        # if t % 10 == 0:
        print('iter %s:' % t, 'loss = %.3f' % loss)
        



#%%

# =================================================================================
# ======================= Testing data with "real" waveform =======================
# =================================================================================


batch_size = 100
w_data =  HDF5Dataset(path + 'waveform_real_GW170817.hdf')

for i in range(int(len(w_data[0][0])/batch_size)):
    w_real = w_data[0][0][i:i+batch_size]
    
    w_real = torch.from_numpy(w_real)
    w_real = w_real.type(torch.FloatTensor)
    w_real = w_real.view(-1,1)
    
    x = flow.sample(batch_size, w_real).detach().numpy()



