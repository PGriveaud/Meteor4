#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:53:19 2020

@author: pipoune
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import h5py
import matplotlib.pyplot as plt
plt.close('all')



class HDF5Dataset(Dataset):
    
     def __init__(self, h5path):
         self.h5path = h5path

     def __getitem__(self, index):
         with h5py.File(self.h5path, 'r') as f:

             dataset_name = 'data_' + str(index)
             d = f[dataset_name]

             data = d[:]

             ra = d.attrs['ra'] #/(2.0*np.pi)
             dec = d.attrs['dec'] #/(2.0*np.pi)

             return data, ra, dec

     def __len__(self):
         with h5py.File(self.h5path, 'r') as f:
             return len(f.keys())
         


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
    
    def train(self, inputs, epochs, learning_rate):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs): 
            running_loss = 0
            
            for i, (data, ra, dec) in enumerate(inputs):
                
                x_ra = ra.type(torch.FloatTensor)
                x_dec = dec.type(torch.FloatTensor)
                waveform = data.type(torch.FloatTensor)
                
                x = torch.cat([x_ra.view(-1,1), x_dec.view(-1,1)], dim=1)
                # print(x.shape)
                # print(waveform.shape)
                loss = - self.log_prob(x,waveform).mean()
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                
                running_loss += loss.item()
        
            if epochs >= 101:
                if epoch % 100 == 99:
                    print('[Epoch %d] loss: %.3f' %
                                  (epoch + 1, running_loss/len(inputs)))
            else:
                print('[Epoch %d] loss: %.3f' %
                                      (epoch + 1, running_loss/len(inputs)))           
        print('Done Training')
   


#%%


path = '/Users/pipoune/Desktop/M2/METEOR_4/Datasets/'
train_dataset = HDF5Dataset(path + 'waveforms_ra_dec_2500.hdf')
inputs = DataLoader(train_dataset, batch_size=100, shuffle=True)


nets = lambda: nn.Sequential(nn.Linear(6146,10), nn.LeakyReLU(), nn.Linear(10,10), nn.LeakyReLU(), nn.Linear(10,2), nn.Tanh())
nett = lambda: nn.Sequential(nn.Linear(6146,10), nn.LeakyReLU(), nn.Linear(10,10), nn.LeakyReLU(), nn.Linear(10,2))
masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
prior = distributions.MultivariateNormal(torch.zeros(2).type(torch.FloatTensor), torch.eye(2).type(torch.FloatTensor)) #torch.eye = ones()


flow = RealNVP(nets, nett, masks, prior)

flow.train(inputs, 200, 1e-4)



#%%

# =================================================================================
# ======================= Getting posterior for "real" waveform ===================
# =================================================================================



w_data =  HDF5Dataset(path + 'waveform_real_GW170817.hdf')
w_real = w_data[0][0]

ra = w_data[0][1]
dec = w_data[0][2]

batch_size = 100
dtype = torch.FloatTensor

x_wf_repeat = torch.from_numpy(np.tile(w_real, (batch_size, 1)))

# Initialise samples
samples = flow.sample(batch_size, x_wf_repeat.type(dtype)).detach().numpy()

# Make a lot of samples, you can make a loop here to produce more samples
for i in range(0, 1000):
    x = flow.sample(batch_size, x_wf_repeat.type(dtype)).detach().numpy()
    samples = np.vstack([samples, x])


# To plot you can use the corner package
import corner

truths = np.vstack([ra, dec])
figure = corner.corner(samples,labels=["Right ascension","Declination"],show_titles=True,truths=truths, color='red', bins=50)
plt.savefig('posterior.png')




#%%

# x = samples
# z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 1000)

# plt.scatter(z[:,0], z[:,1], s=2)
# plt.scatter(x[:, 0], x[:, 1], c='r', s = 2)
# plt.title(r'$X \sim p(X)$')







