#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:53:19 2020

@author: pipoune
"""

import numpy as np
import torch
import torch.nn as nn
from torch import distributions
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import h5py
import matplotlib.pyplot as plt
import glob


"""
/!\ This script must be working with the script "Training_realNVP.py"

"""


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
        dev = "cuda:0"
        dtype = torch.cuda.FloatTensor
else:
        dev = "cpu"
        dtype = torch.FloatTensor

print('device = ', dev)


# =================================================================================
# ===========================  Loading Trained Model  =============================
# =================================================================================

from Training_realNVP import HDF5Dataset, RealNVP

path_script = '/Users/pipoune/Desktop/M2/METEOR_4/'

batch_size = 100

Din = 6146 
nets = lambda: nn.Sequential(nn.Linear(Din,10), nn.LeakyReLU(), nn.Linear(10,10), nn.LeakyReLU(), nn.Linear(10,2), nn.Tanh())
nett = lambda: nn.Sequential(nn.Linear(Din,10), nn.LeakyReLU(), nn.Linear(10,10), nn.LeakyReLU(), nn.Linear(10,2))
masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
prior = distributions.MultivariateNormal(torch.zeros(2).type(dtype), torch.eye(2).type(dtype)) #torch.eye = ones()


flow = RealNVP(nets, nett, masks, prior)
flow.to(torch.device(dev))
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4)

checkpoint = torch.load(path_script + 'trained_model_20.pth')
flow.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']


# =================================================================================
# ======================= Getting posterior for "real" waveform ===================
# =================================================================================


path_realwf = '/Users/pipoune/Desktop/M2/METEOR_4/Datasets/'
w_data =  HDF5Dataset(path_realwf + 'waveform_real_GW170817.hdf')
w_real = w_data[0][0]

ra = w_data[0][1]*2*np.pi
dec = w_data[0][2]*2*np.pi - np.pi

x_wf_repeat = torch.from_numpy(np.tile(w_real, (batch_size, 1)))

samples = flow.sample(batch_size, x_wf_repeat.type(dtype)).detach().numpy()
for i in range(0, 1000):
    x = flow.sample(batch_size, x_wf_repeat.type(dtype)).detach().numpy()
    samples = np.vstack([samples, x])


# To plot you can use the corner package
import corner

samples[:,0] = samples[:,0]*2*np.pi
samples[:,1] = samples[:,1]*2*np.pi - np.pi


truths = np.vstack([ra, dec])
figure = corner.corner(samples,labels=["Right ascension","Declination"],show_titles=True,truths=truths, \
                       color='black', bins=50, quantiles=[])
plt.savefig('posterior.png', dpi = 300)



plt.figure(figsize = [7,7])
figure = corner.hist2d(samples[:,0], samples[:,1], bins = 100)
z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 10000)
plt.scatter(z[:,0], z[:,1], s=2, label='Normal distribution')
plt.xlabel('Right Ascension')
plt.ylabel('Declination')
plt.savefig('Comparison_post_normal.png', dpi = 300)







