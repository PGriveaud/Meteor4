#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:10:18 2020

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
from torch.utils.data import TensorDataset
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
        
    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self, x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp
        
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize,1))
        logp = self.prior.log_prob(z)
        x = self.g(z.view(batchSize,-1))
        return x


# nets can be FCN with input 2 and outputs 2
nets = lambda: nn.Sequential(nn.Linear(2,10), nn.LeakyReLU(), nn.Linear(10,10), nn.LeakyReLU(), nn.Linear(10,2), nn.Tanh())
nett = lambda: nn.Sequential(nn.Linear(2,10), nn.LeakyReLU(), nn.Linear(10,10), nn.LeakyReLU(), nn.Linear(10,2))
masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
prior = distributions.MultivariateNormal(torch.zeros(2).type(torch.FloatTensor), torch.eye(2).type(torch.FloatTensor)) #torch.eye = ones()

flow = RealNVP(nets, nett, masks, prior)


noisy_moons = datasets.make_moons(n_samples=100, noise=.05)[0].astype(np.float32)
noisy_moons = torch.from_numpy(noisy_moons)


optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4)
    
for t in range(5001):    
    
    loss = -flow.log_prob(noisy_moons).mean()
    
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    
    if t % 500 == 0:
        print('iter %s:' % t, 'loss = %.3f' % loss)
        
        
noisy_moons = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
torch_noisy_moons = torch.from_numpy(noisy_moons)

z = flow.f(torch_noisy_moons)[0].detach().numpy()
plt.subplot(221)
plt.scatter(z[:, 0], z[:, 1])
plt.title(r'$z = f(X)$')

z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 1000)
plt.subplot(222)
plt.scatter(z[:, 0], z[:, 1])
plt.title(r'$z \sim p(z)$')

plt.subplot(223)
x = noisy_moons
plt.scatter(x[:, 0], x[:, 1], c='r')
plt.title(r'$X \sim p(X)$')

x = flow.sample(100).detach().numpy()
plt.subplot(224)
plt.scatter(x[:, 0], x[:,1], c='r')
plt.title(r'$X = g(z)$')


