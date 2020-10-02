#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:31:25 2020

@author: pipoune
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import h5py
import glob

"""
# =================================================================================
# ============================   Loading waveforms   ==============================
# =================================================================================

import glob
from  pycbc.types.timeseries import load_timeseries

dataset = []
for file in glob.glob(path+'/waveforms_H1*.hdf'):
    data = load_timeseries(file)
    dataset.append(data)



# for i in range(0,len(dataset), 10):
#     det_strain = dataset[i]
#     plt.plot(det_strain.sample_times, det_strain, label=f'{i}')
# plt.legend()
# plt.xlabel('Time(s)')
# plt.ylabel('Strain')

"""
# %%

path = '/Users/pipoune/Desktop/M2/METEOR_4/Datasets'


files_list = glob.glob(path+'/waveforms_H1*.hdf')


class HDF5Dataset(Dataset):

     def __init__(self, h5path):
         
         self.h5path = h5path

     def __getitem__(self, index):
        
         with h5py.File(self.h5path, 'r') as f:

             dataset_name = 'data' + str(index)  # do not understand this line             
             
             d = f[dataset_name]
             
             data = d[:]

             mass = d.attrs['mass']
             beta = d.attrs['ra']/(2.0*np.pi)
             lambd = d.attrs['dec']/(2.0*np.pi)

             return data, mass, beta, lambd

     def __len__(self):

         with h5py.File(self.h5path, 'r') as f:
             return len(f.keys())
         
bob = HDF5Dataset(files_list[0])


# dataset = []           
# for file in files_list:
#     data = HDF5Dataset(file)
#     dataset.append(data)

# trainloader = DataLoader(dataset)











