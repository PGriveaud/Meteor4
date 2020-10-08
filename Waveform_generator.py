#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 12:07:52 2020

@author: pipoune
"""


import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
import h5py

plt.close('all')
# =================================================================================
# ============================ Saving function in hdf format  =====================
# =================================================================================

def save(datalist, data_params, path):
        with h5py.File(path, 'a') as f:
            for i in range(len(datalist)):
                data = datalist[i]
                params = data_params[i]
                ds = f.create_dataset(f'data_{i}', data=data)
                # ds.attrs['start_time'] = float(data.start_time)
                # ds.attrs['delta_t'] = float(data.delta_t)
                ds.attrs['ra'] = float(params['ra'])
                ds.attrs['dec'] = float(params['dec'])
                ds.attrs['mass'] = float(params['mass'])


# =================================================================================
# ============================ Creation of waveforms ==============================
# =================================================================================

path = '/Users/pipoune/Desktop/M2/METEOR_4/Datasets/'

datalist = []
data_params = [] 


def waveform_generation_CNN(masses, ra, dec):
    for m in masses:
        #Generate a waveform with a given component mass; assumed equal mass, nonspinning
        hp, hc = get_td_waveform(approximant="TaylorF2", 
                                 mass1=m, mass2=m, 
                                 f_lower=40.0, delta_t=1.0/1024)
        
        # Map to how a detector would see the signal for a specific sky location and time
        right_ascension = ra # Radians
        declination = dec
        polarization = 0
        hp.start_time = hc.start_time = 0 # GPS seconds
        params = {'ra':ra, 'dec': dec, 'mass':m}

        
        for name in ['L1', 'H1', 'V1']:
            det = Detector(name)
            det_strain = det.project_wave(hp, hc, 
                                         right_ascension, declination,
                                         polarization)
            det_strain = det_strain.crop(40,9)
            datalist.append(det_strain)
            data_params.append(params)
            plt.plot(det_strain.sample_times, det_strain, label=f'{name}')
            
    # save(datalist, data_params, path + 'waveforms_dataset_mass_croped_testset.hdf')
    
    
def real_waveform_generation(ra, dec):
  m = 1.5
    #Generate a waveform with a given component mass; assumed equal mass, nonspinning
  hp, hc = get_td_waveform(approximant="TaylorF2", 
                             mass1=m, mass2=m, 
                             f_lower=40.0, delta_t=1.0/(1024))
    
    # Map to how a detector would see the signal for a specific sky location and time
  right_ascension = ra # Radians
  declination = dec
  polarization = 0

  hL, hH, hV = [], [], []
  for name in ['L1', 'H1', 'V1']:
        det = Detector(name)
        det_strain = det.project_wave(hp, hc, 
                                     right_ascension, declination,
                                     polarization)
    
        det_strain = det_strain.time_slice(-12, -10) #need len to be ca 2000
        # print(len(det_strain))
        if name == 'L1':
            hL = (det_strain.numpy())
        if name == 'H1':
            hH = (det_strain.numpy())
        else: 
            hV = (det_strain.numpy())
  concat = np.concatenate((hL,hH,hV), axis = 0 )
  # print(len(concat))
  datalist.append(concat)
  # print(np.shape(datalist))
  params = {'ra':ra, 'dec': dec, 'mass':m}
  data_params.append(params)
  # print(len(data_params))
  save(datalist, data_params, path + 'waveform_real_GW170817.hdf')
                   
    
def waveform_generation_NVP(mass, ra_range, dec_range):
    m = mass
    for ra in ra_range:
        for dec in dec_range:
            #Generate a waveform with a given component mass; assumed equal mass, nonspinning
            hp, hc = get_td_waveform(approximant="TaylorF2", 
                                     mass1=m, mass2=m, 
                                     f_lower=40.0, delta_t=1.0/(1024))
            
            # Map to how a detector would see the signal for a specific sky location and time
            right_ascension = ra # Radians
            declination = dec
            polarization = 0
            # hp.start_time = hc.start_time = 0 # GPS seconds
            hL, hH, hV = [], [], []
            for name in ['L1', 'H1', 'V1']:
                det = Detector(name)
                det_strain = det.project_wave(hp, hc, 
                                             right_ascension, declination,
                                             polarization)

                det_strain = det_strain.time_slice(-12, -10) #need len to be ca 2000
                # print(len(det_strain))
                if name == 'L1':
                    hL = (det_strain.numpy())
                if name == 'H1':
                    hH = (det_strain.numpy())
                else: 
                    hV = (det_strain.numpy())
            concat = np.concatenate((hL,hH,hV), axis = 0 )
            # print(len(concat))
            datalist.append(concat)
            # print(np.shape(datalist))
            params = {'ra':ra, 'dec': dec, 'mass':m}
            data_params.append(params)
    # print(len(data_params))
    save(datalist, data_params, path + 'waveform_real_GW170817.hdf')
            



N_samples = 30
ra_range = np.random.uniform(0, 2*np.pi, N_samples) 
dec_range = np.random.uniform(0, 2*np.pi, N_samples)

mass = 1.5

# waveform_generation_CNN(masses_test, ra, dec)

# waveform_generation_NVP(mass, ra_range, dec_range)

ra_real = 3.44
dec_real = 0.41

# real_waveform_generation(ra_real, dec_real)


#%%

# =================================================================================
# ============================ Simple Load and plot  ==============================
# =================================================================================


# path = '/Users/pipoune/Desktop/M2/METEOR_4/Datasets/'

# import glob
# from  pycbc.types.timeseries import load_timeseries

# # dataset = []
# # for file in glob.glob(path+'/waveforms_H1*.hdf'):
# #     data = load_timeseries(file)
#     # dataset.append(data)

# file = '/Users/pipoune/Desktop/M2/METEOR_4/Datasets/waveforms_dataset_mass_croped_testset.hdf'
# data = load_timeseries(file)


# for i in range(0,len(dataset), 10):
#     det_strain = dataset[i]
#     plt.plot(det_strain.sample_times, det_strain, label=f'{i}')
# plt.legend()
# plt.xlabel('Time(s)')
# plt.ylabel('Strain')

