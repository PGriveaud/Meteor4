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


# =================================================================================
# ============================ Saving function in hdf format  =====================
# =================================================================================

def save(datalist, data_params, path):
        with h5py.File(path, 'a') as f:
            for i in range(len(datalist)):
                data = datalist[i]
                params = data_params[i]
                ds = f.create_dataset(f'data_{i}', data=data)
                ds.attrs['start_time'] = float(data.start_time)
                ds.attrs['delta_t'] = float(data.delta_t)
                ds.attrs['ra'] = float(params['ra'])
                ds.attrs['dec'] = float(params['dec'])
                ds.attrs['mass'] = float(params['mass'])


# =================================================================================
# ============================ Creation of waveforms ==============================
# =================================================================================

path = '/Users/pipoune/Desktop/M2/METEOR_4/Datasets/'

datalist = []
data_params = [] 
def waveform_generation(masses, ra, dec):
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
            datalist.append(det_strain)
            data_params.append(params)
            
    save(datalist, data_params, path + 'wavefroms_dataset_mass.hdf')
            
ra = 1.7
dec = 1.7
masses = np.arange(1.4, 1.7, .005)

waveform_generation(masses, ra, dec)


