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
import os as h5py



# =================================================================================
# ============================ Saving function in hdf format  =====================
# =================================================================================

def save(self, path, group = None):
        """
        Parameters
        ----------
        path: string
            Destination file path. Must end with either .hdf, .npy or .txt.
        group: string
            Additional name for internal storage use. Ex. hdf storage uses
            this as the key value.
        """

        key = 'data' if group is None else group
        with h5py.File(path, 'a') as f:
            ds = f.create_dataset(key, data=self.numpy(),
                                  compression='gzip',
                                  compression_opts=9, shuffle=True)
            ds.attrs['start_time'] = float(self.start_time)
            ds.attrs['delta_t'] = float(self.delta_t)
            ds.attrs['ra'] = float(self.right_ascension)
            ds.attrs['dec'] = float(self.declination)
            ds.attrs['mass'] = float(self.m)
            



# =================================================================================
# ============================ Creation of waveforms ==============================
# =================================================================================


import decimal

path = '/Users/pipoune/Desktop/M2/METEOR_4/Datasets'

def waveform_generation(masses, ra, dec):
    m_idx = (1.4-0.005)*1000
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

        
        m_idx += 0.005*1000
        m_idx = decimal.Decimal(m_idx)
        m_idx = int(round(m_idx,3))
        
        for name in ['L1', 'H1', 'V1']:
            det = Detector(name)
            det_strain = det.project_wave(hp, hc, 
                                         right_ascension, declination,
                                         polarization)
            # det_strain.save(path + f'/waveforms_{name}_{m_idx}.hdf')
            
            
ra = 1.7
dec = 1.7
masses = np.arange(1.4, 1.7, .005)

waveform_generation(masses, ra, dec)

