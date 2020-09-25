#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:31:25 2020

@author: pipoune
"""

from pycbc.waveform import get_fd_waveform
import numpy as np

# We will try different component masses and see which gives us the largest 
chirp_mass = np.arange(1.1966, 1.1996, .0001)
masses = np.arange(1.4, 1.7, .001)

# Variables to store when we've found the max
hmax, smax, tmax, mmax, nsnr = None, {}, {}, 0, 0
snrs = []

for m in masses:
    #Generate a waveform with a given component mass; assumed equal mass, nonspinning
    hp, hc = get_fd_waveform(approximant="TaylorF2", 
                             mass1=m, mass2=m, 
                             f_lower=20, delta_f=0.1)
    
  