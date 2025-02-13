# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:03:42 2023

@author: coren
"""

import numpy as np
import matplotlib.pyplot as plt

def normalize(x):
    """
    Scaling function that ensures the absolute maximum value of x 
    is between 0.5 and 1.

    Parameters:
    x (numpy array): Input signal.

    Returns:
    x_n (numpy array): Scaled signal.
    k (float): Scaling factor (power of 2).
    """
    
    # Compute the scaling factor k based on the maximum absolute value of x.
    # A small epsilon (10^(-8)) is added to prevent log2(0) issues.
    k = np.ceil(np.log2(np.max(np.abs(x)) + 10**(-8)))
    
    # Scale x by 2^(-k) to bring its absolute maximum value within [0.5, 1].
    x_n = x * 2**(-k)
    
    return x_n, k  # Return the scaled signal and the scaling factor.

if __name__ == "__main__":

    # Define parameters
    N = 128   # Number of samples
    T = 0.02  # Total duration (in seconds)
    
    # Generate a time vector from 0 to T with N points
    t = np.linspace(0, T - T/N, N)


    x_test=np.array([0.86509438, 0.91655342, 0.97511559, 1.02775841, 1.07692841,
           1.11907425, 1.16587669, 1.21267914, 1.25600876, 1.30517899,
           1.34614082, 1.38710289, 1.43161652, 1.48078651, 1.5193808 ,
           1.56034264, 1.59783188, 1.64345055, 1.67857201, 1.72071785,
           1.75355049, 1.78867218, 1.82024208, 1.85417953, 1.88464462,
           1.91621452, 1.95133598, 1.97943329, 2.00516281, 2.03097107,
           2.05086021, 2.07193314, 2.09766266, 2.12228714, 2.14446487,
           2.1655378 , 2.18542694, 2.20184326, 2.2112352 , 2.22409996,
           2.24162132, 2.25566986, 2.26506203, 2.27327019, 2.28147835,
           2.28960754, 2.29434311, 2.2978157 , 2.30018349, 2.29552689,
           2.29552689, 2.28960754, 2.28960754, 2.28029457, 2.28147835,
           2.27327019, 2.25922165, 2.24864571, 2.23112434, 2.21707581,
           2.20184326, 2.19016251, 2.17374596, 2.1620652 , 2.14446487,
           2.12457596, 2.10587082, 2.08827049, 2.07074913, 2.04849243,
           2.02978729, 2.00516281, 1.98180107, 1.95954414, 1.93610367,
           1.90800636, 1.88109306, 1.85070694, 1.82024208, 1.79798515,
           1.76870407, 1.73129379, 1.69262054, 1.65284248, 1.60951286,
           1.5650782 , 1.52174858, 1.47257835, 1.42340836, 1.37076553,
           1.32041153, 1.2712413 , 1.21504692, 1.16003632, 1.09800133,
           1.02775841, 0.95167488, 0.87203979, 0.7807711 , 0.69531944,
           0.62626007, 0.57592186, 0.53261581, 0.50335052, 0.46471687,
           0.43779579, 0.41321068, 0.39214554, 0.37107262, 0.34766373,
           0.32542259, 0.30786186, 0.28796494, 0.27157219, 0.25518745,
           0.23762672, 0.21889778, 0.20016907, 0.18260834, 0.16622337,
           0.15451881, 0.14047028, 0.12407753, 0.10886101, 0.09598824,
           0.0819397 , 0.06672295, 0.05384651])
    
    
    x_test_n,kx=normalize(x_test)
    

    plt.figure(figsize=(10,4), dpi=80)
    plt.plot(t,x_test,lw=2,label='signal')
    plt.plot(t,x_test_n,lw=2,label='signal sclale')
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.title("kx={}".format(kx))
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    

   
