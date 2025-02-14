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
    
    # Import test signal generation function
    from get_test_signal import get_RTE_signal as get_signal
    
    # Define parameters
    fs = 6400  # Sampling frequency in Hz
    fn = 50    # Nominal signal frequency in Hz
    N = 128   # Number of samples
    T = 0.02  # Total duration (in seconds)
    
    # Input parameters
    id_signal = 1     # id of signals to be encoded
    id_phase = 0      # id of signal phases (u1,u2,u3,i1,i3,i3)
    id_w = 20         # Number of non-overlapping windows per signal

    x_test = get_signal(id_signal)[id_phase][id_w*N:(id_w+1)*N]
    

    # Generate a time vector from 0 to T with N points
    t = np.linspace(0, fn/fs - fn/fs/N, N)
    x_test_n,kx=normalize(x_test)
    

    plt.figure(figsize=(10,4), dpi=80)
    plt.plot(t,x_test,lw=2,label='signal')
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.title("id signal={}, id phase={}, id w={}".format(id_signal,id_phase,id_w))
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    

    plt.figure(figsize=(10,4), dpi=80)
    plt.plot(t,x_test_n,lw=2,label='signal sclale, kx={:.0f}'.format(kx))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.title("id signal={}, id phase={}, id w={}".format(id_signal,id_phase,id_w))
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    
   
