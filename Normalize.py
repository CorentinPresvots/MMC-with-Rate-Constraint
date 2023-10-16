# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:03:42 2023

@author: coren
"""
import numpy as np
import matplotlib.pyplot as plt


def normalize(x):
    k=np.ceil(np.log2(np.max(np.abs(x))+10**(-8)))
    x_n = x * 2**(-k)
    return x_n, k  # Retourne x mis à l'échelle et la valeur de k

# Programme principal
if __name__ == "__main__":


    N=128
    T=0.02
    t=np.linspace(0,T-T/N,N)
    
    x_test=np.array([1.2*np.cos(2*np.pi*50*t[i]+np.pi).real for i in range(N)])
    
    x_test=np.array([4.4, 4.6, 4.3, 4.5, 4.9, 4.4, 4.8, 4.4, 4.1, 4.3, 3.9, 4.2, 3.8, 3.9, 3.7, 3.4, 3.7, 3.6, 3.4, 3.4, 3.3, 3.2, 3.0, 2.9, 2.8, 2.6, 2.4, 2.4, 2.3, 2.0, 2.0, 1.9, 1.8, 1.8, 1.8, 1.8, 1.5, 1.2, 1.2, 1.2, 1.0, 0.9, 0.5, 0.5, 0.4, 0.2, 0.1, -0.5, -0.6, -0.7, -0.9, -0.9, -1.2, -1.3, -1.2, -1.5, -1.5, -1.4, -1.6, -1.6, -1.6, -1.8, -1.7, -1.9, -2.5, -3.8, -3.7, -3.6, -3.9, -3.2, -3.5, -3.4, -3.0, -3.4, -3.0, -3.2, -3.1, -2.8, -3.1, -2.8, -2.8, -2.8, -2.5, -2.8, -2.4, -2.5, -2.6, -2.0, -2.0, -1.9, -1.7, -1.6, -1.3, -1.3, -1.0, -0.9, -0.6, -0.5, -0.3, 0.0, 0.1, 0.1, 0.4, 0.6, 0.9, 0.9, 1.1, 1.0, 1.2, 1.4, 1.5, 1.6, 1.9, 2.0, 2.1, 2.1, 2.3, 2.5, 2.5, 2.5, 2.8, 2.8, 2.9, 2.9, 3.2, 3.8, 4.3, 4.5])
    x_test_n,k=normalize(x_test)
    
    
    plt.figure(figsize=(10,4), dpi=80)
    plt.plot(t,x_test,lw=2,label='signal sinusoïdal')
    plt.plot(t,x_test_n,lw=2,label='signal sinusoïdal normalisé')
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
