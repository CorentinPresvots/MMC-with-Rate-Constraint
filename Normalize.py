# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:03:42 2023

@author: coren
"""
import numpy as np
import matplotlib.pyplot as plt


def normalize(x):
    # Vérifie si n'importe quelle valeur de x est égale à 0.0
    #print(x.any()==0.0)
    if x.any()==0.0:
        print("ok")
        k = 0  # k est initialisé à 0
        return x, k  # Si une valeur de x est 0.0, retourne x inchangé et k
    
    # Trouve la valeur maximale en valeur absolue dans x
    x_max = max(abs(x))
    
    if x_max >= 1:
        k = 0
        # Trouve la puissance de 2 (k) pour laquelle x_max est supérieure ou égale à 2^k
        while x_max > 2**k:
            k += 1
    else:
        k = 0
        # Trouve la puissance de 2 (k) pour laquelle x_max est inférieure à 2^k
        while x_max < 2**k:
            k -= 1
        k += 1  # Augmente k de 1 pour que la valeur absolue de x_max soit entre 0.5 et 1

    # Mise à l'échelle de x en multipliant par 2^-k
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