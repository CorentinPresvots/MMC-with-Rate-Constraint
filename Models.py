# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:55:28 2023

@author: coren
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math


class Models:
    def __init__(self,fn=50,fs=6400,N=128,verbose=False):
        #inputs
        self.fn=fn # fréquence nominale du réseau électrique
        self.fs=fs # fréquence d'échantillonnage
        self.N=N # nombre d'échantillons
        self.verbose = verbose
        
        #constant 
        self.Ts=1/fs
        self.T=N/fs # durée d'une fenêtre
        self.Tn=1/fn # durée d'une période
        self.t=np.linspace(0,(N-1)*self.Ts,N)
        
        
        #if (self.verbose):
        #    print("T",self.T)
     
class Model_sin(Models): 
    def __init__(self,fn=50,fs=6400,N=128,verbose=False):
    #    print("N",self.N)
        super().__init__(fn,fs,N,verbose) 
        #if self.verbose:
        #    print("T",self.T)
     
    def get_theta_sin_ini(self,y):
        a=np.std(y)*np.sqrt(2)

        index_max_y=list(y).index(np.max(y))
        delta=self.t[index_max_y]
        
        phi=-(delta*2*math.pi)/self.Tn 
        
        ### phi est ramené dans l'intervalle - pi pi
        while abs(phi)>np.pi:
            phi-=np.sign(phi)*2*np.pi
     

        if self.verbose:
            print("max y: {:.2f}".format(np.max(y)))
            print("index max y {}".format(index_max_y))
            print("delta t: {:.2f}".format(delta))
            print("phi: {:.2f}".format(phi))
     
        
        return [a,self.fn,phi]
        
    def get_model_sin(self,t,*theta):

        return theta[0]*np.cos(2*math.pi*theta[1]*t+theta[2])

    def get_theta_sin(self,y):
        theta_ini=self.get_theta_sin_ini(y) # theta0
        if self.verbose:
            print("theta ini", theta_ini)
        hat_theta, _ = curve_fit(self.get_model_sin,self.t,y,p0=theta_ini)
        return hat_theta






class Model_poly(Models):
    def __init__(self,fn=50,fs=6400,N=128,verbose=False):
        super().__init__(fn,fs,N,verbose) 
        #if self.verbose:
        #    print("T",self.T)


    def get_theta_poly_ini(self, y, order):
       
        return [np.mean(y)] + [0.] * (order)
    
    
    
    
    def get_model_poly(self, t, *theta):
        order = len(theta) - 1
        
        model = np.zeros_like(t)
        
        for i in range(order+1):
            chebyshev_term = np.polynomial.chebyshev.Chebyshev([0] * i + [1])  # Polynôme de Tchebychev d'ordre i
 
            model += theta[i] * chebyshev_term(2 * t / self.T - 1)
           
        return model
    
    
    
    
    def get_theta_poly(self, y, order):
        theta_ini = self.get_theta_poly_ini(y, order)
        
        if self.verbose:
            print("theta ini", theta_ini)
        hat_theta, _ = curve_fit(self.get_model_poly, self.t, y, p0=theta_ini)
        return hat_theta


  
# Programme principal
if __name__ == "__main__":
    from Measures import get_snr
    verbose = True
    N=128 
    fn=50
    fs=6400
    
    t=np.linspace(0,(N-1)/fs,N)
    
 
    
    model_sin=Model_sin(fn,fs,N,verbose)
    
    sigma=0.1 # écart type du bruit introduit dans le signal test
    
    ###############  test model sinusoïdal
    
    a=np.random.uniform(0.5,1)
    f=np.random.uniform(fn-1,fn+1)
    phi=np.random.uniform(-math.pi,math.pi)
    theta=[a,f,phi]
    
    
    x_sin=model_sin.get_model_sin(t,*theta)+np.random.normal(0,sigma,N) 
    
    
    print("theta_hat",model_sin.get_theta_sin(x_sin))
    x_sin_hat=model_sin.get_model_sin(t,*model_sin.get_theta_sin(x_sin))
    
    x_sin_ini=model_sin.get_model_sin(t,*model_sin.get_theta_sin_ini(x_sin))
    
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_sin,lw=2,label='x')
    plt.plot(t,x_sin_ini,lw=2,label='x ini, SNR={:.1f} dB'.format(get_snr(x_sin,x_sin_ini)))
    plt.plot(t,x_sin_hat,lw=2,label='x hat, SNR={:.1f} dB'.format(get_snr(x_sin,x_sin_hat)))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Modèle sinusoidal")
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
   
    ###############  test polynôme d'ordre k
    order=2
    theta=np.random.uniform(-1,1,order+1)
  
    model_poly=Model_poly(fn,fs,N,verbose)
    
    x_poly=model_poly.get_model_poly(t,*theta)+np.random.normal(0,sigma,N)
    
    x_poly_hat=model_poly.get_model_poly(t,*model_poly.get_theta_poly(x_poly,order))
    
    x_poly_ini=model_poly.get_model_poly(t,*model_poly.get_theta_poly_ini(x_poly,order))
    
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_poly,lw=2,label='x')
    plt.plot(t,x_poly_ini,lw=2,label='x ini, SNR={:.1f} dB'.format(get_snr(x_poly,x_poly_ini)))
    plt.plot(t,x_poly_hat,lw=2,label='x hat, SNR={:.1f} dB'.format(get_snr(x_poly,x_poly_hat)))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Modèle polynomial d'ordre {}".format(order))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()    
   
  