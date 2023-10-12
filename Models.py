# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:48:58 2023

@author: presvotscor
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
import time


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
        
    def norm(self,x,y):
        error = np.sum(np.square(y - x))
        #error = np.sum(np.abs(y - x))
        return error
 
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
        #print("theta",theta)
        return theta[0]*np.cos(2*math.pi*theta[1]*t+theta[2])

    
    def cost_function_sin(self,theta,y):
        x=self.get_model_sin(self.t,*theta)
        return self.norm(x,y)
        
    def get_theta_sin(self,y):
        theta_ini=self.get_theta_sin_ini(y) # theta0
        if self.verbose:
            print("theta ini", theta_ini)
            
        #bounds = [(amin, amax), (fmin, fmax), (phimin, phimax)]

        result = minimize(self.cost_function_sin, theta_ini, y, method='SLSQP')#, bounds=bounds)


            
        #hat_theta, _ = curve_fit(self.get_model_sin,self.t,y,p0=theta_ini)
        #print("[*result.x]",[*result.x])
        return [*result.x]






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
    
    
    def cost_function_poly(self,theta,y):
        x=self.get_model_poly(self.t,*theta)
        return self.norm(x,y)
    
    def get_theta_poly(self, y, order):
        theta_ini = self.get_theta_poly_ini(y, order)
        
        if self.verbose:
            print("theta ini", theta_ini)
            
            
        #bounds = [(amin, amax), (fmin, fmax), (phimin, phimax)]

        result = minimize(self.cost_function_poly, theta_ini, y, method='SLSQP')#, bounds=bounds)

 
        #hat_theta, _ = curve_fit(self.get_model_sin,self.t,y,p0=theta_ini)
        return [*result.x]
            



class Model_pred_samples(Models): 
    def __init__(self,fn=50,fs=6400,N=128,verbose=False):
        #print("N",self.N)
        super().__init__(fn,fs,N,verbose) 
        #if self.verbose:
        #print("T",self.T)
     

    def get_m_theta_pred_samples(self,N_p,eta,sigma):
        
        
        yp=np.array([0.75*np.cos(2*np.pi*self.fn*k*(1/self.fs)) for k in range(3*self.N)])+np.random.normal(0,sigma,3*self.N)
        X=self.get_X(yp[0:2*self.N],N_p,eta)
        m_theta_pred_samples=self.get_theta_pred_samples(X,yp[2*self.N:])
        return m_theta_pred_samples

    
    def get_model_pred_samples(self,X,*alpha):
        #print(np.size(X,1))
        #print(np.array(alpha))
        #print("alpha.reshape((np.size(X,1),1))",np.array(alpha).reshape((np.size(X,1),1)))
       
        #print(X)
        x_rec=X @ np.array(alpha).reshape((np.size(X,1),1))
        x_rec=x_rec.reshape(self.N)
        #print(x_rec_test)        
        
        return x_rec    
            

    def get_X(self,y_p,N_p,eta):
        

        X=np.zeros((self.N,N_p))

        for i in range(self.N):
            A=np.array(y_p[self.N+i-eta-N_p+1:self.N+i-eta+1])
            #print(A)
            #print(A[::-1])
            X[i]=A[::-1]
        #print(np.linalg.norm(X, axis=0))
        #X_n=(X-np.mean(X,axis=0))/np.linalg.norm(X, axis=0) # np.linalg.norm(X, axis=1, keepdims=True)#
        #print(X_n)
        #print(x1_rec[(k-1)*N:k*N])
        return X

    def get_theta_pred_samples(self,X,y):
        
        
        # Calculer a=(A^T*A)^(-1)A^T*B
        #tpsa = time.perf_counter()
        #hat_alpha = (np.linalg.inv(X.T @ X) @ X.T @ y.reshape((self.N,1)))
        #hat_alpha=hat_alpha.reshape(np.size(hat_alpha,0))
        #tpsb = time.perf_counter()
        #print("time 1",tpsb-tpsa)
        
        #print(np.linalg.inv(X.T @ X) @  X.T  @  X.T[0])
       
        
        #tpsa = time.perf_counter()
        hat_alpha, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        hat_alpha=hat_alpha.reshape(np.size(hat_alpha,0))
        #tpsb = time.perf_counter()
        #print("time 2",tpsb-tpsa)
        
        
        
        
        #hat_alpha[-1]=1-np.sum(hat_alpha[0:N_p-1])
        #print(hat_alpha.reshape(N_p))
        return hat_alpha




  
# Programme principal
if __name__ == "__main__":
    from Measures import get_snr,get_snr_l1
    from subsampling import dynamic_subsample
    verbose = False
    N=128 
    fn=50
    fs=6400
    
    t=np.linspace(0,(N-1)/fs,N)
    
 
    
    model_sin=Model_sin(fn,fs,N,verbose)
    
    sigma=0.001 # écart type du bruit introduit dans le signal test
    
    
    
    
    
    """
    test model sinusoïdal
    """
   
    
    a=np.random.uniform(0.5,1)
    f=np.random.uniform(fn-1,fn+1)
    phi=np.random.uniform(-math.pi,math.pi)
    theta=[a,f,phi]
    
    
    x_sin=model_sin.get_model_sin(t,*theta)+np.random.normal(0,sigma,N) 
    #x_sin[20]=20
    #x_sin[40]=15
    #x_sin[60]=20
    
    
    #print("theta hat sin",model_sin.get_theta_sin(x_sin))
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
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_sin-x_sin_hat,lw=2,label='x hat, SNR={:.1f} dB'.format(get_snr(x_sin,x_sin_hat)))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Erreur de reconstrucction")
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    """
    test polynôme d'ordre k
    """
    
    order=3
 
    theta=np.random.uniform(-1,1,order+1)
  
    model_poly=Model_poly(fn,fs,N,verbose)
    
    
    x_poly=model_poly.get_model_poly(t,*theta)+np.random.normal(0,sigma,N)
    #x_poly[20]=20
    #x_poly[40]=15
    #x_poly[60]=20
    print("theta poly hat=",model_poly.get_theta_poly(x_poly,order))
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
   
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_poly-x_poly_hat,lw=2,label='x hat, SNR={:.1f} dB'.format(get_snr(x_poly,x_poly_hat)))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Erreur de reconstruction poar le modèle polynomial d'ordre {}".format(order))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()    
     
   
    
   
    

    
   
    
   
    
    """
    test pred samples
    """
    
    eta=1
    N_p=5 # ordre du prédicteur
    

    pred=Model_pred_samples(fn,fs,N,verbose)
    
    model_sin=Model_sin(fn,fs,N,verbose)

    t_p=np.linspace(-(N*2)/fs,0-1/fs,N*2)
    t=np.linspace(0,(N-1)/fs,N)
    



    ########################################

    nb_test=1000
    sigma=0.1

    alpha_L=np.zeros((nb_test,N_p))
    SNR=np.zeros(nb_test)
    #On créer la matrice X à partir des x_rec, on cherche à prédir les échantillons (x_{kN+1}-x_{(k+1)N})
    
    for i in range(nb_test):
        
    
        a=np.random.uniform(0.5,1)
        f=np.random.uniform(fn-0.1,fn+0.1)
        phi=np.random.uniform(-math.pi,math.pi)
        theta=[a,f,phi]
        
        theta2=[0.*a/8,3*f,phi]
        
        x_p=model_sin.get_model_sin(t_p,*theta)+model_sin.get_model_sin(t_p,*theta2)+np.random.normal(0,sigma,2*N) 
        x=model_sin.get_model_sin(t,*theta)+model_sin.get_model_sin(t,*theta2)+np.random.normal(0,sigma,N)     
        
        
        X=pred.get_X(x_p, N_p, eta)
        alpha_hat=pred.get_theta_pred_samples(X,x)
        #print(np.sum( alpha_hat))
        #print(alpha_L[i])
        #print(alpha_hat.reshape(N_p))
        alpha_L[i]=alpha_hat
        
        
        x_rec=pred.get_model_pred_samples(X,*alpha_hat)
        SNR[i]=get_snr(x, x_rec)
    
        if i==0:
            
            
            SNR1=get_snr(x, x_p[N:2*N])
            plt.figure(figsize=(15,8), dpi=150)
            plt.plot(t_p,x_p,lw=1,label="x_p")
            plt.plot(t,x,lw=1,label="x")
            plt.plot(t,x_p[N:],lw=1,label="x_rec_test, SNR={:.1f} dB".format(SNR1))
            plt.xlabel('t [s]')
            plt.ylabel('amplitude')
            plt.legend()
            plt.title("Superposition de la fenêtre précédente pour reconstruire x")
            plt.grid(which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show() 
                
            plt.figure(figsize=(15,8), dpi=100)
            plt.plot(t_p,x_p,lw=1,label="xp")
            plt.plot(t,x,lw=1,label="x")
            plt.plot(t,x_rec,lw=1,label="x rec,  SNR={:.1f} dB".format(SNR[i]))
            plt.xlabel('t [s]')
            plt.ylabel('amplitude')
            plt.legend()
            plt.title("Reconstruction avec eta={}, Np={}".format(eta,N_p))
            plt.grid(which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show() 
    
 
    m_theta_pred_samples=pred.get_m_theta_pred_samples(N_p,eta,sigma)
    #print("m_theta_pred_samples",m_theta_pred_samples)
    for i in range(N_p):     
        plt.figure(figsize=(15,8), dpi=100)
        plt.hist(alpha_L[:,i],bins=int(nb_test/4),label="{}".format(i))
        plt.xlabel('amplitude')
        plt.ylabel('Nombre')
        plt.legend()
        plt.title("Reconstruction avec eta={}, Np={}, m={:.2f},  m_est={:.2f}, sigma={:.2f},".format(eta,N_p,np.mean(alpha_L[:,i]),m_theta_pred_samples[i],np.std(alpha_L[:,i])))
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()          
            

    
