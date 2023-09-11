# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 07:55:58 2023

@author: coren
"""

import numpy as np
import matplotlib.pyplot as plt
import math



from Quantization import Quantizer
from Measures import get_snr
from Bits_allocation import Allocation_sin,Allocation_poly
from Models import Model_sin,Model_poly
 



class Model_Encoder(Model_sin,Model_poly,Allocation_sin,Allocation_poly,Quantizer):
    def __init__(self,fn=50,fs=6400,N=128,w_sin=[0.5,0.2,2*np.pi],m_sin=[0.75,50,0],w_poly=[2]*(16+1),verbose=False):
        self.verbose = verbose
       
        #print("bm",self.bm) 
        
        super().__init__(fn,fs,N) 
        
        self.t=np.linspace(0,(N-1)/fs,N)
        
        #print("fn",self.fn)
        #print("fs",self.fs)
        #print("N",self.N)
      

        self.w_sin=w_sin
        self.m_sin=m_sin
        Allocation_sin.__init__(self,w_sin,m_sin)
        #print("Allocation_sin w",self.w)
        #print("Allocation_sin m",self.m)

            
        self.w_poly=w_poly
        
        Allocation_poly.__init__(self,w_poly)
        #print("Allocation_poly w",self.w)
                   
        
        Quantizer.__init__(self)
        #print("self.b",self.b)
        #print("Q W",self.w)
        #print("Q m",self.m)
        #constant 
        #self.Ts=1/fs
        #self.T=N/fs # durée d'une fenêtre
        #self.Tn=1/fn # durée d'une période
        #self.t=np.linspace(0,(N-1)*self.Ts,N)
        

        
    def get_theta_sin_tilde(self,theta_sin_hat,bm):
         
         al_sin=self.get_allocation_sin(bm)
         
         
         #print("allocation bits pour bm = {} bits: {}".format(bm,al_sin))
         
         #theta_sin_hat=self.get_theta_sin(x)
         #print("theta sin hat = {:.2f},{:.2f},{:.2f}".format(*theta_sin_hat))
                  
         
         ############ quantification
         
         code_theta_sin_tilde=[0]*bm
         theta_sin_tilde=[0]*3
         
         ptr=0
         for i in range(3): 
             theta_sin_ind=self.get_ind(theta_sin_hat[i],al_sin[i],self.w_sin[i],self.m_sin[i])
             
             code_theta_sin_tilde[ptr:ptr+al_sin[i]]=self.get_code(theta_sin_ind,al_sin[i]) # codage entropique de theta_sin_tilde
             
             
             ptr+=al_sin[i]
             
             theta_sin_tilde[i]=self.get_q(theta_sin_ind,al_sin[i],self.w_sin[i],self.m_sin[i])
         
        
         #print("theta sin tilde = {:.2f},{:.2f},{:.2f}".format(*theta_sin_tilde))

         return theta_sin_tilde,code_theta_sin_tilde

    

    def get_theta_poly_tilde(self,theta_poly_hat,bm):
         
         order=len(theta_poly_hat)-1
         #self.w=self.w_poly[0:order+1]
      
         al_poly=self.get_allocation_poly(bm,order)
         
     
         #al_poly=[int(al_poly[i]) for i in range(order+1)]
         ############ quantification
         code_theta_poly_tilde=[0]*bm
         theta_poly_tilde=[0]*(order+1)
         
         ptr=0
         for i in range(order+1): 
             
             theta_poly_ind=self.get_ind(theta_poly_hat[i],al_poly[i],self.w_poly[i],0)
             
             code_theta_poly_tilde[ptr:ptr+al_poly[i]]=self.get_code(theta_poly_ind,al_poly[i]) # codage entropique de theta_poly_tilde
             ptr+=al_poly[i]             
             
             
             theta_poly_tilde[i]=self.get_q(theta_poly_ind,al_poly[i],self.w_poly[i],0)
         
            
         #print("theta poly tilde",["{:.2f}".format(theta_poly_tilde[i]) for i in range(order+1)])
                  

         return theta_poly_tilde,code_theta_poly_tilde
     
    
    def best_model(self,x,bm):
           
        SNR_best=-100
        m_best="sin"
        #theta_hat_best=[]
        #theta_tilde_best=[]
        code_theta_tilde_best=[]
        x_rec_best=np.zeros(self.N)
        
        
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(t,x,lw=2,label='x')
        
        
        #### test sin
        if np.abs(np.mean(x))<0.1: # test pour discriminer le modèle sin si la moyenne du signal n'est pas nul
            
            theta_sin_hat_test=self.get_theta_sin(x)
            theta_sin_tilde_test,code_theta_sin_tilde_test=self.get_theta_sin_tilde(theta_sin_hat_test,bm)
            x_sin_tilde_test=self.get_model_sin(self.t,*theta_sin_tilde_test) 
            SNR_test=get_snr(x,x_sin_tilde_test)
            
        else:
            theta_sin_tilde_test=[0,0,0]
            code_theta_sin_tilde_test=[0]*bm
            theta_sin_hat_test=[0,0,0]
            x_sin_tilde_test=np.zeros(self.N)
            SNR_test=get_snr(x,x_sin_tilde_test)
        
        if SNR_test>SNR_best:
            SNR_best=SNR_test
            m_best="sin"
            #theta_hat_best=theta_sin_hat_test
            #theta_tilde_best=theta_sin_tilde_test
            code_theta_tilde_best=code_theta_sin_tilde_test
            x_rec_best= x_sin_tilde_test
        
        
        
        plt.plot(t,x_sin_tilde_test,lw=2,label='x tilde, SNR={:.1f} dB, bm={} bits'.format(get_snr(x_sin,x_sin_tilde_test),bm))
       
        

        #### test poly
        for i in [0,2,4,6]:
            theta_poly_hat_test=self.get_theta_poly(x,i)
            theta_poly_tilde_test,code_theta_poly_tilde_test=self.get_theta_poly_tilde(theta_poly_hat_test,bm)
            x_poly_tilde_test=self.get_model_poly(self.t,*theta_poly_tilde_test) 
            SNR_test=get_snr(x,x_poly_tilde_test)   
            
            if SNR_test>SNR_best:
                SNR_best=SNR_test
                m_best="poly_{}".format(i)
                #theta_hat_best=theta_poly_hat_test
                #theta_tilde_best=theta_poly_tilde_test
                code_theta_tilde_best=code_theta_poly_tilde_test
                x_rec_best= x_poly_tilde_test
            
            plt.plot(self.t,x_poly_tilde_test,lw=2,label='x poly_{}, SNR={:.1f} dB, bm={} bits'.format(i,get_snr(x,x_poly_tilde_test),bm))
    
        plt.xlabel('t [s]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title("Modèles")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
    
        return SNR_best,m_best,x_rec_best,code_theta_tilde_best
        
     

class Model_Decoder(Model_sin,Model_poly,Allocation_sin,Allocation_poly,Quantizer):
    def __init__(self,fn=50,fs=6400,N=128,w_sin=[0.5,0.2,2*np.pi],m_sin=[0.75,50,0],w_poly=[2]*(16+1),verbose=False):
        self.verbose = verbose
       
        #print("bm",self.bm) 
        
        super().__init__(fn,fs,N) 
        
        self.t=np.linspace(0,(N-1)/fs,N)
        
        #print("fn",self.fn)
        #print("fs",self.fs)
        #print("N",self.N)
      

        self.w_sin=w_sin
        self.m_sin=m_sin
        Allocation_sin.__init__(self,w_sin,m_sin)
        #print("Allocation_sin w",self.w)
        #print("Allocation_sin m",self.m)

            
        self.w_poly=w_poly
        
        Allocation_poly.__init__(self,w_poly)
        #print("Allocation_poly w",self.w)
                   
        
        Quantizer.__init__(self)
        #print("self.b",self.b)
        #print("Q W",self.w)
        #print("Q m",self.m)
        #constant 
        #self.Ts=1/fs
        #self.T=N/fs # durée d'une fenêtre
        #self.Tn=1/fn # durée d'une période
        #self.t=np.linspace(0,(N-1)*self.Ts,N)
        

        
    def get_theta_sin_tilde(self,code,bm):
         
         al_sin=self.get_allocation_sin(bm)
         
         ############ quantification
         
         theta_sin_tilde=[0]*3
         
         ptr=0
         for i in range(3): 
             
             theta_sin_ind=self.get_inv_code(code[ptr:ptr+al_sin[i]],al_sin[i])
             
             ptr+=al_sin[i]
             
             theta_sin_tilde[i]=self.get_q(theta_sin_ind,al_sin[i],self.w_sin[i],self.m_sin[i])
            
         #print("theta sin tilde = {:.2f},{:.2f},{:.2f}".format(*theta_sin_tilde))

         return theta_sin_tilde

    

    def get_theta_poly_tilde(self,code,bm,order):
         
         

         al_poly=self.get_allocation_poly(bm,order)
         
         ############ quantification

         theta_poly_tilde=[0]*(order+1)
         
         ptr=0
         for i in range(order+1): 
             
             theta_poly_ind=self.get_inv_code(code[ptr:ptr+al_poly[i]],al_poly[i])
             
            
             ptr+=al_poly[i]             
             
             
             theta_poly_tilde[i]=self.get_q(theta_poly_ind,al_poly[i],self.w_poly[i],0)
         
            
         #print("theta poly tilde",["{:.2f}".format(theta_poly_tilde[i]) for i in range(order+1)])
                  

         return theta_poly_tilde      
     
    

    def best_model_dec(self,m,code,bm):
        if m=='sin':
            theta_tilde=self.get_theta_sin_tilde(code,bm)
            model_rec=self.get_model_sin(self.t,*theta_tilde) 
            return model_rec
        else :
       
            order=int(m[5:])
            theta_tilde=self.get_theta_poly_tilde(code,bm,order)
            model_rec=self.get_model_poly(self.t,*theta_tilde) 
            return model_rec




# Programme principal
if __name__ == "__main__":

    
    verbose = False
    N=128
    fn=50
    fs=6400
    
    
    
    t=np.linspace(0,(N-1)/fs,N)
    
    bm=0 ### nombre de bits total pour coder theta
    
    sigma=0.1 # écart type du bruit introduit dans le signal test
        
    
    
    ####################### initialisation class Model_Encoder
    m_theta_sin=[0.75,fn,0]
    w_theta_sin=[0.5,0.2,2*np.pi]
    
    w_theta_poly=[2]*(8+1)
    
    m=Model_Encoder(fn=fn,fs=fs,N=N,w_sin=w_theta_sin,m_sin=m_theta_sin,w_poly=w_theta_poly,verbose=verbose)
    
    
    
    
    #################### on créer un signal de test sinusoidal bruité   
    a=np.random.uniform(0.5,1)
    f=np.random.uniform(fn- w_theta_sin[1]/2,fn+w_theta_sin[1]/2)
    phi=np.random.uniform(-math.pi,math.pi)
    
    theta_sin=[a,f,phi]
    print("theta sin: {:.2f},{:.2f},{:.2f}".format(*theta_sin))

    model_sin=Model_sin(fn,fs,N) # initialisation de la classe qui créer les modèles sinusoïdaux
    
    x_sin=model_sin.get_model_sin(t,*theta_sin)+np.random.normal(0,sigma,N) 
    
     
    
    #####################   Codage de x_sin sur bm bits
    theta_sin_hat=m.get_theta_sin(x_sin)
    print("theta sin hat: {:.2f},{:.2f},{:.2f}".format(*theta_sin_hat))
    
    theta_sin_tilde,_=m.get_theta_sin_tilde(theta_sin_hat,bm)
    print("theta sin tilde: {:.2f},{:.2f},{:.2f}".format(*theta_sin_tilde))
    
    x_sin_hat=m.get_model_sin(t,*theta_sin_hat) 
    x_sin_tilde=m.get_model_sin(t,*theta_sin_tilde) 
    

    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_sin,lw=2,label='x')
    plt.plot(t,x_sin_hat,lw=2,label='x hat, SNR={:.1f} dB'.format(get_snr(x_sin,x_sin_hat)))
    plt.plot(t,x_sin_tilde,lw=2,label='x tilde, SNR={:.1f} dB, bm={} b'.format(get_snr(x_sin,x_sin_tilde),bm))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Modèle sinusoidal")
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
        

    
    #################### on créer un signal de test polynomial bruité   
    order=4 # ordre du polynome

    model_poly=Model_poly(fn,fs,N,order) # initialisation de la classe qui créer les modèles polynomiaux
    
    theta_poly=np.random.uniform(-1,1,order+1)
    print("theta poly",["{:.2f}".format(theta_poly[i]) for i in range(order+1)])
                  
    x_poly=model_poly.get_model_poly(t,*theta_poly)+np.random.normal(0,sigma,N) 
    
    
    
    #####################   Codage de x_poly sur bm bits
    theta_poly_hat=m.get_theta_poly(x_poly,order)     
    print("theta poly hat",["{:.2f}".format(theta_poly_hat[i]) for i in range(order+1)])
                   
    theta_poly_tilde,_=m.get_theta_poly_tilde(theta_poly_hat,bm)
    print("theta poly tilde",["{:.2f}".format(theta_poly_tilde[i]) for i in range(order+1)])

    x_poly_hat=m.get_model_poly(t,*theta_poly_hat) 
    
    x_poly_tilde=m.get_model_poly(t,*theta_poly_tilde) 

            
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_poly,lw=2,label='x')
    plt.plot(t,x_poly_hat,lw=2,label='x hat, SNR={:.1f} dB'.format(get_snr(x_poly,x_poly_hat)))
    plt.plot(t,x_poly_tilde,lw=2,label='x tilde, SNR={:.1f} dB, bm={} bits'.format(get_snr(x_poly,x_poly_tilde),bm))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Modèle polynomial")
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
        
    
       
    



    ######################### test best models

    x_test=x_sin
    best_SNR,best_model,x_dec_enc,code=m.best_model(x_test, bm)
    print("best SNR: {:.1f} dB.".format(best_SNR),"best model:",best_model,"bm={}, code={}, len(code)={}".format(bm,code,len(code)))
    



    ######################### test best models dec


    m_dec=Model_Decoder(fn=fn,fs=fs,N=N,w_sin=w_theta_sin,m_sin=m_theta_sin,w_poly=w_theta_poly,verbose=verbose)
    
    x_dec=m_dec.best_model_dec(best_model, code, bm)
         
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_test,lw=2,label='x')
    plt.plot(t,x_dec_enc,lw=2,label='x dec encoder')
    plt.plot(t,x_dec,lw=2,label='x dec decoder, SNR_enc={:.2f} SNR_dec={:.2f} dB'.format(best_SNR,get_snr(x_test,x_dec)))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Modèle polynomial")
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
                