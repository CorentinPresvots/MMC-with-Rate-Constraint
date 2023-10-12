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
from Bits_allocation import Allocation_sin,Allocation_poly,Allocation_pred_samples
from Models import Model_sin,Model_poly,Model_pred_samples

 
 
import warnings
warnings.filterwarnings('ignore' )


class Model_Encoder(Model_sin,Model_poly,Model_pred_samples,Allocation_sin,Allocation_poly,Allocation_pred_samples,Quantizer):
    def __init__(self,fn=50,fs=6400,N=128,verbose=False):
        self.verbose = verbose
       
        #print("bm",self.bm) 
        
        super().__init__(fn,fs,N) 
        
        self.t=np.linspace(0,(N-1)/fs,N)

      

        Allocation_sin.__init__(self)
   

        # polynomial model
        Allocation_poly.__init__(self)

            
        #pred model 
        Allocation_pred_samples.__init__(self)
        
        
        
        
        Quantizer.__init__(self)

        
    def get_theta_sin_tilde(self,theta_sin_hat,bm,m_theta_sin,w_theta_sin):
         
         al_sin=self.get_allocation_sin(bm,m_theta_sin,w_theta_sin)
         
         
         #print("allocation bits pour bm = {} bits: {}".format(bm,al_sin))
         
         #theta_sin_hat=self.get_theta_sin(x)
         #print("theta sin hat = {:.2f},{:.2f},{:.2f}".format(*theta_sin_hat))
                  
         
         ############ quantification
         
         code_theta_sin_tilde=[0]*bm
         theta_sin_tilde=[0]*3
         
         ptr=0
         for i in range(3): 
             theta_sin_ind=self.get_ind(theta_sin_hat[i],al_sin[i],w_theta_sin[i],m_theta_sin[i])
             
             code_theta_sin_tilde[ptr:ptr+al_sin[i]]=self.get_code(theta_sin_ind,al_sin[i]) # codage entropique de theta_sin_tilde
             
             
             ptr+=al_sin[i]
             
             theta_sin_tilde[i]=self.get_q(theta_sin_ind,al_sin[i],w_theta_sin[i],m_theta_sin[i])
         
        
         #print("theta sin tilde = {:.2f},{:.2f},{:.2f}".format(*theta_sin_tilde))

         return theta_sin_tilde,code_theta_sin_tilde

    

    def get_theta_poly_tilde(self,theta_poly_hat,bm,m_theta_poly,w_theta_poly):
         
         order=len(theta_poly_hat)-1
         #self.w=self.w_poly[0:order+1]
      
         al_poly=self.get_allocation_poly(bm,w_theta_poly)
         
     
         #al_poly=[int(al_poly[i]) for i in range(order+1)]
         ############ quantification
         code_theta_poly_tilde=[0]*bm
         theta_poly_tilde=[0]*(order+1)
         
         ptr=0
         for i in range(order+1): 
             
             theta_poly_ind=self.get_ind(theta_poly_hat[i],al_poly[i],w_theta_poly[i],m_theta_poly[i])
             
             code_theta_poly_tilde[ptr:ptr+al_poly[i]]=self.get_code(theta_poly_ind,al_poly[i]) # codage entropique de theta_poly_tilde
             ptr+=al_poly[i]             
             
             
             theta_poly_tilde[i]=self.get_q(theta_poly_ind,al_poly[i],w_theta_poly[i],m_theta_poly[i])
         
            
         #print("theta poly tilde",["{:.2f}".format(theta_poly_tilde[i]) for i in range(order+1)])
                  

         return theta_poly_tilde,code_theta_poly_tilde
     
    
    def get_theta_pred_samples_tilde(self,theta_pred_samples_hat,bm,m_theta_pred_samples,w_theta_pred_samples):
         
         order=len(theta_pred_samples_hat)
       
      
         al_pred_samples=self.get_allocation_pred_samples(bm,m_theta_pred_samples,w_theta_pred_samples)
         
     
         
         ############ quantification
         code_theta_pred_samples_tilde=[0]*bm
         theta_pred_samples_tilde=[0]*order
         
         ptr=0
         for i in range(order): 
             
             theta_pred_samples_ind=self.get_ind(theta_pred_samples_hat[i],al_pred_samples[i],w_theta_pred_samples[i],m_theta_pred_samples[i])
             
             code_theta_pred_samples_tilde[ptr:ptr+al_pred_samples[i]]=self.get_code(theta_pred_samples_ind,al_pred_samples[i]) # codage entropique de theta_poly_tilde
             ptr+=al_pred_samples[i]             
             
             
             theta_pred_samples_tilde[i]=self.get_q(theta_pred_samples_ind,al_pred_samples[i],w_theta_pred_samples[i],m_theta_pred_samples[i])
         
         
         #theta_pred_samples_tilde[-1]=1-np.sum(theta_pred_samples_tilde[0:order-1])
         #print("theta poly tilde",["{:.2f}".format(theta_poly_tilde[i]) for i in range(order+1)])
                  

         return theta_pred_samples_tilde,code_theta_pred_samples_tilde
     
     
    """ 
    def best_model(self,x,bm):
           
        
        
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(t,x,lw=2,label='x')
        
        
        #### test sin
        #print(np.abs(np.mean(x)))
        #if np.abs(np.mean(x))<0.1: # test pour discriminer le modèle sin si la moyenne du signal n'est pas nul
            
        theta_sin_hat_test=self.get_theta_sin(x)
        theta_sin_tilde_test,code_theta_sin_tilde_test=self.get_theta_sin_tilde(theta_sin_hat_test,bm,m_theta_sin,w_theta_sin)
        x_sin_tilde_test=self.get_model_sin(self.t,*theta_sin_tilde_test) 
        SNR_test=get_snr(x,x_sin_tilde_test)
 
        #if SNR_test>SNR_best:
        SNR_best=SNR_test
        m_best="sin"
        #theta_hat_best=theta_sin_hat_test
        #theta_tilde_best=theta_sin_tilde_test
        code_theta_tilde_best=code_theta_sin_tilde_test
        x_rec_best= x_sin_tilde_test
    
    
        
        plt.plot(t,x_sin_tilde_test,lw=2,label='x sin, SNR={:.2f} dB, bm={} bits'.format(get_snr(x_sin,x_sin_tilde_test),bm))
       
        

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
        """
     

class Model_Decoder(Model_sin,Model_poly,Model_pred_samples,Allocation_sin,Allocation_poly,Allocation_pred_samples,Quantizer):
    def __init__(self,fn=50,fs=6400,N=128,verbose=False):
        self.verbose = verbose
       
        #print("bm",self.bm) 
        
        super().__init__(fn,fs,N) 
        
        self.t=np.linspace(0,(N-1)/fs,N)
        


        
        Allocation_sin.__init__(self)


        
        Allocation_poly.__init__(self)
        #print("Allocation_poly w",self.w)
           
        Allocation_pred_samples.__init__(self)        
        
        Quantizer.__init__(self)
        #print("self.b",self.b)
        #print("Q W",self.w)
        #print("Q m",self.m)
        #constant 
        #self.Ts=1/fs
        #self.T=N/fs # durée d'une fenêtre
        #self.Tn=1/fn # durée d'une période
        #self.t=np.linspace(0,(N-1)*self.Ts,N)
        

        
    def get_theta_sin_tilde(self,code,bm,m_theta_sin,w_theta_sin):
         
         al_sin=self.get_allocation_sin(bm,m_theta_sin,w_theta_sin)
         
         ############ quantification
         
         theta_sin_tilde=[0]*3
         
         ptr=0
         for i in range(3): 
             
             theta_sin_ind=self.get_inv_code(code[ptr:ptr+al_sin[i]],al_sin[i])
             
             ptr+=al_sin[i]
             
             theta_sin_tilde[i]=self.get_q(theta_sin_ind,al_sin[i],w_theta_sin[i],m_theta_sin[i])
            
         #print("theta sin tilde = {:.2f},{:.2f},{:.2f}".format(*theta_sin_tilde))

         return theta_sin_tilde

    

    def get_theta_poly_tilde(self,code,bm,m_theta_poly,w_theta_poly):
         
     
         order=len(w_theta_poly)-1
         al_poly=self.get_allocation_poly(bm,w_theta_poly)
         
         ############ quantification

         theta_poly_tilde=[0]*(order+1)
         
         ptr=0
         for i in range(order+1): 
             
             theta_poly_ind=self.get_inv_code(code[ptr:ptr+al_poly[i]],al_poly[i])
             
            
             ptr+=al_poly[i]             
             
             
             theta_poly_tilde[i]=self.get_q(theta_poly_ind,al_poly[i],w_theta_poly[i],m_theta_poly[i])
         
            
         #print("theta poly tilde",["{:.2f}".format(theta_poly_tilde[i]) for i in range(order+1)])
                  

         return theta_poly_tilde      

    def get_theta_pred_samples_tilde(self,code,bm,m_theta_pred_samples,w_theta_pred_samples):
         
         
         order=len(w_theta_pred_samples)
         al_pred_samples=self.get_allocation_pred_samples(bm,m_theta_pred_samples,w_theta_pred_samples)
         
         ############ quantification

         theta_pred_samples_tilde=[0]*order
         
         ptr=0
         for i in range(order): 
             
             theta_pred_samples_ind=self.get_inv_code(code[ptr:ptr+al_pred_samples[i]],al_pred_samples[i])
             
            
             ptr+=al_pred_samples[i]             
             
             
             theta_pred_samples_tilde[i]=self.get_q(theta_pred_samples_ind,al_pred_samples[i],w_theta_pred_samples[i],m_theta_pred_samples[i])
         
         
         #theta_pred_samples_tilde[-1]=1-np.sum(theta_pred_samples_tilde[0:order-1])
         #print("theta poly tilde",["{:.2f}".format(theta_poly_tilde[i]) for i in range(order+1)])
                  

         return theta_pred_samples_tilde      
     
    
    """
    def best_model_dec(self,m,code,bm):
        if m=='sin':
            theta_tilde=self.get_theta_sin_tilde(code,bm)
            model_rec=self.get_model_sin(self.t,*theta_tilde) 
            return model_rec
        elif m=="none":
            
            model_rec=np.zeros(self.N) 
            return model_rec
        else :
       
            order=int(m[5:])
            theta_tilde=self.get_theta_poly_tilde(code,bm,order)
            model_rec=self.get_model_poly(self.t,*theta_tilde) 
            return model_rec
    """



# Programme principal
if __name__ == "__main__":
    from Normalize import normalize
    
    verbose = False
    N=128
    fn=50
    fs=6400
    
    
    
    t=np.linspace(0,(N-1)/fs,N)
    
    bm=10 ### nombre de bits total pour coder theta
    
    sigma=0.001 # écart type du bruit introduit dans le signal test
        
    
    
    ####################### initialisation class Model_Encoder


    
    m=Model_Encoder(fn=fn,fs=fs,N=N,verbose=verbose)
    
    
    
    
    #################### on créer un signal de test sinusoidal bruité   
    
    m_theta_sin=[0.75,fn,0]
    w_theta_sin=[0.5,0.2,2*np.pi]
    
    
    a=np.random.uniform(m_theta_sin[0]-0.5*w_theta_sin[0],m_theta_sin[0]+0.5*w_theta_sin[0])
    f=np.random.uniform(m_theta_sin[1]-0.5*w_theta_sin[1],m_theta_sin[1]+0.5*w_theta_sin[1])
    phi=np.random.uniform(m_theta_sin[2]-0.5*w_theta_sin[2],m_theta_sin[2]+0.5*w_theta_sin[2])
    
    theta_sin=[a,f,phi]
    print("theta sin: {:.2f},{:.2f},{:.2f}".format(*theta_sin))

    model_sin=Model_sin(fn,fs,N) # initialisation de la classe qui créer les modèles sinusoïdaux
    
    x_sin=model_sin.get_model_sin(t,*theta_sin)+np.random.normal(0,sigma,N) 
    
     
    
    #####################   Codage de x_sin sur bm bits
    theta_sin_hat=m.get_theta_sin(x_sin)
    print("theta sin hat: {:.2f},{:.2f},{:.2f}".format(*theta_sin_hat))
    
    theta_sin_tilde,_=m.get_theta_sin_tilde(theta_sin_hat,bm,m_theta_sin,w_theta_sin)
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
   
    m_theta_poly=[0]*(order+1)
    w_theta_poly=[2]*(order+1)
    model_poly=Model_poly(fn,fs,N,order) # initialisation de la classe qui créer les modèles polynomiaux
    
    theta_poly=[np.random.uniform(-0.5*w_theta_poly[k],0.5*w_theta_poly[k]) for k in range(order+1)]
    print("theta poly",["{:.2f}".format(theta_poly[i]) for i in range(order+1)])
                  
    x_poly=model_poly.get_model_poly(t,*theta_poly)+np.random.normal(0,sigma,N) 
    
    
    
    #####################   Codage de x_poly sur bm bits
    theta_poly_hat=m.get_theta_poly(x_poly,order)     
    print("theta poly hat",["{:.2f}".format(theta_poly_hat[i]) for i in range(order+1)])
                   
    theta_poly_tilde,_=m.get_theta_poly_tilde(theta_poly_hat,bm,m_theta_poly,w_theta_poly)
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
        
    
    
    
    
    
    #################### on créer un signal de test sinusoidal buité comportant des harmoniques  
    m_theta_sin=[0.75,fn,0]
    w_theta_sin=[0.5,0.2,2*np.pi]
    
    
    a=np.random.uniform(m_theta_sin[0]-0.5*w_theta_sin[0],m_theta_sin[0]+0.5*w_theta_sin[0])
    f=np.random.uniform(m_theta_sin[1]-0.5*w_theta_sin[1],m_theta_sin[1]+0.5*w_theta_sin[1])
    phi=np.random.uniform(m_theta_sin[2]-0.5*w_theta_sin[2],m_theta_sin[2]+0.5*w_theta_sin[2])
    
    theta_sin=[a,f,phi]
    
    
    m_theta_sin2=[0*m_theta_sin[0]/8,3*fn,0]
    w_theta_sin2=[w_theta_sin[0]/8,w_theta_sin[1]/8,w_theta_sin[2]/8]
    
    
    a2=np.random.uniform(m_theta_sin2[0]-0.5*w_theta_sin2[0],m_theta_sin2[0]+0.5*w_theta_sin2[0])
    f2=np.random.uniform(m_theta_sin2[1]-0.5*w_theta_sin2[1],m_theta_sin2[1]+0.5*w_theta_sin2[1])
    phi2=np.random.uniform(m_theta_sin2[2]-0.5*w_theta_sin2[2],m_theta_sin2[2]+0.5*w_theta_sin2[2])
    
    
    
    
    theta_sin2=[a2,f2,phi2]
    #print("theta sin: {:.2f},{:.2f},{:.2f}".format(*theta_sin))

    model_sin=Model_sin(fn,fs,N) # initialisation de la classe qui créer les modèles sinusoïdaux
    
    t_pred_samples=np.linspace(0,(3*N-1)/fs,3*N)
    x_sin_H=model_sin.get_model_sin(t_pred_samples,*theta_sin)+model_sin.get_model_sin(t_pred_samples,*theta_sin2)+np.random.normal(0,sigma,3*N) 
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t_pred_samples[0:2*N],x_sin_H[0:2*N],lw=2,label='xp')
    plt.plot(t_pred_samples[2*N:],x_sin_H[2*N:],lw=2,label='x')
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(" x test pred samples")
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
               
    
    #####################   Codage de x_pred sur bm bits
    
    N_p=4
    eta=1
    bm=40
    
    X=m.get_X(x_sin_H[0:2*N],N_p,eta)
    theta_pred_samples_hat=m.get_theta_pred_samples(X,x_sin_H[2*N:])     
    print("theta pred samples hat",["{:.2f}".format(theta_pred_samples_hat[i]) for i in range(N_p)])
    #print(np.shape(X))
    # estimation de m_theta_pred_samples
    
    
    

    m_theta_pred_samples=m.get_m_theta_pred_samples(N_p,eta,sigma)
    w_theta_pred_samples=[1]*N_p
    
    print("theta pred samples m",["{:.2f}".format(m_theta_pred_samples[i]) for i in range(N_p)])
    
    theta_pred_samples_tilde,_=m.get_theta_pred_samples_tilde(theta_pred_samples_hat,bm,m_theta_pred_samples,w_theta_pred_samples)
    print("theta pred samples tilde",["{:.2f}".format(theta_pred_samples_tilde[i]) for i in range(N_p)])


    x_pred_samples_hat=m.get_model_pred_samples(X,*theta_pred_samples_hat) 
    x_pred_samples_tilde=m.get_model_pred_samples(X,*theta_pred_samples_tilde) 

            
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_sin_H[2*N:],lw=2,label='x')
    plt.plot(t,x_pred_samples_hat,lw=2,label='x hat, SNR={:.1f} dB'.format(get_snr(x_sin_H[2*N:],x_pred_samples_hat)))
    plt.plot(t,x_pred_samples_tilde,lw=2,label='x tilde, SNR={:.1f} dB, bm={} bits'.format(get_snr(x_sin_H[2*N:],x_pred_samples_tilde),bm))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Modèle preditif samples")
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
               
    


    """
    ######################### test best models

    x_test=x_sin#normalize(np.array([37.797, 40.045, 42.603, 44.903, 47.052, 48.893, 50.938, 52.983, 54.876, 57.024, 58.814, 60.603, 62.548, 64.697, 66.383, 68.172, 69.81, 71.803, 73.338, 75.179, 76.614, 78.148, 79.528, 81.01, 82.341, 83.721, 85.255, 86.483, 87.607, 88.734, 89.603, 90.524, 91.648, 92.724, 93.693, 94.614, 95.483, 96.2, 96.61, 97.172, 97.938, 98.552, 98.962, 99.321, 99.679, 100.034, 100.241, 100.393, 100.497, 100.293, 100.293, 100.034, 100.034, 99.628, 99.679, 99.321, 98.707, 98.245, 97.479, 96.866, 96.2, 95.69, 94.972, 94.462, 93.693, 92.824, 92.007, 91.238, 90.472, 89.5, 88.683, 87.607, 86.586, 85.614, 84.59, 83.362, 82.186, 80.859, 79.528, 78.555, 77.276, 75.641, 73.952, 72.214, 70.321, 68.379, 66.486, 64.338, 62.19, 59.89, 57.69, 55.541, 53.086, 50.683, 47.972, 44.903, 41.579, 38.1, 34.112, 30.379, 27.362, 25.162, 23.27, 21.992, 20.304, 19.128, 18.053, 17.133, 16.212, 15.19, 14.218, 13.451, 12.581, 11.865, 11.149, 10.382, 9.564, 8.746, 7.978, 7.262, 6.751, 6.137, 5.421, 4.756, 4.194, 3.58, 2.915, 2.353]))[0]#x_sin
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
                
    """