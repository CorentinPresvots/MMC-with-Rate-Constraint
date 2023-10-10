# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 19:09:09 2023

@author: coren
"""


import numpy as np
import matplotlib.pyplot as plt
#import math


from Antonini import Antonini_Encoder,Antonini_Decoder
from Khan_EZW import Khan_Encoder,Khan_Decoder
#from Quantization import Quantizer
from Measures import get_snr
#from Bits_allocation import Allocation_sin,Allocation_poly
from Models import Model_sin,Model_poly
from scipy.fftpack import dct,idct
import pywt 

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class Residual_Encoder(Antonini_Encoder,Khan_Encoder):
    def __init__(self,N=128):
        
        self.N=N
        M=9
        
        
        
        initial_occurrence_first_Antonini=[1,1]
        initial_occurrence_second_Antonini=[1,1,1,1,1]
        Antonini_Encoder.__init__(self,
                                  M,
                                  initial_occurrence_first=initial_occurrence_first_Antonini,
                                  initial_occurrence_second=initial_occurrence_second_Antonini,
                                  adaptive=True,
                                  verbose_AE=False) 
        
        #print(self.initial_occurrence_second)
        
        
        
        #initialisation paramètres ondelettes
        self.wav_test = 'coif5'
        self.level = int(np.ceil(np.log2(N)))
        self.mode= 'periodization'
        
        initial_occurrence_first_Khan=[1,1]
        initial_occurrence_second_Khan=[1,1,1,1]
        Khan_Encoder.__init__(self,
                              level=self.level,
                              M=M,
                              initial_occurrence_first_Khan=initial_occurrence_first_Khan,
                              initial_occurrence_second_Khan=initial_occurrence_second_Khan,
                              adaptive_Khan=True,
                              verbose_KE=False) 
        #print(self.initial_occurrence_second)
        
    def best_residual(self,r,br):
        """
        
        Fonction qui teste toute les méthodes de compression pour un budget de bits br et qui renvoie la meilleur méthode de compression ainsi que le code binaire associé au codage du résidu
        

        Parameters
        ----------
        r : list float
            Suite de coeffieint à coder 
        br : int
            budget de bits pour coder r

        Returns
        -------
        SNR_best : float
            best SNR entre le résidu r et le résidu reconstruit
        l_best : 'str'
            nom de la méthode de compression retenu
        r_rec_best : liste float
           Suitte de coefficient reconstruit.
        code_best : list int
            code binaire represantant r_rec_best sur br bits

        """

    
        ################## Antonini DCT
        l_best="DCT+BPC"
        coefs=dct(r/self.N)
        code_best=self.get_code_res_Antonini(coefs,br)
        r_rec_best=idct(self.coefs_rec)/2
        SNR_best=get_snr(r, r_rec_best)
        
        
        
        
        
        
        
        ################## Khan EZW
        coefs = pywt.wavedec(r, self.wav_test, mode=self.mode, level=self.level)
        
        coefs_L = []
         
        # Pour chaque niveau de décomposition
        for i in range(self.level + 1):
            # Ajouter les coefficients d'approximation et de détail à la liste avec la forme de x_test
            coefs_L.extend(coefs[i]/self.level)
    
    
        code=self.get_code_res_Khan(coefs_L,br)
        
        
        coefs_L_rec=self.coefs_rec
        coefs_rec=[np.array(coefs_L_rec[:int(self.N/2**self.level)])*self.level]
        start=int(self.N/2**self.level)
        # Pour chaque niveau de décomposition
        for i in range(self.level ):
            # Ajouter les coefficients d'approximation et de détail à la liste avec la forme de x_test
            end=2*start
            coefs_rec.append(np.array(coefs_L_rec[start:end])*self.level)
            start=end

        r_rec = pywt.waverec(coefs_rec, self.wav_test, self.mode)
        SNR=get_snr(r, r_rec)
        
        """    
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(r,lw=2,label='r')
        plt.plot(r_rec_best,lw=2,label='r Antonini, SNR={:.2f} dB'.format(get_snr(r,r_rec_best)))
        plt.plot(r_rec,lw=2,label='r Khan, SNR={:.2f} dB'.format(get_snr(r,r_rec)))
        plt.xlabel('t [s]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
        """

        if SNR>SNR_best:
            SNR_best=SNR
            l_best="DWT+BPC"
            code_best=code
            r_rec_best=r_rec
        
        
        
        
        
        
        ################## VAE
        """
        en cours d'écriture
        """
        
        
        return SNR_best,l_best,r_rec_best,code_best
         


class Residual_Decoder(Antonini_Decoder,Khan_Decoder):
    def __init__(self,N=128):
        
        
        
        M=9
        initial_occurrence_first_Antonini=[1,1]
        initial_occurrence_second_Antonini=[1,1,1,1,1]
        Antonini_Decoder.__init__(self,
                                  N=N,
                                  M=M,
                                  initial_occurrence_first=initial_occurrence_first_Antonini,
                                  initial_occurrence_second=initial_occurrence_second_Antonini,
                                  adaptive=True,
                                  verbose_AD=False) 
        

        
        
        #initialisation paramètres ondelettes
        self.wav_test = 'coif5'
        self.level = int(np.ceil(np.log2(N)))
        self.mode= 'periodization'
        
        initial_occurrence_first_Khan=[1,1]
        initial_occurrence_second_Khan=[1,1,1,1]
        Khan_Decoder.__init__(self,
                              level=self.level,
                              N=N,
                              M=M,
                              initial_occurrence_first_Khan=initial_occurrence_first_Khan,
                              initial_occurrence_second_Khan=initial_occurrence_second_Khan,
                              adaptive_Khan=True,
                              verbose_KD=False)         
           
        
        
        
        
    def best_residual_dec(self,l,code,br):
        """

        Parameters
        ----------
        code_best : list int
            code binaire represantant r_rec_best sur br bits

        Returns
        -------
        r_rec_best : liste float
           Suitte de coefficient reconstruit.
        """
        
        
        if l=="DCT+BPC":
            
            coefs_rec=self.get_coefs_rec_Antonini(code,br)
            r_rec=idct(coefs_rec)/2
            
            return r_rec
        elif l=="DWT+BPC":
           
  
            coefs_L=self.get_coefs_rec_Khan(code,br)
            
            
            coefs_rec=[np.array(coefs_L[:int(self.N/2**self.level)])*self.level]
            start=int(self.N/2**self.level)
            # Pour chaque niveau de décomposition
            for i in range(self.level ):
                # Ajouter les coefficients d'approximation et de détail à la liste avec la forme de x_test
                
                end=2*start

                coefs_rec.append(np.array(coefs_L[start:end])*self.level)
                start=end

            r_rec = pywt.waverec(coefs_rec, self.wav_test, self.mode)
            return r_rec
            

            
            
            
            
            





# Programme principal
if __name__ == "__main__":
    from Normalize import normalize
    br=100

    

    N=128 
    fn=50
    fs=6400
    
    t=np.linspace(0,(N-1)/fs,N)
    
    
    sigma=0.1 # écart type du bruit introduit dans le signal test
    

    ###############  test polynôme d'ordre k
    order=12
    theta=np.random.uniform(-0.2,0.2,order+1) #[-0.4, -0.3,  0.2 , 0.05 , 0.8 ,-0.3]#
   
    model_poly=Model_poly(fn,fs,N,verbose=False)
    #"""
    x_test=model_poly.get_model_poly(t,*theta)+np.random.normal(0,sigma,N)
    x_test,_=normalize(x_test)
    #print([x_test[i] for i in range(N)])
    #"""
    #x_test=np.array([-0.8220495282783156, -0.28917539422070276, -0.6124236443917166, -0.20027159169057407, -0.2582012885074851, -0.494013054746981, -0.23794653365466895, -0.21838493540961199, 0.020274643314751206, 0.1869902117074793, -0.17001156783439805, -0.27326496241142556, 0.11836583037640167, 0.06756384615269038, 0.021081438386239285, 0.1641387030005226, -0.13717266966216024, 0.1333221897097512, 0.16152771013533435, 0.22303725424710028, -0.22318858085884857, -0.327447241410199, 0.07190177964272063, 0.13248378696204394, -0.23381991009709271, -0.353281282927286, -0.07734257533211247, -0.5763329713730724, 0.07592210968488128, -0.19074849617819511, -0.257383723928038, -0.2892886257042585, -0.4961555196677412, -0.7465246434623125, -0.46931610386907985, -0.24692011660406654, -0.2897066379928933, -0.41213371801576437, -0.36330833571865034, -0.40077484051989687, -0.5904802628180847, -0.8015612878924238, -0.3141262073397876, -0.17021084696876343, -0.8991949637421731, -0.8229223037485929, -0.5134072337539463, -0.34775929773376957, -0.49334777526274304, -0.6486710577525765, -0.13076623821638644, -0.5794750051772641, -0.6231523857325385, -0.3885728979561846, -0.6071774546031615, -0.44625695821142275, -0.3015026488596481, -0.3252864796486484, -0.4358797848843281, -0.2449171213271782, -0.47000141435068454, -0.5615829583903403, -0.13733851140757947, 0.026790858585140076, -0.42007835094061713, -0.3588882626639315, -0.201594297878031, -0.08661284859685564, 0.11275118195395961, -0.36064276858842614, -0.3538460716506815, -0.09517041150014983, 0.15805550505057123, -0.10028055730288611, 0.12419617796852095, -0.07788333053189485, 0.16325087968889024, -0.11074003072523098, 0.06137397585175708, -0.36361240998217126, -0.43445749869374994, -0.06537635446635798, -0.19629857776210383, -0.4812074681973786, -0.5615083986896692, -0.623916344483784, -0.7260624311371752, -0.6558287283366218, -0.17244894442668707, -0.558394200850944, -0.08048407149586356, -0.3906155640460682, -0.35461758919727343, -0.6059402011864325, -0.31052801838613164, -0.681417935292894, -0.41734556696891595, -0.7677732298763374, -0.29129146939969214, -0.4185744959737806, -0.7080474709692754, -0.9042828692807413, -0.4101593071819347, -0.735823012904156, -0.5675320135431005, -0.7636061480474159, -0.8182438428785721, -0.9503189649317518, -0.7969211411709143, -0.6081751126955858, -0.6810503754073287, -0.7451210827946033, -0.885827969086885, -0.8174128354135606, -0.9464377376343682, -0.9362585995364869, -0.80047271029774, -0.9083336081013951, -0.6233147268536664, -0.8197436616038912, -0.5372651698617508, -0.435946186654593, -0.7888429611281009, -0.39649850791951813, -0.4219902179857482, 0.2151715009728392, 0.11301385627588058, 0.03962931899812844])
    #### ○bug antonini ne s'arrete pas   
    
    
    
    
    
    
    
    
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_test,lw=2,label='x test')
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    



    ############################# encodage résidu sur br bits

    
    l=Residual_Encoder(N)
        
        
        
    best_SNR,best_method,x_dec_enc,code=l.best_residual(x_test, br)
    print("best SNR: {:.1f} dB.".format(best_SNR),"best method:",best_method,"br={}, code={}, len(code)={}".format(br,code,len(code)))
    



    ######################### test best residual dec

    
    l_dec=Residual_Decoder(N)
    
    x_dec=l_dec.best_residual_dec(best_method, code,br)
         
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_test,lw=2,label='x')
    plt.plot(t,x_dec_enc,lw=2,label='x dec encoder')
    plt.plot(t,x_dec,lw=2,label='x dec decoder, SNR_enc={:.2f} SNR_dec={:.2f} dB'.format(best_SNR,get_snr(x_test,x_dec)))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Résidu reconstruit")
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
                
