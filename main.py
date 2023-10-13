# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:13:54 2023

@author: presvotscor
"""


#from  codage_model import Model_Encoder,Model_Decoder
#from  codage_residu import Residual_Encoder,Residual_Decoder
#from  Allocation_two_stages import Allocation_sin_bx_br,Allocation_poly_bx_br,Allocation_pred_samples_bx_br

#from Quantization import Quantizer

import time

from  MMC import Encode_one_window,Decode_one_window
from  Measures import get_snr


import numpy as np
import matplotlib.pyplot as plt




verbose = False

N=128 # size of window
fn=50 # nominal frequency
fs=6400 # samples frequency
 
btot=128 # number of total bits  to code each window

nb_w=100 # number of encoded window






   
"""
# Open the data
"""
from get_RTE_test_signal import get_RTE_signal

v1,v2,v3,i1,i2,i3=get_RTE_signal()









"""
###### Models in the set of models
"""

Model_used = {
"sin": {},
"poly": {},
"pred samples": {},
"pred para": {}
}




# Sinusoidals family

m_theta_sin1=[0.75,fn,0]
w_theta_sin1=[0.5,0.2,2*np.pi]
Model_used["sin"]["sin-1"]=[m_theta_sin1,w_theta_sin1]


m_theta_sin2=[0.75,fn,0]
w_theta_sin2=[0.5,0.05,2*np.pi]
Model_used["sin"]["sin-2"]=[m_theta_sin2,w_theta_sin2]



# Polynomials family
for order in range(9):
    w_theta_poly=[2]*(order+1)
    Model_used["poly"]["poly-{}".format(order)]=[order,w_theta_poly]    
      
  

#Predictives samples family

w_Np=[0.1,0.3,0.5,1.5,1.5,1.5] # dynamic of alpha for predicive samples
for eta in range(0,2):
    for N_p in range(1,len(w_Np)+1):
    
        w_theta_pred=[w_Np[N_p-1]]*N_p
        Model_used["pred samples"]["pred samp-{}-{}".format(N_p,eta)]=[N_p,eta,w_theta_pred]    
             

#Predictives parametric family

factor_w=[2,5,10,50,100,500,1000] # dynamic of alpha for predicive samples

for factor in range(len(factor_w)):
   
    Model_used["pred para"]["pred para-{}".format(factor_w[factor])]=[factor_w[factor]]    
    
    
    
# bilan models used :

name_model= ['none']

print("family: sinusoidal")
for model in Model_used["sin"].items():
    name=model[0]
    name_model.append(name)
    print("name of model: {},".format(name),"m theta: [{:.2f}, {:.2f}, {:.2f}], ".format(*Model_used["sin"][name][0]),"w theta: [{:.2f}, {:.2f}, {:.2f}]".format(*Model_used["sin"][name][1]))

print("family: polynomial")
for model in Model_used["poly"].items():
    name=model[0]
    name_model.append(name)
    order=Model_used["poly"][name][0]
    print("name of model: {}, ".format(name),"order of poly: {}, ".format(order),"w theta: {}".format(Model_used["poly"][name][1]))

print("family: predictive samples")
for model in Model_used["pred samples"].items():
    name=model[0]
    name_model.append(name)
    order=Model_used["pred samples"][name][0]
    eta=Model_used["pred samples"][name][1]
    w_theta = [np.round(100 * Model_used["pred samples"][name][2][k]) / 100 for k in range(order)]

    print(f"name of model: {name:6}, order of predictor: {order:1}, shift eta: {eta:1}, ", end='')
    print("w theta: {}".format(w_theta))

print("family: predictive parametrics")
for model in Model_used["pred para"].items():
    name=model[0]
    name_model.append(name)
    factor=Model_used["pred para"][name][0]
   
    print(f"name of model: {name:14}, factor : {factor:4}")


       







#initialisation MMC
MMC_coder=Encode_one_window(fn,fs,N,Model_used,verbose)


MMC_decoder=Decode_one_window(fn,fs,N,Model_used,verbose)
 








x_rec=np.zeros(N*nb_w)
x_model=np.zeros(N*nb_w)
r=np.zeros(N*nb_w)
r_rec=np.zeros(N*nb_w)



SNR=[0]*nb_w
SNR_r=[0]*nb_w
R_r=[0]*nb_w
SNR_m=[0]*nb_w
R_m=[0]*nb_w
M=[0]*nb_w
L=[0]*nb_w
#### start encodage 
x_p=np.zeros(3*N)


x_test=list(v1)#copie du signal à compresser



tps1 = time.perf_counter()
for w in range(0,nb_w): 
  
    x=np.array(x_test[w*N:(w+1)*N])
    
    
    ##### coder
    code=MMC_coder.MMC_enc(x,x_p,btot)
    
    
    
    
    ##### decoder
    x_rec[w*N:(w+1)*N]=MMC_decoder.MMC_dec(code,x_p,btot)
    
    SNRdec=get_snr(x,MMC_decoder.x_rec_dec)
    SNRenc=get_snr(x,MMC_coder.x_rec_best)
    SNR_m_=MMC_coder.SNR_model_best 
    SNR_r_=MMC_coder.SNR_residual_best 
    b_used=len(code)   
    m=MMC_coder.m_best 
    l=MMC_coder.l_best 
    b_kx=MMC_coder.b_kx_best
    b_kr=MMC_coder.b_kr_best
    bm=MMC_coder.bm
    bl=MMC_coder.bl
    b_bx=MMC_coder.b_bx_best
    bh=b_kx+b_kr+bm+bl+b_bx 
    bx=MMC_coder.bx_best
    br=MMC_coder.br_best 
    kx=MMC_coder.kx_best  
    kr=MMC_coder.kr_best 

    
    if SNRenc!=SNRdec:
        
        print("ERREUR !!!!!!!!!!!!!!!!!!!!!!!!!!")
     
    print(f"window={w+1:3}, SNRenc={SNRenc:5.2f} dB, SNRdec={SNRdec:2.2f} dB, SNR_m={SNR_m_:4.1f} dB, SNR_r={SNR_r_:4.1f} dB, b_used/btot={b_used:3}/{btot:3}, m={m:14}, l={l:7}, bh=b_kx+b_kr+bm+bl+b_bx={b_kx:1}+{b_kr:1}+{bm:1}+{bl:1}+{b_bx:1}={bh:2} b, bx={bx:3} b, br={br:3} b, kx={kx:1}, kr={kr:2}")
        
    
                    


    
    #mise à jour de x_p
    x_p[0:N]=x_p[N:2*N]
    x_p[N:2*N]=x_p[2*N:3*N]
    x_p[2*N:3*N]=x_rec[w*N:(w+1)*N]


    x_model[w*N:(w+1)*N]=MMC_decoder.x_model_dec
    
    r[w*N:(w+1)*N]=x-MMC_decoder.x_model_dec
    r_rec[w*N:(w+1)*N]=MMC_decoder.x_residual_dec
    
    SNR[w]=get_snr(x,MMC_decoder.x_rec_dec)
    SNR_m[w]=get_snr(x,MMC_decoder.x_model_dec)
    SNR_r[w]=get_snr(x-MMC_decoder.x_model_dec,MMC_decoder.x_residual_dec)
    

    R_m[w]=MMC_decoder.bx_dec
    R_r[w]=MMC_decoder.br_dec
    M[w]=MMC_coder.label_model[m]
    L[w]=MMC_coder.label_residual[l]



    if False:

        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(x,lw=2,label='x')
        plt.plot(MMC_decoder.x_model_dec,lw=2,label='m: {}, SNR_m={:.1f} dB'.format(MMC_decoder.m_dec,get_snr(x,MMC_decoder.x_model_dec)))
        plt.xlabel('ind')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title("Window index {}, bx={} bits".format(w+1,MMC_decoder.bx_dec))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()

        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(x-MMC_decoder.x_model_dec,lw=2,label='r')
        plt.plot(MMC_decoder.x_residual_dec,lw=2,label='l: {}, SNR_r={:.1f} dB'.format(MMC_decoder.l_dec,get_snr(x-MMC_decoder.x_model_dec,MMC_decoder.x_residual_dec)))
        plt.xlabel('ind')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title("Window index {}, br={} bits".format(w+1,MMC_decoder.br_dec))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()        


        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(x,lw=2,label='x')
        plt.plot(x_rec[w*N:(w+1)*N],lw=2,label='x rec, SNR={:.1f} dB'.format(get_snr(x,x_rec[w*N:(w+1)*N])))
        plt.xlabel('ind')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title("Window index {}, btot used={} bits, btot={} bits".format(w+1,len(code),btot))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()    
        
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(x-x_rec[w*N:(w+1)*N],lw=2,label='error')
        plt.xlabel('t [s]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title("Window index {}, btot used={} bits, btot max ={} bits".format(w+1,len(code),btot))
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()  
    

tps2 = time.perf_counter()
print("times to encode the {} windows: {:.2f} s".format(nb_w,tps2 - tps1))
#### First and second stage   


plt.figure(figsize=(8,4), dpi=100)
plt.plot(x_test[0:nb_w*N],lw=2,label='x')
plt.plot(x_rec,lw=2,label='x_rec')
plt.xlabel('ind sample')
plt.ylabel('Voltage (kV)')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show() 

plt.figure(figsize=(8,4), dpi=100)
plt.plot(x_test[0:nb_w*N]-x_rec,lw=2,label='x-x_rec')
plt.xlabel('ind sample')
plt.ylabel('Voltage (kV)')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show() 


plt.figure(figsize=(8,4), dpi=100)
plt.plot(SNR,'-o',lw=2,label='SNR MMC')
plt.xlabel('ind window')
plt.ylabel('SNR (dB)')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()    



#### first stage
plt.figure(figsize=(8,4), dpi=100)
plt.plot(x_test[0:nb_w*N],lw=2,label='x')
plt.plot(x_model,lw=2,label='x_model')
plt.xlabel('ind sample')
plt.ylabel('Voltage (kV)')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show() 

plt.figure(figsize=(8,4), dpi=100)
plt.plot(x_test[0:nb_w*N]-x_model,lw=2,label='x-x_model')
plt.xlabel('ind sample')
plt.ylabel('Voltage (kV)')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show() 

plt.figure(figsize=(8,4), dpi=100)
plt.plot(SNR_m,'-o',lw=2,label='SNR model MMC')
plt.xlabel('ind window')
plt.ylabel('SNR (dB)')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()  

plt.figure(figsize=(8,4), dpi=100)
plt.plot(R_m,'-o',lw=2,label='Number of bits to code the model MMC')
plt.xlabel('ind window')
plt.ylabel('bm (bits)')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show() 






#### Second stage
plt.figure(figsize=(8,4), dpi=100)
plt.plot(r,lw=2,label='r')
plt.plot(r_rec,lw=2,label='r_rec')
plt.xlabel('ind sample')
plt.ylabel('Voltage (kV)')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show() 

plt.figure(figsize=(8,4), dpi=100)
plt.plot(r-r_rec,lw=2,label='r-r_rec')
plt.xlabel('ind sample')
plt.ylabel('Voltage (kV)')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show() 

plt.figure(figsize=(8,4), dpi=100)
plt.plot(SNR_r,'-o',lw=2,label='SNR residual MMC')
plt.xlabel('ind window')
plt.ylabel('SNR (dB)')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()  

plt.figure(figsize=(8,4), dpi=100)
plt.plot(R_r,'-o',lw=2,label='Number of bits to code the residual MMC')
plt.xlabel('ind window')
plt.ylabel('br (bits)')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()  

### Model used
yticks_labels = name_model 
yticks_positions = np.arange(len(yticks_labels))
plt.figure(figsize=(10,7), dpi=100)
plt.plot(M,'o',lw=2,label='model index')
plt.xlabel('ind window')
plt.ylabel('Model index')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.yticks(yticks_positions, yticks_labels)
plt.show() 

### Transform used
yticks_labels = ['DCT+BPC','DWT+BPC']
yticks_positions = np.arange(len(yticks_labels))
plt.figure(figsize=(8,4), dpi=100)
plt.plot(L,'o',lw=2,label='Transform used')
plt.xlabel('ind window')
plt.ylabel('Transform index')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.yticks(yticks_positions, yticks_labels)
plt.show() 

print("mean SNR={:.2f} dB".format(np.mean(SNR)))
print("mean SNR_m={:.2f} dB".format(np.mean(SNR_m)))
print("mean b_x={:.2f} b".format(np.mean(R_m)))
print("mean SNR_r={:.2f} dB".format(np.mean(SNR_r)))
print("mean b_r={:.2f} b".format(np.mean(R_r)))


