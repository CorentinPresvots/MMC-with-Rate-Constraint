# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 19:41:57 2023

@author: presvotscor
"""


import numpy as np
import matplotlib.pyplot as plt


from Normalize import normalize


from codage_model import Model_Encoder,Model_Decoder
from codage_residu import Residual_Encoder,Residual_Decoder
from Allocation_two_stages import Allocation_sin_bx_br,Allocation_poly_bx_br,Allocation_pred_samples_bx_br

#from Models import Model_sin,Model_poly
from Measures import get_snr,get_quality,my_bin,my_inv_bin



class Encode_one_window(Model_Encoder,Residual_Encoder,Allocation_sin_bx_br,Allocation_poly_bx_br,Allocation_pred_samples_bx_br):
    def __init__(self,fn=50,fs=6400, N=128,Model_used={},verbose=False):
        
        #memory of previous encoded window
        
        

        self.family_best_p="none" #previous family used
        self.theta_tilde_best_p=[] # previous parametric vector
        self.m_best_p="none"
        
        
        
        
        self.Model_used=Model_used # dictionnaire des modèles utilisés ainsi que leurs caractéristiques respéctives
        

        ### on labélise les modèles
        self.label_model={}
        ind_m=0
        self.label_model["none"]=ind_m
        ind_m+=1
        
        
        
        #### models sin
        self.name_model_sin=[]
        self.m_theta_model_sin=[]
        self.w_theta_model_sin=[]
        for model in self.Model_used["sin"].items():
            self.name_model_sin.append(model[0])
            self.m_theta_model_sin.append(Model_used["sin"][model[0]][0])
            self.w_theta_model_sin.append(Model_used["sin"][model[0]][1])
            self.label_model[model[0]]=ind_m
            ind_m+=1
        
        #### models poly
        self.name_model_poly=[]
        self.order_model_poly=[]
        self.w_theta_model_poly=[]
        for model in self.Model_used["poly"].items():
            self.name_model_poly.append(model[0])
            self.order_model_poly.append(Model_used["poly"][model[0]][0])
            self.w_theta_model_poly.append(Model_used["poly"][model[0]][1])
            self.label_model[model[0]]=ind_m
            ind_m+=1
        
        #### models pred samples
        self.name_model_pred_samples=[]
        self.order_model_pred_samples=[]
        self.eta_model_pred_samples=[]
        self.m_theta_model_pred_samples=[]
        self.w_theta_model_pred_samples=[]
        for model in self.Model_used["pred samples"].items():
            self.name_model_pred_samples.append(model[0])
            self.order_model_pred_samples.append(Model_used["pred samples"][model[0]][0])
            

            self.eta_model_pred_samples.append(Model_used["pred samples"][model[0]][1])
            self.w_theta_model_pred_samples.append(Model_used["pred samples"][model[0]][2])
            self.label_model[model[0]]=ind_m
            ind_m+=1
            
            

        #### models pred para
        self.name_model_pred_para=[]
        self.factor_model_pred_para=[]
        
        for model in self.Model_used["pred para"].items():
            self.name_model_pred_para.append(model[0])
            self.factor_model_pred_para.append(Model_used["pred para"][model[0]][0])
            
            
            self.label_model[model[0]]=ind_m
            ind_m+=1                                
        
        
        
        self.nb_model_sin=len(self.name_model_sin) # nombre de modèles sin
        self.nb_model_poly=len(self.name_model_poly) # nombre de modèles poly
        self.nb_model_pred_samples=len(self.name_model_pred_samples) # nombre de modèles pred samples
        self.nb_model_pred_para=len(self.name_model_pred_para) # nombre de modèles pred parametric
        
        self.nb_model=1+self.nb_model_sin+self.nb_model_poly+self.nb_model_pred_samples+self.nb_model_pred_para # nombre de modèles
        
        
        Model_Encoder.__init__(self,fn,fs,N,False) 
        Residual_Encoder.__init__(self,N)   
        
        ### on labélise les érsidus
        self.label_residual={}
        self.label_residual["DCT+BPC"]=0
        self.label_residual["DWT+BPC"]=1
        self.nb_residual=len(self.label_residual)
        
        
        
        Allocation_sin_bx_br.__init__(self,N,fs,False) 
        Allocation_poly_bx_br.__init__(self,N,fs,False) 
        Allocation_pred_samples_bx_br.__init__(self,N,fs,False) 
        
        #self.nb_test=32 # nombre de test réalisé autour de bx et br théorique déterminé complexité: pour un modèle on test -self.nb_test + bx_opt à self.nb_test + bx_opt

        
        #self.list_btot=[32,64,96,128,160,192,224,256]
    
    
        
    
        
        ##################### budget de bits servant à décoder le signal
        #self.b_btot=0#int(np.ceil(np.log2(len(self.list_btot)))) # nombre de bits pour coder btot
        
        self.bm=int(np.ceil(np.log2(self.nb_model))) # nombre de bits pour coder le modèle tous les polynomes d'ordre 0 à 8 + sin +none
        
        self.bl=int(np.ceil(np.log2(self.nb_residual)))  # nombre de bits pour coder la méthode de résidu actuelement: DCT, DWT
       

        self.b_kx=5 # nombre de bits pour coder kx, 0 si modèle pred_samples
        
        self.b_kr=4 # nombre de bits pour coder kr, 0 si le modèle sélectionné est none
        
        
        
        # nombr de bits max pour coder chaque modèle
        self.nb_max_bit_theta=8 # nombre de bits maximale par coefficient
        self.nb_max_bit_theta_pred=4
        
    
        
        
        self.b_bx_sin=[int(np.ceil(np.log2(3*self.nb_max_bit_theta)))]*self.nb_model_sin
        self.b_bx_poly=[int(np.ceil(np.log2((self.order_model_poly[k]+1)*self.nb_max_bit_theta))) for k in range(self.nb_model_poly)]
        self.b_bx_pred_samples=[int(np.ceil(np.log2(self.order_model_pred_samples[k]*self.nb_max_bit_theta))) for k in range(self.nb_model_pred_samples)]
        self.b_bx_pred_para=[0]*(self.nb_model_pred_para)
        
        #int(np.ceil(np.log2(np.max([1,len(self.theta_tilde_best_p)])*self.nb_max_bit_theta_pred))) 
        #                            for k in range(self.nb_model_pred_para)]
        
       
        #self.b_bx_pred_samples=[int(np.ceil(np.log2(np.max([1,(self.order_model_pred_samples[k]-1)*self.nb_max_bit_theta])))) 
        #                            for k in range(self.nb_model_pred_samples)]
        
        
        
        self.bx_sin_max=[2**self.b_bx_sin[k]-1 for k in range(self.nb_model_sin)]
        self.bx_poly_max=[2**self.b_bx_poly[k]-1 for k in range(self.nb_model_poly)]
        self.bx_pred_samples_max=[2**self.b_bx_pred_samples[k]-1 for k in range(self.nb_model_pred_samples)]
        self.bx_pred_para_max=[0]*self.nb_model_pred_para    
        
        
        

        
        
    def ini_MMC_enc(self,metric):
        #################### grandeurs optimals meilleurs modèle + meilleur méthode de compression de résidu

        
        
        

        
        self.m_best='none'
        self.l_best="DCT+BPC"
        
        self.theta_hat_best=[]
        self.theta_tilde_best=[]
        
        self.code_theta_tilde_best=[]
        self.code_residual_best=[]
        
        self.x_model_best=np.zeros(self.N)
        self.x_residual_best=np.zeros(self.N)
        self.x_rec_best=np.zeros(self.N)
        
        
        if metric=="SNR" or metric=="SNR_L1":
            c=0

        else :
            c=10**10
        
        self.quality_model_best=c
        self.quality_residual_best=c
        self.quality_best=c
        
        
        
        
        self.b_bx_best=0
        
        self.bx_best=0
        self.br_best=0
        
        self.b_kx_best=0 # nombre de bits pour coder kr, 0 si le modèle sélectionné est none
        self.b_kr_best=0 # nombre de bits pour coder kr, 0 car le modèle sélectionné est none
        
        self.bl_best=1 # nombre de bits pour coder la méthode de résidu actuelement: DCT,DWT
        
        self.btot_best=10**10
        self.kx_best=0
        self.kr_best=0
        
        self.family_best="none"
        
        self.code=[]
        
        
    """
    def trouver_racine(self,dictionnaire,name_model):
        for cle, valeur in dictionnaire.items():
            if isinstance(valeur, dict):
                if name_model in valeur:
                    return cle
                else:
                    racine = self.trouver_racine(valeur,name_model)
                    if racine:
                        return cle
        return None

    """    
        
    def MMC_enc(self,x,x_p,metric,quality,btot):
        
      
        self.ini_MMC_enc(metric)
        #print("self.quality_best",self.quality_best)
        

     
        """
        print("b_bx_sin",  self.b_bx_sin,"bx_max=",self.bx_sin_max)
        print("b_bx_poly", self.b_bx_poly,"bx_max=",self.bx_poly_max)
        print("b_bx_pred_samples", self.b_bx_pred_samples,"bx_max=",self.bx_pred_samples_max)
        print("b_bx_pred_para", self.b_bx_pred_para,"bx_max=",self.bx_pred_para_max)
        """
        

        ##################### normalisation de x
        _,kx_=normalize(x)

        if kx_>=2**self.b_kx:
            kx_=2**self.b_kx-1
        
        if kx_<0:
            kx_==0
            
           
        x_n=x*2**(-kx_)
        
        self.x_n=x_n 
        
        
        #if self.bm+self.bl>=btot:
        #    return []
        
        
        #print("kx",kx_)
        """
        print("self.nb_model_sin",self.nb_model_sin)
        print("self.nb_model_sin",self.nb_model_poly)
        print("self.nb_model_sin",self.nb_model_pred_samples)
        """
        ##################### détermination de tous les theta_hat modèles pour les modèles conssidérés:  sin, poly 0,...,6
        

        theta_sin_hat=[self.get_theta_sin(x_n)]*self.nb_model_sin 
        theta_poly_hat=[self.get_theta_poly(x_n,self.order_model_poly[k]) for k in range(self.nb_model_poly)]

  
        # model pred samples
        _,kx_p=normalize(x_p[2*self.N:3*self.N])
        x_p_n=x_p*2**(-kx_) 
        X_pred_samples=[self.get_X(x_p_n[self.N:3*self.N],self.order_model_pred_samples[k],self.eta_model_pred_samples[k]) for k in range(self.nb_model_pred_samples)]
        theta_pred_samples_hat=[self.get_theta_pred_samples(X_pred_samples[k],x_n) for k in range(self.nb_model_pred_samples)] 
        
        #m_theta_model_pred_samples=[self.get_m_theta_pred_samples(self.order_model_pred_samples[k],self.eta_model_pred_samples[k],0) for k in range(self.nb_model_pred_samples)] 
        #"""
        if self.family_best_p!="pred samples":
            m_theta_model_pred_samples=[self.get_m_theta_pred_samples(self.order_model_pred_samples[k],self.eta_model_pred_samples[k],0) for k in range(self.nb_model_pred_samples)] 
        
        else :   
            X_pred_samples2=[self.get_X(x_p_n[0:2*self.N],self.order_model_pred_samples[k],self.eta_model_pred_samples[k]) for k in range(self.nb_model_pred_samples)]
            m_theta_model_pred_samples=[self.get_theta_pred_samples(X_pred_samples2[k],x_p_n[2*self.N:3*self.N])for k in range(self.nb_model_pred_samples)] 
        #"""

        
        
        
        theta_pred_para_hat=[0]*self.nb_model_pred_para
        m_theta_pred_para=[0]*self.nb_model_pred_para
        w_theta_pred_para=[0]*self.nb_model_pred_para
        if self.family_best_p=="sin":
            theta_pred_para_hat=[theta_sin_hat[0]]*self.nb_model_pred_para#[self.get_theta_sin(x_n) for k in range(self.nb_model_pred_para)]
            m_theta_pred_para=[self.theta_tilde_best_p]*self.nb_model_pred_para# for k in range(self.nb_model_pred_para)]
            

            w_theta_sin_p=self.Model_used[self.family_best_p][self.m_best_p][1]
            w_theta_pred_para=[[w_theta_sin_p[i]/self.factor_model_pred_para[k] for i in range(len(w_theta_sin_p))] for k in range(self.nb_model_pred_para)]
        
        

        elif self.family_best_p=="pred samples":
            order=self.Model_used[self.family_best_p][self.m_best_p][0]
            eta=self.Model_used[self.family_best_p][self.m_best_p][1]
            
            
            X_pred_para=self.get_X(x_p_n[self.N:3*self.N],order,eta)
            
            
            theta_pred_para_hat=[self.get_theta_pred_samples(X_pred_para,x_n)]*(self.nb_model_pred_para)
            
            
            #X_pred_samples2_p=self.get_X(x_p_n[0:2*self.N],order,eta)
            #m_theta_pred_para=[self.get_theta_pred_samples(X_pred_samples2_p,x_p_n[2*self.N:3*self.N])]*self.nb_model_pred_samples

            
            m_theta_pred_para=[self.theta_tilde_best_p]*self.nb_model_pred_para
           

            w_theta_pred_samples_p=self.Model_used[self.family_best_p][self.m_best_p][2]
            w_theta_pred_para=[[w_theta_pred_samples_p[i]/self.factor_model_pred_para[k] for i in range(order)] for k in range(self.nb_model_pred_para)]
        elif self.family_best_p=="poly":
            order=self.Model_used[self.family_best_p][self.m_best_p][0]
            theta_pred_para_hat=[self.get_theta_poly(x_n,order)]* self.nb_model_pred_para
            m_theta_pred_para=[self.theta_tilde_best_p]*self.nb_model_pred_para
            
            w_theta_poly_p=self.Model_used[self.family_best_p][self.m_best_p][1]
            w_theta_pred_para=[[w_theta_poly_p[i]/self.factor_model_pred_para[k] for i in range(order+1)] for k in range(self.nb_model_pred_para)]
       
        #print("self.theta_tilde_best_p",self.theta_tilde_best_p)
        """
       
        
        t2=np.linspace(0,(3*self.N-1)*(1/self.fs),3*self.N)
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(t2[0:2*self.N],x_p_n[0:2*self.N],lw=2,label='two prvious w')
        plt.plot(t2[2*self.N:3*self.N],x_p_n[2*self.N:3*self.N],lw=2,label='previous w')
        plt.plot(t2[2*self.N:3*self.N],x_n,lw=2,label='curent w SNR ={}'.format(get_snr(x_p_n[2*self.N:3*self.N],x_n)))
        plt.xlabel('t [s]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()  
        """
            
        
        #print("theta_sin_hat",theta_sin_hat)
        #print("theta_poly_hat",theta_poly_hat[7])
        #print("theta_pred_samples_hat",theta_pred_samples_hat)
        #print("theta_pred_para_hat",theta_pred_para_hat)
        

        ###################################### détermination des modèles où les paramètres ont été estimés
        
        """
        x_sin_hat=[self.get_model_sin(self.t,*theta_sin_hat[k]) for k in range(self.nb_model_sin)]
        
        x_poly_hat=[self.get_model_poly(self.t,*theta_poly_hat[k]) for k in range(self.nb_model_poly)]
        
        
        x_pred_samples_hat=[self.get_model_pred_samples(X_pred_samples[k],*theta_pred_samples_hat[k]) for k in range(self.nb_model_pred_samples)]
        
        
        x_pred_para_hat=[[0]*self.N]*self.nb_model_pred_para        
        if self.family_best_p=="sin":
            x_pred_para_hat=[self.get_model_sin(self.t,*theta_pred_para_hat[k]) for k in range(self.nb_model_pred_para)]
        elif self.family_best_p=="poly":
            x_pred_para_hat=[self.get_model_poly(self.t,*theta_pred_para_hat[k]) for k in range(self.nb_model_pred_para)]
        elif self.family_best_p=="pred samples":
            x_pred_para_hat=[self.get_model_pred_samples(X_pred_para,*theta_pred_para_hat[k]) for k in range(self.nb_model_pred_para)]
        """  

        
        """
        for k in range(self.nb_model_pred_samples):
            #print("m_theta_model_pred_samples",[np.round(1000*m_theta_model_pred_samples[k][i])/1000 for i in range(self.order_model_pred_samples[k])],"eta={}".format(self.eta_model_pred_samples[k]))
            #print("x_theta_model_pred_samples",[np.round(1000*theta_pred_samples_hat[k][i])/1000 for i in range(self.order_model_pred_samples[k])],"eta={}".format(self.eta_model_pred_samples[k]))
            print("error",[np.round(1000*(m_theta_model_pred_samples[k][i]-theta_pred_samples_hat[k][i]))/1000 for i in range(self.order_model_pred_samples[k])])
        """
        
        """
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(x_n,lw=1,label='x')
        
        
        for k in range(self.nb_model_sin):
            plt.plot(x_sin_hat[k],lw=1,label='name={}, SNR={:.2f} dB'.format(self.name_model_sin[k],get_snr(x_n,x_sin_hat[k])))
        plt.xlabel('ind')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()    
        
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(x_n,lw=1,label='x')
        
        for k in range(self.nb_model_poly):
            plt.plot(x_poly_hat[k],lw=1,label='name={}, SNR={:.2f} dB'.format(self.name_model_poly[k],get_snr(x_n,x_poly_hat[k])))
        plt.xlabel('ind')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()    
        
        
        
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(x_n,lw=1,label='x')
        for k in range(self.nb_model_pred_samples):
            plt.plot(x_pred_samples_hat[k],lw=1,label='name={}, SNR={:.2f} dB'.format(self.name_model_pred_samples[k],get_snr(x_n,x_pred_samples_hat[k])))
        plt.xlabel('ind')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()    
        
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(x_n,lw=1,label='x')
        plt.plot(x_pred_para_hat[0],lw=1,label='name={}, name_p={}, SNR={:.2f} dB'.format(self.name_model_pred_para[0],self.m_best_p,get_snr(x_n,x_pred_para_hat[0])))
        plt.xlabel('ind')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()  
        """        
      











        """
        #################### détermination de l'erreur due à l'estimation des paramétre 
        SE_sin_hat=[np.sum((x_n-x_sin_hat[k])**2)  for k in range(self.nb_model_sin)]
        SE_poly_hat=[np.sum((x_n-x_poly_hat[k])**2) for k in range(self.nb_model_poly)]
        SE_pred_samples_hat=[np.sum((x_n-x_pred_samples_hat[k])**2) for k in range(self.nb_model_pred_samples)]
        SE_pred_para_hat=[np.sum((x_n-x_pred_para_hat[k])**2) for k in range(self.nb_model_pred_para)]

        ###################### estimation de bx pour chaque modèle utilisé
        bx_sin_hat=[self.get_sin_bx_br(SE_sin_hat[k],
                            btot-self.bm-self.bl-self.b_kx-self.b_kr-self.b_bx_sin[k],
                            self.m_theta_model_sin[k],self.w_theta_model_sin[k])[0] for k in range(self.nb_model_sin)]
        bx_poly_hat=[self.get_poly_bx_br(SE_poly_hat[k],
                            btot-self.bm-self.bl-self.b_kx-self.b_kr-self.b_bx_poly[k],
                            self.w_theta_model_poly[k])[0] for k in range(self.nb_model_poly)]
        bx_pred_samples_hat=[self.get_pred_samples_bx_br(SE_pred_samples_hat[k],
                            btot-self.bm-self.bl-1*self.b_kx-self.b_kr-self.b_bx_pred_samples[k],
                            m_theta_model_pred_samples[k],self.w_theta_model_pred_samples[k],x_p_n[self.N:3*self.N],self.eta_model_pred_samples[k])[0] for k in range(self.nb_model_pred_samples)]

        
        
        bx_pred_para_hat=[0]*self.nb_model_pred_para
        if self.family_best_p=="sin":
            bx_pred_para_hat=[self.get_sin_bx_br(SE_pred_para_hat[k],
                                btot-self.bm-self.bl-1*self.b_kx-self.b_kr-self.b_bx_pred_para[k],
                                m_theta_pred_para[k],w_theta_pred_para[k])[0] for k in range(self.nb_model_pred_para)]
        elif self.family_best_p=="poly":
            bx_pred_para_hat=[self.get_poly_bx_br(SE_pred_para_hat[k],
                                btot-self.bm-self.bl-1*self.b_kx-self.b_kr-self.b_bx_pred_para[k],
                                w_theta_pred_para[k])[0] for k in range(self.nb_model_pred_para)]
        elif self.family_best_p=="pred samples":
            bx_pred_para_hat=[self.get_pred_samples_bx_br(SE_pred_para_hat[k],
                                btot-self.bm-self.bl-1*self.b_kx-self.b_kr-self.b_bx_pred_para[k],
                                m_theta_pred_para[k],w_theta_pred_para[k],x_p_n[self.N:3*self.N],eta)[0] for k in range(self.nb_model_pred_para)]

        """
        
        
        
        
        
        
        
        
        
        
        
        
        """
        print("bx_sin_hat",bx_sin_hat)
        print("bx_poly_hat",bx_poly_hat)
        print("bx_pred_samples_hat",bx_pred_samples_hat)
        print("bx_pred_para_hat",bx_pred_para_hat,self.m_best_p)
        """
        #################### début test modèle
    
        
        


        for bx_tot in range(btot): #(25,26): Pour fixer le nombre de bits servant à coder le modèle



            if metric=="SNR" or metric=="SNR_L1":
                quality_model=0
            else :
                quality_model=10**10
                
            to_do_second_stage=0 
            
            
            
            ### TESTs Pred samples
            for k in range(self.nb_model_pred_samples):

                bx_test=bx_tot-1*self.b_kx-self.b_kr-self.b_bx_pred_samples[k]

                if bx_test>=0 and bx_test<=2**self.b_bx_pred_samples[k]-1:

                    theta_pred_samples_tilde_test,code_theta_pred_samples_tilde_test=self.get_theta_pred_samples_tilde(theta_pred_samples_hat[k],bx_test,m_theta_model_pred_samples[k],self.w_theta_model_pred_samples[k])
                    x_pred_samples_tilde_test=self.get_model_pred_samples(X_pred_samples[k],*theta_pred_samples_tilde_test) 
                    
                    quality_model_test=get_quality(x,x_pred_samples_tilde_test*2**(kx_),metric)
                    
                    test=False
                    if metric=="SNR" or metric=="SNR_L1":
                        if quality_model_test>=quality_model:
                            test=True
 
                    else :
                        if quality_model_test<=quality_model:
                            test=True

                    
                    if test:    
                        family_best="pred samples"
                        m=self.name_model_pred_samples[k]
                        b_bx=self.b_bx_pred_samples[k]
                        bx=bx_test
                        #print("bx",bx,m)
                        theta_hat=theta_pred_samples_hat[k]
                        theta_tilde=theta_pred_samples_tilde_test
                        
                        code_theta_tilde=code_theta_pred_samples_tilde_test
                        
                        x_model=x_pred_samples_tilde_test
                        
                        quality_model=quality_model_test

                        b_kr=self.b_kr
                        

                        b_kx=self.b_kx
                        kx=kx_
                        
                        to_do_second_stage=1               
    

            ### TESTs Pred para
            for k in range(self.nb_model_pred_para):
    
                bx_test=bx_tot-1*self.b_kx-self.b_kr-self.b_bx_pred_para[k]
                
                
                if metric=="SNR" or metric=="SNR_L1":
                    quality_model_test=-1
                else :
                    quality_model_test=10**10+1
                

                if bx_test>=0 and  bx_test<=2**self.b_bx_pred_para[k]-1:

                    if self.family_best_p=="sin":
                        theta_pred_para_tilde_test,code_theta_pred_para_tilde_test=self.get_theta_sin_tilde(theta_pred_para_hat[k],bx_test,m_theta_pred_para[k],w_theta_pred_para[k])
                        x_pred_para_tilde_test=self.get_model_sin(self.t,*theta_pred_para_tilde_test) 
                        
                        quality_model_test=get_quality(x,x_pred_para_tilde_test*2**(kx_),metric)
                        
                    
                    elif self.family_best_p=="pred samples":
                        theta_pred_para_tilde_test,code_theta_pred_para_tilde_test=self.get_theta_pred_samples_tilde(theta_pred_para_hat[k],bx_test,m_theta_pred_para[k],w_theta_pred_para[k])
                        x_pred_para_tilde_test=self.get_model_pred_samples(X_pred_para,*theta_pred_para_tilde_test) 
                        
                        quality_model_test=get_quality(x,x_pred_para_tilde_test*2**(kx_),metric)
                       
                    elif self.family_best_p=="poly":
                        theta_pred_para_tilde_test,code_theta_pred_para_tilde_test=self.get_theta_poly_tilde(theta_pred_para_hat[k],bx_test,m_theta_pred_para[k],w_theta_pred_para[k])
                        x_pred_para_tilde_test=self.get_model_poly(self.t,*theta_pred_para_tilde_test) 
                        
                        quality_model_test=get_quality(x,x_pred_para_tilde_test*2**(kx_),metric)
                     
                    """
                    print("theta",np.round(100*np.array(theta_pred_para_hat[k]))/100,"m_theta",np.round(100*np.array(m_theta_pred_para[k]))/100,"w_theta",np.round(100*np.array(w_theta_pred_para[k]))/100)
                    
                    print("tilde",np.round(100*np.array(theta_pred_para_tilde_test))/100)
                    print("bx_tot = {}, bx = {}, quality ={} para {}, b_bx_pred_para = {}".format(bx_tot,bx_test,quality_model_test,k,self.b_bx_pred_para[k]))
                    
                    plt.figure(figsize=(8,4), dpi=100)
                    plt.plot(x,lw=1,label='x')
                    plt.plot(np.array(x_pred_para_hat[k])*2**(kx_),lw=1,label='name={}, hat, {}={:.2f} dB'.format(self.name_model_pred_para[k],metric,get_quality(x_n,np.array(x_pred_para_hat[k]),metric)))
                    plt.plot(x_pred_para_tilde_test*2**(kx_),lw=1,label='name={}, tilde, {}={:.2f} dB'.format(self.name_model_pred_para[k],metric,get_quality(x_n,x_pred_para_tilde_test,metric)))
                    plt.xlabel('ind')
                    plt.ylabel('Amplitude')
                    plt.legend()
                    plt.grid( which='major', color='#666666', linestyle='-')
                    plt.minorticks_on()
                    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                    plt.show()  
                    """  
                    
                    
                    test=False
                    if metric=="SNR" or metric=="SNR_L1":
                        if quality_model_test>=quality_model:
                            test=True
                    else :
                        if quality_model_test<=quality_model:
                            test=True
                        
                    if test:
                        
                        family_best="pred para"
                        m=self.name_model_pred_para[k]
                        b_bx=self.b_bx_pred_para[k]
                        bx=bx_test

                        theta_hat=theta_pred_para_hat[k]
                        theta_tilde=theta_pred_para_tilde_test
                        
                        code_theta_tilde=code_theta_pred_para_tilde_test
                        
                        x_model=x_pred_para_tilde_test
                        
                        quality_model=quality_model_test
                        b_kr=self.b_kr
                        
                        b_kx=self.b_kx
                        kx=kx_
                        
                        to_do_second_stage=1   



                        
            # test sin
            for k in range(self.nb_model_sin):   
       

                bx_test=bx_tot-self.b_kx-self.b_kr-self.b_bx_sin[k]
             
                #print("bx_test sin",bx_test)
                if bx_test>=0  and  bx_test<=2**self.b_bx_sin[k]-1:
    
                    theta_sin_tilde_test,code_theta_sin_tilde_test=self.get_theta_sin_tilde(theta_sin_hat[k],bx_test,self.m_theta_model_sin[k],self.w_theta_model_sin[k])
                    x_sin_tilde_test=self.get_model_sin(self.t,*theta_sin_tilde_test) 
                    quality_model_test=get_quality(x,x_sin_tilde_test*2**(kx_),metric)
                    
                    if metric=="SNR" or metric=="SNR_L1":
                        if quality_model_test>=quality_model:
                            test=True
                        else : 
                            test=False
                    else :
                        if quality_model_test<=quality_model:
                            test=True
                        else : 
                            test=False
                    if test:    
        
                        family_best="sin"
                        m=self.name_model_sin[k]
                        b_bx=self.b_bx_sin[k]
                        bx=bx_test
                        
                        theta_hat=theta_sin_hat[k]
                        theta_tilde=theta_sin_tilde_test
                        
                        code_theta_tilde=code_theta_sin_tilde_test
                        
                        x_model=x_sin_tilde_test
                        
                        quality_model=quality_model_test
                   
                        b_kr=self.b_kr
                        
                        b_kx=self.b_kx
                        kx=kx_
                        

                        to_do_second_stage=1
                        
              
            ### TESTs POLY
            for k in range(self.nb_model_poly):

                    
                bx_test=bx_tot-self.b_kx-self.b_kr-self.b_bx_poly[k]
        
                if  bx_test>=0 and bx_test<=2**self.b_bx_poly[k]-1:
                    theta_poly_tilde_test,code_theta_poly_tilde_test=self.get_theta_poly_tilde(theta_poly_hat[k],bx_test,[0]*(self.order_model_poly[k]+1),self.w_theta_model_poly[k])
                    x_poly_tilde_test=self.get_model_poly(self.t,*theta_poly_tilde_test) 
                    quality_model_test=get_quality(x,x_poly_tilde_test*2**(kx_),metric)
                   
                    if metric=="SNR" or metric=="SNR_L1":
                        if quality_model_test>=quality_model:
                            test=True
                        else : 
                            test=False
                    else :
                        if quality_model_test<=quality_model:
                            test=True
                        else : 
                            test=False
                    if test:                        
                        family_best="poly"
                        m=self.name_model_poly[k]
                        b_bx=self.b_bx_poly[k]
                        bx=bx_test
                        #print("bx",bx,m)
                        theta_hat=theta_poly_hat[k]
                        theta_tilde=theta_poly_tilde_test
                        
                        code_theta_tilde=code_theta_poly_tilde_test
                        
                        x_model=x_poly_tilde_test
                        
                        quality_model=quality_model_test

                        b_kr=self.b_kr
                        
                        b_kx=self.b_kx
                        kx=kx_
                        

                        to_do_second_stage=1
            
            
            
            
            #Test None model
            bx_test=bx_tot-self.b_kx-0*self.b_kr
            if bx_test==0:
                x_model_test=np.zeros(self.N)
                quality_model_test=get_quality(x,x_model_test*2**(kx_),metric)  
                
                if metric=="SNR" or metric=="SNR_L1":
                    if quality_model_test>=quality_model:
                        test=True
                    else : 
                        test=False
                else :
                    if quality_model_test<=quality_model:
                        test=True
                    else : 
                        test=False
                if test:    
            
                    family_best="none"
                    m='none'
                    
                    b_bx=0
                    bx=bx_test
                    
                    theta_hat=[]
                    theta_tilde=[]
                    
                    code_theta_tilde=[]
                    
                    x_model=x_model_test
                    quality_model=quality_model_test
                    
                    b_kr=0
                    b_kx=self.b_kx
                    
                    kr=0
   
                    kx=kx_
                    
                    to_do_second_stage=1
            #if to_do_second_stage :  
            #    br_test=btot-self.bm-self.bl-b_kx-b_bx-bx-b_kr
            #    print("bx_tot = {}, bx = {}, br= {}, quality = {}, model ={}".format( bx_tot,bx,br_test,quality_model,m))
            
            
            """
            second stage
            """
            #print("to_do_second_stage",to_do_second_stage,SNR_model,SNR_model_test)
            if to_do_second_stage==1: # si un modèle pour ce bx donné à donné de meilleurs résultats on calcul le second étage
                                 
                r=x_n-x_model # définition du résidu
                
                br_max=btot-self.bm-self.bl-b_kx-b_bx-bx-b_kr
                #print("br encoder",br_max)
                
                ########## normalisation de r
                _,kr=normalize(r)
                if -kr>=2**b_kr:
                    kr=-(2**b_kr-1)
                if kr>0:
                    kr=0
                r_n=r*2**(-kr)
            
                #print("rn max",np.max(np.abs(r_n)))
                
                quality_model=get_quality(x,x_model*2**(kx_),metric)
                if metric=="SNR" or metric=="SNR_L1": 
                    
                    if quality_model>=quality:
                        b_kr=0
                        kr=0
                        x_residual=np.zeros(self.N)
                        code_residual=[]
                        l="DCT+BPC"
                       
                    else:
                        quality_r_target=quality-quality_model
                        l,x_residual,code_residual=self.best_residual(r_n,metric,quality_r_target,br_max)
                        
                else :
                    if quality_model<=quality:
                        b_kr=0
                        kr=0
                        x_residual=np.zeros(self.N)
                        code_residual=[]
                        l="DCT+BPC"
                        #print("etrange")
                    else:
                    
                        quality_r_target=quality*2**(-kx_-kr)
                        #print(" quality_r_target", quality_r_target)
                        l,x_residual,code_residual=self.best_residual(r_n,metric,quality_r_target,br_max)
                        #print("corect")
                        
                        """
                        plt.figure(figsize=(8,4), dpi=100)
                        plt.plot(r_n,lw=1,label='r_n')
                        plt.plot(x_residual,lw=1,label='r rec, {} = {:.5f}, target = {} '.format(metric,get_quality(r_n,x_residual,metric),quality_r_target))
                        plt.xlabel('ind')
                        plt.ylabel('Amplitude')
                        plt.legend()
                        plt.grid( which='major', color='#666666', linestyle='-')
                        plt.minorticks_on()
                        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                        plt.show()  
                    
                        plt.figure(figsize=(8,4), dpi=100)
                        plt.plot(r*2**(kx_),lw=1,label='r')
                        plt.plot(x_residual*2**(kx_+kr),lw=1,label='r rec, {} = {:.5f}, target = {:.5f}, code = {} bits / {} '.format(metric,get_quality(r*2**(kx_),x_residual*2**(kx_+kr),metric),quality,len(code_residual),br_max))
                        plt.xlabel('ind')
                        plt.ylabel('Amplitude')
                        plt.legend()
                        plt.grid( which='major', color='#666666', linestyle='-')
                        plt.minorticks_on()
                        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                        plt.show()  
                        """
                btot_test=b_kx+self.bm+b_bx+bx+b_kr+self.bl+len(code_residual)
                quality_test=get_quality(x,x_model*2**(kx_)+x_residual*2**(kx_+kr),metric)
                
                
                test_quality_test=False
                test_quality_best=False
                if metric=="SNR" or metric=="SNR_L1":
                    if quality_test>=quality:
                        test_quality_test=True
                    if self.quality_best>=quality:
                        test_quality_best=True
                else :
                    if quality_test<=quality:
                        test_quality_test=True
                    if self.quality_best<=quality:
                        test_quality_best=True
                    
                   
                
        
                actualization=False
                ## cas 1 : test_quality_test = True et quality best == True -> on actualise si btot_test est plus petit b_tot_best
                if  test_quality_test ==True and  test_quality_best==True:
                    if btot_test<self.btot_best:
                        actualization=True
                        
                ## cas 2 : test_quality_test = False et quality best == False -> on actualise si quality test est meilleur que qualité best
                if  test_quality_test ==False and  test_quality_best==False:
                    if metric=="SNR" or metric=="SNR_L1":
                        if quality_test>self.quality_best:
                            actualization=True      
                    else :
                        if quality_test<self.quality_best:
                            actualization=True 
                        
                    
                ## cas 3 : test_quality_test = True et quality best == False -> on actualise 
                if  test_quality_test ==True and  test_quality_best==False:
                    actualization=True
               
                ## cas 4 : test_quality_test = False et quality best == True -> on actualise pas
               
                #print(actualization,quality_test,self.quality_best)
                #print(bx,br_max)
                
                if actualization :
              
                    

                    self.m_best=m
                    self.l_best=l
                    self.b_bx_best=b_bx
                    
                    self.bx_best=bx
                    self.br_best=len(code_residual)
                    
                    self.b_kx_best=b_kx
                    self.b_kr_best=b_kr
                    
                    self.btot_best=self.b_kx_best+self.bm+self.b_bx_best+self.bx_best+self.b_kr_best+self.bl+self.br_best
                
                    self.code_theta_tilde_best=code_theta_tilde
                    self.code_residual_best=code_residual
                    
                    self.kx_best=kx
                    self.kr_best=kr
                    #print("self.kr_best",self.kr_best)
                    self.theta_hat_best=theta_hat
                    self.theta_tilde_best=theta_tilde
                    
                    #print(self.m_best,"max enc",np.max(x_model),"kx",kx,"kx_",kx_,"kx_p",kx_p)
                    self.x_model_best=x_model*2**kx_
                    self.x_residual_best=x_residual*2**(kx_+kr)
                   
                    self.x_rec_best=self.x_model_best+self.x_residual_best
                    
                    self.quality_model_best=get_quality(x,self.x_model_best,metric)
                    self.quality_residual_best=get_quality(x,self.x_residual_best,metric)
                    self.quality_best=get_quality(x,self.x_rec_best,metric)
                    
                    
                    self.family_best=family_best
        
             
        if self.family_best!="pred para":# and self.family_best!="none":
            self.family_best_p=self.family_best
            self.m_best_p=self.m_best
            self.b_bx_pred_para=[int(np.ceil(np.log2(np.max([1,len(self.theta_tilde_best)])*self.nb_max_bit_theta_pred)))]*self.nb_model_pred_para
          
            self.bx_pred_para_max=[2**self.b_bx_pred_para[k]-1 for k in range(self.nb_model_pred_para)]
        
            
        
        
        
        
        
        
        #if self.family_best!="none":   
        self.theta_tilde_best_p=self.theta_tilde_best
        

        #print("self.bx_pred_para_max enc",self.bx_pred_para_max)
                                
                                 
        code_m=my_bin(self.label_model[self.m_best],self.bm)        
        #print("code_m",code_m)
              

        code_kx=my_bin(self.kx_best,self.b_kx_best)
        #print("code_kx",code_kx)
        
        
        #print("self.bx_best",self.bx_best,"self.b_bx_best",self.b_bx_best)
        code_bx=my_bin(self.bx_best,self.b_bx_best)
        #print("code_bx",code_bx)
        
        code_kr=my_bin(-self.kr_best,self.b_kr_best)
        #print("code_kr",code_kr)
        
        
        code_l=my_bin(self.label_residual[self.l_best],self.bl)
        #print("code_l",code_l)
        

        code=code_m+code_kx+code_bx+self.code_theta_tilde_best+code_kr+code_l+self.code_residual_best
        #print("len(code)",len(code),btot)
        
        
        #print("SNR_m",self.SNR_model_best)
        #print("SNR_r",self.SNR_residual_best)

        #self.family_best_p=self.trouver_racine(self.Model_used,self.m_best)
        #print(self.family_best_p)
        #print("max enc",np.max(self.x_model_best))
        #print("theta_tilde_enc",self.theta_tilde_best)
        
        """
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(bx_L,SNR_L,"-o",lw=1,label='SNR bx')
        plt.xlabel('bx+b_bx+b_kx+b_kr')
        plt.ylabel('SNR tot')
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()  
        """
        self.code=code
        
        return code



class Decode_one_window(Model_Decoder,Residual_Decoder):
    def __init__(self,fn=50,fs=6400, N=128,Model_used={},verbose=False):
        
        
   
        self.family_dec_p="none" #previous family used
        self.theta_tilde_dec_p=[] # previous parametric vector
        self.m_dec_p="none"
        
        
        
        self.Model_used=Model_used # dictionnaire des modèles utilisés ainsi que leurs caractéristiques respéctives
        
        ### on labélise les modèles
        self.label_model={}
        ind_m=0
        self.label_model[ind_m]="none"
        ind_m+=1
        
        
        
        #### models sin
        self.name_model_sin=[]
        self.m_theta_model_sin=[]
        self.w_theta_model_sin=[]
        for model in self.Model_used["sin"].items():
            self.name_model_sin.append(model[0])
            self.m_theta_model_sin.append(Model_used["sin"][model[0]][0])
            self.w_theta_model_sin.append(Model_used["sin"][model[0]][1])
            self.label_model[ind_m]=model[0]
            ind_m+=1
        
        #### models poly
        self.name_model_poly=[]
        self.order_model_poly=[]
        self.w_theta_model_poly=[]
        for model in self.Model_used["poly"].items():
            self.name_model_poly.append(model[0])
            self.order_model_poly.append(Model_used["poly"][model[0]][0])
            self.w_theta_model_poly.append(Model_used["poly"][model[0]][1])
            self.label_model[ind_m]=model[0]
            ind_m+=1
        
        #### models pred samples
        self.name_model_pred_samples=[]
        self.order_model_pred_samples=[]
        self.eta_model_pred_samples=[]
       
        self.w_theta_model_pred_samples=[]
        for model in self.Model_used["pred samples"].items():
            self.name_model_pred_samples.append(model[0])
            self.order_model_pred_samples.append(Model_used["pred samples"][model[0]][0])
            self.eta_model_pred_samples.append(Model_used["pred samples"][model[0]][1])
            
            self.w_theta_model_pred_samples.append(Model_used["pred samples"][model[0]][2])
            self.label_model[ind_m]=model[0]
            ind_m+=1
                                
        
        #### models pred para
        self.name_model_pred_para=[]
        self.factor_model_pred_para=[]
        
        for model in self.Model_used["pred para"].items():
            self.name_model_pred_para.append(model[0])
            self.factor_model_pred_para.append(Model_used["pred para"][model[0]][0])
            
            
            self.label_model[ind_m]=model[0]
            ind_m+=1                                
        
        
        
        
        self.nb_model_sin=len(self.name_model_sin) # nombre de modèles sin
        self.nb_model_poly=len(self.name_model_poly) # nombre de modèles poly
        self.nb_model_pred_samples=len(self.name_model_pred_samples) # nombre de modèles pred samples
        self.nb_model_pred_para=len(self.name_model_pred_para) # nombre de modèles pred parametric
        
        self.nb_model=1+self.nb_model_sin+self.nb_model_poly+self.nb_model_pred_samples+self.nb_model_pred_para # nombre de modèles
        
        
        Model_Decoder.__init__(self,fn,fs,N,False) 
        Residual_Decoder.__init__(self,N)   
        
        ### on labélise les érsidus
        self.label_residual={}
        self.label_residual[0]="DCT+BPC"
        self.label_residual[1]="DWT+BPC"
        self.nb_residual=len(self.label_residual)
        
        
        
        #self.list_btot=[32,64,96,128,160,192,224,256]
    

        ##################### budget de bits servant à décoder le signal
        #self.b_btot=0#int(np.ceil(np.log2(len(self.list_btot)))) # nombre de bits pour coder btot
        
        self.bm=int(np.ceil(np.log2(self.nb_model))) # nombre de bits pour coder le modèle tous les polynomes d'ordre 0 à 8 + sin +none
        
        self.bl=int(np.ceil(np.log2(self.nb_residual))) # nombre de bits pour coder la méthode de résidu actuelement: DCT, DWT
        

        self.b_kx=5 # nombre de bits pour coder kx, 0 si modèle pred_samples
        
        self.b_kr=4 # nombre de bits pour coder kr, 0 si le modèle sélectionné est none
        
        
        # nombr de bits max pour coder chaque modèle
        self.nb_max_bit_theta=8 # nombre de bits max par coefficient
        self.nb_max_bit_theta_pred=4
        


        
        self.b_bx_sin=[int(np.ceil(np.log2(3*self.nb_max_bit_theta)))]*self.nb_model_sin
        self.b_bx_poly=[int(np.ceil(np.log2((self.order_model_poly[k]+1)*self.nb_max_bit_theta))) for k in range(self.nb_model_poly)]
        self.b_bx_pred_samples=[int(np.ceil(np.log2(self.order_model_pred_samples[k]*self.nb_max_bit_theta))) for k in range(self.nb_model_pred_samples)]
        self.b_bx_pred_para=[0]*self.nb_model_pred_para
        
    
        
        self.b_bx=[0]+ self.b_bx_sin+self.b_bx_poly+self.b_bx_pred_samples+self.b_bx_pred_para
        
        
        
        self.bx_sin_max=[2**self.b_bx_sin[k]-1 for k in range(self.nb_model_sin)]
        self.bx_poly_max=[2**self.b_bx_poly[k]-1 for k in range(self.nb_model_poly)]
        self.bx_pred_samples_max=[2**self.b_bx_pred_samples[k]-1 for k in range(self.nb_model_pred_samples)]
        self.bx_pred_para_max=[0]*self.nb_model_pred_para        
        
        
        
        

        
        #################### grandeurs optimals meilleurs modèle + meilleur méthode de compression de résidu

    def ini_MMC_dec(self):
        self.m_dec='none'
        self.l_dec="DCT+BPC"
        
        self.theta_tilde_dec=[]
        

        
        self.x_model_dec=np.zeros(self.N)
        self.x_residual_dec=np.zeros(self.N)
        self.x_rec_dec=np.zeros(self.N)
        
        self.b_bx_dec=0
        self.bx_dec=0
        self.br_dec=0
        
        self.b_kx_dec=self.b_kx # nombre de bits pour coder kx
        self.b_kr_dec=0 # nombre de bits pour coder kr
        self.kx_dec=0
        self.kr_dec=0            
    
        self.family_dec="none"
    
    def trouver_racine(self,dictionnaire,name_model):
        for cle, valeur in dictionnaire.items():
            if isinstance(valeur, dict):
                if name_model in valeur:
                    return cle
                else:
                    racine = self.trouver_racine(valeur,name_model)
                    if racine:
                        return cle
        return None



    def MMC_dec(self,code,x_p):
        """
        print("b_bx_sin",  self.b_bx_sin,"bx_max=",self.bx_sin_max)
        print("b_bx_poly", self.b_bx_poly,"bx_max=",self.bx_poly_max)
        print("b_bx_pred_samples", self.b_bx_pred_samples,"bx_max=",self.bx_pred_samples_max)
        print("b_bx_pred_para", self.b_bx_pred_para,"bx_max=",self.bx_pred_para_max)
        """
        self.ini_MMC_dec()

        #decodage 
        
        ptr=0

        #btot=self.list_btot[int(my_inv_bin(code[0:ptr+b_btot]))]
        #ptr+=b_btot
        #print("btot", btot)
        
        label_model=int(my_inv_bin(code[ptr:ptr+self.bm]))
        #print("family model", label_model)
        
        self.m_dec=self.label_model[label_model]
        ptr+=self.bm
        #print("m_dec", self.m_dec)
        
        

        

        
        
        self.family_dec="none"
        if self.m_dec!="none":
            self.family_dec=self.trouver_racine(self.Model_used,self.m_dec)
        #print("family model",self.family_dec)    
        
        #if self.family_dec=="pred samples" or self.family_dec=="pred para" :
        #    self.b_kx_dec=0
        #    self.kx_dec=normalize(x_p[2*self.N:3*self.N])[1]
            
            
            """
            t2=np.linspace(0,(3*self.N-1)*(1/self.fs),3*self.N)
            plt.figure(figsize=(8,4), dpi=100)
            plt.plot(t2[0:2*self.N],x_p[self.N:3*self.N]*2**(-self.kx_dec),lw=2,label='kx_p={}'.format(self.kx_dec))
            plt.xlabel('t [s]')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show()  
            """

        #else :
        self.kx_dec=int(my_inv_bin(code[ptr:ptr+self.b_kx_dec]))
            
        ptr+=self.b_kx_dec
        #print("kx", self.kx_dec)
         
        
      
        self.b_bx_dec=self.b_bx[label_model]
        #print("b_bx", self.b_bx_dec)

        self.bx_dec=int(my_inv_bin(code[ptr:ptr+self.b_bx_dec]))
        ptr+=self.b_bx_dec
        #print("bx", self.bx_dec)
        
        

        if self.family_dec=="sin":
            m_theta_sin=self.Model_used[self.family_dec][self.m_dec][0]
            w_theta_sin=self.Model_used[self.family_dec][self.m_dec][1]
            self.theta_tilde_dec=self.get_theta_sin_tilde(code[ptr:ptr+self.bx_dec],self.bx_dec,m_theta_sin,w_theta_sin)
            self.x_model_dec=self.get_model_sin(self.t,*self.theta_tilde_dec)*2**self.kx_dec 
            


        elif self.family_dec=="poly":
            order_poly=self.Model_used[self.family_dec][self.m_dec][0]
            w_theta_poly=self.Model_used[self.family_dec][self.m_dec][1]
            self.theta_tilde_dec=self.get_theta_poly_tilde(code[ptr:ptr+self.bx_dec],self.bx_dec,[0]*(order_poly+1),w_theta_poly)
            self.x_model_dec=self.get_model_poly(self.t,*self.theta_tilde_dec)*2**self.kx_dec   
            

        elif self.family_dec=="pred samples":
            #print("self.kx_dec",self.kx_dec)
            order_pred_samples=self.Model_used[self.family_dec][self.m_dec][0]
            eta_pred_samples=self.Model_used[self.family_dec][self.m_dec][1]
            X=self.get_X(x_p[self.N:3*self.N]*2**(-self.kx_dec),order_pred_samples,eta_pred_samples)
            

            
            if self.family_dec_p!="pred samples":
                m_theta_pred_samples=self.get_m_theta_pred_samples(order_pred_samples,eta_pred_samples,0)
            
            else :   
                X_pred_samples2=self.get_X(x_p[0:2*self.N]*2**(-self.kx_dec),order_pred_samples, eta_pred_samples) 
                m_theta_pred_samples=self.get_theta_pred_samples(X_pred_samples2,x_p[2*self.N:3*self.N]*2**(-self.kx_dec))
     
            w_theta_pred_samples=self.Model_used[self.family_dec][self.m_dec][2]
            self.theta_tilde_dec=self.get_theta_pred_samples_tilde(code[ptr:ptr+self.bx_dec],self.bx_dec,m_theta_pred_samples,w_theta_pred_samples)
            self.x_model_dec=self.get_model_pred_samples(X,*self.theta_tilde_dec)*2**self.kx_dec

            
        elif self.family_dec=="pred para":   
            if self.family_dec_p=="sin":
                m_theta_sin=self.theta_tilde_dec_p
                
                w_theta_sin_p=self.Model_used[self.family_dec_p][self.m_dec_p][1]
                w_theta_sin= [w_theta_sin_p[i]/self.Model_used[self.family_dec][self.m_dec][0] for i in range(3)]
                self.theta_tilde_dec= self.get_theta_sin_tilde(code[ptr:ptr+self.bx_dec],self.bx_dec,m_theta_sin,w_theta_sin)
                self.x_model_dec=self.get_model_sin(self.t,*self.theta_tilde_dec)*2**self.kx_dec 
                
            elif self.family_dec_p=="poly":
                
                order=self.Model_used[self.family_dec_p][self.m_dec_p][0]
                m_theta_poly=self.theta_tilde_dec_p
                w_theta_poly_p=self.Model_used[self.family_dec_p][self.m_dec_p][1]
                w_theta_poly=[w_theta_poly_p[i]/self.Model_used[self.family_dec][self.m_dec][0] for i in range(order+1)]
                self.theta_tilde_dec=self.get_theta_poly_tilde(code[ptr:ptr+self.bx_dec],self.bx_dec,m_theta_poly,w_theta_poly)
                self.x_model_dec=self.get_model_poly(self.t,*self.theta_tilde_dec)*2**self.kx_dec  
            
            elif self.family_dec_p=="pred samples":
                #print("self.kx_dec",self.kx_dec)
                order_pred_samples=self.Model_used[self.family_dec_p][self.m_dec_p][0]
                eta_pred_samples=self.Model_used[self.family_dec_p][self.m_dec_p][1]
                X=self.get_X(x_p[self.N:3*self.N]*2**(-self.kx_dec),order_pred_samples,eta_pred_samples)



                #X_pred_samples2_p=self.get_X(x_p[0:2*self.N]*2**(-self.kx_dec),order_pred_samples,eta_pred_samples)
                #m_theta_pred_samples=self.get_theta_pred_samples(X_pred_samples2_p,x_p[2*self.N:3*self.N]*2**(-self.kx_dec))




                m_theta_pred_samples=self.theta_tilde_dec_p
                
                
                
                
                
                w_theta_pred_samples_p=self.Model_used[self.family_dec_p][self.m_dec_p][2]
                w_theta_pred_samples=[w_theta_pred_samples_p[i]/self.Model_used[self.family_dec][self.m_dec][0] for i in range(order_pred_samples)]

                self.theta_tilde_dec= self.get_theta_pred_samples_tilde(code[ptr:ptr+self.bx_dec],self.bx_dec,m_theta_pred_samples,w_theta_pred_samples)
                self.x_model_dec=self.get_model_pred_samples(X,*self.theta_tilde_dec)*2**self.kx_dec
                #print("np.max(self.x_model_dec)",np.max(self.x_model_dec))
                        
        ptr+=self.bx_dec 
        
        
        if self.family_dec!="pred para":# and self.family_dec!="none":  
            self.family_dec_p=self.family_dec#previous family used
           
            self.m_dec_p=self.m_dec
            

            
            self.b_bx_pred_para=[int(np.ceil(np.log2(np.max([1,len(self.theta_tilde_dec)])*self.nb_max_bit_theta_pred)))]*self.nb_model_pred_para
            
            
            self.bx_pred_para_max=[2**self.b_bx_pred_para[k]-1 for k in range(self.nb_model_pred_para)]
            
            self.b_bx=[0]+ self.b_bx_sin+self.b_bx_poly+self.b_bx_pred_samples+self.b_bx_pred_para    
        
        #if self.family_dec!="none": 
        self.theta_tilde_dec_p=self.theta_tilde_dec # previous parametric vector
        #print("self.bx_pred_para_max dec",self.bx_pred_para_max)
        
        if self.m_dec!="none":
            self.b_kr_dec=self.b_kr
        #print("b_kr_dec", self.b_kr_dec)
        
        
        
        self.kr_dec=-int(my_inv_bin(code[ptr:ptr+self.b_kr_dec]))
        ptr+=self.b_kr_dec
        #print("kr", self.kr_dec)

        label_residual=int(my_inv_bin(code[ptr:ptr+self.bl]))
        #print("label_residual",label_residual)

        self.l_dec=self.label_residual[label_residual]
        ptr+=self.bl
        
        
        
        

        #SNR_r=quality-self.SNR_model_dec
        
        #self.br_dec=btot-self.bm-self.bl-self.b_kx_dec-self.b_kr_dec-self.b_bx_dec-self.bx_dec
        

        
        self.x_residual_dec=np.array(self.best_residual_dec(self.l_dec,code[ptr:]))*2**(self.kx_dec+self.kr_dec)
        #print("br",len(code[ptr:]))
        
        self.x_rec_dec=self.x_model_dec+self.x_residual_dec
        
        

                
        return  self.x_rec_dec
            
        