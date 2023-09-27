# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:13:54 2023

@author: presvotscor
"""
import numpy as np
from Normalize import normalize


from codage_model import Model_Encoder,Model_Decoder
from codage_residu import Residual_Encoder,Residual_Decoder
from Allocation_two_stages import Allocation_sin_bx_br,Allocation_poly_bx_br

#from Models import Model_sin,Model_poly
from Measures import get_snr,my_bin,my_inv_bin


class Encode_one_window(Model_Encoder,Residual_Encoder,Allocation_sin_bx_br,Allocation_poly_bx_br):
    def __init__(self,fn=50,fs=6400, N=128,verbose=False):

        # initialisation de la classe Model_Encoder
        w_sin=[0.5,0.2,2*np.pi]
        m_sin=[0.75,fn,0]
        w_poly=[2]*(16+1)      
        
        
        self.list_poly_used=[0,1,2,3,4,5,6,7,8] # ordre des modèles polynomiaux testés
        
        
        
        
        Model_Encoder.__init__(self,fn,fs,N,w_sin,m_sin,w_poly,False) 
        
   
        Residual_Encoder.__init__(self,N)   
        
        Allocation_sin_bx_br.__init__(self,N,fs,w_sin,m_sin,False) 
        Allocation_poly_bx_br.__init__(self,N,fs,w_poly,False) 
    
        
        self.nb_test=5 # nombre de test réalisé autour de bx et br théorique déterminé complexité: pour un modèle on test -self.nb_test + bx_opt à self.nb_test + bx_opt
        self.nb_max_bit_theta=10 # nombre de bits maximale par coefficient
        
        #self.list_btot=[32,64,96,128,160,192,224,256]
    
    
        
    
        
        ##################### budget de bits servant à décoder le signal
        #self.b_btot=0#int(np.ceil(np.log2(len(self.list_btot)))) # nombre de bits pour coder btot
        self.b_kx=5 # nombre de bits pour coder kx
        self.bm=int(np.ceil(np.log2(len(self.list_poly_used)+2))) # nombre de bits pour coder le modèle tous les polynomes d'ordre 0 à 8 + sin +none
        
        self.bl=3 # nombre de bits pour coder la méthode de résidu actuelement: DCT,DWT
        
        self.bh=self.b_kx+self.bm+self.bl# +b_btot estimation de bx et br
        
        
        self.b_kr=3 # nombre de bits pour coder kr, 0 si le modèle sélectionné est none
        self.b_bx_sin=int(np.ceil(np.log2(3*self.nb_max_bit_theta)))
        self.b_bx_poly=[int(np.ceil(np.log2((self.list_poly_used[k]+1)*self.nb_max_bit_theta))) for k in range(len(self.list_poly_used))]
        #print("b_bx_sin",  self.b_bx_sin)
        #print("b_bx_poly", self.b_bx_poly)
    
        
    def ini_MMC_enc(self):
        #################### grandeurs optimals meilleurs modèle + meilleur méthode de compression de résidu
        self.SNR_model_best=0
        self.SNR_residual_best=0
        self.SNR_best=0
        
        self.m_best='none'
        self.l_best="Antonini_DCT"
        
        self.theta_hat_best=[]
        self.theta_tilde_best=[]
        
        self.code_theta_tilde_best=[]
        self.code_residual_best=[]
        
        self.x_model_best=np.zeros(self.N)
        self.x_residual_best=np.zeros(self.N)
        self.x_rec_best=np.zeros(self.N)
        
        self.b_bx_best=0
        
        self.bx_best=0
        self.br_best=0
        
        
        self.b_kr_best=0 # nombre de bits pour coder kr, 0 si le modèle sélectionné est none
        
        self.bl_best=self.bl # nombre de bits pour coder la méthode de résidu actuelement: DCT,DWT
        

        self.kx_best=0
        self.kr_best=0
        
    def MMC_enc(self,x,btot):
        
        self.ini_MMC_enc()
        
        ##################### normalisation de x
        _,kx=normalize(x)

        if kx>=2**self.b_kx:
            kx=2**self.b_kx-1
        
        if kx<0:
            kx==0
        x_n=x*2**(-kx)
        
        
   
        
        ##################### détermination de tous les theta_hat modèles pour les modèles conssidérés:  sin, poly 0,...,6
        theta_sin_hat=self.get_theta_sin(x_n)

        
        theta_poly_hat=[self.get_theta_poly(x_n,self.list_poly_used[k]) for k in range(len(self.list_poly_used))]
        
        ##################### reconstruction des modèles avec les paramètres estimés
        x_sin_hat=self.get_model_sin(self.t,*theta_sin_hat) 
        x_poly_hat=[self.get_model_poly(self.t,*theta_poly_hat[k]) for k in range(len(self.list_poly_used))]
        
        """
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(t,x_n,lw=1,label='x')
        plt.plot(t, x_sin_hat,lw=1,label='model,theta=[{:.2f},{:.2f},{:.2f}]'.format(*theta_sin_hat))
        plt.xlabel('t [s]')
        plt.ylabel('Amplitude')
        plt.title("window: {}".format(w))
        plt.legend()
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()    
        """
        #################### détermination de l'erreur due à l'estimation des paramétre 
        SE_sin_hat=np.sum((x_n-x_sin_hat)**2)
        SE_poly_hat=[np.sum((x_n-x_poly_hat[k])**2) for k in range(len(self.list_poly_used))]
        


        ###################### estimation de bx pour chaque modèle utilisé
        bx_sin_hat=self.get_sin_bx_br(SE_sin_hat,btot-self.bh-self.b_bx_sin)[0]
        bx_poly_hat=[self.get_poly_bx_br(SE_poly_hat[k],btot-self.bh-self.b_bx_poly[k],self.list_poly_used[k])[0] for k in range(len(self.list_poly_used))]

        #print("bx_sin_opt=", bx_sin_hat)
        #print("bx_poly_opt=", bx_poly_hat)
        #################### début test modèle
    
    
    
    
        m='none'
        b_bx=0#self.b_bx_poly[self.list_poly_used[0]]
        bx=0
        
        theta_hat=[]
        theta_tilde=[]
        
        code_theta_tilde=[]
        
        x_model=np.zeros(self.N)
        
        SNR_model=get_snr(x_n,x_model)       
        
        b_kr=0
        to_do_second_stage=1
        

        
        for bx_test in range(btot): #(25,26): Pour fixer le nombre de bits servant à coder le modèle
            
                
            #print("SNR_none={:.2f} dB".format(SNR_model))
            #print("sin", bx_sin_hat-self.nb_test,bx_sin_hat+self.nb_test)
            ### TEST SIN 
            if  bx_sin_hat-self.nb_test<=bx_test and bx_test<= bx_sin_hat+self.nb_test and bx_test<=btot-self.bh-self.b_bx_sin and bx_test<=2**self.b_bx_sin-1:

                theta_sin_tilde_test,code_theta_sin_tilde_test=self.get_theta_sin_tilde(theta_sin_hat,bx_test)
                #print("theta_sin_tilde_test",theta_sin_tilde_test)
                x_sin_tilde_test=self.get_model_sin(self.t,*theta_sin_tilde_test) 
                SNR_model_test=get_snr(x_n,x_sin_tilde_test)
                
                if SNR_model_test>SNR_model:
                    m='sin'
                    b_bx=self.b_bx_sin
                    bx=bx_test
                    
                    theta_hat=theta_sin_hat
                    theta_tilde=theta_sin_tilde_test
                    
                    code_theta_tilde=code_theta_sin_tilde_test
                    
                    x_model=x_sin_tilde_test
                    
                    SNR_model=SNR_model_test
                    #print("SNR_sin={:.2f} dB".format(SNR_model))
                    b_kr=self.b_kr
                    to_do_second_stage=1
                    
              
            ### TESTs POLY
            for k in range(len(self.list_poly_used)):
                #print("bx_poly_hat[k]+-self.nb_test",bx_poly_hat[k]-self.nb_test,bx_test,bx_poly_hat[k]+self.nb_test)
                #print("bx_poly_hat[k]+self.nb_test",bx_poly_hat[k]+self.nb_test)
                if  bx_poly_hat[k]-self.nb_test<=bx_test and bx_test<=bx_poly_hat[k]+self.nb_test and bx_test<=btot-self.bh-self.b_bx_poly[k] and bx_test<=2**self.b_bx_poly[k]-1:
    
                    theta_poly_tilde_test,code_theta_poly_tilde_test=self.get_theta_poly_tilde(theta_poly_hat[k],bx_test)
                    x_poly_tilde_test=self.get_model_poly(self.t,*theta_poly_tilde_test) 
                    SNR_model_test=get_snr(x_n,x_poly_tilde_test)
                    #print("SNR_model_test",SNR_model_test)
                    if SNR_model_test>SNR_model:
                        m='poly-{}'.format(self.list_poly_used[k])
                        b_bx=self.b_bx_poly[self.list_poly_used[k]]
                        bx=bx_test
                        #print("bx",bx,m)
                        theta_hat=theta_poly_hat[k]
                        theta_tilde=theta_poly_tilde_test
                        
                        code_theta_tilde=code_theta_poly_tilde_test
                        
                        x_model=x_poly_tilde_test
                        
                        SNR_model=SNR_model_test
                        #print("SNR_poly_{}={:.2f} dB".format(k,SNR_model))
                        b_kr=self.b_kr
                        to_do_second_stage=1
                    
          
            
            """
            second stage
            """
            
            if to_do_second_stage==1: # si un modèle pour ce bx donné à donné de meilleurs résultats on calcul le second étage
                r=x_n-x_model # définition du résidu
                
                
                br=btot-self.bh-bx-b_bx-b_kr
 
                
                ########## normalisation de r
                _,kr=normalize(r)
                if -kr>=2**b_kr:
                    kr=-(2**b_kr-1)
                if kr>0:
                    kr=0
                r_n=r*2**(-kr)
            
            
                SNR_residual,l,x_residual,code_residual=self.best_residual(r_n,br)
                to_do_second_stage=0
                
                #print("SNR_m={:.2f} dB, SNR_r={:.2f} dB, SNR_tot={:.2f} dB".format(SNR_model,SNR_residual,SNR_residual+SNR_model))
                if SNR_residual+SNR_model>self.SNR_best:
                    
                    
                    self.SNR_model_best=SNR_model
                    self.SNR_residual_best=SNR_residual
                    self.SNR_best=SNR_residual+SNR_model
                    self.m_best=m
                    self.l_best=l
                    self.b_bx_best=b_bx
                    
                    self.bx_best=bx
                    self.br_best=len(code_residual)
                    
                    
                    self.b_kr_best=b_kr
                    
                    
                    self.code_theta_tilde_best=code_theta_tilde
                    self.code_residual_best=code_residual
                    
                    self.kx_best=kx
                    self.kr_best=kr
                    
                    self.theta_hat_best=theta_hat
                    self.theta_tilde_best=theta_tilde
                    
                    
                    self.x_model_best=x_model*2**kx
                    self.x_residual_best=x_residual*2**(kx+kr)
                    self.x_rec_best=self.x_model_best+self.x_residual_best
                    
 
                

        code_kx=my_bin(self.kx_best,self.b_kx)
        #print("code_kx",code_kx)
        
        if self.m_best=="none":
            code_m=my_bin(0,self.bm)        
        elif self.m_best=="sin":
            code_m=my_bin(1,self.bm)
        else :
            code_m=my_bin(int(self.m_best[5:])+2,self.bm)
        #print("code_m",code_m)
        
        code_bx=my_bin(self.bx_best,self.b_bx_best)
        #print("code_bx",code_bx)
        
        if self.m_best!='none': ## on ne code kr que si on n'a pas eu none comme modèle
            code_kr=my_bin(-self.kr_best,self.b_kr_best)
        else :  code_kr=[]
        #print("code_kr",code_kr)
        
        if self.l_best=="Antonini_DCT":
            code_l=my_bin(0,self.bl)
        elif self.l_best=="Khan_DWT":
            code_l=my_bin(1,self.bl)
        #print("code_l",code_l)
        

        code=code_kx+code_m+code_bx+self.code_theta_tilde_best+code_kr+code_l+self.code_residual_best
        #print("len(code)",len(code),btot)
        
        return code



class Decode_one_window(Model_Decoder,Residual_Decoder):
    def __init__(self,fn=50,fs=6400, N=128,verbose=False):


        # initialisation de la classe Model_Decoder
        w_sin=[0.5,0.2,2*np.pi]
        m_sin=[0.75,fn,0]
        w_poly=[2]*(16+1)      
        
        self.list_poly_used=[0,1,2,3,4,5,6,7,8]# ordre des modèles polynomiaux testés
       
        
        Model_Decoder.__init__(self,fn,fs,N,w_sin,m_sin,w_poly,False) 
        
        
        Residual_Decoder.__init__(self,N)   
 
    
       
        self.nb_max_bit_theta=10 # nombre de bits maximale par coefficient

        #self.list_btot=[32,64,96,128,160,192,224,256]
    
    
    
        ##################### budget de bits servant à décoder le signal
        #self.b_btot=0#int(np.ceil(np.log2(len(self.list_btot)))) # nombre de bits pour coder btot
        self.b_kx=5 # nombre de bits pour coder kx
        self.bm=int(np.ceil(np.log2(len(self.list_poly_used)+2))) # nombre de bits pour coder le modèle
        
        self.bl=3 # nombre de bits pour coder la méthode de résidu
        
        self.bh=self.b_kx+self.bm+self.bl# +b_btot estimation de bx et br
        
        self.b_kr=3 # nombre de bits pour coder 
        self.b_bx_sin=int(np.ceil(np.log2(3*self.nb_max_bit_theta)))
        self.b_bx_poly=[int(np.ceil(np.log2((self.list_poly_used[k]+1)*self.nb_max_bit_theta))) for k in range(len(self.list_poly_used))]

    
        #################### grandeurs optimals meilleurs modèle + meilleur méthode de compression de résidu
    def ini_MMC_dec(self):
        self.m_dec='none'
        self.l_dec="Antonini_DCT"
        
        self.theta_tilde_dec=[]
        
        self.code_theta_tilde_dec=[]
        self.code_residual_dec=[]
        
        self.x_model_dec=np.zeros(self.N)
        self.x_residual_dec=np.zeros(self.N)
        self.x_rec_dec=np.zeros(self.N)
        
        self.b_bx_dec=0
        self.bx_dec=0
        self.br_dec=0
        
        self.b_kr_dec=0 # nombre de bits pour coder 
        self.kx_dec=0
        self.kr_dec=0            
    
   
    def MMC_dec(self,code,btot):
        
        self.ini_MMC_dec()

        #decodage 
        
        ptr=0

        #btot=self.list_btot[int(my_inv_bin(code[0:ptr+b_btot]))]
        #ptr+=b_btot
        #print("btot", btot)
        
        self.kx_dec=int(my_inv_bin(code[ptr:ptr+self.b_kx]))
        ptr+=self.b_kx
        #print("kx", self.kx_dec)
        
        
        self.m_dec=my_inv_bin(code[ptr:ptr+self.bm])
        ptr+=self.bm
        
        
        if self.m_dec==0:
            self.b_bx_dec=0
            self.m_dec='none'
            
            
        
        elif self.m_dec==1:
            self.b_bx_dec=int(np.ceil(np.log2(3*self.nb_max_bit_theta)))
            self.m_dec='sin'
            self.b_kr_dec=self.b_kr
        else:
            order=int(self.m_dec-2)
            self.b_bx_dec=int(np.ceil(np.log2((order+1)*self.nb_max_bit_theta)))
            self.m_dec='poly-{}'.format(order)
            self.b_kr_dec=self.b_kr
        #print("m", self.m_dec)
        #print("b_bx",self.b_bx_dec)        
        
        self.bx_dec=int(my_inv_bin(code[ptr:ptr+self.b_bx_dec]))
        ptr+=self.b_bx_dec
        #print("bx", self.bx_dec)
        
        #print("code_m=",code[ptr:ptr+bx])
        self.x_model_dec=np.array(self.best_model_dec(self.m_dec,code[ptr:ptr+self.bx_dec],self.bx_dec))*2**self.kx_dec
        ptr+=self.bx_dec  
        
        
        if self.m_dec!='none':
            self.kr_dec=-int(my_inv_bin(code[ptr:ptr+self.b_kr_dec]))
            ptr+=self.b_kr_dec
            #print("kr", self.kr_dec)
        
        self.l_dec=int(my_inv_bin(code[ptr:ptr+self.bl]))
        ptr+=self.bl
        if self.l_dec==0:
            self.l_dec="Antonini_DCT"
        elif self.l_dec==1:
            self.l_dec="Khan_DWT"
        #print("l", self.l_dec)
        
        

                
        self.br_dec=btot-self.bh-self.b_bx_dec-self.bx_dec-self.b_kr_dec
        #print("br",self.br_dec)
        
        
        self.x_residual_dec=np.array(self.best_residual_dec(self.l_dec,code[ptr:],self.br_dec))*2**(self.kx_dec+self.kr_dec)
        
        
        self.x_rec_dec=self.x_model_dec+self.x_residual_dec
        
        
                
        return  self.x_rec_dec
            
        
        
        
        
        

            

# Programme principal
if __name__ == "__main__":
    import matplotlib.pyplot as plt
  
    #from get_RTE_test_signal import x_test_RTE
    N=128 # taille d'une fenêtre
    fn=50 # fréquence nominale du réseau
    fs=6400 # fréquence d'échantillonnage
     
    btot=128

    verbose = False


    t=np.linspace(0,(N-1)/fs,N)
    
    
    nb_w=100

    
    
    #initialisation MMC
    MMC_coder=Encode_one_window(fn,fs,N,verbose)
    MMC_decoder=Decode_one_window(fn,fs,N,verbose)
 
    
   
    
    # Ouvrir le fichier en mode lecture
    with open('data_test.txt', 'r') as file:
        # Lire le contenu du fichier
        content = file.read()

    # Extraire la liste à partir du contenu du fichier
    # En supposant que la liste est toujours dans la forme indiquée
    start = content.find('[')
    end = content.find(']')
    if start != -1 and end != -1:
        # Extraire la partie entre crochets et la diviser en éléments individuels
        list_str = content[start + 1:end]
        x_test_RTE = [float(item.strip()) for item in list_str.split(',')]



    
    x_rec=np.zeros(N*nb_w)
    x_model=np.zeros(N*nb_w)
    r=np.zeros(N*nb_w)
    r_rec=np.zeros(N*nb_w)
    
    SNR=[0]*nb_w
    SNR_r=[0]*nb_w
    SNR_m=[0]*nb_w
    M=[0]*nb_w
    L=[0]*nb_w
    #### start encodage    
    for w in range(nb_w): 
      
        x=np.array(x_test_RTE[w*N:(w+1)*N])
        
        
        ##### coder
        code=MMC_coder.MMC_enc(x,btot)
        
        
        SNR_=MMC_coder.SNR_best 
        b_used=len(code)   
        m=MMC_coder.m_best 
        l=MMC_coder.l_best 
        b_kx=MMC_coder.b_kx
        b_kr=MMC_coder.b_kr_best
        bm=MMC_coder.bm
        bl=MMC_coder.bl
        b_bx=MMC_coder.b_bx_best
        bh=b_kx+b_kr+bm+bl+b_bx 
        bx=MMC_coder.bx_best
        br=MMC_coder.br_best 
        kx=MMC_coder.kx_best  
        kr=MMC_coder.kr_best 

 
        print(f"window={w+1:3}, SNR={SNR_:5.1f} dB, b_used/btot={b_used:3}/{btot:3}, m={m:6}, l={l:12}, bh=b_kx+b_kr+bm+bl+b_bx={b_kx:1}+{b_kr:1}+{bm:1}+{bl:1}+{b_bx:1}={bh:2} b, bx={bx:2} b, br={br:3} b, kx={kx:1}, kr={kr:2}")
            
        
                        
    
        ##### decoder
        x_rec[w*N:(w+1)*N]=MMC_decoder.MMC_dec(code,btot)
        



        x_model[w*N:(w+1)*N]=MMC_decoder.x_model_dec
        
        r[w*N:(w+1)*N]=x-MMC_decoder.x_model_dec
        r_rec[w*N:(w+1)*N]=MMC_decoder.x_residual_dec
        
        SNR[w]=MMC_coder.SNR_best
        SNR_m[w]=MMC_coder.SNR_model_best
        SNR_r[w]=MMC_coder.SNR_residual_best
        
        if MMC_decoder.m_dec=='sin':
            M[w]=1
        elif MMC_decoder.m_dec=='none':
            M[w]=0
        else :
            M[w]=int(MMC_decoder.m_dec[5:])+1
        
        if MMC_decoder.l_dec=='Khan_DWT':
            L[w]=1

        



        if None:
            plt.figure(figsize=(8,4), dpi=100)
            plt.plot(t,x,lw=2,label='x')
            plt.plot(t,MMC_decoder.x_model_dec,lw=2,label='model rec, m: {}, SNR={:.1f} dB'.format(MMC_decoder.m_dec,get_snr(x,MMC_decoder.x_model_dec)))
            plt.plot(t,x_rec[w*N:(w+1)*N],lw=2,label='x rec, l: {}, SNR={:.1f} dB'.format(MMC_decoder.l_dec,get_snr(x,x_rec[w*N:(w+1)*N])))
            plt.xlabel('t [s]')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.title("Window index {}, btot={} bits, btot max ={} bits".format(w+1,len(code),btot))
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show()
    
            plt.figure(figsize=(8,4), dpi=100)
            plt.plot(t,x-MMC_decoder.x_model_dec,lw=2,label='residual')
            plt.plot(t,MMC_decoder.x_residual_dec,lw=2,label='x rec, l: {}, SNR={:.1f} dB'.format(MMC_decoder.l_dec,get_snr(x-MMC_decoder.x_model_dec,MMC_decoder.x_residual_dec)))
            plt.xlabel('t [s]')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.title("Window index {}, btot={} bits, btot max ={} bits".format(w+1,len(code),btot))
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show()        
    
            plt.figure(figsize=(8,4), dpi=100)
            plt.plot(t,x-x_rec[w*N:(w+1)*N],lw=2,label='error')
            plt.xlabel('t [s]')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.title("Window index {}, btot={} bits, btot max ={} bits".format(w+1,len(code),btot))
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show()  
        

      
        
#### First and second stage   

plt.figure(figsize=(8,4), dpi=100)
plt.plot(x_test_RTE[0:nb_w*N],lw=2,label='x')
plt.plot(x_rec,lw=2,label='x_rec')
plt.xlabel('ind sample')
plt.ylabel('Voltage (kV)')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show() 

plt.figure(figsize=(8,4), dpi=100)
plt.plot(x_test_RTE[0:nb_w*N]-x_rec,lw=2,label='x-x_rec')
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
plt.plot(x_test_RTE[0:nb_w*N],lw=2,label='x')
plt.plot(x_model,lw=2,label='x_model')
plt.xlabel('ind sample')
plt.ylabel('Voltage (kV)')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show() 

plt.figure(figsize=(8,4), dpi=100)
plt.plot(x_test_RTE[0:nb_w*N]-x_model,lw=2,label='x-x_model')
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


### Model used

yticks_labels = ['None','sin','poly-0','poly-1','poly-2','poly-3','poly-4','poly-5','poly-6','poly-7','poly-8']
yticks_positions = np.arange(len(yticks_labels))
plt.figure(figsize=(8,4), dpi=100)
plt.plot(M,'o',lw=2,label='model index')
plt.xlabel('ind window')
plt.ylabel('Model index')
plt.legend()
plt.grid( which='major', color='#666666', linestyle='-')
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.yticks(yticks_positions, yticks_labels)
plt.show() 

### Transform used
yticks_labels = ['DCT','DWT']
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





