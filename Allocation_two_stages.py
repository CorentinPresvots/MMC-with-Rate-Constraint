# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:40:12 2023

@author: coren
"""



########################################################

#allocation de bits entre modèle et résidu en supposant que le résidu est gaussien


import numpy as np
import matplotlib.pyplot as plt


from Bits_allocation import Allocation_sin,Allocation_poly
from codage_model import Model_Encoder
from Models import Model_sin,Model_poly
from Measures import get_snr


class Allocation_sin_bx_br(Allocation_sin):
    
    def __init__(self,N=128,fs=6400,w_theta_sin=[0.5,0.2,2*np.pi],m_theta_sin=[0.75,50,0],verbose_Allocation_sin_bx_br=False):
        self.N=N
        self.fs=fs
        self.verbose_Allocation_sin_bx_br=verbose_Allocation_sin_bx_br
        super().__init__(w_theta_sin,m_theta_sin)
        
        
        

    def get_eqmp_e_sin(self,SEemp,bx,br):
        """
        Parameters
        ----------
        SEemp : float 
            somme des erreurs quadratiques entre le signal et le modèle
        bx : int
            Nombre de bits servant à coder le modèle
        br : int
            nombre de bits servant à coder le résidu

        Returns
        -------
        Fonction qui détermine 'eqmp':  l'erreur introduite par le modèle et la quantification de ces paramètres et "e" l'erreur totale entre le signal initiale et le signal reconstruit sur bx +br bits

        """
    
        al_sin=self.get_allocation_sin(bx)# détermination de l'allocation optimale de bits

        delta=[self.w_theta_sin[k]*2**(-al_sin[k]) for k in range(3)]

        SEeq=(self.N/24)*(delta[0]**2+(self.w_theta_sin[0]**2/12+self.m_theta_sin[0]**2)*((4/3)*((np.pi**2*self.N**2)/self.fs**2)*delta[1]**2+delta[2]**2))

        SEeqmp=SEemp+SEeq

        SEe=(SEemp+SEeq)*2**(-2*br/self.N)
        
        return SEeqmp,SEe 
            
        
    def get_sin_bx_br(self,SEemp,btot):
        """
        Parameters
        ----------
        SEemp : float 
            somme des erreurs quadratiques entre le signal et le modèle
        btot : int
            nombre de bits tot

        Returns
        -------
        Fonction qui détermine bx et br théorique en testant les btot repartition de bits (bx,br)

        """
    
        
        SEeqmp=[0]*btot
        SEe=[0]*btot
        for bx_test in range(btot):
            br_test=btot-bx_test
            
            SEeqmp_,SEe_=self.get_eqmp_e_sin(SEemp,bx_test,br_test)
            
            SEeqmp[bx_test]=SEeqmp_
            SEe[bx_test]=SEe_
            
        bx_opt=SEe.index(np.min(SEe))
        br_opt=btot-bx_opt
        

        if self.verbose_Allocation_sin_bx_br:
            
            plt.figure(figsize=(8,4), dpi=100)
            plt.plot(np.log(SEeqmp),lw=2,label='SEeqmp')
            plt.plot(np.log(SEe),lw=2,label='SEe, bx min={}'.format(bx_opt))
            plt.xlabel('bx')
            plt.ylabel('Magnitude sin')
            plt.legend()
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show()    
               
           
           
        
        return bx_opt,br_opt,SEeqmp,SEe
    


class Allocation_poly_bx_br(Allocation_poly):
    
    def __init__(self,N=128,fs=6400,w_theta_poly=[2],verbose_Allocation_poly_bx_br=False):
        self.N=N
        self.fs=fs
        self.verbose_Allocation_poly_bx_br=verbose_Allocation_poly_bx_br
        super().__init__(w_theta_poly)
        

    def get_eqmp_e_poly(self,SEemp,bx,br,order):
        """
        Parameters
        ----------
        SEemp : float 
            somme des erreurs quadratiques entre le signal et le modèle
        bx : int
            Nombre de bits servant à coder le modèle
        br : int
            nombre de bits servant à coder le résidu

        Returns
        -------
        Fonction qui détermine 'eqmp':  l'erreur introduite par le modèle et la quantification de ces paramètres et "e" l'erreur totale entre le signal initiale et le signal reconstruit sur bx +br bits

        """
    
        al_poly=self.get_allocation_poly(bx,order)# détermination de l'allocation optimale de bits
        
        delta=[self.w_theta_poly[k]*2**(-al_poly[k]) for k in range(order+1)]

        SEeq=self.N*np.sum([self.c[k]*delta[k]**2/12 for k in range(order+1)])

        SEeqmp=SEemp+SEeq

        SEe=(SEemp+SEeq)*2**(-2*br/self.N)
        
        """
        if bx in[0,10,127]:
            print("al_poly={}, bx={}, br={}, SEeqmp={:.2f}, SEe={:.2f}, 2**(-2*br/self.N))={:.4f}".format(al_poly,bx,br,SEeqmp,SEe,2**(-2*br/self.N)))
        
        """
        return SEeqmp,SEe 
            
        
    def get_poly_bx_br(self,SEemp,btot,order):
        """
        Parameters
        ----------
        SEemp : float 
            somme des erreurs quadratiques entre le signal et le modèle
        btot : int
            nombre de bits tot

        Returns
        -------
        Fonction qui détermine bx et br théorique en testant les btot repartition de bits (bx,br)

        """
    
        
        SEeqmp=[0]*btot
        SEe=[0]*btot
        for bx_test in range(btot):
            br_test=btot-bx_test
            
            SEeqmp_,SEe_=self.get_eqmp_e_poly(SEemp,bx_test,br_test,order)
            
            SEeqmp[bx_test]=SEeqmp_
            SEe[bx_test]=SEe_
            
        bx_opt=SEe.index(np.min(SEe))
        br_opt=btot-bx_opt
        

        if self.verbose_Allocation_poly_bx_br:
            
            plt.figure(figsize=(8,4), dpi=100)
            plt.plot(np.log(SEeqmp),lw=2,label='SEeqmp')
            plt.plot(np.log(SEe),lw=2,label='SEe, bx min={}'.format(bx_opt))
            plt.xlabel('bx')
            plt.ylabel('Magnitude poly order {}'.format(order))
            plt.legend()
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show()    
        
        return bx_opt,br_opt,SEeqmp,SEe
    
    


# Programme principal
if __name__ == "__main__":
    from Antonini3 import Antonini_Encoder
    from Normalize import normalize
    from scipy.fftpack import dct,idct

    verbose = False
    N=128
    fn=50
    fs=6400
    
    btot=128 # nombre de bits servant à coder toute le signal
    nb_sign=20 # nombre signaux testés, servant à trouver des valeurs moyennes
    sigma=0.01 # bruit ajouté aux signaux de test
    
    order=4 # ordre du polynome testé
    
    t=np.linspace(0,(N-1)/fs,N) # vecteur de temps
    

    ##################### initialisation codage Antonini
    AE=Antonini_Encoder(M=9,initial_occurrence_first=[1,1],\
                     initial_occurrence_second=[1,1,1,1,1],\
                         adaptive=False,verbose_AE=False)


    ##################### initialisation allocation de bits optimale entre les deux étages pour le modèle sinusoïdal
    m_theta_sin=[0.75,fn,0]
    w_theta_sin=[0.5,0.2,2*np.pi]
    allocation_sin_bx_br=Allocation_sin_bx_br(N,fs,w_theta_sin,m_theta_sin,verbose_Allocation_sin_bx_br=verbose)
    
    ##################### initialisation allocation de bits optimale entre les deux étages pour le modèle sinusoïdal
    w_theta_poly=[2]*(8+1)
    allocation_poly_bx_br=Allocation_poly_bx_br(N,fs,w_theta_poly,verbose_Allocation_poly_bx_br=verbose)
    
        
    ###################### initialisation class codeur modele
    m=Model_Encoder(fn=fn,fs=fs,N=N,w_sin=w_theta_sin,m_sin=m_theta_sin,w_poly=w_theta_poly,verbose=verbose)
    
    
    
    
    #####################
    # Code test pour visualiser si l'allocation de bits théorique entre les deux étages de comression cole à la réalité pour le modèle sinusoïdal
    #####################

    
    SEemp_real=np.zeros(nb_sign) # erreur quadratique entre le siglal réel et le signal éstimé
    SEemp_theo=N*sigma**2 # erreur quadratique théorique de l'erreur d'estimation des paramètres pour un signal sinusoidal
    
    SEeq_real=np.zeros((nb_sign,btot)) # erreur quadratique entre le siglal estimé et le signal quantifié
    #SEeq_theor=np.zeros(nb_sign) # erreur quadratique entre le siglal estimé et le signal quantifié théroque
    
    SEeqmp_real=np.zeros((nb_sign,btot))#[[0]*nb_sign]*btot  # erreur quadratique entre le signal réel et le signal reconstruit à partir du modèle quantifié réel
    SEeqmp_theo=np.zeros((nb_sign,btot))   # erreur quadratique entre le signal réel et le signal reconstruit à partir du modèle quantifié éstimé
    
    SEe_real=np.zeros((nb_sign,btot)) #erreur quadratique totale entre le signal réel et le signal reconstruit réel
    SEe_theo=np.zeros((nb_sign,btot)) #erreur quadratique totale entre le signal réel et le signal reconstruit théorique
    
    for sign in range(nb_sign):
    
        """
        Création d'un signal de test polynomial
        """

        a=np.random.uniform(0.5,1)
        f=np.random.uniform(fn- w_theta_sin[1]/2,fn+w_theta_sin[1]/2)
        phi=np.random.uniform(-np.pi,np.pi)
        
        theta_sin=[a,f,phi]
        #print("theta sin: {:.2f},{:.2f},{:.2f}".format(*theta_sin))
    
        model_sin=Model_sin(fn,fs,N)
        
        x_test=model_sin.get_model_sin(t,*theta_sin)+np.random.normal(0,sigma,N) 
        
        
        
        """
        estimation des paramètres
        """
        theta_sin_hat=m.get_theta_sin(x_test) # estimation des paramètres polynomiaux
        # print("theta sin hat: {:.2f},{:.2f},{:.2f}".format(*theta_sin_hat))
        x_sin_hat=m.get_model_sin(t,*theta_sin_hat) # modèle déterminé

        emp_real=x_test-x_sin_hat # erreur entre le signal et le modèle

        SEemp_real[sign]=np.sum((emp_real)**2) 
        


        for bx_test in range(btot):
            
            br_test=btot-bx_test # nombre de bits max servant à coder le résidu
            

            """
            Premier étage de compression
            """
            theta_sin_tilde,_=m.get_theta_sin_tilde(theta_sin_hat,bx_test) #Quantification de theta_sin_hat sur bx bits
            x_sin_tilde=m.get_model_sin(t,*theta_sin_tilde) # Construction du modèle sin avec theta_sin_tilde
            #print("bx",bx_test,"theta_hat",theta_sin_hat)
            #print("bx",bx_test,"theta_tilde",theta_sin_tilde)
            eq_real=x_sin_hat-x_sin_tilde # calcul de l'erreur quadratique entre le modèle où les paramètres ont été estimé et où les paramètre ont été quantifié
            SEeq_real[sign][bx_test]=np.sum(eq_real**2)
        
            r=x_test-x_sin_tilde #résidu obtenu en sortie du premier étage de compression
            SEeqmp_real[sign][bx_test]=np.sum(r**2) 
        
            """
            Second étage de compression
            """
                  
            
             
            """
            coefs_r=dct(r/N) # on effectue une DCT
            coefs_r_n,k=normalize(coefs_r) # on normalise les coefficients DCT
           
            #print("coefs min {:.2f} max coefs {:.2f} ".format(np.min(coefs_r_n),np.max(coefs_r_n)))
            
            code,nb_coefs=AE.get_code_res(coefs_r_n,br_test) # on encode les coeffs avec le codeur d'Antonini
            
            coefs_r_rec=AE.coefs_rec*2**(k) # coefs reconstruit su br bits
            r_rec=idct(coefs_r_rec)/2
            """
             
           
            r_n,k=normalize(r) # on normalise les coefficients DCT
            coefs_r=dct(r_n/N) # on effectue une DCT
            
            #print("coefs min {:.2f} max coefs {:.2f} ".format(np.min(coefs_r),np.max(coefs_r)))
            
            code=AE.get_code_res(coefs_r,br_test) # on encode les coeffs avec le codeur d'Antonini
            coefs_r_rec=AE.coefs_rec # coefs reconstruit su br bits
            
            r_rec=idct(coefs_r_rec)*2**(k-1)
            
            
            
            x_sin_rec=x_sin_tilde+r_rec# signal rec apres les deux étages 
            e_real=x_test-x_sin_rec # erreur de reconstruction total
            SEe_real[sign][bx_test]=np.sum(e_real**2)
            
           
            SEeqmp_theo_,SEe_theo_=allocation_sin_bx_br.get_eqmp_e_sin(SEemp_real[sign],bx_test,br_test)# theo ou real
            
            SEeqmp_theo[sign][bx_test]=SEeqmp_theo_
            SEe_theo[sign][bx_test]=SEe_theo_
            
            #if bx_test<=10:
            #print("bx={}".format(bx_test),"br={}".format(len(code)),"btot={}".format(bx_test+len(code)),"SNR_x={:.2f}".format(get_snr(x_test,x_poly_tilde)),"SNR_r={:.2f}".format(get_snr(r,r_rec)),"SNR_tot={:.2f}".format(get_snr(x_test,x_poly_rec)))
            
            if bx_test in [20] and sign==0:
            
                plt.figure(figsize=(8,4), dpi=100)
                plt.plot(t,x_test,lw=2,label='x')
                plt.plot(t,x_sin_hat,lw=2,label='x hat, SNR={:.1f} dB'.format(get_snr(x_test,x_sin_hat)))
                plt.plot(t,x_sin_tilde,lw=2,label='x tilde, SNR={:.1f} dB, bx={} b'.format(get_snr(x_test,x_sin_tilde),bx_test))
                plt.plot(t,x_sin_rec,lw=2,label='x rec, SNR={:.1f} dB, bx={}+len(code)={}={} b'.format(get_snr(x_test,x_sin_rec),bx_test,len(code),bx_test+len(code)))
                plt.xlabel('t [s]')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.title("Modèle sinusoïdal")
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show()
                
                """
                plt.figure(figsize=(8,4), dpi=100)
                plt.plot(coefs_r,lw=2,label='coefs')
                plt.plot(coefs_r_rec,lw=2,label='coefs rec, SNR={:.1f} dB, br={} b, len(code)={} b'.format(get_snr(r,r_rec),br_test,len(code)))
                plt.xlabel('ind')
                plt.ylabel('coefs')
                plt.legend()
                plt.title("coefficient dct du résidu pour le modèle sin")
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show()
                
                
                plt.figure(figsize=(8,4), dpi=100)
                plt.plot(t,r,lw=2,label='r')
                plt.plot(t,r_rec,lw=2,label='r rec, SNR={:.1f} dB, br={} b, len(code)={} b'.format(get_snr(r,r_rec),br_test,len(code)))
                plt.xlabel('t [s]')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.title("Résidu reconstruit")
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show()
                """
            
            
   
    SEeqmp_real_m=[np.mean(SEeqmp_real[:,k]) for k in range(btot)]
    SEeqmp_theo_m=[np.mean(SEeqmp_theo[:,k]) for k in range(btot)]
    
   
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(np.log(SEeqmp_real_m),lw=2,label='SEeqmp real')
    plt.plot(np.log(SEeqmp_theo_m),lw=2,label='SEeqmp theoritical')
    plt.xlabel('bx')
    plt.ylabel('Magnitude')
    plt.title('log(eqmp real mean) and log(eqmp theo mean) with sine model')
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()    
   
    
    SEe_real_m=[np.mean(SEe_real[:,k]) for k in range(btot)]
    SEe_theo_m=[np.mean(SEe_theo[:,k]) for k in range(btot)]
    
   
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(np.log(SEe_real_m),lw=2,label='SEe real, bx opt={}'.format(SEe_real_m.index(min(SEe_real_m))))
    plt.plot(np.log(SEe_theo_m),lw=2,label='SEe theoritical, bx min={}'.format(SEe_theo_m.index(min(SEe_theo_m))))
    plt.xlabel('bx')
    plt.ylabel('Magnitude')
    plt.title('log(e real mean) and log(e theo mean) with sine')
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()    
   
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(np.log(SEe_real_m),lw=2,label='SEe real, bx opt={}'.format(SEe_real_m.index(min(SEe_real_m))))
    plt.xlabel('bx')
    plt.ylabel('Magnitude')
    plt.title('log(e real mean) with sin of order {}')
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()       
   
     
    #print("sin",SEemp_real)
    print("sin","SEemp_real mean={:.4f}".format(np.mean(SEemp_real)),"SEemp_theo={:.4f}".format(SEemp_theo))
       

    bx_opt,br_opt=allocation_sin_bx_br.get_sin_bx_br(np.mean(SEemp_real),btot)[0:2]
    print("sine model, bx opt = {}, br opt = {}, btot = {} bits".format(bx_opt,br_opt,btot))
    
    
     
           
    
    
    
    
    
    

    #####################   début du codage de x_test avec un modèle polynomial 
    


    SEemp_real=np.zeros(nb_sign) # erreur quadratique entre le siglal réel et le signal éstimé
    SEemp_theo=N*sigma**2/(order+1) # erreur quadratique théorique de l'erreur d'estimation des paramètres pour un polynôme
    
    SEeq_real=np.zeros((nb_sign,btot)) # erreur quadratique entre le siglal estimé et le signal quantifié
    #SEeq_theor=np.zeros(btot) # erreur quadratique entre le siglal estimé et le signal quantifié théroque
    
    SEeqmp_real=np.zeros((nb_sign,btot))#[[0]*nb_sign]*btot  # erreur quadratique entre le signal réel et le signal reconstruit à partir du modèle quantifié réel
    SEeqmp_theo=np.zeros((nb_sign,btot))   # erreur quadratique entre le signal réel et le signal reconstruit à partir du modèle quantifié éstimé
    
    SEe_real=np.zeros((nb_sign,btot)) #erreur quadratique totale entre le signal réel et le signal reconstruit réel
    SEe_theo=np.zeros((nb_sign,btot)) #erreur quadratique totale entre le signal réel et le signal reconstruit théorique
    
    
    for sign in range(nb_sign):
    
        """
        Création d'un signal de test polynomial
        """

        theta_poly=np.random.uniform(-1,1,order+1)
      
        model_poly=Model_poly(fn,fs,N,verbose)
        
        x_test=model_poly.get_model_poly(t,*theta_poly)+np.random.normal(0,sigma,N)
        x_test,_=normalize(x_test)
        
        
        """
        estimation des paramètres
        """
        theta_poly_hat=m.get_theta_poly(x_test,order) # estimation des paramètres polynomiaux
        # print("theta poly hat: {:.2f},{:.2f},{:.2f}".format(*theta_poly_hat))
        x_poly_hat=m.get_model_poly(t,*theta_poly_hat) # modèle déterminé

        emp_real=x_test-x_poly_hat # erreur entre le signal et le modèle

        SEemp_real[sign]=np.sum((emp_real)**2) 
        

        for bx_test in range(btot):
            br_test=btot-bx_test # nombre de bits max servant à coder le résidu
            
            """
            Premier étage de compression
            """
            theta_poly_tilde,_=m.get_theta_poly_tilde(theta_poly_hat,bx_test) #Quantification de theta_sin_hat sur bx bits
            x_poly_tilde=m.get_model_poly(t,*theta_poly_tilde) # Construction du modèle sin avec theta_sin_tilde
        
            eq_real=x_poly_hat-x_poly_tilde # calcul de l'erreur quadratique entre le modèle où les paramètres ont été estimé et où les paramètre ont été quantifié
            SEeq_real[sign][bx_test]=np.sum(eq_real**2)
        
            r=x_test-x_poly_tilde #résidu obtenu en sortie du premier étage de compression
            SEeqmp_real[sign][bx_test]=np.sum(r**2) 
        
            """
            Second étage de compression
            """
            
            
            """
            coefs_r=dct(r/N) # on effectue une DCT
            coefs_r_n,k=normalize(coefs_r) # on normalise les coefficients DCT

            #print("coefs min {:.2f} max coefs {:.2f} ".format(np.min(coefs_r_n),np.max(coefs_r_n)))
    
            code,nb_coefs=AE.get_code_res(coefs_r_n,br_test) # on encode les coeffs avec le codeur d'Antonini
            
            coefs_r_rec=AE.coefs_rec*2**(k) # coefs reconstruit su br bits
            r_rec=idct(coefs_r_rec)/2
            """
        
       
            r_n,k=normalize(r) # on normalise les coefficients DCT
            coefs_r=dct(r_n/N) # on effectue une DCT
            
            #print("coefs max {:.2f}".format(np.max(np.abs(np.array(coefs_r)))))
    
            code=AE.get_code_res(coefs_r,br_test) # on encode les coeffs avec le codeur d'Antonini
            coefs_r_rec=AE.coefs_rec # coefs reconstruit su br bits
            
            r_rec_n=idct(coefs_r_rec)/2
            r_rec=r_rec_n*2**(k)
            
        
        
            x_poly_rec=x_poly_tilde+r_rec# signal rec apres les deux étages 
            e_real=x_test-x_poly_rec # erreur de reconstruction total
            SEe_real[sign][bx_test]=np.sum(e_real**2)
            

            SEeqmp_theo_,SEe_theo_=allocation_poly_bx_br.get_eqmp_e_poly(SEemp_real[sign],bx_test,br_test,order)# theo ou real
        
            SEeqmp_theo[sign][bx_test]=SEeqmp_theo_
            SEe_theo[sign][bx_test]=SEe_theo_
        
            #if bx_test<=10:
            #print("bx={}".format(bx_test),"br={}".format(len(code)),"btot={}".format(bx_test+len(code)),"SNR_x={:.2f}".format(get_snr(x_test,x_poly_tilde)),"SNR_r={:.2f}".format(get_snr(r,r_rec)),"SNR_tot={:.2f}".format(get_snr(x_test,x_poly_rec)))
            
            if bx_test in [20] and sign==0:
                
                plt.figure(figsize=(8,4), dpi=100)
                plt.plot(t,x_test,lw=2,label='x')
                plt.plot(t,x_poly_hat,lw=2,label='x hat, SNR={:.1f} dB'.format(get_snr(x_test,x_poly_hat)))
                plt.plot(t,x_poly_tilde,lw=2,label='x tilde, SNR={:.1f} dB, bx={} b'.format(get_snr(x_test,x_poly_tilde),bx_test))
                plt.plot(t,x_poly_rec,lw=2,label='x rec, SNR={:.1f} dB, bx={}+len(code)={}={} b'.format(get_snr(x_test,x_poly_rec),bx_test,len(code),bx_test+len(code)))
                plt.xlabel('t [s]')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.title("Modèle polynomial d'ordre {}".format(order))
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show()
                
                """
                plt.figure(figsize=(8,4), dpi=100)
                plt.plot(coefs_r,lw=2,label='coefs')
                plt.plot(coefs_r_rec,lw=2,label='coefs rec,SNR={:.1f} dB, br={} b, len(code)={} b'.format(get_snr(r,r_rec),br_test,len(code)))
                plt.xlabel('ind')
                plt.ylabel('coefs')
                plt.legend()
                plt.title("coefficient dct")
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show()
                
            
                plt.figure(figsize=(8,4), dpi=100)
                plt.plot(t,r_n,lw=2,label='r')
                plt.plot(t,r_rec_n,lw=2,label='r rec, SNR={:.1f} dB, br={} b, len(code)={} b'.format(get_snr(r,r_rec),br_test,len(code)))
                plt.xlabel('t [s]')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.title("Résidu rec")
                plt.grid( which='major', color='#666666', linestyle='-')
                plt.minorticks_on()
                plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                plt.show()
                """
        


    SEeqmp_real_m=[np.mean(SEeqmp_real[:,k]) for k in range(btot)]
    SEeqmp_theo_m=[np.mean(SEeqmp_theo[:,k]) for k in range(btot)]
    

    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(np.log(SEeqmp_real_m),lw=2,label='SEeqmp real')
    plt.plot(np.log(SEeqmp_theo_m),lw=2,label='SEeqmp theoritical')
    plt.xlabel('bx')
    plt.ylabel('Magnitude')
    plt.title('log(eqmp real mean) and log(eqmp theo mean) with polynom of order {}'.format(order))
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()    
       
    
    SEe_real_m=[np.mean(SEe_real[:,k]) for k in range(btot)]
    SEe_theo_m=[np.mean(SEe_theo[:,k]) for k in range(btot)]
    

    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(np.log(SEe_real_m),lw=2,label='SEe real, bx opt={}'.format(SEe_real_m.index(min(SEe_real_m))))
    plt.plot(np.log(SEe_theo_m),lw=2,label='SEe theoritical, bx min={}'.format(SEe_theo_m.index(min(SEe_theo_m))))
    plt.xlabel('bx')
    plt.ylabel('Magnitude')
    plt.title('log(e real mean) and log(e theo mean) with polynom of order {}'.format(order))
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()    
       
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(np.log(SEe_real_m),lw=2,label='SEe real, bx opt={}'.format(SEe_real_m.index(min(SEe_real_m))))
    plt.xlabel('bx')
    plt.ylabel('Magnitude')
    plt.title('log(e real mean) with polynom of order {}'.format(order))
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()       
       
    
    #print("poly",SEemp_real)
    print("poly order {}".format(order),"SEemp_real mean={:.4f}".format(np.mean(SEemp_real)),"SEemp_theo={:.4f}".format(SEemp_theo))
   

    bx_opt,br_opt=allocation_poly_bx_br.get_poly_bx_br(np.mean(SEemp_real),btot,order)[0:2]
    print("poly order {}, bx opt = {}, br opt = {}, btot = {} bits".format(order,bx_opt,br_opt,btot))
    
    
    
    
    
    
    
    
    
    