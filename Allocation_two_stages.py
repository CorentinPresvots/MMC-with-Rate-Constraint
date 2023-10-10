# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:40:12 2023

@author: coren
"""



########################################################

#allocation de bits entre modèle et résidu en supposant que le résidu est gaussien


import numpy as np
import matplotlib.pyplot as plt


from Bits_allocation import Allocation_sin,Allocation_poly,Allocation_pred_samples
from codage_model import Model_Encoder
from Models import Model_sin,Model_poly,Model_pred_samples
 
from Measures import get_snr


class Allocation_sin_bx_br(Allocation_sin):
    
    def __init__(self,N=128,fs=6400,verbose_Allocation_sin_bx_br=False):
        self.N=N
        self.fs=fs
        self.verbose_Allocation_sin_bx_br=verbose_Allocation_sin_bx_br
        super().__init__()
        
        
        

    def get_eqmp_e_sin(self,SEemp,bx,br,m_theta_sin,w_theta_sin):
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
    
        al_sin=self.get_allocation_sin(bx,m_theta_sin,w_theta_sin)# détermination de l'allocation optimale de bits

        delta=[w_theta_sin[k]*2**(-al_sin[k]) for k in range(3)]

        SEeq=(self.N/24)*(delta[0]**2+(w_theta_sin[0]**2/12+m_theta_sin[0]**2)*((4/3)*((np.pi**2*self.N**2)/self.fs**2)*delta[1]**2+delta[2]**2))

        SEeqmp=SEemp+SEeq

        SEe=(SEemp+SEeq)*2**(-2*br/self.N)
        
        return SEeqmp,SEe 
            
        
    def get_sin_bx_br(self,SEemp,btot,m_theta_sin,w_theta_sin):
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
            
            SEeqmp_,SEe_=self.get_eqmp_e_sin(SEemp,bx_test,br_test,m_theta_sin,w_theta_sin)
            
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
    
    def __init__(self,N=128,fs=6400,verbose_Allocation_poly_bx_br=False):
        self.N=N
        self.fs=fs
        self.verbose_Allocation_poly_bx_br=verbose_Allocation_poly_bx_br
        super().__init__()
        

    def get_eqmp_e_poly(self,SEemp,bx,br,w_theta_poly):
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
        order=len(w_theta_poly)-1
        al_poly=self.get_allocation_poly(bx,w_theta_poly)# détermination de l'allocation optimale de bits
        
        delta=[w_theta_poly[k]*2**(-al_poly[k]) for k in range(order+1)]

        SEeq=self.N*np.sum([self.c[k]*delta[k]**2/12 for k in range(order+1)])

        SEeqmp=SEemp+SEeq

        SEe=(SEemp+SEeq)*2**(-2*br/self.N)
        
        """
        if bx in[0,10,127]:
            print("al_poly={}, bx={}, br={}, SEeqmp={:.2f}, SEe={:.2f}, 2**(-2*br/self.N))={:.4f}".format(al_poly,bx,br,SEeqmp,SEe,2**(-2*br/self.N)))
        
        """
        return SEeqmp,SEe 
            
        
    def get_poly_bx_br(self,SEemp,btot,w_theta_poly):
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
            
            SEeqmp_,SEe_=self.get_eqmp_e_poly(SEemp,bx_test,br_test,w_theta_poly)
            
            SEeqmp[bx_test]=SEeqmp_
            SEe[bx_test]=SEe_
            
        bx_opt=SEe.index(np.min(SEe))
        br_opt=btot-bx_opt
        

        if self.verbose_Allocation_poly_bx_br:
            order=len(w_theta_poly)-1
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
    



class Allocation_pred_samples_bx_br(Allocation_pred_samples):
    
    def __init__(self,N=128,fs=6400,verbose_Allocation_pred_samples_bx_br=False):
        self.N=N
        self.fs=fs
        self.verbose_Allocation_pred_samples_bx_br=verbose_Allocation_pred_samples_bx_br
        super().__init__()
        

    def get_eqmp_e_pred_samples(self,SEemp,bx,br,m_theta_pred_samples,w_theta_pred_samples,previous_samples,eta):
        
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
        order=len(w_theta_pred_samples)
        al_pred_samples=self.get_allocation_pred_samples(bx,m_theta_pred_samples,w_theta_pred_samples)# détermination de l'allocation optimale de bits
        #print("al_pred_samples",al_pred_samples,bx)
        delta=[w_theta_pred_samples[k]*2**(-al_pred_samples[k]) for k in range(order)]
        
        SEeq=np.sum([np.sum(np.array(previous_samples[self.N-eta-k:2*self.N-eta-k])**2)*delta[k]**2/12 for k in range(order)])

        SEeqmp=SEemp+SEeq
        
        #print(SEeq)
        SEe=(SEemp+SEeq)*2**(-2*br/self.N)
        
        """
        if bx in[0,10,127]:
            print("al_poly={}, bx={}, br={}, SEeqmp={:.2f}, SEe={:.2f}, 2**(-2*br/self.N))={:.4f}".format(al_poly,bx,br,SEeqmp,SEe,2**(-2*br/self.N)))
        
        """
        return SEeqmp,SEe 
            
        
    def get_pred_samples_bx_br(self,SEemp,btot,m_theta_pred_samples,w_theta_pred_samples,x_pre,eta):
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
            
            SEeqmp_,SEe_=self.get_eqmp_e_pred_samples(SEemp,bx_test,br_test,m_theta_pred_samples,w_theta_pred_samples,x_pre,eta)
            
            SEeqmp[bx_test]=SEeqmp_
            SEe[bx_test]=SEe_
            
        bx_opt=SEe.index(np.min(SEe))
        br_opt=btot-bx_opt
        

        if self.verbose_Allocation_pred_samples_bx_br:
            order=len(w_theta_pred_samples)-1
            plt.figure(figsize=(8,4), dpi=100)
            plt.plot(np.log(SEeqmp),lw=2,label='SEeqmp')
            plt.plot(np.log(SEe),lw=2,label='SEe, bx min={}'.format(bx_opt))
            plt.xlabel('bx')
            plt.ylabel('Magnitude pred_samples order {}'.format(order))
            plt.legend()
            plt.grid( which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show()    
        
        return bx_opt,br_opt,SEeqmp,SEe


    


# Programme principal
if __name__ == "__main__":
    from Antonini import Antonini_Encoder
    from Normalize import normalize
    from scipy.fftpack import dct,idct

    verbose = False
    N=128
    fn=50
    fs=6400
    
    btot=128 # nombre de bits servant à coder toute le signal
    nb_sign=40 # nombre signaux testés, servant à trouver des valeurs moyennes
    sigma=0.01 # bruit ajouté aux signaux de test
    
    
    
    t=np.linspace(0,(N-1)/fs,N) # vecteur de temps
    

    ##################### initialisation codage Antonini
    AE=Antonini_Encoder(M=9,initial_occurrence_first=[1,1],\
                     initial_occurrence_second=[1,1,1,1,1],\
                         adaptive=True,verbose_AE=False)


    ##################### initialisation allocation de bits optimale entre les deux étages pour le modèle sinusoïdal

    allocation_sin_bx_br=Allocation_sin_bx_br(N,fs,verbose_Allocation_sin_bx_br=verbose)
    
    ##################### initialisation allocation de bits optimale entre les deux étages pour le modèle sinusoïdal
    

    allocation_poly_bx_br=Allocation_poly_bx_br(N,fs,verbose_Allocation_poly_bx_br=verbose)
    
    ##################### initialisation allocation de bits optimale entre les deux étages pour le modèle pred samples
    

    allocation_pred_samples_bx_br=Allocation_pred_samples_bx_br(N,fs,verbose_Allocation_pred_samples_bx_br=verbose)
    
                
    ###################### initialisation class codeur modele
    m=Model_Encoder(fn=fn,fs=fs,N=N,verbose=verbose)
    
    ##################### initialisation class modèle sin modele poly
    #model_sin=Model_sin(fn,fs,N)
    
    #model_poly=Model_poly(fn,fs,N)
    
    
    #Model_pred_samples=Model_pred_samples(fn,fs,N)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    #####################   début du codage de x_test avec un modèle pred_samples
    N_p=3
    eta=0
    

    SEemp_real=np.zeros(nb_sign) # erreur quadratique entre le siglal réel et le signal éstimé
    SEemp_theo=10*N*sigma**2/(N_p) # erreur quadratique théorique de l'erreur d'estimation des paramètres pour un polynôme
    
    SEeq_real=np.zeros((nb_sign,btot)) # erreur quadratique entre le siglal estimé et le signal quantifié
    #SEeq_theor=np.zeros(btot) # erreur quadratique entre le siglal estimé et le signal quantifié théroque
    
    SEeqmp_real=np.zeros((nb_sign,btot))#[[0]*nb_sign]*btot  # erreur quadratique entre le signal réel et le signal reconstruit à partir du modèle quantifié réel
    SEeqmp_theo=np.zeros((nb_sign,btot))   # erreur quadratique entre le signal réel et le signal reconstruit à partir du modèle quantifié éstimé
    
    SEe_real=np.zeros((nb_sign,btot)) #erreur quadratique totale entre le signal réel et le signal reconstruit réel
    SEe_theo=np.zeros((nb_sign,btot)) #erreur quadratique totale entre le signal réel et le signal reconstruit théorique
    
    
    
    
    m_theta_sin=[0.75,fn,0]
    w_theta_sin=[0.5,0.2,2*np.pi]
    
    
    fact=20
    m_theta_sin2=[m_theta_sin[0]/fact,3*fn,0]
    w_theta_sin2=[w_theta_sin[0]/fact,w_theta_sin[1]/fact,w_theta_sin[2]/fact]
    
    
    
    t_pred_samples=np.linspace(0,(3*N-1)/fs,3*N)
    
    

    yp=np.array([np.cos(2*np.pi*m_theta_sin[1]*k*(1/fs)) for k in range(3*N)]) 
    m_theta_pred_samples=m.get_m_theta_pred_samples(N_p,eta,0)
    w_theta_pred_samples=[1]*N_p    
    
    print("m_theta_pred_samples", [np.round(1000*m_theta_pred_samples[k])/1000 for k in range(N_p)])
    for sign in range(nb_sign):
    
        """
        Création d'un signal de test avec harmonique
        """
    
        a=np.random.uniform(m_theta_sin[0]-0.5*w_theta_sin[0],m_theta_sin[0]+0.5*w_theta_sin[0])
        f=np.random.uniform(m_theta_sin[1]-0.5*w_theta_sin[1],m_theta_sin[1]+0.5*w_theta_sin[1])
        phi=np.random.uniform(m_theta_sin[2]-0.5*w_theta_sin[2],m_theta_sin[2]+0.5*w_theta_sin[2])
        
        theta_sin=[a,f,phi]
        
    
        a2=np.random.uniform(m_theta_sin2[0]-0.5*w_theta_sin2[0],m_theta_sin2[0]+0.5*w_theta_sin2[0])
        f2=3*f#np.random.uniform(m_theta_sin2[1]-0.5*w_theta_sin2[1],m_theta_sin2[1]+0.5*w_theta_sin2[1])
        phi2=np.random.uniform(m_theta_sin2[2]-0.5*w_theta_sin2[2],m_theta_sin2[2]+0.5*w_theta_sin2[2])
        
        
        theta_sin2=[a2,f2,phi2]
   


    
        x_sin_H=m.get_model_sin(t_pred_samples,*theta_sin)+m.get_model_sin(t_pred_samples,*theta_sin2)+np.random.normal(0,sigma,3*N) 
    
    
        x_test=x_sin_H[2*N:]
        x_test_pre=x_sin_H[0:2*N]
        
        """
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(t_pred_samples[0:2*N],x_test_pre,lw=2,label='xp')
        plt.plot(t_pred_samples[2*N:],x_test,lw=2,label='x')
        plt.xlabel('t [s]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title(" x test pred samples")
        plt.grid( which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
        """              
        
        
        
        
        """
        estimation des paramètres
        """
        theta_pred_samples_hat,X=m.get_theta_pred_samples(x_test,x_test_pre,N_p,eta) # estimation des paramètres polynomiaux
        #print("theta pred_samples hat: {:.2f},{:.2f}".format(*theta_pred_samples_hat))
        x_pred_samples_hat=m.get_model_pred_samples(X,*theta_pred_samples_hat) # modèle déterminé

        emp_real=x_test-x_pred_samples_hat # erreur entre le signal et le modèle

        SEemp_real[sign]=np.sum((emp_real)**2) 
        
         
        
        
        

        
        
        for bx_test in range(btot):
            br_test=btot-bx_test # nombre de bits max servant à coder le résidu
            
            """
            Premier étage de compression
            """
            theta_pred_samples_tilde,_=m.get_theta_pred_samples_tilde(theta_pred_samples_hat,bx_test,m_theta_pred_samples,w_theta_pred_samples) #Quantification de theta_sin_hat sur bx bits
            #### test si sum alpha k=1
            #theta_pred_samples_tilde[-1]=1-sum(theta_pred_samples_tilde[0:N_p-1])
            
            #print(np.sum(theta_pred_samples_tilde))
            
            
            x_pred_samples_tilde=m.get_model_pred_samples(X,*theta_pred_samples_tilde) # Construction du modèle sin avec theta_sin_tilde
            

            eq_real=x_pred_samples_hat-x_pred_samples_tilde #erreur quadratique entre le modèle où les paramètres ont été estimé et où les paramètre ont été quantifié
            SEeq_real[sign][bx_test]=np.sum(eq_real**2)
             
            r=x_test-x_pred_samples_tilde #résidu obtenu en sortie du premier étage de compression
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
    
            code=AE.get_code_res_Antonini(coefs_r,br_test) # on encode les coeffs avec le codeur d'Antonini
            coefs_r_rec=AE.coefs_rec # coefs reconstruit su br bits
            
            r_rec_n=idct(coefs_r_rec)/2
            r_rec=r_rec_n*2**(k)
            
        
        
            x_pred_samples_rec=x_pred_samples_tilde+r_rec# signal rec apres les deux étages 
            e_real=x_test-x_pred_samples_rec # erreur de reconstruction total
            SEe_real[sign][bx_test]=np.sum(e_real**2)
            

            SEeqmp_theo_,SEe_theo_=allocation_pred_samples_bx_br.get_eqmp_e_pred_samples(SEemp_real[sign],bx_test,br_test,m_theta_pred_samples,w_theta_pred_samples,x_test_pre,eta)# theo ou real
        
            SEeqmp_theo[sign][bx_test]=SEeqmp_theo_
            SEe_theo[sign][bx_test]=SEe_theo_
        
            #if bx_test<=10:
            #print("bx={}".format(bx_test),"br={}".format(len(code)),"btot={}".format(bx_test+len(code)),"SNR_x={:.2f}".format(get_snr(x_test,x_poly_tilde)),"SNR_r={:.2f}".format(get_snr(r,r_rec)),"SNR_tot={:.2f}".format(get_snr(x_test,x_poly_rec)))
            
            if bx_test in [10] and sign==0:
                #print(bx_test)
          
               
                print("theta_pred_samples_hat",[np.round(1000*theta_pred_samples_hat[k])/1000 for k in range(N_p)])
                print("theta_pred_samples_tilde",[np.round(1000*theta_pred_samples_tilde[k])/1000 for k in range(N_p)])
                print("bx={} bits, SNR_hat={:.2f}, SNR_tilde={:.2f}".format(bx_test,get_snr(x_test,x_pred_samples_hat),get_snr(x_test,x_pred_samples_tilde)))
                
                plt.figure(figsize=(8,4), dpi=100)
                plt.plot(t,x_test,lw=2,label='x')
                plt.plot(t,x_pred_samples_hat,lw=2,label='x hat, SNR={:.1f} dB'.format(get_snr(x_test,x_pred_samples_hat)))
                plt.plot(t,x_pred_samples_tilde,lw=2,label='x tilde, SNR={:.1f} dB, bx={} b'.format(get_snr(x_test,x_pred_samples_tilde),bx_test))
                plt.plot(t,x_pred_samples_rec,lw=2,label='x rec, SNR={:.1f} dB, bx={}+len(code)={}={} b'.format(get_snr(x_test,x_pred_samples_rec),bx_test,len(code),bx_test+len(code)))
                plt.xlabel('t [s]')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.title("Modèle pred samples d'ordre {}".format(N_p))
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
    plt.title('log(eqmp real mean) and log(eqmp theo mean) with pred samples of order {}'.format(N_p))
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
    plt.title('log(e real mean) and log(e theo mean) with pred samples of order {}'.format(N_p))
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()    
       
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(np.log(SEe_real_m),lw=2,label='SEe real, bx opt={}'.format(SEe_real_m.index(min(SEe_real_m))))
    plt.xlabel('bx')
    plt.ylabel('Magnitude')
    plt.title('log(e real mean) with pred samples of order {}'.format(N_p))
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()       
       
    
    #print("poly",SEemp_real)
    print("pred_samples order {}".format(N_p),"SEemp_real mean={:.4f}".format(np.mean(SEemp_real)),"SEemp_theo={:.4f}".format(SEemp_theo))
   

    bx_opt,br_opt=allocation_pred_samples_bx_br.get_pred_samples_bx_br(np.mean(SEemp_real),btot,m_theta_pred_samples,w_theta_pred_samples,x_test_pre,eta)[0:2]
    print("pred order {}, bx opt = {}, br opt = {}, btot = {} bits".format(N_p,bx_opt,br_opt,btot))
    

    
    
    
    
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
    
    m_theta_sin=[0.75,fn,0]
    w_theta_sin=[0.5,0.2,2*np.pi]
    
    
    for sign in range(nb_sign):
    
        """
        Création d'un signal de test polynomial
        """
        a=np.random.uniform(m_theta_sin[0]-0.5*w_theta_sin[0],m_theta_sin[0]+0.5*w_theta_sin[0])
        f=np.random.uniform(m_theta_sin[1]-0.5*w_theta_sin[1],m_theta_sin[1]+0.5*w_theta_sin[1])
        phi=np.random.uniform(m_theta_sin[2]-0.5*w_theta_sin[2],m_theta_sin[2]+0.5*w_theta_sin[2])
        

        
        theta_sin=[a,f,phi]
        #print("theta sin: {:.2f},{:.2f},{:.2f}".format(*theta_sin))
    
        
        
        x_test=m.get_model_sin(t,*theta_sin)+np.random.normal(0,sigma,N) 
        
        
        
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
            theta_sin_tilde,_=m.get_theta_sin_tilde(theta_sin_hat,bx_test,m_theta_sin,w_theta_sin) #Quantification de theta_sin_hat sur bx bits
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
            
            code=AE.get_code_res_Antonini(coefs_r,br_test) # on encode les coeffs avec le codeur d'Antonini
            coefs_r_rec=AE.coefs_rec # coefs reconstruit su br bits
            
            r_rec=idct(coefs_r_rec)*2**(k-1)
            
            
            
            x_sin_rec=x_sin_tilde+r_rec# signal rec apres les deux étages 
            e_real=x_test-x_sin_rec # erreur de reconstruction total
            SEe_real[sign][bx_test]=np.sum(e_real**2)
            
           
            SEeqmp_theo_,SEe_theo_=allocation_sin_bx_br.get_eqmp_e_sin(SEemp_real[sign],bx_test,br_test,m_theta_sin,w_theta_sin)# theo ou real
            
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
       

    bx_opt,br_opt=allocation_sin_bx_br.get_sin_bx_br(np.mean(SEemp_real),btot,m_theta_sin,w_theta_sin)[0:2]
    print("sine model, bx opt = {}, br opt = {}, btot = {} bits".format(bx_opt,br_opt,btot))
    
    
     
           
    
    
    
    
    
    

    #####################   début du codage de x_test avec un modèle polynomial 
    order=3
    

    SEemp_real=np.zeros(nb_sign) # erreur quadratique entre le siglal réel et le signal éstimé
    SEemp_theo=N*sigma**2/(order+1) # erreur quadratique théorique de l'erreur d'estimation des paramètres pour un polynôme
    
    SEeq_real=np.zeros((nb_sign,btot)) # erreur quadratique entre le siglal estimé et le signal quantifié
    #SEeq_theor=np.zeros(btot) # erreur quadratique entre le siglal estimé et le signal quantifié théroque
    
    SEeqmp_real=np.zeros((nb_sign,btot))#[[0]*nb_sign]*btot  # erreur quadratique entre le signal réel et le signal reconstruit à partir du modèle quantifié réel
    SEeqmp_theo=np.zeros((nb_sign,btot))   # erreur quadratique entre le signal réel et le signal reconstruit à partir du modèle quantifié éstimé
    
    SEe_real=np.zeros((nb_sign,btot)) #erreur quadratique totale entre le signal réel et le signal reconstruit réel
    SEe_theo=np.zeros((nb_sign,btot)) #erreur quadratique totale entre le signal réel et le signal reconstruit théorique
    
    w_theta_poly=[2]*(order+1)
    for sign in range(nb_sign):
    
        """
        Création d'un signal de test polynomial
        """

        theta_poly=[np.random.uniform(-0.5*w_theta_poly[k],0.5*w_theta_poly[k]) for k in range(order+1)]
      
        
        
        x_test=m.get_model_poly(t,*theta_poly)+np.random.normal(0,sigma,N)
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
            theta_poly_tilde,_=m.get_theta_poly_tilde(theta_poly_hat,bx_test,[0]*(order+1),w_theta_poly) #Quantification de theta_sin_hat sur bx bits
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
    
            code=AE.get_code_res_Antonini(coefs_r,br_test) # on encode les coeffs avec le codeur d'Antonini
            coefs_r_rec=AE.coefs_rec # coefs reconstruit su br bits
            
            r_rec_n=idct(coefs_r_rec)/2
            r_rec=r_rec_n*2**(k)
            
        
        
            x_poly_rec=x_poly_tilde+r_rec# signal rec apres les deux étages 
            e_real=x_test-x_poly_rec # erreur de reconstruction total
            SEe_real[sign][bx_test]=np.sum(e_real**2)
            

            SEeqmp_theo_,SEe_theo_=allocation_poly_bx_br.get_eqmp_e_poly(SEemp_real[sign],bx_test,br_test,w_theta_poly)# theo ou real
        
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
   

    bx_opt,br_opt=allocation_poly_bx_br.get_poly_bx_br(np.mean(SEemp_real),btot,w_theta_poly)[0:2]
    print("poly order {}, bx opt = {}, br opt = {}, btot = {} bits".format(order,bx_opt,br_opt,btot))
    
    
    
    
    
    
    
    
    
    