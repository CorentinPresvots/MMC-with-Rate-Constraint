# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:53:45 2023

@author: coren
"""


import numpy as np
from scipy.optimize import fsolve


class Allocation:
    def __init__(self,verbose=False):
        pass
        #self.max_bin_per_theta=16
        
    
        
    def round_allocation(self,L,bm):
        """
        Arrondit les éléments d'un vecteur L de flottants pour obtenir un vecteur d'entiers positifs dont la somme est égale à bm,
        en minimisant l'erreur entre L et le vecteur arrondi.
        
        Args:
            L (array-like): Vecteur de flottants à arrondir.
            bm (float): Cible de la somme du vecteur arrondi.
            verbose (bool): Afficher des informations de débogage (par défaut False).
            
        Returns:
            array: Vecteur d'entiers positifs dont la somme est égale à bm.
        """
        if self.verbose:
            print("L:", L, "Somme:", np.sum(L))
            
        # Mettre les éléments négatifs à zéro
        root = np.maximum(L, 0)
        
        # Réajuster l'allocation en fonction de la somme de root
        total_allocated = np.sum(root)
        if total_allocated != 0:
            root *= bm / total_allocated
        
        # Arrondir les éléments de root
        #root =np.round(root)#np.array([int(np.round(root[i])) for i in range(len(root))])
        root=list(map(int, np.round(root)))
        if self.verbose:
            print("root:", root, "Somme:", np.sum(root))
        
        # Vérification de si la cible bm est atteinte
        ecart = np.sum(root) - bm
        
        if ecart == 0:
            if self.verbose:
                print("Écart = 0, root:", root)
            return root
        
        if ecart > 0:  # Il faut enlever des bits pour atteindre la cible
            res = [root[i] - L[i] for i in range(len(root))]
            
            if self.verbose:
                print("Écart > 0, res:", res)
                
            while np.sum(root) != bm:
                index_max = res.index(max(res))
                if self.verbose:
                    print("Index max:", index_max)
                root[index_max] -= 1
                res[index_max] -= 1
            
            if self.verbose:
                print("Écart > 0, root:", root)
            
            return root
        
        if ecart < 0:  # Il faut ajouter des bits pour atteindre la cible
            res = [root[i] - L[i] for i in range(len(root))]
            
            if self.verbose:
                print("Écart < 0, res:", res)
                
            while np.sum(root) != bm:
                index_min = res.index(min(res))
                if self.verbose:
                    print("Index min:", index_min)
                root[index_min] += 1
                res[index_min] += 1
            
            if self.verbose:
                print("Écart < 0, root:", root)
            
            return root
     
        
 
    
    
    

class Allocation_sin(Allocation):
    def __init__(self,verbose=False):
       
        self.verbose = verbose
        super().__init__() 

    def FUNCallocation_sin(self,a,bm,m_theta_sin,w_theta_sin):
        
        return [-w_theta_sin[0]**2*(2**(-2*a[0]))*np.log2(2)/12+a[3],     
              -w_theta_sin[1]**2*(2**(-2*a[1]))*np.log2(2)*(w_theta_sin[0]**2/12+m_theta_sin[0]**2)*4*np.pi**2/(12*3*m_theta_sin[1]**2)+a[3],
              -w_theta_sin[2]**2*(2**(-2*a[2]))*np.log2(2)*(w_theta_sin[0]**2/12+m_theta_sin[0]**2)/12+a[3],
              a[0]+a[1]+a[2]-bm]       
        
        
    def get_allocation_sin(self,bm,m_theta_sin,w_theta_sin):
        
        #if bm>12*3:
        #    root=self.round_allocation([bm/3,bm/3,bm/3], bm)  
        #    return [min([self.max_bin_per_theta,root[i]]) for i in range(3)]  
        #ajouter un bout de code pour renvoyer [bm/3,bm/3,bm/3] si bm>=40
        
        
        #print("bm get allocation sin w m",bm,self.w,self.m)
        # Résolution du système d'équations avec fsolve
        root = fsolve(self.FUNCallocation_sin,[bm/3,bm/3,bm/3,1],args=(bm,m_theta_sin,w_theta_sin))#,maxfev=100
        root=root[0:3]
        #print("root",root)
        if (self.verbose):
            print("allocation sin {:.2f},{:.2f},{:.2f} pour bm: {} bits".format(root[0],root[1],root[2],bm))
            print("sum allocation sin: {:.2f}".format(np.sum(root)))
          
            
        root=self.round_allocation(root, bm)      

        if (self.verbose):
            print("allocation sin {:.0f},{:.0f},{:.0f} pour bm: {} bits".format(root[0],root[1],root[2],bm))
            print("sum allocation sin: {:.0f}".format(np.sum(root)))
                    
        
        return root   
    
class Allocation_poly(Allocation):
    def __init__(self,verbose=False):
        #inputs
        self.verbose = verbose
        

        self.c=[1,1/3,7/15,17/35,31/63,49/99,71/143,0.5]+[0.5]*64
        super().__init__() 
        
    
     
    def FUNCallocation_poly(self,a,bm,w_theta_poly):
        
        order=len(w_theta_poly)-1
        f=[0]*(order+2)
        for i in range(order+1):

            f[i]=-self.c[i]*w_theta_poly[i]**2*(2**(-2*a[i]))*np.log2(2)/order+a[order+1]     
        f[-1]= np.sum(a[0:order+1])-bm 
          
        return f     
        
        
    def get_allocation_poly(self,bm,w_theta_poly):
        
        order=len(w_theta_poly)-1
        if order==0:
            
            return [bm]
        
        #if bm>12*(order+1):
        #    root=self.round_allocation([bm/(order+1)]*(order+1), bm)  
        #    return [min([self.max_bin_per_theta,root[i]]) for i in range(order+1)]  
        
        # Résolution du système d'équations avec fsolve
        root = fsolve(self.FUNCallocation_poly,[bm/(order+1)]*(order+2),args=(bm,w_theta_poly))
        root=root[0:order+1]
        if (self.verbose):
            print("allocation poly {} pour bm: {} bits".format(root,bm))
            print("sum allocation poly: {:.2f}".format(np.sum(root)))
          
            
        root=self.round_allocation(root,bm)       

        if (self.verbose):
            print("allocation poly {} pour bm: {} bits".format(root,bm))
            print("sum allocation poly: {:.2f}".format(np.sum(root)))
                        
        
        return root


class Allocation_pred_samples(Allocation):
    def __init__(self,verbose=False):
        #inputs
        self.verbose = verbose
        

        super().__init__() 
        
    
     
        
    def get_allocation_pred_samples(self,bm,m_theta_pred_samples,w_theta_pred_samples):
        
        

        order=len(w_theta_pred_samples)
        
        if order==1:
            return [bm]
        
        root=self.round_allocation([bm/(order)]*(order), bm)  
        #A=np.array(w_theta_pred_samples)**2/12+np.array(m_theta_pred_samples)**2
        #root=self.round_allocation(bm*(A)/np.sum(A), bm)
        return root 
    






# Programme principal
if __name__ == "__main__":

 

    verbose = False
    N=128
    fn=50*2
    fs=6400
    
    bm=10 # 40 ne veux pas 
    
    m_theta_sin=[0.75,fn,0]
    w_theta_sin=[0.5,0.2,2*np.pi]
    
    allocation_sin=Allocation_sin()
    
    al_sin=allocation_sin.get_allocation_sin(bm,m_theta_sin,w_theta_sin)
    print("allocation sin {}, sum bits: {}, bm: {}".format(al_sin,np.sum(al_sin),bm))
    
    
    ################ test polynome
    
    order=6
    
    w_theta_poly=[2]*(order+1)
    allocation_poly=Allocation_poly()
    

    al_poly=allocation_poly.get_allocation_poly(bm,w_theta_poly)
    print("allocation poly d'ordre {}: {}, sum bits: {} bits, bm: {} bits".format(order,al_poly,np.sum(al_poly),bm))
    
    ################## test pred_sample
    
    order=5
    
    m_theta_pred_samples=[k for k in range(order)]#[0]*order
    w_theta_pred_samples=[1]*order
    
    allocation_pred_samples=Allocation_pred_samples()
    
    al_pred=allocation_pred_samples.get_allocation_pred_samples(bm,m_theta_pred_samples,w_theta_pred_samples)
    print("allocation pred d'ordre {}: {}, sum bits: {} bits, bm: {} bits".format(order,al_pred,np.sum(al_pred),bm))
    
    

    