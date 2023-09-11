# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 08:53:36 2023

@author: coren
"""

import numpy as np
import matplotlib.pyplot as plt
from Measures import my_bin,my_inv_bin

class Quantizer: 
    """
    Classe implantant un quantificateur mid tread
    """   
    
    def __init__(self,verbose=False):
        pass
     

    def get_ind(self,x,b,w,m): 
        if b==0:
            return 0
            
        ind_max=2**(b-1)-1 # index du palier maximal
        delta=w/(2**b-1) # pas de quantification  
        

        ind=np.round((x-m)/delta)
        
        # troncature si x n'appartient pas à l'intervalle [-w/2 ,w/2]
        if ind<-ind_max:
            ind=-ind_max
        if ind>ind_max:
            ind=ind_max
    
        return ind   


    def get_q(self,ind,b,w,m): 
        if b==0:
            return m
        delta=w/(2**b-1) # pas de quantification 
        return delta*ind+m 
                   


        
            
    def get_code(self,ind,b):
        if b==0:
            return []
        
        ind_pos=2**(b-1)-1+ind # index positif à coder allant de 
        
        #print("ind_pos",ind_pos)
        
        """Convertit un nombre en binaire"""
        code=my_bin(ind_pos,b)
    
    
        return code
     
    def get_inv_code(self,code,b):
        
        if b==0:
            return 0

        ind_pos=my_inv_bin(code)
        #print("ind_pos",ind_pos)
            
        return ind_pos-2**(b-1)+1

    



# Programme principal
if __name__ == "__main__":
    b=10
    #print("b,int(b)",b,int(b))
    w=2
    m=0
    
    if b==0:
        ind_max=0
        delta=0
    else :
        ind_max=2**(b-1)-1
        delta=w/(2**b-1)
    
    
    verbose = False
    
    x=np.array([i/10+m for i in range(-int(w*10),int(w*10))])
    
    
    
    q_x=Quantizer(verbose)
    
    x_ind_q=np.zeros(len(x))
    x_q=np.zeros(len(x))
    

    for i in range(len(x)):
        x_ind_q[i]=q_x.get_ind(x[i],b,w,m)
        print("ind={}".format( x_ind_q[i]),"b={}".format(b))
        code=q_x.get_code(x_ind_q[i],b)
        print("code={}, len(code)={}".format(code,len(code)))
        ind_rec=q_x.get_inv_code(code,b)
        print("ind real={}, ind rec={}".format(x_ind_q[i],ind_rec))
        x_q[i]=q_x.get_q(ind_rec,b,w,m)
   
   
        
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(x,x_ind_q,lw=2,label='index de quantification de x')
    plt.xlabel('valeurs de x')
    plt.ylabel('index du palier')
    plt.legend()
    plt.title("b = {} bits, nombre de paliers: {}, delta = {:.2f}, intervalle: [{:.2f},{:.2f}]".format(b,2*ind_max+1,delta,-w/2,w/2))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    
    plt.figure(figsize=(8,8), dpi=100)
    plt.plot(x,x,lw=2,label='x')
    plt.plot(x,x_q,lw=2,label='coefficients quantifiés de x')
    plt.xlabel('valeurs de x')
    plt.ylabel('valeurs de x_q')
    plt.axis("equal")
    plt.legend()
    plt.title(" b = {} bits, delta = {:.2f}, intervalle  [{:.2f},{:.2f}]".format(b,delta,-w/2,w/2))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(x,x-x_q,lw=2)
    plt.xlabel('valeur de x')
    plt.ylabel('erreur de quantification')
    plt.title('erreur entre x et x_q')
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid( which='minor', color='#999999', linestyle='-', alpha=0.2) 
    plt.show()



    





"""
class Quantizer: 
    
    Classe implantant un quantificateur mid tread
      
    
    def __init__(self,b=3,w=2,m=0,verbose=False):
        #input
       
        self.b = max([1,b]) # nombre de bits
        self.w = w # dynamique
        self.m = m # moyenne
        #constant
        self.ind_max=2**(self.b-1)-1 # index du palier maximal
        self.delta=self.w/(2**self.b-1) # pas de quantification  
        
    

    def get_ind_q(self,x,b=None,w=None,m=None): 
        # fonction qui stoque l'index correspondnant au palier de quantification dans la liste samples_ind
        # et les coefficient quantifié dans la liste samples_q
        
        #mise à jours des constantes si b est entrée
        if b is not None:
            self.b = max([b,1])
            
            self.ind_max=2**(self.b-1)-1 # index du palier maximal
            self.delta=self.w/(2**self.b-1) # pas de quantification 
            
            
        #mise à jour des constantes si w est entrée    
        if w is not None:
            self.w = w
            
            self.delta=self.w/(2**self.b-1) # pas de quantification 
            
        if m is not None :
            self.m=m
        
        
            
        ind=np.round((x-self.m)/self.delta)
        
        # troncature si x n'appartient pas à l'intervalle [-w/2 ,w/2]
        if ind<-self.ind_max:
            ind=-self.ind_max
        if ind>self.ind_max:
            ind=self.ind_max
        return ind
    
    def get_q(self,ind):
        

        return self.delta*ind+self.m
        



# Programme principal
if __name__ == "__main__":
    b=4
    w=1
    m=0
    verbose = False
    
    x=np.array([i/100+m for i in range(-int(w*100),int(w*100))])
    
    
    
    q_x=Quantizer(b,w,m,verbose)
    
    x_ind_q=np.zeros(len(x))
    x_q=np.zeros(len(x))
    
    b=2
    w=1
    m=0
    for i in range(len(x)):
        x_ind_q[i]=q_x.get_ind_q(x[i],b,w,m)
        x_q[i]=q_x.get_q(x_ind_q[i])
   
        
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(x,x_ind_q,lw=2,label='index de quantification de x')
    plt.xlabel('valeurs de x')
    plt.ylabel('index du palier')
    plt.legend()
    plt.title("b = {} bits, nombre de paliers: {}, delta = {:.2f}, intervalle: [{:.2f},{:.2f}]".format(q_x.b,2*q_x.ind_max+1,q_x.delta,-q_x.w/2,q_x.w/2))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    
    plt.figure(figsize=(8,8), dpi=100)
    plt.plot(x,x,lw=2,label='x')
    plt.plot(x,x_q,lw=2,label='coefficients quantifiés de x')
    plt.xlabel('valeurs de x')
    plt.ylabel('valeurs de x_q')
    plt.axis("equal")
    plt.legend()
    plt.title(" b = {} bits, delta = {:.2f}, intervalle  [{:.2f},{:.2f}]".format(b,q_x.delta,-w/2,w/2))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(x,x-x_q,lw=2)
    plt.xlabel('valeur de x')
    plt.ylabel('erreur de quantification')
    plt.title('erreur entre x et x_q')
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid( which='minor', color='#999999', linestyle='-', alpha=0.2) 
    plt.show()


"""











"""
class Quantizer: 
   
    Classe implantant un quantificateur mid tread
      
    
    def __init__(self,b=1,w=1,m=0,verbose=False):
        
        #input
        self.b = max([1,b]) # nombre de bits
        self.w = w # dynamique
        self.m = m # moyenne
        #constant
        self.ind_max=2**(self.b-1)-1 # index du palier maximal
        self.delta=self.w/(2**self.b-1) # pas de quantification  
        self.verbose = verbose
        
        #variables
        self.samples_ind = [] # liste d'index contenant les paliers de quantificiation de la source à quantifier
        self.samples_q = [] # liste des coefficients quantifié de la source à quantifier
        
        
        if (self.verbose):
            print("Intervalle de quantification: [{:.2f},{:.2f}]".format(-self.w/2,self.w/2))
            print("Nombre de bits: b = {}".format(self.b))
            print("Pas de quantification: delta = {:.2f}".format(self.delta))
    
    
    def reset(self):
        self.samples_ind = [] # liste d'index contenant les paliers de quantificiation de la source à quantifier
        self.samples_q = [] # liste des coefficients quantifié de la source à quantifier
        

    def q(self,x,b=None,w=None,m=None): 
        # fonction qui stoque l'index correspondnant au palier de quantification dans la liste samples_ind
        # et les coefficient quantifié dans la liste samples_q
        
        #mise à jours des constantes si b et W sont entrée
        if b is not None and w is not None:
            self.b = max([b,1])
            self.w = w
            
            
            self.ind_max=2**(self.b-1)-1 # index du palier maximal
            self.delta=self.w/(2**self.b-1) # pas de quantification 
            
        if m is not None :
            self.m=m
            
            
            
        ind=np.round((x-self.m)/self.delta)
        
        # troncature si x n'appartient pas à l'intervalle [-w/2 ,w/2]
        if ind<-self.ind_max:
            ind=-self.ind_max
        if ind>self.ind_max:
            ind=self.ind_max
        
        self.samples_ind.append(ind)
        self.samples_q.append(self.delta*ind+self.m)
        
        if (self.verbose):
            print("x = {:.3f}, x_ind = {}, x_q = {:.3f}".format(x,int(self.samples_ind[-1]),self.samples_q[-1]))
    
    
    def quantization_symbols(self,samples):
        self.reset()
        for element in samples:
            self.q(element)


# Programme principal
if __name__ == "__main__":
    b=4
    w=2
    verbose = False
    
    x=np.array([i/100 for i in range(-int(w*100),int(w*100))])
    
    
    
    q_x=Quantizer(b,w,verbose)
    
    q_x.quantization_symbols(x)
        
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(x,q_x.samples_ind,lw=2,label='index de quantification de x')
    plt.xlabel('valeurs de x')
    plt.ylabel('index du palier')
    plt.legend()
    plt.title("b = {} bits, nombre de paliers: {}, delta = {:.2f}, intervalle: [{:.2f},{:.2f}]".format(q_x.b,2*q_x.ind_max+1,q_x.delta,-q_x.w/2,q_x.w/2))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    
    plt.figure(figsize=(8,8), dpi=100)
    plt.plot(x,x,lw=2,label='x')
    plt.plot(x,q_x.samples_q,lw=2,label='coefficients quantifiés de x')
    plt.xlabel('valeurs de x')
    plt.ylabel('valeurs de x_q')
    plt.axis("equal")
    plt.legend()
    plt.title(" b = {} bits, delta = {:.2f}, intervalle  [{:.2f},{:.2f}]".format(b,q_x.delta,-w/2,w/2))
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(x,x-np.array(q_x.samples_q),lw=2)
    plt.xlabel('valeur de x')
    plt.ylabel('erreur de quantification')
    plt.title('erreur entre x et x_q')
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid( which='minor', color='#999999', linestyle='-', alpha=0.2) 
    plt.show()


"""




