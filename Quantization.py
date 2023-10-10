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
    b=3
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
        #print("ind={}".format( x_ind_q[i]),"b={}".format(b))
        code=q_x.get_code(x_ind_q[i],b)
        #print("code={}, len(code)={}".format(code,len(code)))
        ind_rec=q_x.get_inv_code(code,b)
        #print("ind real={}, ind rec={}".format(x_ind_q[i],ind_rec))
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



    




