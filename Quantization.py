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
        if b<1.:
            return 0
            
        ind_max=2**(b-1)-1 # index du palier maximal
        delta=w/(2**b-1) # pas de quantification  
        
        #print(delta,x-m,b)
        ind=np.round((x-m)/delta)
        #"""
        # troncature si x n'appartient pas à l'intervalle [-w/2 ,w/2]
        
        """
        if ind<-ind_max:
            #print(x,m,w)
            #print("!!!!!!!!!!!!!!!!!!!!!!!!!")
            ind=-np.round(((w-m)/2)**2/ind_max)
        if ind>ind_max:
            #print(x,m,w)
            #print("!!!!!!!!!!!!!!!!!!!!!!!!!")
            ind=-np.round(((w-m)/2)**2/ind_max)
        """
        return ind   


    def get_q(self,ind,b,w,m): 
        if b<1.:
            return m
        delta=w/(2**b-1) # pas de quantification 
        return delta*ind+m 
                   


        
            
    def get_code(self,ind,b):
        if b<1.:
            return []
        
        ind_pos=2**(b-1)-1+ind # index positif à coder allant de 
        
        #print("ind_pos",ind_pos)
        
        """Convertit un nombre en binaire"""
        code=my_bin(ind_pos,b)
    
    
        return code
     
    def get_inv_code(self,code,b):
        
        if b<1:
            return 0

        ind_pos=my_inv_bin(code)
        #print("ind_pos",ind_pos)
            
        return ind_pos-2**(b-1)+1

    



    #################### uniform



    def get_ind_u(self,x,b,w,m): 
        if b==0:
            return 0
            

        delta=w/(2**b) # pas de quantification  
        

        ind=np.floor((x-m)/delta)
        
        
        #"""
        ind=min(ind,round(w/(2*delta))-1)
        ind=max(ind,round(-w/(2*delta)))
        #"""
        
        """
        # troncature si x n'appartient pas à l'intervalle [-w/2 ,w/2]
     
        if ind>round(w/(2*delta))-1:
            print(x,m,w,ind,round(w/(2*delta))-1,b)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!")
            ind=round(w/(2*delta))-1
        if ind<round(-w/(2*delta)):
            print(x,m,w,ind,round(-w/(2*delta)))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!")
            ind=round(-w/(2*delta))
        """
        
        
        return ind   


    def get_q_u(self,ind,b,w,m): 
        if b==0:
            return m
        delta=w/(2**b) # pas de quantification 
        return delta*(ind+0.5)+m 
                   


        
            
    def get_code_u(self,ind,b):
        if b==0:
            return []
        
        #print("ind",ind)
        ind_pos=2**(b-1)+ind # index positif à coder allant de 
        
        #print("ind_pos enc",ind_pos)
        
        """Convertit un nombre en binaire"""
        code=my_bin(ind_pos,b)
    
    
        return code
     
    def get_inv_code_u(self,code,b):
        
        if b==0:
            return 0

        ind_pos=my_inv_bin(code)
        #print("ind_pos dec",ind_pos)
        
        #print("ind_dec",ind_pos-2**(b-1))
            
        return ind_pos-2**(b-1)

# Programme principal
if __name__ == "__main__":
    b=3
    #print("b,int(b)",b,int(b))
    w=2
    m=1
    
    if b==0:
        ind_max=0
        delta=0
    else :
        ind_max=2**(b-1)-1
        delta=w/(2**b-1)
    
    
    verbose = False
    
    x=np.array([i/100+m for i in range(-int(w*120),int(w*120))])
    
    
    
    q_x=Quantizer(verbose)
    
    x_ind_q=np.zeros(len(x))
    x_q=np.zeros(len(x))
    

    ############## Q Uniform 
    
    delta=w/(2**b-1)
    for i in range(len(x)):
        x_ind_q[i]=q_x.get_ind(x[i],b,w,m)
        #print("ind={}".format( x_ind_q[i]),"b={}".format(b))
        code=q_x.get_code(x_ind_q[i],b)
        #print("code={}, len(code)={}".format(code,len(code)))
        ind_rec=q_x.get_inv_code(code,b)
        #print("ind real={}, ind rec={}".format(x_ind_q[i],ind_rec))
        x_q[i]=q_x.get_q(ind_rec,b,w,m)
        x_q[i]=q_x.get_q(x_ind_q[i],b,w,m)
   
   
        
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(x,x_ind_q,lw=2,label='index de quantification de x')
    plt.xlabel('valeurs de x')
    plt.ylabel('index du palier')
    plt.legend()
    plt.title("b = {} bits, nombre de paliers: {}, delta = {:.2f}, intervalle: [{:.2f},{:.2f}]".format(b,2**b-1,delta,-w/2,w/2))
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


    ############## Q Uniform 
    
    delta=w/(2**b)
    for i in range(len(x)):
        x_ind_q[i]=q_x.get_ind_u(x[i],b,w,m)
        print("ind={}".format( x_ind_q[i]),"b={}".format(b))
        code=q_x.get_code_u(x_ind_q[i],b)
        print("code={}, len(code)={}".format(code,len(code)))
        ind_rec=q_x.get_inv_code_u(code,b)
        print("ind real={}, ind rec={}".format(x_ind_q[i],ind_rec))
        x_q[i]=q_x.get_q_u(ind_rec,b,w,m)
        x_q[i]=q_x.get_q_u(x_ind_q[i],b,w,m)
   
   
        
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(x,x_ind_q,lw=2,label='index de quantification de x')
    plt.xlabel('valeurs de x')
    plt.ylabel('index du palier')
    plt.legend()
    plt.title("b = {} bits, nombre de paliers: {}, delta = {:.2f}, intervalle: [{:.2f},{:.2f}]".format(b,2**b+1,delta,-w/2,w/2))
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
    




