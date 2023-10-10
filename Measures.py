# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:20:49 2023

@author: coren
"""

import numpy as np
from collections import Counter
from subsampling import dynamic_subsample

def entropy(p):
    return -np.sum([p[i]*np.log2(p[i]+10**(-8)) for i in range(len(p))])

def get_frequency(liste_symboles):
    size_liste_symbole=len(liste_symboles)
    compteur = Counter(liste_symboles)
    alphabet = list(compteur.keys())
    frequency = [compteur[symbole]/size_liste_symbole for symbole in alphabet]
    return alphabet, frequency
        
def get_snr(signal, signal_rec):
    # Calculer la puissance du signal
    
    signal_power = np.sum(np.square(signal))
    
    # Calculer la puissance du bruit
    noise_power = np.sum(np.square(signal-signal_rec))
    
    # Calculer le SNR en décibels (dB)
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    return snr_db
    
def get_snr_l1(signal, signal_rec):
    # Calculer la summ des valeurs absolues du signal
    
    signal_power = np.sum(np.abs(signal))
    
    # Calculer de la somme des valeurs absolue du bruit
    noise_power = np.sum(np.abs(signal-signal_rec))
    
    # Calculer le ? en décibels (dB)
    snr_l1_db = 20 * np.log10(signal_power / noise_power)
    return snr_l1_db

    
def my_bin(ind,b):
    """
    

    Parameters
    ----------
    ind : int
        entié à représenter en binaire
    b : int
        Nombre de bits permettant de representer ind

    Returns
    -------
    code : liste
        liste de 1 et de 0 allant des poids faible au poids forts

    """
    
    q = -1
    code = [0]*b
    
    i=0
    while i <b:
        q = ind // 2
        r = ind % 2
        code[i]=int(r)
        ind = q
        i+=1    
    return code

def my_inv_bin(code):
    """

    Parameters
    ----------
    code : liste
       liste de 1 et de 0 allant des poids faible au poids forts

    Returns
    -------
    val : int
        valeurs du mot de code binaire code en base 10

    """

    ind_pos=0.
    for k in range(len(code)):
        ind_pos+=code[k]*2**k
    
    return ind_pos
    







def encode_variable_length(ind):
    """

    Parameters
    ----------
    ind : int 
       Entier que l'on cherche à représenter en binaire'

    Returns
    -------
    code : int
       code est constitué de : ceil(log2(ind))+(ind en base 2) bits 

    """

    
    if ind < 0:
        raise ValueError("L'entier doit être positif.")
    
    print("ind",ind)
    len_ind=int(np.ceil(np.log2(ind+10**(-8))))
    print("longueur binaire de ind",len_ind)
    
    
    code_ind=my_bin(ind,len_ind)
    print("code servant à représenter ind",code_ind)
    
    
    code=[1]*(len_ind-1)+[0]+code_ind

    return code

def decode_variable_length(code):
    """

    Parameters
    ----------
    code: list int 
       
    Returns
    -------
    ind : int
       fonction inverse de la fonction encode_variable_length

    """
    
    len_ind=1
    i=0
    while code[i]!=0:
        len_ind+=1
        i+=1
    print("longueur binaire du nombre de bits servant à décoder ind",len_ind)  
    
    ### décodage de len_ind
    ind=my_inv_bin(code[i+1:i+1+len_ind])
    print("ind",ind)
    
    

        
    return ind



def curve_tex(t,x,threshold):
    if threshold>0:
        ind= dynamic_subsample(x,threshold)
         
        t_d=t[ind]
        x_d=x[ind]
    else :
        t_d=t
        x_d=x
     
    for k in range(len(x_d)):
        print("({},{}) ".format(np.round(10000*t_d[k])/10000,np.round(100*x_d[k])/100), end='') 
    print("ok")    
    print("ok") 
    print("nb element={}/{}".format(len(x_d),len(x)))


        
        
if __name__ == "__main__":
    
    
    from Source import Generation_Source
    
    
    ### Génération d'une source alléatoire
    N=20
    alphabet = ['AA', 'BB', 'CC']
    probability = [0.6,0.1,0.3]
   
    
    
    S=Generation_Source(alphabet,probability)
    
    S.generate_source(N)
    
    x=S.source
    

    for i in range(len(alphabet)):
        print("sym: {}, p real: {:.2f}, p source: {:.2f}".format(alphabet[i],probability[i],S.probability_source[i]))
    
    alphabet_source,probability_source=get_frequency(x)
    print("alphabet_source",alphabet_source)
    print("probability_source",probability_source)   
    
    H=entropy(probability_source)
    print("entropy source: {:.2f}".format(H))
    
    
    print("my_bin",my_bin(40,150),"my_inv_bin",my_inv_bin(my_bin(40,150)))
    
    
    ########## bin et inv bin
    b=100
    ind=5.764607523034235e+18
    a=my_bin(ind, b)
    print("a", a,len(a))
    ind_rec=my_inv_bin(a)
    print("ind_rec",ind_rec)
    
    
    
    
    
    ##### codage à longueure variable :
        
    # Exemple d'utilisation
    num_to_encode = 50
    encoded = encode_variable_length(num_to_encode)
    decoded = decode_variable_length(encoded)
    
    print(f"Entier original : {num_to_encode}")
    print(f"Train binaire encodé : {encoded}")
    print(f"Entier décodé : {decoded}")
            
     
