# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:15:34 2023

@author: coren
"""


import random

from itertools import accumulate


class Generation_Source:
    
    def __init__(self,alphabet=[],probability=[],verbose=False):
        #inputs
        self.alphabet=list(alphabet) # alphabet de la source
        self.probability=list(probability)# probabilité d'apparition des symboles
        self.verbose = verbose
        
        
        #constants
        self.size_alphabet=len(self.alphabet)
        self.cumulate_probability=list(accumulate(self.probability)) # probabilités cumulé
        
        
        #variables
        self.probability_source=[0]*self.size_alphabet
        self.source=[]
  
        
        if (self.verbose):
            print(f'alphabet: {self.alphabet}')
            print(f'probability: {self.probability}')
            print(f'cumulate probability: {self.cumulate_probability}')
   
    def reset(self):
        
        self.probability_source=[0]*self.size_alphabet
        self.source=[]
  
    
    def generate_one_symbole(self):
        r = random.random()  # Générer un nombre aléatoire entre 0 et 1
        for i in range(self.size_alphabet):
            if r < self.cumulate_probability[i]:
                return i

                
        
    def generate_source(self,size_source):
        self.source=[0]*size_source
        for n in range(size_source):
            i=self.generate_one_symbole()
            
            self.source[n]=self.alphabet[i]
            self.probability_source[i]+=1
        
        for i in range(self.size_alphabet):
            self.probability_source[i]/=size_source
            
        #self.cumulate_probability_real=list(accumulate(self.probability_real))
         

# Exemple d'utilisation : générer 20 symboles aléatoires
if __name__ == "__main__":
    verbose=False
    N=20
    alphabet = ['A', 'B', 'C']
    probability = [0.8,0.05,0.15]
    
    S=Generation_Source(alphabet,probability,verbose)
    
    S.generate_source(N)
    
    x=S.source
    
    print('Source à coder',x)
    
    for i in range(len(alphabet)):
        
        print("sym: {}, p real: {:.2f}, p source: {:.2f}".format(alphabet[i],probability[i],S.probability_source[i]))


