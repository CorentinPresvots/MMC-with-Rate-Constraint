# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 13:34:00 2023

@author: coren
"""

from itertools import accumulate



class Aritmetic_Encoder:
    """
    Classe implantant un codeur arithmétique adaptatif
    M : precision
    source : source à coder constituée de symboles quelconques
    
    code : suite de bits codée 
    """
    
    def __init__(self,M=9,alphabet=[],initial_occurrence=[],adaptive=True,verbose=False):
        
        #input class
        self.M=M
        self.alphabet=list(alphabet) # alphabet des symboles pouvant être présent dans la source à coder
        self.initial_occurrence=list(initial_occurrence)
        self.adaptive=adaptive # booléen indiquant si le codeur est adaptatif où non
        self.verbose = verbose # booléen indiquant si il faut afficher tous les prints
        
        
        
        
        #constant
        self.size_alphabet=len(alphabet) # nombre de symboles différents potentièlement présent dans la source à coder x
        self.full=2**M
        self.half = 2**(M-1)
        self.quater = 2**(M-2)
        self.threequater = 3*self.quater 
        
        
        
        
        #variables 
        self.occurrence=list(initial_occurrence)
        self.cumulate_occurrence=list(accumulate(initial_occurrence))
        self.code = [] # sortie binaire représantant x
        self.l=0
        self.h=2**M
        self.nb_sym_encoded=0
        self.follow = 0
      
        
      
        if (self.verbose):
            print(f'Intervalle initial: [{self.l},{self.h}], follow = {self.follow}')
            print("occurrence",self.occurrence)
            print("cumulate_occurrence",self.cumulate_occurrence)
            
            
    def reset(self): 
        
        self.occurrence=list(self.initial_occurrence)
        self.cumulate_occurrence=list(accumulate(self.initial_occurrence))
        self.code = [] # sortie binaire représantant x
        self.l=0
        self.h=self.full
        self.nb_sym_encoded=0
        self.follow = 0          
        
            
    def symbol_labelise(self,symbol):
        return self.alphabet.index(symbol) 


       
    def encode_one_symbol(self,symbol):
        
        # Codage d'un symbole unique 
        
        x=self.symbol_labelise(symbol) # symbol à coder constitué d'entier allant de 0 à len(alphabet)
            
        s_hight=self.cumulate_occurrence[x]
        s_low=self.cumulate_occurrence[x]-self.occurrence[x]
        
        
        full_occurrence=self.cumulate_occurrence[-1]
        
        Range=self.h-self.l
        
        
        self.h=self.l+Range*s_hight // full_occurrence
        self.l=self.l+Range*s_low // full_occurrence
        
        if (self.adaptive):
            self.occurrence[x]+=1
        
            for i in range(x,len(self.occurrence)):
                self.cumulate_occurrence[i]+=1
                
        if (self.verbose):
            print("Codage du symbole {}: [{},{}], follow = {}, occurrence = {}, cumulate occurrence= {}".format(x,self.l,self.h,self.follow,self.occurrence,self.cumulate_occurrence))
        

        to_be_dilated = True
      
        while to_be_dilated:
            if self.h<self.half:
                self.l=2*self.l
                self.h=2*self.h
                self.code += [0]+[1]*self.follow
                self.follow=0
                to_be_dilated = True
                if (self.verbose):
                    print(f'Dilatation (basse): [{self.l},{self.h}], ajout de {[0]+[1]*self.follow} au code, follow = {self.follow}')
            elif (self.l>=self.half):
                self.l = self.l*2-self.full;
                self.h = self.h*2-self.full;
                self.code += [1]+[0]*self.follow
                self.follow=0
                to_be_dilated = True
                if (self.verbose):
                    print(f'Dilatation (haute): [{self.l},{self.h}], ajout de {[1]+[0]*self.follow} au code,  follow = {self.follow}')
            elif (self.l>=self.quater)&(self.h<self.threequater):
                self.l = self.l*2 - self.half
                self.h = self.h*2 - self.half
                self.follow = self.follow + 1
                to_be_dilated = True
                if (self.verbose):
                    print(f'Dilatation (centrale): [{self.l},{self.h}], follow = {self.follow}')
            else:
                to_be_dilated = False

        
    def finish(self):
        if (self.l<self.quater):
            self.code += [0]+[1]*(self.follow+1)
            if (self.verbose):
                print(f'Terminaison, l<2^(M-2): ajout de {[0]+[1]*(self.follow+1)} au code')
        else:
            self.code += [1]+[0]*(self.follow+1)
            if (self.verbose):
                print(f'Terminaison, l<2^(M-2): ajout de {[1]+[0]*(self.follow+1)} au code')
       
        
    #### encodage de la source x
    def encode_symbols(self,source):
        # fonction qui encode toute la source x
        

        size_source=len(source) # taille de la source à coder
                    
    
        self.reset()    
        
        
        for i in range(size_source):

            self.encode_one_symbol(source[i])    
        self.finish()
        
    def encode_symbols_under_constraint_R(self,source,R):
        # fonction qui encode les k premiers élément de la source x en respectant la contrainte de débit R
        
        self.reset()
        
        
        
        while 1:
            
            #test 
            
            occurrence_test=list(self.occurrence)
            cumulate_occurrence_test=list(self.cumulate_occurrence)
            
            code_test = list(self.code) 
            
            l_test=self.l
            h_test=self.h
            
            follow_test=self.follow       
            
            nb_sym_encoded_test=self.nb_sym_encoded          
            
            
            
            
            
            
            self.encode_one_symbol(source[self.nb_sym_encoded]) 
            self.nb_sym_encoded+=1
            
            
            if (len(self.code)+self.follow+2>R):
                
                self.occurrence=list(occurrence_test)
                self.cumulate_occurrence=list(cumulate_occurrence_test)
                
                
                self.code = list(code_test)# sortie binaire représantant x
                
                self.l=l_test
                self.h=h_test
                
                self.follow = follow_test        
                
                
                self.nb_sym_encoded=nb_sym_encoded_test
                
                break
                
        self.finish()
        





    
class Aritmetic_Decoder:
    """
    Classe implantant un decodeur arithmétique
    code : suite de bits codée
    M : precision
    """
    
    def __init__(self,M=9,alphabet=[],initial_occurrence=[],adaptive=True,verbose=False):
    
        #input
        self.M=M
        self.alphabet=list(alphabet)
        self.initial_occurrence=list(initial_occurrence)
        self.adaptive=adaptive
        self.verbose = verbose
        
        
        
        
        #Constants
        self.full = 2**M
        self.half = 2**(M-1)
        self.quater = 2**(M-2)
        self.threequater = 3*2**(M-2)  
        self.size_alphabet=len(alphabet) # nombre de symboles différents potentièlement présent dans la source à coder x
        
        
        
        
        # variables
        self.codeword = 0 # Entier associé aux M premiers bits de la chaine codée
        self.count = M

        self.occurrence=list(initial_occurrence)
        self.cumulate_occurrence=list(accumulate(self.occurrence))
        
        self.message = []
        self.l=0
        self.h=self.full



       

        
        

        if (self.verbose):
            print(f'Suite des bits codés : {self.code}')
            print(f'Intervalle initial : [{self.l},{self.h}]')
            print(f'Valeur courante à décoder : {self.codeword}')
            print(f'Occurrence : {self.occurrence}')
            print(f'Cumulate occurrence : {self.cumulate_occurrence}')
    
    
    def ini_codeword(self,code):
    
        for i in range(self.M):
            if i<len(code):
                self.codeword += code[i]*2**(M-1-i)
       
    def reset(self):
        self.codeword = 0 # Entier associé aux M premiers bits de la chaine codée
        self.count = self.M
        
        self.occurrence=list(self.initial_occurrence)
        self.cumulate_occurrence=list(accumulate(self.occurrence))
        
        self.message = []
        self.l=0
        self.h=self.full
        
        
        
        
        
    def decode_one_symbol(self,code):
    # Decodage d'un symbole unique
        
        full_occurrence=self.cumulate_occurrence[-1]
        for ind in range(self.size_alphabet):
            
            s_hight=self.cumulate_occurrence[ind]
            s_low=self.cumulate_occurrence[ind]-self.occurrence[ind]
            
            Range=self.h-self.l
            
            h0=self.l+Range*s_hight // full_occurrence
            l0=self.l+Range*s_low // full_occurrence
                
            #print("val:",self.codeword,"l:", l0, "h:", h0)
            
            if (l0<=self.codeword) & (self.codeword<h0):
                self.message.append(self.alphabet[ind])
                

                     
                self.l=l0
                self.h=h0
                
                if (self.adaptive):
                    self.occurrence[ind]+=1
                
                    for i in range(ind,self.size_alphabet):
                        self.cumulate_occurrence[i]+=1
                    
                break
            
        if (self.verbose):
            print("Décodage du symbole {}: [{},{}], occ = {}, cum occ= {}".format(self.message[-1],self.l,self.h,self.occurrence,self.cumulate_occurrence))
            
                
                
        to_be_dilated = True
      
        while to_be_dilated:
            if (self.h<self.half):
                if (self.verbose):
                    print(f'[{self.l},{self.h}[ inclus dans [0,{self.half}[')
                self.l=2*self.l
                self.h=2*self.h
                self.codeword = 2*self.codeword
                if (self.count<len(code)):
                    self.codeword += code[self.count]
                    self.count += 1
                if (self.verbose):
                    print(f'Intervalle après dilatation : [{self.l},{self.h}]')
                    if (self.count<len(code)):
                        print(f'Suite des bits restant à décoder : {code[self.count:]}')
                    print(f'Valeur courante à décoder : {self.codeword}')
                                    
                to_be_dilated = True
            elif (self.l>=self.half):
                if (self.verbose):
                    print(f'[{self.l},{self.h}[ inclus dans [{self.half},0[')
                self.l = self.l*2-self.full
                self.h = self.h*2-self.full
                self.codeword = 2*self.codeword - self.full
                if (self.count<len(code)):
                    self.codeword += code[self.count]
                    self.count += 1
                if (self.verbose):
                    print(f'Intervalle après dilatation : [{self.l},{self.h}]')
                    if self.count<len(code):
                        print(f'Suite des bits restant à décoder : {code[self.count:]}')
                    print(f'Valeur courante à décoder : {self.codeword}')

                to_be_dilated = True
            elif (self.l>=self.quater) & (self.h<self.threequater):
                if (self.verbose):
                    print(f'[{self.l},{self.h}[ inclus dans [{self.quater},{self.threequater}[')
                self.l = self.l*2 - self.half
                self.h = self.h*2 - self.half
                self.codeword = 2*self.codeword -self.half
                if (self.count<len(code)):
                    self.codeword += code[self.count]
                    self.count += 1
                if (self.verbose):
                    print(f'Intervalle après dilatation : [{self.l},{self.h}]')
                    if (self.count<len(code)):
                        print(f'Suite des bits restant à décoder : {self.code[self.count:]}')
                    print(f'Valeur courante à décoder : {self.codeword}')

                to_be_dilated = True
            else:
                to_be_dilated = False
        
        
    #### decodage de la source x
    def decode_symbols(self,code,size_source):
        # fonction qui decode size_source symboles
        
        self.reset()
        self.ini_codeword(code)
                
        for i in range(size_source):
            self.decode_one_symbol(code)    
         
               
# Programme principal

if __name__ == "__main__":
    from Source import Generation_Source
    from Measures import get_frequency,entropy
    
    M = 9
    verbose = False
    adaptive = True
    
    ### Génération d'une source alléatoire
    N=20
    alphabet = ['A', 'B', 'C']
    probability = [0.6,0.1,0.3]
    initial_occurrence = [1]*len(alphabet)
    
    S=Generation_Source(alphabet,probability)
    
    S.generate_source(N)
    
    x=S.source
    
    
    for i in range(len(alphabet)):
        print("sym: {}, p: {:.2f}, p source: {:.2f}".format(alphabet[i],probability[i],S.probability_source[i]))
    
   
    
    ### Codage de la source sans contrainte
    
    
    
    alphabet_source,probability_source=get_frequency(x)
    #print("alphabet_source",alphabet_source)
    #print("probability_source",probability_source)   
    
    H=entropy(probability_source)
    print("entropy source: {:.2f}".format(H))
    
    
    
    

    
    print("\n\nCodeur sans contrainte")
 
    ac = Aritmetic_Encoder(M,alphabet,initial_occurrence,adaptive,verbose)
    
    ac.encode_symbols(x)
    
    print('Longueur du code: ',len(ac.code),' bits,', 'Entropie: {:.2f} bits'.format(H*N))
    
    
    
    print("\n\nDecodeur sans contrainte")
    
    
    ad = Aritmetic_Decoder(M,alphabet,initial_occurrence,adaptive,verbose)
    
    ad.decode_symbols(ac.code,N)
    
    print('Occurrence codeur',ac.occurrence)
    print('Occurrence décodeur',ad.occurrence)
    
    print('Source à coder',x)
    print('Source décodé ',ad.message)




    ### Codage de la source avec contrainte
    
    R= len(ac.code)-5 # contrainte de débit
    print("\n\nCodeur avec contrainte")
 
    ac = Aritmetic_Encoder(M,alphabet,initial_occurrence,adaptive,verbose)
    

    ac.encode_symbols_under_constraint_R(x,R)
   
    
    alphabet_source,probability_source=get_frequency(x[0:ac.nb_sym_encoded])
    #print("alphabet_source",alphabet_source)
    #print("probability_source",probability_source)   
    
    H=entropy(probability_source)
    print("entropy source: {:.2f}".format(H))
    
    
    
    
    
    
    
    
    print("nombre de symboles codé {} sur {}".format(ac.nb_sym_encoded,N))
    print('Longueur du code: ',len(ac.code),' bits,', 'contrainte: ', R,' bits,', 'Entropie: {:.2f} bits'.format(H*ac.nb_sym_encoded))
    
    
    
    print("\n\nDecodeur avec contrainte")
    
    
    ad = Aritmetic_Decoder(M,alphabet,initial_occurrence,adaptive,verbose)
    
    ad.decode_symbols(ac.code,ac.nb_sym_encoded)
    
    print('Occurrence codeur',ac.occurrence)
    print('Occurrence décodeur',ad.occurrence)
    
    print('Source à coder',x[0:ac.nb_sym_encoded])
    print('Source décodée',ad.message)
