# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:20:37 2023

@author: coren
"""

class Context_Aritmetic_Encoder:
    """
    Classe codant un codeur arithmétique de context 
    M : precision

    code : suite de bits codée 
    """
    
    def __init__(self,M=9,verbose=False):
        
        #input class
        self.M=M
        self.verbose = verbose # booléen indiquant si il faut afficher tous les prints
        
 
        #constant
        self.full=2**M
        self.half = 2**(M-1)
        self.quater = 2**(M-2)
        self.threequater = 3*self.quater 
        
        #variables 
        self.l=0
        self.h=2**M
        self.follow = 0
        
  
    def reset_Context_Aritmetic_Encoder(self): 
        
        self.l=0
        self.h=self.full
        self.follow = 0          
        
            
    def encode_one_symbol(self,x,occurrence,cumulate_occurrence):
        
        code=[]
        # Codage d'un symbole unique 

        s_hight=cumulate_occurrence[x]
        s_low=cumulate_occurrence[x]-occurrence[x]
        
        
        full_occurrence=cumulate_occurrence[-1]
        
        Range=self.h-self.l
        
        
        self.h=self.l+Range*s_hight // full_occurrence
        self.l=self.l+Range*s_low // full_occurrence
        
 
        
        to_be_dilated = True
      
        while to_be_dilated:
            if self.h<self.half:
                self.l=2*self.l
                self.h=2*self.h
                code += [0]+[1]*self.follow
                self.follow=0
                to_be_dilated = True
                if (self.verbose):
                    print(f'Dilatation (basse): [{self.l},{self.h}], ajout de {[0]+[1]*self.follow} au code, follow = {self.follow}')
            elif (self.l>=self.half):
                self.l = self.l*2-self.full;
                self.h = self.h*2-self.full;
                code += [1]+[0]*self.follow
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
        return code
            

        
    def finish(self,l,follow):
        code=[]
        if (l<self.quater):
            code += [0]+[1]*(follow+1)
            if (self.verbose):
                print(f'Terminaison, l<2^(M-2): ajout de {[0]+[1]*(follow+1)} au code')
        else:
            code += [1]+[0]*(follow+1)
            if (self.verbose):
                print(f'Terminaison, l<2^(M-2): ajout de {[1]+[0]*(follow+1)} au code')
       
        return code

        

  
class Context_Aritmetic_Decoder:
    """
    Classe décodant un decodeur de context arithmétique
    code : suite de bits coder
    M : precision
    """
    
    def __init__(self,M=9,verbose=False):
    
        #input
        self.M=M
        self.verbose = verbose
        
    
        #Constants
        self.full = 2**M
        self.half = 2**(M-1)
        self.quater = 2**(M-2)
        self.threequater = 3*2**(M-2)  

         
        
        # variables
        self.codeword = 0 # Entier associé aux M premiers bits de la chaine codée
        self.count = M
        self.message = []
        self.l=0
        self.h=self.full

    
    def ini_codeword(self,code):
    
        for i in range(self.M):
            if i<len(code):
                self.codeword += code[i]*2**(self.M-1-i)
       
    def reset_Context_Aritmetic_Decoder(self):
        self.codeword = 0 # Entier associé aux M premiers bits de la chaine codée
        self.count = self.M
        
        self.message = []
        self.l=0
        self.h=self.full
        
        

    def decode_one_symbol(self,code,alphabet,occurrence,cumulate_occurrence):
    # Decodage d'un symbole unique
        
        full_occurrence=cumulate_occurrence[-1]
        for ind in range(len(alphabet)):
            
            s_hight=cumulate_occurrence[ind]
            s_low=cumulate_occurrence[ind]-occurrence[ind]
            
            Range=self.h-self.l
            
            h0=self.l+Range*s_hight // full_occurrence
            l0=self.l+Range*s_low // full_occurrence
                
            #print("val:",self.codeword,"l:", l0, "h:", h0)
            
            if (l0<=self.codeword) & (self.codeword<h0):
                self.message.append(alphabet[ind])
                
                self.l=l0
                self.h=h0
                
                break
            

        
                
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


        return self.message[-1]  
    
