# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 19:10:30 2023

@author: coren
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate
from Measures import my_bin,my_inv_bin

from Context_Arithmetic import Context_Aritmetic_Encoder,Context_Aritmetic_Decoder
 

class Antonini_Encoder(Context_Aritmetic_Encoder):
    def __init__(self,M=9,initial_occurrence_first=[1,1],
                 initial_occurrence_second=[1,1,1,1,1],
                 adaptive=True,verbose_AE=False):
        # Constructeur de la classe Antonini_Encoder
        
        # Initialisation des paramètres spécifiques à Antonini_Encoder
        
        self.initial_occurrence_first = initial_occurrence_first # Probabilité initiale équivalente pour la première passe.
        self.initial_occurrence_second = initial_occurrence_second  # Probabilité initiale équivalente pour la seconde passe.
        self.verbose_AE = verbose_AE # Mode débogage pour Antonini_Encoder.
        self.adaptive = adaptive # 0: le codeur arithmétique n'est pas adaptatif, 1: le codeur arithmétique est adaptatif.
        
        
        
        # ... Autres constantes, initialisations, etc. ...
        self.alphabet_first = ['R', 'S'] # Alphabet de la première passe : 'R' (0 non significatif), 'S' (1 significatif).
        self.alphabet_second = ['0', '1', '-', '+', 'E']  # Alphabet de la seconde passe : '0' (0), '1' (1), '+' (longueur de séquence), '-' (polarité), 'E' (fin).
        
        # Initialisation du codeur arithmétique de contexte adaptatif en appelant le constructeur de la classe parente.
        super().__init__(M)
        
        # Sauvegarde des valeurs courantes de l'état du codeur arithmétique.
        self.memoire_l = self.l
        self.memoire_h = self.h
        self.memoire_follow = self.follow
         

    
    def reset_Antonini_Encoder(self,coefs,br):
        # Réinitialisation des paramètres pour commencer un nouvel encodage
        
        self.reset_Context_Aritmetic_Encoder()  # Réinitialisation des variables de la classe Context_Aritmetic_Encoder.
        self.N = len(coefs)  # Détermination de la longueur du vecteur à coder.
        self.code = [] # Initialisation de la suite binaire qui sera retournée en tant que résultat du codage.
        self.flag = np.zeros(self.N) # Drapeau indiquant si un coefficient a déjà été considéré significatif. 0 : jamais significatif, 1 : au moins une fois significatif.
        self.coefs = list(coefs) # Initialisation de la liste des coefficients à coder.
        self.res = list(self.coefs) # Initialisation du résidu qui sera mis à jour après le codage d'un coefficient significatif.
        self.coefs_rec = np.zeros(self.N) # Initialisation des coefficients reconstruits.
        self.threshold = 0.5 # Initialisation du seuil à 0.5. # Étant donné que les coefficients sont compris entre -1 et 1, le seuil est initialisé à 0.5.
        self.symbol = []  # Liste pour stocker les symboles codés.
        self.occurrence_first = list(self.initial_occurrence_first)  # Initialisation de l'occurrence pour la première passe, qui peut être adaptée au fur et à mesure du codage.
        self.occurrence_second = list(self.initial_occurrence_second)  # Initialisation de l'occurrence pour la seconde passe, qui peut être adaptée au fur et à mesure du codage.
        self.cumulate_occurrence_first = list(accumulate(self.initial_occurrence_first)) # Calcul des cumuls des occurrences pour la première passe et la seconde passe.
        self.cumulate_occurrence_second = list(accumulate(self.initial_occurrence_second))
        self.nb_coefs = 0  # Nombre de coefficients significatif codé
        self.br = br # Contrainte de débit.
        self.occurrence_first_true = list(self.initial_occurrence_first) # Occurrence réelle d'apparition des symboles pour la première passe.
        self.occurrence_second_true = list(self.initial_occurrence_second) # Occurrence réelle d'apparition des symboles pour la seconde passe.
    
    
        ###### ici sont définis le nombre de coefficients maximale codable dépendnant de br
        nb_coefs_max=br/2
        self.nb_bits_coefs_max=int(np.ceil(np.log2(max([1,nb_coefs_max]))))
        self.nb_coefs_max=2**self.nb_bits_coefs_max-1
        #print("Antonini, br={}, nb_bits_coefs_max={}, nb_coefs_max={}".format(br,self.nb_bits_coefs_max,self.nb_coefs_max))
        #print("br",br)
        #print("self.nb_coefs_max",self.nb_coefs_max)
        #print("self.nb_bits_coefs_max",self.nb_bits_coefs_max)
        
       
        
        
    def get_symbol_first(self):
        """
        Fonction qui encode les coefficients de la première passe pour le plan de bits courant
        

        Returns 1 ou 0 si la longueur du mot de code servant à coder le résidu devient suppérieur à la contrainte br
        -------
        int
            DESCRIPTION.

        """
  
        for i in range(self.N):# on balaye tous les flags
            if self.flag[i]!=0:
                if self.res[i]>=0:
                       
                    x=1 ## corespond à l'indice de "S"  dans le dictionnaire dans le dictionnaire de la première passe
                    #print("x S",x,"occurrence_first",self.occurrence_first,"cumulate_occurrence_first",self.cumulate_occurrence_first)
                    
                    code_first=self.encode_one_symbol(x,self.occurrence_first,self.cumulate_occurrence_first)
                    
                    if len(self.code)+len(code_first)+self.follow+2<=self.br-self.nb_bits_coefs_max and self.nb_coefs+1<=self.nb_coefs_max :
                        #### mise à jours des variables 
                        self.symbol.append("S") # on ajoute le symbole à la liste
                        self.coefs_rec[i]+=self.threshold/2# mise à jour du coef rec
                        self.res[i]-=self.threshold/2
                        self.code.extend(code_first)
                        
                        if self.adaptive:
                            self.occurrence_first[x]+=1 # mise à jours du dictionnaire si la contrainte de débit est respécté
                            #print("self.cumulate_occurrence_first av S",self.cumulate_occurrence_first)
                            self.cumulate_occurrence_first[x]+=1 # mise à jour de cumulate occurrence
                            #print("self.cumulate_occurrence_first ap S",self.cumulate_occurrence_first)
                        
                        self.occurrence_first_true[x]+=1# mise à jours du dictionnaire qui s'incrémente si la contrainte de débit est respécté
                            
                        self.nb_coefs+=1
                        
                        
                        
                        
                        
                        self.memoire_l=self.l
                        self.memoire_h=self.h
                        self.memoire_follow=self.follow  
                        
                        
                        if False:
                            print("first pass +")
                            plt.figure(figsize=(8,4), dpi=100)
                            plt.plot(np.array(self.coefs),lw=2,label='coefs')
                            plt.plot(np.array(self.coefs_rec),lw=2,label='coefs+1')
                            plt.xlabel('ind')
                            plt.ylabel('Amplitude')
                            plt.legend()
                            plt.grid( which='major', color='#666666', linestyle='-')
                            plt.minorticks_on()
                            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                            plt.show()  
                            
                            
                            
                            x_test=idct(np.array(self.coefs))/2
        
                            x_rec_test=idct(np.array(self.coefs_rec))/2
                            
                            
                          
                            plt.figure(figsize=(8,4), dpi=100)
                            plt.plot(x_test,lw=2,label='x')
                            plt.plot(x_rec_test,lw=2,label='x_rec SNR={:.2f}'.format(get_snr(x_test,x_rec_test)))
                            plt.xlabel('ind')
                            plt.ylabel('Amplitude')
                            plt.legend()
                            plt.grid( which='major', color='#666666', linestyle='-')
                            plt.minorticks_on()
                            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                            plt.show()  
                            
                            #print("res")
                            
                            #curve_tex([k for k in range(self.N)],np.array(self.coefs)*2,0)
                            
                            #print("res_rec")
                            #curve_tex([k for k in range(self.N)],np.array(self.coefs_rec)*2,0)                           
                          
                            #print("x")
                            #curve_tex([k for k in range(self.N)],x_test,0)
                            
                            #print("x_rec")
                            #curve_tex([k for k in range(self.N)],x_rec_test,0)
                                                   
                        
                        
                        
                    else :
                        return 0
                        
                else :
                    x=0 # corresond à l'indice "R" dans le dictionaire de la première passe
                    #print("x R",x,"occurrence_first",self.occurrence_first,"cumulate_occurrence_first",self.cumulate_occurrence_first)
                    code_first=self.encode_one_symbol(x,self.occurrence_first,self.cumulate_occurrence_first)
                    #print("x",x,"occurrence_first",self.occurrence_first,"cumulate_occurrence_first",self.cumulate_occurrence_first)
                    
                    if len(self.code)+len(code_first)+self.follow+2<=self.br-self.nb_bits_coefs_max and self.nb_coefs+1<=self.nb_coefs_max :
                        #### mise à jours des variables 
                        self.symbol.append("R") # on ajoute le symbole à la liste
                        self.coefs_rec[i]-=self.threshold/2# mise à jour du coef rec
                        self.res[i]+=self.threshold/2
                        self.code.extend(code_first)
                        
                        
                        if self.adaptive:
                            self.occurrence_first[x]+=1 # mise à jours du dictionnaire si la contrainte de débit est respécté
                            self.cumulate_occurrence_first[0]+=1 # mise à jour de cumulate occurrence
                            self.cumulate_occurrence_first[1]+=1 # mise à jour de cumulate occurrence
                            
                        #print("self.cumulate_occurrence_first ap R",self.cumulate_occurrence_first)
                        self.occurrence_first_true[x]+=1# mise à jours du dictionnaire qui s'incrémente si la contrainte de débit est respécté
                        
                        
                        self.nb_coefs+=1
                        
                        self.memoire_l=self.l
                        self.memoire_h=self.h
                        self.memoire_follow=self.follow  
                        
                        
                        if False:
                            print("first pass -")
                            plt.figure(figsize=(8,4), dpi=100)
                            plt.plot(np.array(self.coefs),lw=2,label='coefs')
                            plt.plot(np.array(self.coefs_rec),lw=2,label='coefs+1')
                            plt.xlabel('ind')
                            plt.ylabel('Amplitude')
                            plt.legend()
                            plt.grid( which='major', color='#666666', linestyle='-')
                            plt.minorticks_on()
                            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                            plt.show()  
                            
                            
                       
                            x_test=idct(np.array(self.coefs))/2
        
                            x_rec_test=idct(np.array(self.coefs_rec))/2
                            
                            
                          
                            plt.figure(figsize=(8,4), dpi=100)
                            plt.plot(x_test,lw=2,label='x')
                            plt.plot(x_rec_test,lw=2,label='x_rec SNR={:.2f}'.format(get_snr(x_test,x_rec_test)))
                            plt.xlabel('ind')
                            plt.ylabel('Amplitude')
                            plt.legend()
                            plt.grid( which='major', color='#666666', linestyle='-')
                            plt.minorticks_on()
                            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                            plt.show()  
                            
                            #print("res")
                            
                            #curve_tex([k for k in range(self.N)],np.array(self.coefs)*2,0)
                            
                            #print("res_rec")
                            #curve_tex([k for k in range(self.N)],np.array(self.coefs_rec)*2,0)                           
                          
                            #print("x")
                            #curve_tex([k for k in range(self.N)],x_test,0)
                            
                            #print("x_rec")
                            #curve_tex([k for k in range(self.N)],x_rec_test,0)
                            
                    else :
                        return 0

        
        return 1    # le plan de bits a entièrement été codé  
        
        
    
    def get_symbol_second(self):
        """
        Fonction qui encode les coefficients de la seconde passe pour le plan de bits courant
        

        Returns 1 ou 0 si la longueur du mot de code servant à coder le résidu devient suppérieur à la contrainte br
        -------
        int
            DESCRIPTION.

        """
        
        count=0 #compteur de "R" pour run length
        #test_symbol_second=[]
        #coefs_rec_second=np.zeros(self.N)
        for i in range(self.N):# on balaye tous les flags
            if self.flag[i]==0 :
                if np.abs(self.res[i])>=self.threshold: # le symbole est significatif
                    
                    # encodage des "R avec run length
                    sym_R=[]
                    code_R=[]
                    occurrence_second=list(self.occurrence_second)
                    occurrence_second_true=list(self.occurrence_second_true)
                    cumulate_occurrence_second=list(self.cumulate_occurrence_second)
                    if count > 0:
                        count_bin=list(bin(count-1)[2:])
                        
                    
                        sym_R=count_bin[::-1]
                        #print("sym_R",sym_R,"count",count)
                        for element in sym_R:
                            x=int(element)
                            #print("x run length",x)
                            code_R_=self.encode_one_symbol(x,occurrence_second,cumulate_occurrence_second)
                            
                            if self.adaptive:
                                occurrence_second[x]+=1

                                for xx in range(x,5):
                                    cumulate_occurrence_second[xx]+=1
                                    
                            #print("cumulate_occurrence_second ap",cumulate_occurrence_second)
                            occurrence_second_true[x]+=1
                            code_R.extend(code_R_)
                        count = 0
                
                    
                    x=int(2+(np.sign(self.res[i])+1)/2) ## corespond à l'indice de "-": 2 ou "+" : 3  dans le dictionnaire de la seconde passe
                    #print("x",x)
                    
                    code_second=self.encode_one_symbol(x,occurrence_second,cumulate_occurrence_second)
                    
                    if len(self.code)+len(code_R)+len(code_second)+self.follow+2<=self.br-self.nb_bits_coefs_max and self.nb_coefs+1<=self.nb_coefs_max :
                        #### mise à jours des variables 
                        self.flag[i]=np.sign(self.res[i]) 
                        self.symbol.extend(sym_R) # on ajoute le symbole à la liste indiquant le nombre de zero à coder avant le prochain symbole significatif
                        if self.flag[i]>0:
                            self.symbol.append("+") # on ajoute le symbole à la liste
                        else :
                            self.symbol.append("-") # on ajoute le symbole à la liste
                        self.coefs_rec[i]+=self.flag[i]*(self.threshold+self.threshold/2)# mise à jour du coef rec
                        self.res[i]-=self.flag[i]*(self.threshold+self.threshold/2)
                        
                        self.code.extend(code_R+code_second)
                        
                        if False:
                            print("second pass")
                            plt.figure(figsize=(8,4), dpi=100)
                            plt.plot(np.array(self.coefs),lw=2,label='coefs')
                            plt.plot(np.array(self.coefs_rec),lw=2,label='coefs+1')
                            plt.xlabel('ind')
                            plt.ylabel('Amplitude')
                            plt.legend()
                            plt.grid( which='major', color='#666666', linestyle='-')
                            plt.minorticks_on()
                            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                            plt.show()  
                            
                            
                            x_test=idct(np.array(self.coefs))/2
        
                            x_rec_test=idct(np.array(self.coefs_rec))/2
                            
                            
                          
                            plt.figure(figsize=(8,4), dpi=100)
                            plt.plot(x_test,lw=2,label='x')
                            plt.plot(x_rec_test,lw=2,label='x_rec SNR={:.2f}'.format(get_snr(x_test,x_rec_test)))
                            plt.xlabel('ind')
                            plt.ylabel('Amplitude')
                            plt.legend()
                            plt.grid( which='major', color='#666666', linestyle='-')
                            plt.minorticks_on()
                            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
                            plt.show()  
                            
                            #print("res")
                            
                            #curve_tex([k for k in range(self.N)],np.array(self.coefs)*2,0)
                            
                            #print("res_rec")
                            #curve_tex([k for k in range(self.N)],np.array(self.coefs_rec)*2,0)                           
                          
                            #print("x")
                            #curve_tex([k for k in range(self.N)],x_test,0)
                            
                            #print("x_rec")
                            #curve_tex([k for k in range(self.N)],x_rec_test,0)
                            
                            
                       
              
                        self.occurrence_second=occurrence_second
                        self.cumulate_occurrence_second=cumulate_occurrence_second
                        self.occurrence_second_true=occurrence_second_true# mise à jours du dictionnaire qui s'incrémente si la contrainte de débit est respécté
                    
                        if self.adaptive:
                            self.occurrence_second[x]+=1 # mise à jours du dictionnaire si la contrainte de débit est respécté
                        
                            for xx in range(x,5):
                                self.cumulate_occurrence_second[xx]+=1 # mise à jour de cumulate occurrence
                            
                        self.occurrence_second_true[x]+=1# mise à jours du dictionnaire qui s'incrémente si la contrainte de débit est respécté
                        
                        self.nb_coefs+=1
                        
                        self.memoire_l=self.l
                        self.memoire_h=self.h
                        self.memoire_follow=self.follow  
                    else :
                        return 0                    
                    
                    
                else : 
                    count+=1
                    #test_symbol_second.append('R')
        
        
        if count>0:
            x=4 #index correspondant à "E"
            code_second=self.encode_one_symbol(x,self.occurrence_second,self.cumulate_occurrence_second)
            if len(self.code)+len(code_second)+self.follow+2<=self.br-self.nb_bits_coefs_max and self.nb_coefs+1<=self.nb_coefs_max :
                #### mise à jours des variables 
 
                self.symbol.extend("E") # on ajoute le symbole à la liste indiquant le nombre de zero à coder avant le prochain symbole significatif
                
               
                self.code.extend(code_second)
                
                if self.adaptive:
                    self.occurrence_second[x]+=1
                    self.cumulate_occurrence_second[x]+=1
                self.occurrence_second_true[x]+=1# mise à jours du dictionnaire qui s'incrémente si la contrainte de débit est respécté
            
                self.nb_coefs+=1
                
                self.memoire_l=self.l
                self.memoire_h=self.h
                self.memoire_follow=self.follow  
            else :
                return 0 
        self.threshold/=2
        return 1 # le plan de bits à entièrement été codé 
            
            
            
      

    def get_code_res_Antonini(self,coefs,br):
        # Obtenir le code résultant et le nombre de bits par coefficient
        self.reset_Antonini_Encoder(coefs,br)
        
        if self.nb_coefs_max==0:
            return []
        
        
        while self.get_symbol_first() and self.get_symbol_second():
            pass
       
        code_end=self.finish(self.memoire_l,self.memoire_follow)
        
        code_nb_coefs=my_bin(self.nb_coefs,self.nb_bits_coefs_max) # on code la longueur
        #print("code_nb_coefs",code_nb_coefs)
        self.code.extend(code_end)
        
        
        ###### on place le train binaire en tête indiquant combien de coefficients ont été encodé
        
        #print(code_nb_coefs)
        #print(self.code)
        #print(code_nb_coefs+self.code)
        
        return code_nb_coefs+self.code
        #return self.code











class Antonini_Decoder(Context_Aritmetic_Decoder):
    def __init__(self,N=128,M=9,initial_occurrence_first=[1,1],
                 initial_occurrence_second=[1,1,1,1,1],
                 adaptive=True,verbose_AD=False):
        
        # Constructeur de la classe Antonini_Decoder
        
        # Initialisation des paramètres spécifiques à Antonini_Decoder
        self.N=N #Longueur du vecteur à coder.
        self.initial_occurrence_first = initial_occurrence_first # Probabilité initiale équivalente pour la première passe.
        self.initial_occurrence_second = initial_occurrence_second # Probabilité initiale équivalente pour la seconde passe.
        self.verbose_AD = verbose_AD # Mode débogage pour Antonini_Decoder.
        self.adaptive = adaptive  # 0: le codeur arithmétique n'est pas adaptatif, 1: le codeur arithmétique est adaptatif.
 
        
        # ... Autres constantes, initialisations, etc. ...
        self.alphabet_first = ['R', 'S']  # Alphabet de la première passe : 'R' (0 non significatif), 'S' (1 significatif).
        self.alphabet_second = ['0', '1', '-', '+', 'E']  # Alphabet de la seconde passe : '0' (0), '1' (1), '+' (longueur de séquence), '-' (polarité), 'E' (fin).
        
        # Initialisation du codeur arithmétique de contexte adaptatif en appelant le constructeur de la classe parente.
        super().__init__(M)
        
    
    def reset_Antonini_Decoder(self,code,br):
        # Réinitialisation des paramètres pour commencer un nouvel encodage

        self.reset_Context_Aritmetic_Decoder() # Réinitialisation des variables de la classe Context_Aritmetic_Encoder.
        
        self.flag=np.zeros(self.N) # drapeau indiquant si le coefficient a déja été significatif. 0 : jamais significatif, 1 : au moins une fois significatif
        self.coefs_rec=np.zeros(self.N) #Initialisation du seuil à 0.5.
        self.threshold=0.5 # les coefficicients sont compris entre -1 et 1 donc le threshold est initialisé à 0.5. Étant donné que les coefficients sont compris entre -1 et 1, le seuil est initialisé à 0.5.       
        self.symbol=[] # Liste pour stocker les symboles décodés.
        self.occurrence_first=list(self.initial_occurrence_first) # Initialisation de l'occurrence pour la première passe, qui peut être adaptée au fur et à mesure du codage.
        self.occurrence_second=list(self.initial_occurrence_second) # Initialisation de l'occurrence pour la seconde passe, qui peut être adaptée au fur et à mesure du codage.
        self.cumulate_occurrence_first=list(accumulate(self.initial_occurrence_first)) # Calcul des cumuls des occurrences pour la première passe et la seconde passe.
        self.cumulate_occurrence_second=list(accumulate(self.initial_occurrence_second)) # Calcul des cumuls des occurrences pour la seconde passe et la seconde passe.
        
        
        
        
        nb_coefs_max=br/2#br
        self.nb_bits_coefs_max=int(np.ceil(np.log2(max([1,nb_coefs_max]))))
        self.nb_coefs_max=2**self.nb_bits_coefs_max-1
        #print("nb_coefs_max",self.nb_coefs_max)
        #print("br",br)
        #print("self.nb_coefs_max",self.nb_coefs_max)
        #print("self.nb_bits_coefs_max",self.nb_bits_coefs_max)
        
        
        
        self.nb_coefs=my_inv_bin(code[0:self.nb_bits_coefs_max])
        
        #print("self.nb_coefs",self.nb_coefs)
        self.nb_coefs_dec=0
        
        
        
        self.code=code[self.nb_bits_coefs_max:] # Détermination de la longueur du vecteur à coder.
        #print("self.code dec",self.code)
    def get_symbol_first(self):
        # Décode un symbole de la première passe
        

        for i in range(self.N):
            if (self.flag[i]!=0):
                symbol_first=self.decode_one_symbol(self.code,self.alphabet_first,self.occurrence_first,self.cumulate_occurrence_first)
                self.symbol.append(symbol_first)
                self.nb_coefs_dec+=1
                
                if symbol_first=="S":
                    x=1
                    if self.adaptive:
                        self.cumulate_occurrence_first[1]+=1
                    self.coefs_rec[i]+=self.threshold/2
                else:
                    x=0
                    if self.adaptive:
                        self.cumulate_occurrence_first[0]+=1
                        self.cumulate_occurrence_first[1]+=1
                    self.coefs_rec[i]-=self.threshold/2
                    
                    
                    
                if self.adaptive:    
                    self.occurrence_first[x]+=1
                
                if self.nb_coefs_dec==self.nb_coefs:
                    
                    return 0
        return 1
                
                    

        
 
        
    def get_symbol_second(self):
        # Décode un symbole de la seconde passe 
        
        count=0
        
        pointeur=0 # pointeur indiquant la position du coefficients courant dans le vecteur coefs 
        cpt=0         
        
        #♦print("np.sum(np.abs(self.flag))",self.flag)
        while 1:
            #print(pointeur)
            ### on décode un symbole
            symbol_second=self.decode_one_symbol(self.code,self.alphabet_second,
                            self.occurrence_second,self.cumulate_occurrence_second)
         
            x=self.alphabet_second.index(symbol_second)
            
            if self.adaptive:
                self.occurrence_second[x]+=1
                for xx in range(x,5):
                    self.cumulate_occurrence_second[xx]+=1

            self.symbol.append(symbol_second)
            
            #print("symbol_second",symbol_second)
    
            if symbol_second == '0':
                cpt+=1
                
            elif symbol_second == '1':
                count += 2**cpt
                cpt+=1
                
            elif symbol_second == '+' or symbol_second == '-' :
                if cpt!=0:
                    count+=1
                
  
                for k in range(count+1):
                    #print(pointeur)
                    while self.flag[pointeur]!=0:
                        pointeur+=1
                    #print(pointeur)
                    if k==count: # on encode le '+' ou '-'
                        if symbol_second == '+':
                            self.coefs_rec[pointeur]+=self.threshold+self.threshold/2
                            self.flag[pointeur]=1
                        else :
                            self.coefs_rec[pointeur]-=self.threshold+self.threshold/2
                            self.flag[pointeur]=-1
                    pointeur+=1
                        
                #print("pointeur",pointeur) 
                #print("flag",self.flag)
                #print("count",count)
                

            
                self.nb_coefs_dec+=1
                if self.nb_coefs_dec==self.nb_coefs:
                    return 0
                
                cpt = 0
                count = 0
                
                #print(np.sum(np.abs(self.flag[pointeur-1:])),self.N-pointeur+1)
               
                if np.sum(np.abs(self.flag[pointeur-1:]))==self.N-pointeur+1:
                    self.threshold/=2
                    if self.nb_coefs_dec==self.nb_coefs:
                        return 0
                    else :
                        return 1
                    
            

            elif symbol_second == 'E':
                self.threshold/=2
                
                self.nb_coefs_dec+=1
                

                if self.nb_coefs_dec==self.nb_coefs:
                    return 0
                else :
                    return 1
     
    
        #self.threshold/=2
    
    
    def get_coefs_rec_Antonini(self,code,br):
        # Obtenir les coefficients reconstruit à partir de la suite binaire code
        
       
        
        self.reset_Antonini_Decoder(code,br)
        self.ini_codeword(self.code)
        
       
            
        
        while self.nb_coefs_dec<self.nb_coefs:
            #print("nb_coefs_dec",self.nb_coefs_dec,"nb_coefs",self.nb_coefs)
            #print("count",self.count,"len(code)",len(code))
            self.get_symbol_first()
            if self.nb_coefs_dec<self.nb_coefs:
                #print("nb_coefs_dec",self.nb_coefs_dec,"nb_coefs",self.nb_coefs)
                #print("count",self.count,"len(code)",len(code))
                self.get_symbol_second()

            #pass
        #print("nb_coefs_dec",self.nb_coefs_dec,"nb_coefs",self.nb_coefs)
        #print("count",self.count,"len(code)",len(code))
        
      
        return self.coefs_rec




# Programme principal
if __name__ == "__main__":
    from Measures import get_snr,entropy
    from Models import Model_poly
    from Normalize import normalize
    from scipy.fftpack import dct,idct
    from Measures import curve_tex

    br=64
    M=9
    
    adaptive =True
    verbose = False
    

    initial_occurrence_first=[1,1]
    initial_occurrence_second=[1,1,1,1,1]
                         
    N=128
    fn=50
    fs=6400
    
    t=np.linspace(0,(N-1)/fs,N)
    
 
    
    
    sigma=0.1 # écart type du bruit introduit dans le signal test
    

    ###############  test polynôme d'ordre k
    order=4
    theta=np.random.uniform(-0.4,0.4,order+1) # 
    #print([np.round(100*theta[k])/100 for k in range(order+1)])
    model_poly=Model_poly(fn,fs,N,verbose)
  
    x_test=model_poly.get_model_poly(t,*theta)+np.random.normal(0,sigma,N)
    #x_test=np.array([0.45, 0.67, 0.63, 1.09, 1.27, 1.7, 1.34, 1.61, 1.79, 1.46, 1.02, 0.72, 0.59, -0.0, -0.65, -0.92, -0.77, -0.74, -0.64, -0.64, -0.61, -0.37, -0.35, -0.17, -0.55, -0.62, -0.29, -0.36, -0.23, 0.07, 0.3, 0.18, 0.15, 0.28, -0.07, -0.0, 0.14, 0.23, 0.51, -0.19, -0.04, 0.31, 0.02, 0.09, -0.06, -0.16, -0.48, -0.89, -0.39, -0.53, -0.71, -0.25, 0.18, 0.62, 0.49, 0.94, 1.41, 1.57, 1.68, 1.66, 1.27, 1.41, 1.15, 0.61, -0.09, -0.46, -0.68, -0.88, -1.12, -1.44, -1.39, -1.71, -1.89, -1.46, -1.13, -0.87, -0.48, 0.05, 0.86, 0.87, 0.77, 0.79, 0.49, 0.69, 0.66, 0.32, 0.25, 0.11, 0.35, 0.52, -0.07, 0.1, 0.02, -0.12, -0.25, -0.23, 0.0, -0.18, 0.12, -0.05, -0.04, -0.13, -0.2, 0.39, 0.19, -0.0, 0.24, 0.01, 0.22, 0.32, 0.33, 0.84, 0.29, 0.48, 0.51, 0.15, -0.08, -0.72, -0.59, -0.89, -1.3, -1.47, -1.78, -1.51, -1.07, -1.1, -1.05, -0.45])
    
    x_test,_=normalize(x_test)

    #x_test=np.zeros(N)
    #x_test[N-1]=0.687
    #x_test[N-2]=-0.4
    
    #x_test=idct(x_test)/2
    
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_test,lw=2,label='x test')
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    coefs=dct(x_test/N)
    coefs_n=coefs
    k=0
    #coefs_n,k=normalize(coefs)


    AE=Antonini_Encoder(M,initial_occurrence_first,initial_occurrence_second,adaptive,verbose_AE=verbose)

    code=AE.get_code_res_Antonini(coefs_n,br)
    

    
    nb_coefs=AE.nb_coefs
    
    #print("code=",code)
    print("longueur des mots de code = {} bits".format(len(code)),"br = {} bits".format(br))
    print("Nb sym codé / Nb sym max = {} / {}".format(nb_coefs,AE.nb_coefs_max))
    #print(AE.symbol)

    occurrence_first_=AE.occurrence_first
    occurrence_second_=AE.occurrence_second
    print("Occurrence des symboles des premières passe",occurrence_first_)
    print("Occurrence des symboles des deuxièmes passe",occurrence_second_)
        
        
    occurrence_first=np.array(AE.occurrence_first_true)-1
    occurrence_second=np.array(AE.occurrence_second_true)-1
    
    if np.sum(occurrence_first)!=0:
        
        p_first=occurrence_first/np.sum(occurrence_first)
        H_first=entropy(p_first)
    else:
        p_first=occurrence_first
        H_first=0
        
    if np.sum(occurrence_second)!=0:
        
        p_second=occurrence_second/np.sum(occurrence_second)
        H_second=entropy(p_second)
    else:
        p_second=occurrence_second
        H_second=0
        
    print("Occurrence des symboles des premières passe",occurrence_first)
    print("Occurrence des symboles des deuxièmes passe",occurrence_second)
    
    print("H first = {:.2f} bits".format(H_first))
    print("H second = {:.2f} bits".format(H_second))

    H_tot=np.sum(occurrence_first)*H_first+np.sum(occurrence_second)*H_second
    print("H tot={:.2f} bits".format(H_tot))
    
    coefs_rec_E_n=AE.coefs_rec
    coefs_rec_E=coefs_rec_E_n*2**k
    x_rec_E=idct(coefs_rec_E)/2
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_test,lw=2,label='x_test')
    plt.plot(t,x_rec_E,lw=2,label='x_rec_Enc, br={} b, R_tot={} b, H_tot={:.1f} b, SNR={:.1f} dB'.format(br,len(code),H_tot,get_snr(x_test,x_rec_E)))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    
    
    coefs_rec_E_n=AE.coefs_rec
    coefs_rec_E=coefs_rec_E_n*2**k
    
    x_rec_E=idct(coefs_rec_E)/2
    #################### décodeur

    AD=Antonini_Decoder(N,M,initial_occurrence_first,initial_occurrence_second,adaptive,verbose_AD=verbose)


    

        
    coefs_rec_D_n=AD.get_coefs_rec_Antonini(code,br)
    
    
    #print(AD.symbol)
    
    coefs_rec_D=coefs_rec_D_n*2**k
    
    

    x_rec_D=idct(coefs_rec_D)/2
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(coefs,lw=2,label='coefs')
    plt.plot(AD.coefs_rec,lw=2,label='coefs rec avec b_used = {} sur br={} bits'.format(len(code),br))
    plt.xlabel('ind')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    
    
    plt.figure(figsize=(8,4), dpi=100)
    plt.plot(t,x_test,lw=2,label='x_test')
    plt.plot(t,x_rec_E,lw=2,label='x_rec_Enc, br={} b, R_tot={} b, H_tot={:.1f} b, SNR={:.1f} dB'.format(br,len(code),H_tot,get_snr(x_test,x_rec_E)))
    plt.plot(t,x_rec_D,lw=2,label='x_rec_Dec, br={} b, R_tot={} b, H_tot={:.1f} b, SNR={:.1f} dB'.format(br,len(code),H_tot,get_snr(x_test,x_rec_D)))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid( which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    
    
    
    
   
    