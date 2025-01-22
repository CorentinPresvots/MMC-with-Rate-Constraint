# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:03:42 2023

@author: coren
"""
import numpy as np
import matplotlib.pyplot as plt
from Quantization import Quantizer
from Measures import my_bin,my_inv_bin

"""
def normalize(x):
    x_max=np.max(np.abs(x))
    k=-np.ceil(np.log2(x_max))
    
    
    
    
    #k=-1+np.ceil(np.log2(1/x_max))
    x_n = x * 2**(k)
    
    return x_n, k  # Retourne x mis à l'échelle et la valeur de k

"""

def normalize(x):
    k=np.ceil(np.log2(np.max(np.abs(x))+10**(-8)))
    x_n = x * 2**(-k)
    return x_n, k  # Retourne x mis à l'échelle et la valeur de k




def scale_mean2(x):
    
    mean=np.mean(x)
    x_mean=np.array(x-mean)
    
    _,k_x=normalize(x_mean)
    
    """
    x_n_1=x*2**(-k_x)
    
    min_=np.min(x_n_1)
    max_=np.max(x_n_1)
    
    plt.figure(figsize=(10,4), dpi=80)
    plt.plot(x_n_1,lw=2,label='signal normalisé max={:.2f}, min={:.2f}, W={:.2f}'.format(max_,min_,max_-min_))
    plt.xlabel('ind')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    
    
    
    mean_x_n_1=mean*2**(-k_x)
    
    print(mean_x_n_1,np.mean(x_n_1))
  
    k_mean=np.ceil(np.log2(mean_x_n_1*8+10**(-8)))
    

    mean_n=mean_x_n_1*2**(-k_mean)
    print("k_mean",k_mean,"mean_n",mean_n)

    
    
    plt.figure(figsize=(10,4), dpi=80)
    plt.plot(x_n_1-mean_x_n_1+mean_x_n_1*2**(-k_mean),lw=2,label='signal normalisé max={:.2f}, min={:.2f}, W={:.2f}'.format(max_,min_,max_-min_))
    plt.xlabel('ind')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    
    
    """
    k_test=0
    while 1:
        _,k_m=normalize(x_mean+mean*2**(-k_test))
        if k_m==k_x:
            break
        k_test+=1
   
    x_n=(x_mean+mean*2**(-k_test))*2**(-k_x)
    
    #print("k_test",k_test)
    return x_n,k_x,k_test

def coder_km(a, b, km_min, km_max): 
    """
    Cette fonction détermine le mot de code binaire le plus court pour représenter km dans [a,b]
    en conssidérant l'intervalle de départ [k_min, k_max].
    A noter que  km_min<a<b<km_max

    Parameters
    ----------
    a : float
        Borne inférieure du nombre réel que l'on souhaite coder.
    b : float
        Borne supérieure du nombre réel que l'on cherche à coder.
    km_min : int
        Borne inférieure de l'intervalle initial.
    km_max : int
        Borne supérieure de l'intervalle initial.

    Returns
    -------
    binary_code : list de 0 et 1
        Le mot de code binaire le plus court pour représenter le nombre réel dans l'intervalle [a, b].
    """

    binary_code = []  # Liste binaire pour stocker le mot de code.
    mid = (km_max + km_min) / 2  # Milieu de l'intervalle initial.

    while 1:  # Boucle infinie, s'arrête lorsque la valeur est correctement codée.

        if a <= mid and mid <= b:
            return binary_code  # Retourne le mot de code lorsque la valeur est correctement codée.

        else:
            if b < mid:
                km_max = mid
                binary_code.append(0)  # Ajoute 0 au mot de code si la valeur est à gauche du milieu.

            else:
                km_min = mid
                binary_code.append(1)  # Ajoute 1 au mot de code si la valeur est à droite du milieu.

        mid = (km_max + km_min) / 2  # Met à jour le milieu de l'intervalle.

      
    
def decoder_km(binary_code, km_min, km_max):
    """
    Cette fonction décode un mot de code binaire en un nombre réel compris entre les bornes km_min, km_max

    Parameters
    ----------
    binary_code : list de 0 et 1
        Le mot de code binaire à décoder.
    km_min : int
        Borne inférieure de l'intervalle initial.
    km_max : int
        Borne supérieure de l'intervalle initial.

    Returns
    -------
    mid : float
        Le nombre réel décodé 
    """

    mid = (km_min+km_max)/2 # Milieu de l'intervalle initial.
   

    for bit in binary_code:
        if bit==0:
            km_max = mid
        else :
            km_min = mid
            
        mid=(km_min+km_max)/2# Met à jour le milieu de l'intervalle.    

    return mid


a=0.2
b=0.4
bound_min=0
bound_max=2**2  
   
code_km=coder_km(a,b,bound_min,bound_max)

km=decoder_km(code_km,bound_min,bound_max)
#print("km",km,code_km)

def scale_mean(x,km_min,km_max):

    x_n1,kx1=normalize(x)
    if np.max(x_n1)>=0.5 and np.min(x_n1)<=-0.5:
        return 0,x_n1,kx1,0,0,[]    
    
    
    x_min=np.min(x)
    x_max=np.max(x)
    

    
    
    kx=np.ceil(np.log2(1/(x_max-x_min)))
    

    """
    plt.figure(figsize=(10,4), dpi=80)
    plt.plot(t,x_test*2**kx,lw=2,label='x x_min={:.2f}, x_max={:.2f}'.format(x_min,x_max))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    """
    """
    print("kx",np.log2(1/(x_max-x_min)),kx,np.log2(2/(x_max-x_min)))
    print("condition dynamique",1,2**kx*(x_max-x_min),2)
    print("2**kx*(x_max)",2**kx*(x_max),"2**kx*x_min",2**kx*(x_min),"diff",2**kx*(x_max-x_min))
    """
    if x_min>0:
        cas=1
    elif x_max<0:
        cas=2
    elif x_min<=0 and x_max>=0:
        cas=3

        
        
    
    
   # print("-1+2**kx*x_max",-1+2**kx*x_max)
   # print("-0.5+2**kx*x_max",-0.5+2**kx*x_max)
    
    #print("0.5-2**kx*x_max",0.5-2**kx*x_max)
    #print("1-2**kx*x_max",1-2**kx*x_max)
    
    if cas==1:
        s=1
        #print("cas 1")
        """
        print("0.5+x_min*2**kx",0.5+x_min*2**kx)
        print("-0.5+x_max*2**kx",-0.5+x_max*2**kx)
        
        print("1+x_min*2**kx",1+x_min*2**kx)
        print("-1+x_max*2**kx",-1+x_max*2**kx)
        """
        

        a=np.max([km_min,np.log2(0.5+x_min*2**kx),np.log2(-1+x_max*2**kx)])
        b=np.min([np.log2(1+x_min*2**kx),np.log2(-0.5+x_max*2**kx)])
      
        #a=np.max([km_min,np.log2(-1+0.5+x_max*2**kx)])
        #b=np.log2(1+x_min*2**kx)#-
        
    elif cas==2:
        s=-1
        
        #print("cas 2")
        a=np.max([km_min,np.log2(0.5-x_max*2**kx),np.log2(-1-x_min*2**kx)]) #np.ceil(-1+np.log2(s*(x_min+x_max)/(x_max-x_min)))
        b=np.min([np.log2(1-x_max*2**kx),np.log2(-0.5-x_min*2**kx)])
        
        #a=np.max([km_min,np.log2(-1-x_min*2**kx)])
        #b=np.log2(1-x_max*2**kx)
    elif cas==3:
        
        if x_max*2**kx>1:
            s=1
            a1=0.5+x_min*2**kx
            a2=-1+x_max*2**kx
            
            a_test=np.max([0,a1,a2])
            #print("a1",a1)
            #print("a2",a2)
            
            b1=1+x_min*2**kx
            b2=-0.5+x_max*2**kx
            
            #print("b1",b1)
            #print("b2",b2)
            
            b_test=np.min([b1,b2])
            a=np.log2(a_test)
            b=np.log2(b_test)
        else: #x_min*2**kx<-1:
            s=-1
            
            a1=0.5-x_max*2**kx
            a2=-1-x_min*2**kx
            
            a_test=np.max([0,a1,a2])
            
            b1=1-x_max*2**kx
            b2=0.5-x_min*2**kx
            
            b_test=np.min([b1,b2])
            
            a=np.log2(a_test)
            b=np.log2(b_test)

            
        
    #print("km min={:.3f}<a={:.4f}<b={:.4f}<km max={:.4f}".format(km_min,a,b,km_max)) 
   
    if km_min<=a and b<=km_max:
        
        code_km=coder_km(a,b,km_min,km_max)
                    
        km=decoder_km(code_km,km_min,km_max)
        #print("km min={:.3f}<a={:.4f}<km={:.4f}<b={:.4f}<km max={:.4f}, len(code_km)={}".format(km_min,a,km,b,km_max,len(code_km))) 

            
        
        x_n = x*2**(kx)-s*2**(km)
        
        #print("x_n max=",np.max(x_n))
        #print("x_n min=",np.min(x_n))
        return 1,x_n, kx,km,s,code_km  # Retourne x mis à l'échelle et la valeur de k        
            
    else :
        
        return 0,x_n1,kx1,0,0,[]    

    

def scale_mean2(x,km_min,km_max):

    x_n1,kx1=normalize(x)
    if np.max(x_n1)>=0.5 and np.min(x_n1)<=-0.5:
        return 0,x_n1,kx1,0,0,[]    
    
    
    x_min=np.min(x)
    x_max=np.max(x)
    

    
    
    kx=np.ceil(np.log2(1/(x_max-x_min)))
    

    """
    plt.figure(figsize=(10,4), dpi=80)
    plt.plot(t,x_test*2**kx,lw=2,label='x x_min={:.2f}, x_max={:.2f}'.format(x_min,x_max))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    """
    """
    print("kx",np.log2(1/(x_max-x_min)),kx,np.log2(2/(x_max-x_min)))
    print("condition dynamique",1,2**kx*(x_max-x_min),2)
    print("2**kx*(x_max)",2**kx*(x_max),"2**kx*x_min",2**kx*(x_min),"diff",2**kx*(x_max-x_min))
    """
    if x_min>0:
        cas=1
    elif x_max<0:
        cas=2
    elif x_min<=0 and x_max>=0:
        cas=3

        
        
    
    
   # print("-1+2**kx*x_max",-1+2**kx*x_max)
   # print("-0.5+2**kx*x_max",-0.5+2**kx*x_max)
    
    #print("0.5-2**kx*x_max",0.5-2**kx*x_max)
    #print("1-2**kx*x_max",1-2**kx*x_max)
    
    if cas==1:
        s=1


      
        a=np.log2(-1+x_max*2**kx)
        b=np.log2(1+x_min*2**kx)
        
    elif cas==2:
        s=-1
        

        a=np.log2(-1-x_min*2**kx)
        b=np.log2(1-x_max*2**kx)
    elif cas==3:
        
        if x_max*2**kx>1:
            s=1
            a1=-1+x_max*2**kx
           
            
            a_test=a1
            a=np.log2(a_test)
            #print("a1",a1)

            
            b1=1+x_min*2**kx
           
            
            #print("b1",b1)

            b_test=b1
            
            b=np.log2(b_test)
        elif x_min*2**kx<-1:
            s=-1
            
            a1=-1-x_min*2**kx
         
            
            a_test=a1
            
            b1=1-x_max*2**kx
          
            
            b_test=b1
            
            a=np.log2(a_test)
            b=np.log2(b_test)
        else :
            a=km_min-1
            b=km_max+1

            

        
    #print("km min={:.3f}<a={:.4f}<b={:.4f}<km max={:.4f}".format(km_min,a,b,km_max)) 
   
    if km_min<=a and b<=km_max:
        
        code_km=coder_km(a,b,km_min,km_max)
                    
        km=decoder_km(code_km,km_min,km_max)
        #print("km min={:.3f}<a={:.4f}<km={:.4f}<b={:.4f}<km max={:.4f}, len(code_km)={}".format(km_min,a,km,b,km_max,len(code_km))) 

            
        
        x_n = x*2**(kx)-s*2**(km)
        
        #print("x_n max=",np.max(x_n))
        #print("x_n min=",np.min(x_n))
        return 1,x_n, kx,km,s,code_km  # Retourne x mis à l'échelle et la valeur de k        
            
    else :
        
        return 0,x_n1,kx1,0,0,[]    


     

def coder_mean(x,mean_max,b_mean):
    
    if b_mean<2:
        return 0,0,[]
    
    mean=np.mean(x)
    sign_mean=np.sign(mean)
    
    delta=mean_max/2**(b_mean)
    if sign_mean*mean<delta:
        return 0,[]
    
    
    
    
    
    code=[int((sign_mean+1)/2)]
    
    Q=Quantizer()
    ind_mean_q=Q.get_ind_u(sign_mean*mean,b_mean-1,mean_max-delta,delta+(mean_max-delta)/2)
    #print("ind_mean_q",ind_mean_q)

    code+=Q.get_code_u(ind_mean_q, b_mean-1)

    return 1,code



def decoder_mean(encode_mean,code,mean_max,b_mean):
    
    if b_mean<2 or encode_mean==0:
        return 0
    
    Q=Quantizer()
    
    delta=mean_max/2**(b_mean)
    mean=Q.get_inv_code_u(code[1:],b_mean-1)
    mean_q=Q.get_q_u(mean,b_mean-1,mean_max-delta,delta+(mean_max-delta)/2)
    
    mean_q*=code[0]*2-1
    return mean_q
    
    
    
    
    




if __name__ == "__main__":


    N=128
    T=0.02
    t=np.linspace(0,T-T/N,N)
    """
    x_test=np.array([10*np.cos(2*np.pi*50*t[i]+np.pi).real for i in range(N)])
    
   
    x_test_n,k=normalize(x_test)

    
    plt.figure(figsize=(10,4), dpi=80)
    #plt.plot(t,x_test,lw=2,label='signal sinusoïdal')
    plt.plot(t,x_test_n,lw=2,label='signal sinusoïdal normalisé')
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    """

    x_test=np.array([37796.55, 40044.83, 42603.45, 44903.45, 47051.72, 48893.1, 50937.93, 52982.76, 54875.86, 57024.14, 58813.79, 60603.45, 62548.28, 64696.55, 66382.76, 68172.41, 69810.34, 71803.45, 73337.93, 75179.31, 76613.79, 78148.28, 79527.59, 81010.34, 82341.38, 83720.69, 85255.17, 86482.76, 87606.9, 88734.48, 89603.45, 90524.14, 91648.28, 92724.14, 93693.1, 94613.79, 95482.76, 96200.0, 96610.34, 97172.41, 97937.93, 98551.72, 98962.07, 99320.69, 99679.31, 100034.48, 100241.38, 100393.1, 100496.55, 100293.1, 100293.1, 100034.48, 100034.48, 99627.59, 99679.31, 99320.69, 98706.9, 98244.83, 97479.31, 96865.52, 96200.0, 95689.66, 94972.41, 94462.07, 93693.1, 92824.14, 92006.9, 91237.93, 90472.41, 89500.0, 88682.76, 87606.9, 86586.21, 85613.79, 84589.66, 83362.07, 82186.21, 80858.62, 79527.59, 78555.17, 77275.86, 75641.38, 73951.72, 72213.79, 70320.69, 68379.31, 66486.21, 64337.93, 62189.66, 59889.66, 57689.66, 55541.38, 53086.21, 50682.76, 47972.41, 44903.45, 41579.31, 38100.0, 34112.41, 30378.97, 27361.72, 25162.41, 23270.34, 21991.72, 20303.79, 19127.59, 18053.45, 17133.1, 16212.41, 15189.66, 14217.93, 13450.69, 12581.38, 11865.17, 11149.31, 10382.07, 9563.79, 8745.52, 7978.28, 7262.41, 6751.03, 6137.24, 5421.03, 4756.21, 4193.79, 3580.0, 2915.17, 2352.59])
    #x_test=np.array([-1380.86, -1432.0, -1176.31, -1278.59, -1227.45, -1278.59, -1278.59, -1227.45, -1227.45, -1329.72, -1278.59, -1227.45, -1227.45, -1329.72, -1176.31, -1074.0, -1227.45, -1380.86, -1278.59, -1278.59, -1227.45, -1176.31, -1227.45, -1432.0, -1329.72, -1227.45, -1329.72, -1380.86, -1176.31, -1329.72, -1278.59, -1432.0, -1329.72, -1329.72, -1329.72, -1329.72, -1380.86, -1432.0, -1432.0, -1380.86, -1329.72, -1278.59, -1380.86, -1329.72, -1432.0, -1380.86, -1227.45, -1432.0, -1432.0, -1483.14, -1432.0, -1534.31, -1380.86, -1432.0, -1534.31, -1432.0, -1483.14, -1432.0, -1432.0, -1483.14, -1380.86, -1380.86, -1534.31, -1534.31, -1483.14, -1483.14, -1483.14, -1483.14, -1534.31, -1432.0, -1483.14, -1432.0, -1585.45, -1534.31, -1534.31, -1329.72, -1585.45, -1380.86, -1483.14, -1432.0, -1329.72, -1380.86, -1329.72, -1483.14, -1329.72, -1585.45, -1432.0, -1380.86, -1483.14, -1380.86, -1380.86, -1483.14, -1432.0, -1329.72, -1329.72, -1432.0, -1329.72, -1380.86, -1380.86, -1278.59, -1432.0, -1329.72, -1278.59, -1432.0, -1432.0, -1380.86, -1534.31, -1380.86, -1329.72, -1380.86, -1380.86, -1278.59, -1329.72, -1380.86, -1380.86, -1227.45, -1329.72, -1329.72, -1278.59, -1227.45, -1125.14, -1278.59, -1278.59, -1278.59, -1278.59, -1278.59, -1278.59, -1227.45])
    #x_test=np.array([230248.28, 230348.28, 229634.48, 228865.52, 227279.31, 225389.66, 222165.52, 219044.83, 215313.79, 210555.17, 205186.21, 199458.62, 193424.14, 186468.97, 179000.0, 171482.76, 163965.52, 155934.48, 147600.0, 138803.45, 129700.0, 120493.1, 110775.86, 100955.17, 90320.69, 79782.76, 69451.72, 58558.62, 47717.24, 36924.14, 25980.69, 14626.9, 3324.31, -7824.83, -19434.48, -30583.79, -41579.31, -52524.14, -63162.07, -74617.24, -85051.72, -95075.86, -105506.9, -115327.59, -125096.55, -134506.9, -143813.79, -152865.52, -160641.38, -168668.97, -176341.38, -182937.93, -189127.59, -194855.17, -200686.21, -205441.38, -209686.21, -213727.59, -217306.9, -220479.31, -223496.55, -225437.93, -227231.03, -228762.07, -229889.66, -230144.83, -229682.76, -228662.07, -227127.59, -225131.03, -222217.24, -219148.28, -215413.79, -210555.17, -205289.66, -199613.79, -193320.69, -186417.24, -178796.55, -171534.48, -163965.52, -155886.21, -147751.72, -138751.72, -129648.28, -120544.83, -110879.31, -101006.9, -90524.14, -79886.21, -69810.34, -58813.79, -47920.69, -36975.86, -25929.66, -14678.28, -3170.9, 7927.24, 19485.52, 30532.41, 41682.76, 52627.59, 63468.97, 74824.14, 85203.45, 95382.76, 105765.52, 115431.03, 125248.28, 134658.62, 143662.07, 152817.24, 160537.93, 168620.69, 176137.93, 182837.93, 189231.03, 194755.17, 200582.76, 205493.1, 209789.66, 213831.03, 217206.9, 220631.03, 223700.0, 225744.83, 227331.03, 228917.24])
    x_test=np.array([511.45, 460.28, 358.0, 460.28, 358.0, 306.86, 358.0, 460.28, 409.14, 409.14, 562.59, 358.0, 255.72, 409.14, 511.45, 511.45, 306.86, 306.86, 255.72, 358.0, 204.57, 306.86, 306.86, 358.0, 255.72, 153.43, 204.57, 153.43, 204.57, 255.72, 0.0, 153.43, 51.14, 102.29, 0.0, 204.57, 51.14, 153.43, -51.14, 0.0, 102.29, 102.29, 102.29, 102.29, -51.14, -102.29, -51.14, 51.14, 0.0, -51.14, 51.14, 0.0, 0.0, -153.43, -51.14, -153.43, -51.14, 0.0, 153.43, -204.57, -51.14, -102.29, -153.43, -153.43, -51.14, -204.57, -255.72, -204.57, -153.43, -102.29, -102.29, -204.57, -255.72, -255.72, -255.72, -204.57, -204.57, -255.72, -306.86, -102.29, -102.29, 51.14, -204.57, -204.57, -204.57, -306.86, -255.72, -102.29, -153.43, -204.57, -255.72, -255.72, -102.29, -204.57, -204.57, 0.0, -102.29, -153.43, -255.72, -255.72, -102.29, -255.72, -204.57, -204.57, -255.72, -204.57, -153.43, -204.57, -306.86, -306.86, -153.43, -306.86, -255.72, -204.57, -255.72, -204.57, -153.43, -153.43, -306.86, -409.14, -102.29, -255.72, -153.43, -204.57, -153.43, -255.72, -306.86, -255.72])
    
    x_test=np.array([100*np.cos(2*np.pi*50*t[i]+np.pi).real+60 for i in range(N)])
    x_test_n,kx=normalize(x_test)
    
    print("kx",kx)
    km_min=0
    km_max=5
 
    
   
    cond,x_test_mean,kx,km,s,code_km=scale_mean(x_test,km_min,km_max)
    #x_rec=(x_test_mean-np.mean(x_test_mean))*2**(k)+np.mean(x_test_mean)*2**(k+k_mean)
   
    """
    plt.figure(figsize=(10,4), dpi=80)
    plt.plot(t,x_test_n,lw=2,label='signal scale')
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)   
    """
    

    
    plt.figure(figsize=(10,4), dpi=80)
    plt.plot(t,x_test_n,lw=2,label='signal sclale')
    plt.plot(t,x_test_mean,lw=2,label='signal scale kx={} + remove mean s={} (nb bits to encode km = {} bits)'.format(kx,s,len(code_km)))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.title("Cond={}".format(cond))
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    
    """
    plt.figure(figsize=(10,4), dpi=80)
    plt.plot(t,x_test,lw=2,label='signal sinusoïdal+shift')
    plt.plot(t,x_rec,lw=2,label='signal rec')
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    """
    
    
    mean_max=200
    b_mean=2
    encode_mean,code=coder_mean(x_test,mean_max,b_mean)
    cst=decoder_mean(encode_mean,code,mean_max,b_mean)
    plt.figure(figsize=(10,4), dpi=80)
    plt.plot(t,x_test,lw=2,label='signal')
    plt.plot(t,np.ones(len(x_test))*cst,lw=2,label='signal scale')
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.title("Cond={}".format(cond))
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    
    
    
    mean_max=10
    b_mean=3
    x_test=[k/10 for k in range(200)]
    
    x_test_q=[0]*len(x_test)
    for k in range(len(x_test)):
        encode_mean,code=coder_mean(x_test[k],mean_max,b_mean)
        
        x_test_q[k]=decoder_mean(encode_mean,code,mean_max,b_mean)
        #print(code)
    
    plt.figure(figsize=(8,8), dpi=100)
    plt.plot(x_test,x_test,lw=2,label='signal')
    plt.plot(x_test,x_test_q,lw=2,label='signal scale')
    plt.axis("equal")
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend()
    plt.title("Cond={}".format(cond))
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)    
    
    
    
    
    