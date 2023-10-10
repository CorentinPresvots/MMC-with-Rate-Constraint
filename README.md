# Multiple-Model Coding (MMC) Scheme

The article is available : [MMC](https://www.researchgate.net/publication/374226674_Multiple-Model_Coding_Scheme_for_Electrical_Signal_Compression)

This code proposes a low-latency Multiple-Model Coding approach to compress sampled electrical signal
waveforms under encoding rate constraints. The approach is window-based. Several parametric waveform models
are put in competition to obtain a first coarse representation of the signal in each considered window. Then, different
residual compression techniques are compared to minimize the residual reconstruction error. The model parameters
are quantized, and the allocation of the rate budget among the two steps is optimized.


## Stage 1: The various competing models include


- **No model (none)**


- **Sinusoidal models**
  - (sin-1) : $p_{\boldsymbol{\theta}}=\mathcal{U}\left(\left(\frac{a-0.75}{0.25},\frac{f-f_\text{n}}{0.2},\frac{\phi}{\pi}\right);\left[-1,1 \right]^3\right)$ 
  - (sin-2) : $p_{\boldsymbol{\theta}}=\mathcal{U}\left(\left(\frac{a-0.75}{0.25},\frac{f-f_\text{n}}{0.05},\frac{\phi}{\pi}\right);\left[-1,1 \right]^3\right)$


- **Polynomial models of order 0 to 8. Mean value of $\boldsymbol{\theta}$ is assumed to be zeros.**  
  - (poly-{0,1,2,3,4,5,6,7,8}) : $p_{\boldsymbol{\theta}}=\mathcal{U}\left(\boldsymbol{\theta};\left[-1,1\right]^{\{0,1,2,3,4,5,6,7,8\}}\right)$

    
- **Parameter predictive models. Mean value of $\boldsymbol{\theta}$ is assumed to be zeros. $i$: index of current window** 
  - (pred para-{2,5,10,50,100,500,1000}) :  $p_{\boldsymbol{\theta}}=\mathcal{U}\left(\{2,5,10,50,100,500,1000\}\times\left(\boldsymbol{\theta}_n-\boldsymbol{\theta}_{n-1}\right);\left[-1,1 \right]^{\text{dim}\left(\boldsymbol{\theta}_{n-1}\right)}\right)$ 


  
- **Sample predictive models. Mean value $\mathbb{E}\left[\boldsymbol{\theta}\right]=\left(m_1,\dots,m_{N_\text{p}}\right)$ is estimated depending of previous encoded window**
  - (pred samples-1-0) : $N_p=1$, $\eta=0$,  $p_{\boldsymbol{\theta}}=\mathcal{U}\left(\frac{\boldsymbol{\theta}-\mathbb{E}\left[\boldsymbol{\theta}\right]}{0.1};\left[-1,1\right]^{1}\right)$
  - (pred samples-1-1) : $N_p=1$, $\eta=1$,  $p_{\boldsymbol{\theta}}=\mathcal{U}\left(\frac{\boldsymbol{\theta}-\mathbb{E}\left[\boldsymbol{\theta}\right]}{0.1};\left[-1,1\right]^{1}\right)$
  - (pred para-2-0) \(N_p=1\), \(\eta=0\) \(\boldsymbol{w}=(0.3,0.3)\)
  - (pred para-2-0) \(N_p=1\), \(\eta=1\) \(\boldsymbol{w}=(0.3,0.3)\)
  - (pred para-3-0) \(N_p=2\), \(\eta=0\) \(\boldsymbol{w}=(0.5,0.5,0.5)\)
  - (pred para-3-1) \(N_p=2\), \(\eta=1\) \(\boldsymbol{w}=(1.5,1.5,1.5)\)
  - (pred para-4-0) \(N_p=3\), \(\eta=0\) \(\boldsymbol{w}=(1.5,1.5,1.5)\)
  - (pred para-4-1) \(N_p=3\), \(\eta=1\) \(\boldsymbol{w}=(1.5,1.5,1.5)\)
  - (pred para-5-0) \(N_p=4\), \(\eta=0\) \(\boldsymbol{w}=(1.5,1.5,1.5,1.5,1.5)\)
  - (pred para-5-1) \(N_p=4\), \(\eta=1\) \(\boldsymbol{w}=(1.5,1.5,1.5,1.5,1.5)\)
  - (pred para-6-0) \(N_p=5\), \(\eta=0\) \(\boldsymbol{w}=(1.5,1.5,1.5,1.5,1.5,1.5)\)
  - (pred para-6-1) \(N_p=5\), \(\eta=1\) \(\boldsymbol{w}=(1.5,1.5,1.5,1.5,1.5,1.5)\)
              - 
## Stage 2: The different competing residual compression methods are:

- Antonini's method (Antonini_DCT)


- Khan's method (Khan_DWT)



# main.py
The main file will compress a 2-second reference signal:

A single-phase voltage signal recorded on the RTE network is considered. This signal was sampled at
6400 Hz (128 samples per nominal frequency period fn = 50 Hz) and consists of 12800 samples, see (link to the signal). The window size is set to 128 samples.

- The size of each window is set to N=128 samples (can be modified in the main).


- The number of coded windows is 12800/N.


- The maximum bit rate to encode each window is b_tot (can be modified in the main).

The main file performs compression for each window; the encoder takes the samples and the b_tot bit rate as input and returns the binary frame corresponding to the compressed window signal on b_tot bits.
From this binary frame, the decoder reconstructs the signal.



# Prerequisites

- numpy


- matplotlib.pyplot


- accumulate from the itertools library


- dct, idct from the scipy.fftpack library


- fsolve from the scipy.optimize library
