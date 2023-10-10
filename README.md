# Multiple-Model Coding Scheme


This code proposes a low-latency Multiple-Model Coding (MMC) approach to compress sampled electrical signal
waveforms under encoding rate constraints. The approach is window-based. Several parametric waveform models
are put in competition to obtain a first coarse representation of the signal in each considered window. Then, different
residual compression techniques are compared to minimize the residual reconstruction error. The model parameters
are quantized, and the allocation of the rate budget among the two steps is optimized.


## Stage 1: The various competing models include


- **No model (none)**


- **Sinusoidal models**
  - (sin-1) $\boldsymbol{m}=(m_a,m_f,m_\phi)=(0.75,f_n,0)$ and $\boldsymbol{w}=(w_a,w_f,w_\phi)=(0.5,0.2,2\pi)$
  - (sin-2) $\boldsymbol{m}=(0.75,f_n,0)$ and $\boldsymbol{w}=(w_a,w_f,w_\phi)=(0.5,0.05,2\pi)$


- **Polynomial models of order 0 to 8. Mean value of \boldsymbol{\theta}\) is assumed to be zeros.**  
  - (poly-0) \(\boldsymbol{w}=(w_{\theta_1})=(2)\)
  - (poly-1) \(\boldsymbol{w}=(w_{\theta_1},w_{\theta_2})=(2,2)\)
  - (poly-2) \(\boldsymbol{w}=(2,2,2)\)
  - ...
  - (poly-8) \(\boldsymbol{w}=(2,2,2,2,2,2,2,2,2)\)
    

- **Parameter predictive models. Mean value of \(\boldsymbol{\theta}\) is assumed to be zeros. \(i\): index of current window** 
  - (pred para-2) \(\boldsymbol{w}^(i)=\boldsymbol{w}^(i-1)/2\)
  - (pred para-5) \(\boldsymbol{w}^(i)=\boldsymbol{w}^(i-1)/5\)
  - (pred para-10) \(\boldsymbol{w}^(i)=\boldsymbol{w}^(i-1)/10\)
  - (pred para-50) \(\boldsymbol{w}^(i)=\boldsymbol{w}^(i-1)/50\)
  - (pred para-100) \(\boldsymbol{w}^(i)=\boldsymbol{w}^(i-1)/100\)
  - (pred para-500) \(\boldsymbol{w}^(i)=\boldsymbol{w}^(i-1)/500\)
  - (pred para-1000) \(\boldsymbol{w}^(i)=\boldsymbol{w}^(i-1)/1000\)

  
- **Sample predictive models. Mean value is estimate depending of previous encoded window**
  - (pred samples-1-0) \(N_p=0\), \(\eta=0\) \(\boldsymbol{w}=(0.1)\)
  - (pred samples-1-1) \(N_p=0\), \(\eta=1\) \(\boldsymbol{w}=(0.1)\)
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
