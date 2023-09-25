# Multiple-Model Coding Scheme


This code proposes a low-latency Multiple-Model Coding (MMC) approach to compress sampled electrical signal
waveforms under encoding rate constraints. The approach is window-based. Several parametric waveform models
are put in competition to obtain a first coarse representation of the signal in each considered window. Then, different
residual compression techniques are compared to minimize the residual reconstruction error. The model parameters
are quantized, and the allocation of the rate budget among the two steps is optimized.



Stage 1: The various competing models include
- No model


- Sinusoidal model


- Polynomial models of order 0 to 8



Stage 2: The different competing residual compression methods are:

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
