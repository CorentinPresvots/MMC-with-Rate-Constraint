# Multiple-Model Coding Scheme


This code proposes a low-latency Multiple-Model Coding (MMC) approach to compress sampled electrical signal
waveforms under encoding rate constraints. The approach is window-based. Several parametric waveform models
are put in competition to obtain a first coarse representation of the signal in each considered window. Then, different
residual compression techniques are compared to minimize the residual reconstruction error. The model parameters
are quantized, and the allocation of the rate budget among the two steps is optimized.



Stage 1: The various competing models include

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



Note that 100 plots will be generated when compiling the main file. These plots depict, for each window, the reconstructed model and the reconstructed signal on b_tot bits. In the main file, you have the option to comment out the following section responsible for displaying the windows:


plt.figure(figsize=(8, 4), dpi=100)  
plt.plot(t, x_test, lw=2, label='x')  
plt.plot(t, x_poly_hat, lw=2, label='x hat, SNR={:.1f} dB'.format(get_snr(x_test, x_poly_hat)))  
plt.plot(t, x_poly_tilde, lw=2, label='x tilde, SNR={:.1f} dB, bx={} b'.format(get_snr(x_test, x_poly_tilde), bx_test))  
plt.plot(t, x_poly_rec, lw=2,label='x rec, SNR={:.1f} dB, bx={}+len(code)={}={} b'.format(get_snr(x_test, x_poly_rec), bx_test, len(code),bx_test + len(code)))  
plt.xlabel('t [s]')  
plt.ylabel('Amplitude')  
plt.legend()  
plt.title("Polynomial Model of Order {}".format(order))  
plt.grid(which='major', color='#666666', linestyle='-')  
plt.minorticks_on()  
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)  
plt.show()  

It is also possible that the compression of the next window may only begin after closing the previously opened window.



# Prerequisites

- numpy


- matplotlib.pyplot


- accumulate from the itertools library


- dct, idct from the scipy.fftpack library


- fsolve from the scipy.optimize library
