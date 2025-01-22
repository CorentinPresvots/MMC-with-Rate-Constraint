# Multiple-Model Coding (MMC) Scheme

> **Citation:**
>
> @article{PresvotsMMC2023  
> author = {Presvôts, Corentin and Kieffer, Michel and Prevost, Thibault and Panciatici, Patrick and Li, Zuxing and Piantanida, Pablo},  
> title = {Multiple-Model Coding Scheme for Electrical Signal Compression},  
> year = {2023},  
> note = {Available at SSRN: [https://ssrn.com/abstract=4584757](https://ssrn.com/abstract=4584757) or [http://dx.doi.org/10.2139/ssrn.4584757](http://dx.doi.org/10.2139/ssrn.4584757)}  
> }

This code proposes a low-latency Multiple-Model Coding approach to compress sampled electrical signal
waveforms under encoding rate constraints. The approach is window-based. Several parametric waveform models
are put in competition to obtain a first coarse representation of the signal in each considered window. Then, different
residual compression techniques are compared to minimize the residual reconstruction error. The model parameters
are quantized, and the allocation of the rate budget among the two steps is optimized.

Article is available at: [MMC](https://www.researchgate.net/publication/374226674_Multiple-Model_Coding_Scheme_for_Electrical_Signal_Compression)
You can integrate the two tables into your README as follows:

---

## Stage 1: The various competing models include

### Model types and their distributions
The following table outlines the model types used in the first stage of the Multiple-Model Coding (MMC) scheme, along with the corresponding a priori distributions for their parameters:

| Model type                  | $p_{\boldsymbol{\theta}^{m}}$                                                                                               |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------|
| **Bypass**                 |                                                                                                                             |
| **Sinusoidal**              | $\mathcal{U}\left(\boldsymbol{\theta}^m;(0.5,40.9,-\pi)^T,(1,50.1,\pi)^T\right)$ |
| **Polynomial, order $K$ with $K\in\left[0,\dots,9\right]$**  | $\mathcal{U}\left(\boldsymbol{\theta}^m;-(\frac{w_1^m}{2}, \dots,\frac{w_{K+1}^m}{2}))^T,(\frac{w_1^m}{2}, \dots, \frac{w_{K+1}^m}{2})^T\right)$ |
| **Sample predictive, order $K$ with $K\in\left[1,2\right]$, $\eta=\in\left[0,1\right]$**| $\mathcal{U}\left(\boldsymbol{\theta}^m;-\frac{1}{2}\boldsymbol{1},\frac{1}{2}\boldsymbol{1}\right)$                      |
| **Parameter predictive**        | $\mathcal{U}\left(\boldsymbol{\delta}\boldsymbol{\theta}^m;-0.1\cdot\boldsymbol{1},0.1\cdot\boldsymbol{1}\right)$          |

In these distributions:
- $\mathcal{U}(\boldsymbol{\theta}^m; \boldsymbol{a}^m, \boldsymbol{b}^m)$ represents a uniform distribution with parameters $\boldsymbol{a}^m$ and $\boldsymbol{b}^m$.
- $\boldsymbol{1}$ is a vector of ones matching the dimension of $\boldsymbol{\theta}^m$.
- For the polynomial model, we choose uniform prior distributions with widths $w_k^m$ for the $k$-th coefficient of the polynomial model $m$. The values of $w_k^m$ were determined to encapsulate 90\% of the total energy of the distribution of $\theta_k^m$, based on an analysis of 20,000 signals from the [Data\_u](https://github.com/rte-france/digital-fault-recording-database).

### Residual compression methods
The second stage of the MMC scheme applies various residual compression methods, each with an associated a priori distribution:

| Method type                | $p_{\boldsymbol{y}^{\ell}}$                                                   |
|----------------------------|-------------------------------------------------------------------------------|
| **Bypass**                 |                                                                               |
| **[DCT + BPC](https://www.academia.edu/26719905/A_new_bitplane_coder_for_scalable_transform_audio_coding)**              | $\mathcal{U}\left(\boldsymbol{y};-\frac{1}{2}\boldsymbol{1},\frac{1}{2}\boldsymbol{1}\right)$ |
| **[DWT + BPC](https://ieeexplore.ieee.org/document/6682511)**              | $\mathcal{U}\left(\boldsymbol{y};-\frac{1}{2}\boldsymbol{1},\frac{1}{2}\boldsymbol{1}\right)$ |


Refer to the detailed sections in the paper for further explanation of each method’s parameters and performance considerations.

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


- pywt


- fsolve from the scipy.optimize library
