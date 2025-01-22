# Multiple-Model Coding (MMC) Scheme for Electrical Signal Compression

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


### Stage 2: Residual compression methods
The second stage of the MMC scheme applies various residual compression methods, each with an associated a priori distribution:

| Method type                | $p_{\boldsymbol{y}^{\ell}}$                                                   |
|----------------------------|-------------------------------------------------------------------------------|
| **Bypass**                 |                                                                               |
| **[DCT + BPC](https://www.academia.edu/26719905/A_new_bitplane_coder_for_scalable_transform_audio_coding)**              | $\mathcal{U}\left(\boldsymbol{y};-\frac{1}{2}\boldsymbol{1},\frac{1}{2}\boldsymbol{1}\right)$ |
| **[DWT + BPC](https://ieeexplore.ieee.org/document/6682511)**              | $\mathcal{U}\left(\boldsymbol{y};-\frac{1}{2}\boldsymbol{1},\frac{1}{2}\boldsymbol{1}\right)$ |


Refer to the detailed sections in the paper for further explanation of each method’s parameters and performance considerations.


## Explanation of `main.py`

`main.py` is designed to encode selected voltage and current signals using a multiple-model coding (MMC) approach. The signals and various parameters can be adjusted directly within the script to tailor the encoding process.

### Signal Selection and Phases
By default, the script uses 12 three-phase voltage signals identified as `85, 91, 98, 176, 189, 195, 282, 287, 316, 337, 371, 380` from the [Data_S](https://github.com/rte-france/digital-fault-recording-database) dataset. These signals are chosen because they correspond to faults. Each signal is one second long and sampled at 6400 Hz, resulting in a total of 100 non-overlapping 20 ms windows per signal.

- The number of signals to encode is controlled by `nb_signal`.
- The number of phases (voltage or current) is specified with `nb_phase`.  
  - For example, `nb_phase=3` encodes only the three voltage phases of each signal.  
  - Setting `nb_phase=6` encodes both three voltage phases and three current phases for all 12 signals.

### Key Parameters
- **Window Size (N):** Each window is by default 128 samples long. This can be modified by changing the `N` value in the script.
- **Number of Coded Windows (nb_w):** The number of windows to be encoded is set to 50 by default. Adjusting `nb_w` controls how many windows per signal are processed.
- **Maximum Bit Rate (n_tot):** The total bit rate allocated for encoding each window is set to 128 bits (equivalent to 1 bit/sample) by default. You can modify $n_tot$ to explore different encoding rates.

### Model Selection
- The set of models ($\mathcal{M}$) used in the encoding process is initialized in the main code.  
- Adding or removing models is as simple as commenting or uncommenting lines where the models are defined.

### Residual Compression Methods
- Residual compression methods ($\mathcal{L}$) are also initialized in the main code.  
- You can exclude certain methods by commenting the corresponding lines in the initialization section.

### Compression Techniques
There are multiple encoding methods available in `main.py`, each offering a different balance between computational complexity and compression performance. Depending on your computational resources and the desired compression quality, you can select one of the following methods:

1. **Exhaustive Search:**  
For each $n_{\text{x}}$, the model parameters are quantized, and all residual compression methods are tested until a distortion constraint is met. The best combination of parameter quantization and residual compression is selected. This approach is the most computationally expensive but achieves the best compression performance.
   - High computational cost (~529×O(Nlog(N))).  
   - Example: 48.07 dB SNR for the 12 signals at 1 bit/sample.

2. **Golden Section Search:**  
This method assumes convexity in the bit allocation problem and uses the golden section search to iteratively narrow the range of $n_{\text{x}}$. It requires fewer function evaluations than exhaustive search, resulting in faster convergence, though with slightly reduced compression quality.
   - Mid-level computational complexity (~186×O(Nlog(N))).  
   - Slightly lower compression performance than exhaustive search, with 47.86 dB SNR at 1 bit/sample.

3. **Sub-optimal Exhaustive Search:**  
This process starts by selecting $n_{\text{x}}$ values and first determining the best model. For the best model, the best residual compression method is then identified. This simplified exhaustive search evaluates fewer configurations, reducing complexity while maintaining reasonable compression performance.
   - Reduced complexity (~72×O(Nlog(N))).  
   - Still offers a decent compression performance of 47.76 dB SNR at 1 bit/sample.

4. **Distortion Model Exhaustive Search:**  
A distortion model is used to identify a subset of candidate models and bit allocations. An exhaustive search is then performed on this reduced set to find the optimal $n_{\text{x}}$, resulting in significant complexity reduction at a modest cost to compression quality.
   - Further reduces complexity (~40×O(Nlog(N))).  
   - Results in 47.70 dB SNR at 1 bit/sample.

5. **Distortion Model Golden Section Search:**  
This approach applies a distortion model to pre-select candidate models and bit allocations, followed by a golden section search to find the optimal $n_{\text{x}}$. It is the least computationally demanding method, trading off some compression performance for speed.
   - The least complex method (~25×O(Nlog(N))).  
   - Fastest option, but with the lowest compression performance (47.56 dB SNR at 1 bit/sample).

### Encoding and Decoding
The main code handles the compression process for each window by:  
1. Using the selected encoding method to compress the samples within a window.  
2. Producing a binary frame that represents the compressed signal.  
3. The decoder then reconstructs the original signal from this binary frame.

This approach provides flexibility to adjust signal selection, model choice, and encoding methods, enabling you to balance computational cost and compression quality to meet your specific needs.

# Prerequisites

- numpy


- matplotlib.pyplot


- accumulate from the itertools library


- dct, idct from the scipy.fftpack library


- pywt


- fsolve from the scipy.optimize library