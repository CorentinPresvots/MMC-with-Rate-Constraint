# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:13:54 2023
@author: presvôts
"""

# Select one optimization method from the five provided below.
# The choice determines the computational complexity and may slightly affect compression performance.
# Generally, higher complexity yields better compression, while lower complexity sacrifices performance for faster processing.

# Method 1: Exhaustive Search (~529 times O(Nlog(N)))
# - This method is the most computationally expensive, running exhaustive tests on all possible configurations.
# - Compression performance is the highest among the options (48.07 dB SNR on data_S2 at 1 bit/sample).

# Method 2: Golden Section Search (~186 times O(Nlog(N)))
# - A mid-range approach that uses the Golden Section algorithm to find the optimal settings more efficiently than exhaustive search.
# - Compression performance is only slightly lower (47.86 dB SNR on data_S2 at 1 bit/sample).

# Method 3: Sub-optimal Exhaustive Search (~72 times O(Nlog(N)))
# - A simplified version of exhaustive search that tests fewer configurations.
# - While less computationally demanding, it still offers decent compression performance (47.76 dB SNR on data_S2 at 1 bit/sample).

#Method 4: Distortion Model Exhaustive Search (~40 times O(Nlog(N)))
#- Estimation of the model and bit distributions between two stages, using a variant of exhaustive search.
#- This approach reduces computational complexity further, but with a slight impact on compression quality (47.70 dB SNR on data_S2 at 1 bit/sample).

#Method 5: Distortion Model Golden Section Search (~25 times O(Nlog(N)))
#- The least complex approach, using a variant of the golden section search to estimate the model and bit distributions across two stages.
#- This method is the fastest among all options but results in the lowest compression performance (47.56 dB SNR on data_S2 at 1 bit/sample).

# Importing the desired optimization method (uncomment one of the following lines based on your choice)

#from Exaustive_search import Encode_one_window, Decode_one_window
#from Golden_section_search import Encode_one_window, Decode_one_window
#from Sub_optimal_exaustive_search import Encode_one_window, Decode_one_window
#from Distortion_model_exaustive_search import Encode_one_window, Decode_one_window
from Distortion_model_golden_section_search import Encode_one_window, Decode_one_window




import time
from Measures import get_snr, get_rmse
import numpy as np
import matplotlib.pyplot as plt

from get_test_signal import get_RTE_signal as get_signal

# General parameters
fs = 6400  # Sampling frequency
fn = 50    # Nominal frequency
N = 128    # Number of samples per window

# Input parameters
nb_signal = 2      # Number of signals to encode
nb_phase = 1        # Number of phases (3 for voltage phases, 6 includes currents)
nb_w = 50           # Number of windows per signal
n_tot = 128         # Rate constraint (total available bits)

"""
Define the set of models to be used
"""
Model_used = {}  # Dictionary to store all models

#########################################################
# Model 1: None (no model used)
id_model = 0


Model_used[id_model] = {
    "name": "none",
    "family": "none",
    "m theta": [],
    "w theta": []
}
id_model += 1  # Increment model ID

#########################################################
# Model 2: Sinusoidal model
# Define prior distribution for sinusoidal parameters
m_theta_sin = [0.75, fn, 0]  # Mean [amplitude, frequency, phase]
w_theta_sin = [0.5, 0.2, 2 * np.pi]  # Bound widths

Model_used[id_model] = {
    "name": "sin-1",
    "family": "sin",
    "m theta": m_theta_sin,
    "w theta": w_theta_sin
}
id_model += 1  # Increment model ID

#########################################################
# Polynomial models of various orders

# Predefined standard deviations for polynomial coefficients
sigma_poly=[[0.215],
        [0.2150, 0.3812],
        [0.2409, 0.3812, 0.4191],
        [0.2409, 0.2812, 0.4192, 0.2539],
        [0.2258, 0.2812, 0.3572, 0.2539, 0.1363],
        [0.2258, 0.2877, 0.3572, 0.2310, 0.1364, 0.0654],
        [0.2268, 0.2877, 0.3611, 0.2310, 0.1264, 0.0655, 0.0426],
        [0.2268, 0.2870, 0.3611, 0.2316, 0.1264, 0.0601, 0.0427, 0.0326],
        [0.2268, 0.2871, 0.3609, 0.2316, 0.1262, 0.0601, 0.0383, 0.0327, 0.0280],
        [0.2268, 0.2871, 0.3609, 0.2317, 0.1261, 0.0593, 0.0384, 0.0289, 0.0281, 0.0244],
        [0.2267, 0.2871, 0.3609, 0.2317, 0.1261, 0.0592, 0.0374, 0.0289, 0.0249, 0.0246, 0.0217],
        [0.2267, 0.2870, 0.3609, 0.2316, 0.1261, 0.0592, 0.0373, 0.0281, 0.0250, 0.0218, 0.0218, 0.0196],
        [0.2267, 0.2871, 0.3609, 0.2316, 0.1260, 0.0592, 0.0373, 0.0280, 0.0242, 0.0218, 0.0193, 0.0198, 0.0181],
        [0.2267, 0.2871, 0.3609, 0.2315, 0.1261, 0.0592, 0.0373, 0.0279, 0.0239, 0.0210, 0.0194, 0.0176, 0.0183, 0.0171]
        ]



for order in range(0, 10):  # Define models up to order 9
    m_theta_poly = [0] * (order + 1)  # Mean coefficients
    w_theta_poly = [
        min(1.55, sigma_poly[order][k] * 7) for k in range(order + 1)
    ]  # Compute bound widths for each coefficient
    print(order,np.array(w_theta_poly)/2)
    Model_used[id_model] = {
        "name": f"poly-{order}",
        "family": "poly",
        "order": order,
        "m theta": m_theta_poly,
        "w theta": w_theta_poly
    }
    id_model += 1  # Increment model ID

#########################################################
# Predictive samples models with various delays and orders
for eta in range(0, 2):  # Test delays of 0 and 1
    for order in range(1, 3):  # Test predictor orders 1 and 2
        m_theta_pred_samples = [0] * order  # Mean coefficients
        w_theta_pred_samples = [2] * order  # Bound widths for coefficients

        Model_used[id_model] = {
            "name": f"samp.-{order}-{eta}",
            "family": "pred samples",
            "model used": 0,  # Reference to the base model
            "order": order,
            "eta": eta,
            "m theta": m_theta_pred_samples,
            "w theta": w_theta_pred_samples
        }
        id_model += 1  # Increment model ID

#########################################################
# Predictive parametric models with scaling factors
for factor in [2, 10]:  # Test scaling factors 2 and 10
    id_previous_model = 0  # Use the "none" model as the base

    # Compute prior distribution for parametric predictive models
    m_theta_pred_para = Model_used[id_previous_model]["m theta"]  # Mean
    w_theta_pred_para = np.array(Model_used[id_previous_model]["w theta"]) / factor  # Scaled bounds

    Model_used[id_model] = {
        "name": f"para.-{factor}",
        "family": "pred para",
        "model used": id_previous_model,
        "factor": factor,
        "m theta": m_theta_pred_para,
        "w theta": w_theta_pred_para
    }
    id_model += 1  # Increment model ID

#########################################################
# Recap: Display all defined models
print("Recap: Set of models")
for id_model in Model_used:
    print(f"id model: {id_model}, name: {Model_used[id_model]['name']}")


"""
Set of methods for residual compression
"""
Residual_used = {}

# Define different residual compression methods
Residual_used[0] = {"name": "none"}         # No residual compression
Residual_used[1] = {"name": "DCT+BPC"}      # Discrete Cosine Transform + Binary Plane Coding
Residual_used[2] = {"name": "DWT+BPC"}      # Discrete Wavelet Transform + Binary Plane Coding

# Recap of residual compression methods
print("Recap: Set of methods for residual compression")
for id_residual in Residual_used:
    print(f"id residual: {id_residual}, name: {Residual_used[id_residual]['name']}")

"""
Initialisation of MMC (Model, Method, Compression)
"""


# Initialize MMC for each phase
dict_MMC_enc = {}  # Dictionary to store encoding instances per phase
dict_MMC_dec = {}  # Dictionary to store decoding instances per phase
for k in range(nb_phase):
    dict_MMC_enc[k] = Encode_one_window(fn, fs, N, Model_used, Residual_used)  # Encoder instance
    dict_MMC_dec[k] = Decode_one_window(fn, fs, N, Model_used, Residual_used)  # Decoder instance

# Initialize storage for reconstructed signals and metrics
x_real = np.zeros((nb_signal, nb_phase, N * nb_w))      # signals (original)
x_rec = np.zeros((nb_signal, nb_phase, N * nb_w))      # Reconstructed signals (model + residual)
x_model = np.zeros((nb_signal, nb_phase, N * nb_w))    # Reconstructed model components
x_residual = np.zeros((nb_signal, nb_phase, N * nb_w)) # Reconstructed residual components

# Performance metrics
SNR = np.zeros((nb_signal, nb_phase, nb_w))       # Signal-to-Noise Ratio (model + residual)
SNR_m = np.zeros((nb_signal, nb_phase, nb_w))     # SNR for model only
SNR_r = np.zeros((nb_signal, nb_phase, nb_w))     # SNR for residual only
RMSE = np.zeros((nb_signal, nb_phase, nb_w))      # Root Mean Square Error

# Bit rate allocation
R_m = np.zeros((nb_signal, nb_phase, nb_w))       # Bit rate for the model
R_r = np.zeros((nb_signal, nb_phase, nb_w))       # Bit rate for the residual
R_h = np.zeros((nb_signal, nb_phase, nb_w))       # Bit rate for headers
R_unused = np.zeros((nb_signal, nb_phase, nb_w))  # Unused bit rate

# Model and method indices
L = np.zeros((nb_signal, nb_phase, nb_w))         # Index of residual compression methods used
M = np.zeros((nb_signal, nb_phase, nb_w))         # Index of models used



#Record the start time
start_time = time.time()
# Loop over signals to encode
for id_signal in range(nb_signal):
    # Retrieve signal components (voltage and current phases)
    v1 = get_signal(id_signal)[0]
    v2 = get_signal(id_signal)[1]
    v3 = get_signal(id_signal)[2]
    i1 = get_signal(id_signal)[3]
    i2 = get_signal(id_signal)[4]
    i3 = get_signal(id_signal)[5]

    # Create input array for MMC (first N*nb_w samples per phase)
    x = [
        v1[:N * nb_w], v2[:N * nb_w], v3[:N * nb_w],
        i1[:N * nb_w], i2[:N * nb_w], i3[:N * nb_w]
    ]

    # Loop over each phase
    for phase in range(nb_phase):
        # Loop over windows
        x_real[id_signal][phase]=x[phase]
        for w in range(nb_w):
            # Extract current window
            x_test = x[phase][w * N:(w + 1) * N]

            # Handle the memory of previously encoded windows
            if w == 0:  # No previous window for the first frame
                x_previous = np.zeros(N * 2)
            elif w == 1:  # For the second frame, only use the last window
                x_previous = np.zeros(N * 2)
                x_previous[N:2 * N] = x_rec[id_signal][phase][(w - 1) * N:w * N]
            else:  # For all other frames, use the last two windows
                x_previous = x_rec[id_signal][phase][(w - 2) * N:w * N]

            # Encode the current window
            dict_MMC_enc[phase].MMC_enc(x_test, x_previous, n_tot)
            SNRenc = get_snr(x_test, dict_MMC_enc[phase].x_rec_enc)

            # Decode the encoded window
            dict_MMC_dec[phase].MMC_dec(dict_MMC_enc[phase].code, x_previous)
            SNRdec = get_snr(x_test, dict_MMC_dec[phase].x_rec_dec)

            # Check for encoding/decoding errors
            if SNRenc != SNRdec:
                print(SNRenc, SNRdec)
                print("ERROR in encoding/decoding process!")

            # Save the outputs
            x_model[id_signal][phase][w * N:(w + 1) * N] = dict_MMC_enc[phase].x_model_enc
            x_residual[id_signal][phase][w * N:(w + 1) * N] = dict_MMC_enc[phase].x_residual_enc
            x_rec[id_signal][phase][w * N:(w + 1) * N] = (
                x_model[id_signal][phase][w * N:(w + 1) * N] +
                x_residual[id_signal][phase][w * N:(w + 1) * N]
            )

            # Calculate performance metrics
            SNR[id_signal][phase][w] = get_snr(x_test, x_rec[id_signal][phase][w * N:(w + 1) * N])
            SNR_m[id_signal][phase][w] = get_snr(x_test, x_model[id_signal][phase][w * N:(w + 1) * N])
            SNR_r[id_signal][phase][w] = get_snr(
                x_test - x_model[id_signal][phase][w * N:(w + 1) * N],
                x_residual[id_signal][phase][w * N:(w + 1) * N]
            )
            RMSE[id_signal][phase][w] = get_rmse(x_test, x_rec[id_signal][phase][w * N:(w + 1) * N])

            # Rate metrics
            R_m[id_signal][phase][w] = dict_MMC_enc[phase].nx_enc
            R_r[id_signal][phase][w] = dict_MMC_enc[phase].nr_enc
            R_h[id_signal][phase][w] = (
                dict_MMC_enc[phase].n_kx_enc +
                dict_MMC_enc[phase].n_kr_enc +
                dict_MMC_enc[phase].nm_enc +
                dict_MMC_enc[phase].nl_enc +
                dict_MMC_enc[phase].n_nx_enc +
                dict_MMC_enc[phase].n_nr_enc
            )
            R_unused[id_signal][phase][w] = n_tot - R_m[id_signal][phase][w] - R_r[id_signal][phase][w] - R_h[id_signal][phase][w]

            # Save the model and method indices
            M[id_signal][phase][w] = dict_MMC_enc[phase].id_model_enc
            L[id_signal][phase][w] = dict_MMC_enc[phase].id_residual_enc

            # Print summary for the current window
            print(f"id={id_signal:3}, w={w+1:3}, ph:{phase+1:1}", end='')
            print(f", n_tot={len(dict_MMC_enc[phase].code):3}", end='')
            print(f", SNR={SNR[id_signal][phase][w]:5.2f} dB", end='')
            print(f", RMSE={RMSE[id_signal][phase][w]:6.2f} V", end='')
            print(f", m={dict_MMC_enc[phase].m_enc:10}, l={dict_MMC_enc[phase].l_enc:7}", end='')
            print(f", SNR_m={SNR_m[id_signal][phase][w]:4.1f} dB, SNR_r={SNR_r[id_signal][phase][w]:4.1f} dB", end='')
            print(f", nh=nm+n_kx+n_nx+nl+n_kr+n_nr={dict_MMC_enc[phase].nm_enc:1}+{dict_MMC_enc[phase].n_kx_enc:1}+{dict_MMC_enc[phase].n_nx_enc:1}+{dict_MMC_enc[phase].nl_enc:1}+{dict_MMC_enc[phase].n_kr_enc:1}+{dict_MMC_enc[phase].n_nr_enc:1}={dict_MMC_enc[phase].nm_enc + dict_MMC_enc[phase].n_kx_enc + dict_MMC_enc[phase].n_nx_enc + dict_MMC_enc[phase].nl_enc + dict_MMC_enc[phase].n_kr_enc + dict_MMC_enc[phase].n_nr_enc:2} b", end='')
            print(f", nx={dict_MMC_enc[phase].nx_enc} b, nr={dict_MMC_enc[phase].nr_enc:4} b", end='') 
            print(f", kx={dict_MMC_enc[phase].kx_enc:3}, kr={dict_MMC_enc[phase].kr_enc:2}", end="")
            print('')
        
        

#Record the end time
end_time = time.time()

#Calculate the elapsed time
elapsed_time = end_time - start_time

print("time to encode {:.0f} signal(s) ({} phase(s)) of {:.2f} seconde(s): {:.2f} secondes".format(nb_signal,nb_phase,N*nb_w/fs,elapsed_time))    
 
print("Number of DCT + DWT transforms evaluated per window: {:.1f}".format(dict_MMC_enc[0].max_size_Nx / (nb_signal * nb_w)))
"""
Visualization of outputs
"""
for id_signal in range(nb_signal):  

    # Define the time vector for visualization
    t = np.linspace(0, (nb_w) * (N - 1) * (1 / fs), nb_w * N)
    
    # Loop through each phase for visualization
    for phase in range(nb_phase):
        #### Reconstructed signal
        plt.figure(figsize=(10, 4), dpi=100)
        plt.plot(t, x_real[id_signal][phase] / 1000, lw=1, label='x (original)')
        plt.plot(t, x_rec[id_signal][phase] / 1000, lw=1, label="x_rec (reconstructed)")
        plt.xlabel('t (s)')
        plt.ylabel('Magnitude x10³')
        plt.legend()
        plt.title(f"Reconstructed Signal: {id_signal}, Phase: {phase + 1}, "
                  f"SNR Mean = {np.mean(SNR[id_signal][phase]):.2f} dB, "
                  f"RMSE Mean = {np.mean(RMSE[id_signal][phase]):.2f} V")
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
    
        #### Reconstruction error
        plt.figure(figsize=(10, 4), dpi=100)
        plt.plot(t, (x_real[id_signal][phase] - x_rec[id_signal][phase]), lw=1, label='x - x_rec (error)')
        plt.xlabel('t (s)')
        plt.ylabel('Magnitude')
        plt.title(f"Reconstruction Error: {id_signal}, Phase: {phase + 1}")
        plt.legend()
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
    
        #### RMSE for each window
        plt.figure(figsize=(10, 4), dpi=100)
        plt.plot([t[k] for k in range(0, nb_w * N, N)], RMSE[id_signal][phase], '-o', lw=1, label='RMSE')
        plt.xlabel('t (s)')
        plt.ylabel('RMSE')
        plt.title(f"RMSE for Each Window, Mean RMSE = {np.mean(RMSE[id_signal][phase]):.0f} V, "
                  f"Signal: {id_signal}, Phase: {phase + 1}")
        plt.legend()
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
    
        #### Reconstructed model
        plt.figure(figsize=(10, 4), dpi=100)
        plt.plot(t, x_real[id_signal][phase] / 1000, lw=1, label='x (original)')
        plt.plot(t, x_model[id_signal][phase] / 1000, lw=1, label='x_model (reconstructed)')
        plt.xlabel('t (s)')
        plt.ylabel('Magnitude x10³')
        plt.legend()
        plt.title(f"Reconstructed Model, Signal: {id_signal}, Phase: {phase + 1}")
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
    
        #### Residual reconstruction
        plt.figure(figsize=(10, 4), dpi=100)
        plt.plot(t, (x_real[id_signal][phase] - x_model[id_signal][phase]) / 1000, lw=1, label='x - x_model (residual)')
        plt.plot(t, x_residual[id_signal][phase] / 1000, lw=1, label='x_residual (reconstructed)')
        plt.xlabel('t (s)')
        plt.ylabel('Magnitude x10³')
        plt.legend()
        plt.title(f"Reconstructed Residuals, Signal: {id_signal}, Phase: {phase + 1}")
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
    
        #### Contribution of two stages (model and residual)
        largeur_barre = (2 * N / 3) / fs  # Bar width for plotting
        plt.figure(figsize=(10, 4), dpi=100)
        plt.bar([t[k] for k in range(0, nb_w * N, N)], R_m[id_signal][phase], width=largeur_barre, color='r')
        plt.bar([t[k] for k in range(0, nb_w * N, N)], R_r[id_signal][phase], width=largeur_barre, bottom=R_m[id_signal][phase], color='b')
        plt.bar([t[k] for k in range(0, nb_w * N, N)], R_h[id_signal][phase], width=largeur_barre, bottom=R_m[id_signal][phase] + R_r[id_signal][phase], color='g')
        plt.bar([t[k] for k in range(0, nb_w * N, N)], R_unused[id_signal][phase], width=largeur_barre, bottom=R_m[id_signal][phase] + R_r[id_signal][phase] + R_h[id_signal][phase], color='y')
        plt.xlabel('t (s)')
        plt.ylabel('Rate')
        plt.legend(["R_m", "R_r", "R_h", "R_unused"])
        plt.title(f"Bits Used, Mean = {np.mean(R_m[id_signal][phase] + R_r[id_signal][phase] + R_h[id_signal][phase]):.2f} bits, "
                  f"Signal: {id_signal}, Phase: {phase + 1}")
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
    
        #### SNR contribution from model and residual
        plt.figure(figsize=(10, 4), dpi=100)
        plt.bar([t[k] for k in range(0, nb_w * N, N)], SNR_m[id_signal][phase], width=largeur_barre, color='r')
        plt.bar([t[k] for k in range(0, nb_w * N, N)], SNR_r[id_signal][phase], width=largeur_barre, bottom=[max(0, SNR_m[id_signal][phase][k]) for k in range(nb_w)], color='b')
        plt.xlabel('t (s)')
        plt.ylabel('SNR (dB)')
        plt.legend(["SNR (Model)", "SNR (Residual)"])
        plt.title(f"SNR Mean: {np.mean(SNR[id_signal][phase]):.2f} dB, Signal: {id_signal}, Phase: {phase + 1}")
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
    
        #### Model index over time
        yticks_labels = [Model_used[id_model]["name"] for id_model in Model_used]
        yticks_positions = np.arange(len(yticks_labels))
        plt.plot([t[k] for k in range(0, nb_w * N, N)], M[id_signal][phase], 'o', lw=1, label='Model Index')
        plt.xlabel('Window Index')
        plt.ylabel('Model Index')
        plt.legend()
        plt.title(f"Model Selection, Signal: {id_signal}, Phase: {phase + 1}")
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.yticks(yticks_positions, yticks_labels)
        plt.show()
    
        #### Residual compression method index over time
        yticks_labels = [Residual_used[id_residual]["name"] for id_residual in Residual_used]
        yticks_positions = np.arange(len(yticks_labels))
        plt.figure(figsize=(10, 4), dpi=100)
        plt.plot([t[k] for k in range(0, nb_w * N, N)], L[id_signal][phase], 'o', lw=1, label='Residual Compression Index')
        plt.xlabel('t (s)')
        plt.ylabel('Residual Compression Method Index')
        plt.legend()
        plt.title(f"Selected Residual Compression Method, Signal: {id_signal}, Phase: {phase + 1}")
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.yticks(yticks_positions, yticks_labels)
        plt.show()
    
