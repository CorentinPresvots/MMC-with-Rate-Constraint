# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 19:09:09 2023

@author: coren
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import pywt

from Antonini import Antonini_Encoder, Antonini_Decoder
from Khan_EZW import Khan_Encoder, Khan_Decoder
from Measures import get_quality, entropy, get_snr
from Normalize import normalize


class Residual_Encoder(Antonini_Encoder, Khan_Encoder):
    def __init__(self, N=128, factor_scale=4):
        """
        Initialize the Residual Encoder class by combining Antonini and Khan encoders.

        Parameters:
        - N: Length of the signal to encode.
        - factor_scale: Scaling factor applied to coefficients.
        """
        self.N = N  # Signal length
        M = 10  # Precision for arithmetic encoding

        # Initialize Antonini encoder with default parameters
        initial_occurrence_first_Antonini = [1, 1]
        initial_occurrence_second_Antonini = [1, 1, 1, 1, 1]
        Antonini_Encoder.__init__(
            self,
            M,
            initial_occurrence_first=initial_occurrence_first_Antonini,
            initial_occurrence_second=initial_occurrence_second_Antonini,
            adaptive=True,
            verbose_AE=False,
        )

        # Initialize parameters for the wavelet encoder (Khan Encoder)
        self.wave_test = "coif5"  # Wavelet type
        self.level = int(np.ceil(np.log2(N)))  # Wavelet decomposition levels
        self.mode = "periodization"  # Wavelet mode

        initial_occurrence_first_Khan = [1, 1]
        initial_occurrence_second_Khan = [1, 1, 1, 1]
        Khan_Encoder.__init__(
            self,
            level=self.level,
            wave_test=self.wave_test,
            M=M,
            initial_occurrence_first_Khan=initial_occurrence_first_Khan,
            initial_occurrence_second_Khan=initial_occurrence_second_Khan,
            adaptive_Khan=True,
            verbose_KE=False,
        )

        self.factor_scale = factor_scale  # Scaling factor for coefficients

    def get_r_DCT_BPC_tilde(self, r, metric, quality, nr, n_sym):
        """
        Encode a residual signal `r` using DCT and Antonini coding.

        Parameters:
        - r: Residual signal to encode.
        - metric: Quality metric ("SNR", "RMSE", or "MSE").
        - quality: Quality target value.
        - nr: Maximum number of bits for encoding.
        - n_sym: Number of symbols for encoding.

        Returns:
        - r_rec_DCT: Reconstructed residual signal after encoding.
        - code_DCT: Encoded bitstream for the residual signal.
        - kr: Scaling factor for normalization.
        - nb_sym_residual: Number of significant symbols encoded.
        """
        # Perform DCT and scale coefficients
        coefs_DCT = dct(r / self.N) / self.factor_scale

        # Normalize the coefficients
        coefs_DCT_n, kr = normalize(coefs_DCT)
       

        # Adjust the quality target based on the metric
        if metric == "RMSE":
            quality_n = quality * 2 ** (-kr) / self.factor_scale
        elif metric == "SNR":
            quality_n = quality
        elif metric == "MSE":
            quality_n = quality * 2 ** (-2 * kr) / self.factor_scale

        # Encode the normalized coefficients using Antonini encoding
        code_DCT, nb_sym_residual = self.get_code_res_Antonini(
            coefs_DCT_n, metric, quality_n, nr, n_sym
        )
        coefs_rec_DCT_n = self.coefs_rec

        # Denormalize the reconstructed coefficients
        coefs_rec_DCT = coefs_rec_DCT_n * self.factor_scale * 2 ** (kr)

        # Perform inverse DCT to reconstruct the residual signal
        r_rec_DCT = self.get_x_rec_Antonini(coefs_rec_DCT)

        # Handle case where all coefficients are zero
        if np.sum(np.abs(coefs_rec_DCT)) == 0:
            r_rec_DCT = np.zeros(self.N)
            code_DCT = []

        # Plot and print additional details if verbose mode is enabled
        if self.verbose:
            # Plot the original and reconstructed coefficients
            plt.figure(figsize=(8, 4), dpi=100)
            plt.plot(coefs_DCT_n, lw=2, label="Original coefficients")
            plt.plot(coefs_rec_DCT_n, lw=2, label="Reconstructed coefficients")
            plt.xlabel("Index")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.title(
                "DCT, Nb sym used / Nb sym max = {} / {}, kr = {}".format(
                    nb_sym_residual, 2 ** n_sym - 1, kr
                )
            )
            plt.grid(which="major", color="#666666", linestyle="-")
            plt.minorticks_on()
            plt.grid(which="minor", color="#999999", linestyle="-", alpha=0.2)
            plt.show()

            # Plot the original and reconstructed residual signals
            plt.figure(figsize=(8, 4), dpi=100)
            plt.plot(r, lw=2, label="Original residual")
            plt.plot(r_rec_DCT, lw=2, label="Reconstructed residual")
            plt.xlabel("Index")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.title("DCT, SNR = {:.2f} dB".format(get_snr(r, r_rec_DCT)))
            plt.grid(which="major", color="#666666", linestyle="-")
            plt.minorticks_on()
            plt.grid(which="minor", color="#999999", linestyle="-", alpha=0.2)
            plt.show()

            # Compute and display entropy details
            p1 = np.array(self.occurrence_first) / np.sum(self.occurrence_first)
            p2 = np.array(self.occurrence_second) / np.sum(self.occurrence_second)
            H1 = np.sum([-p1[k] * np.log2(p1[k] + 1e-16) for k in range(len(p1))])
            H2 = np.sum([-p2[k] * np.log2(p2[k] + 1e-16) for k in range(len(p2))])
            print("p first", p1, "H first = {:.2f} bits".format(H1))
            print("p second", p2, "H second = {:.2f} bits".format(H2))
            print(
                "nr = {} bits, H = {:.2f} bits".format(
                    len(code_DCT),
                    H2 * (-5 + np.sum(self.occurrence_second_true))
                    + H1 * (-2 + np.sum(self.occurrence_first_true)),
                )
            )
            print("symbols", self.symbol, len(self.symbol))

        return r_rec_DCT, code_DCT, kr, nb_sym_residual

    
            
    def get_r_DWT_BPC_tilde(self, r, metric, quality, nr, n_sym):
        """
        Perform residual encoding using Discrete Wavelet Transform (DWT) and Khan encoding.
    
        Parameters:
        - r: Residual signal to encode.
        - metric: Quality metric ("SNR", "RMSE", or "MSE").
        - quality: Quality target value.
        - nr: Maximum number of bits for encoding.
        - n_sym: Number of symbols for encoding.

        Returns:
        - r_rec_DWT: Reconstructed residual signal after encoding.
        - code_DWT: Encoded bitstream for the residual signal.
        - kr: Scaling factor for normalization.
        - nb_sym_residual: Number of significant symbols encoded.
        """
        # Perform Discrete Wavelet Transform (DWT) on the residual signal
        coefs = pywt.wavedec(r, self.wave_test, mode=self.mode, level=self.level)
    
        # Flatten and scale the DWT coefficients
        coefs_DWT = []
        for i in range(self.level + 1):
            # Add approximation and detail coefficients, scaled by level and factor_scale
            coefs_DWT.extend(coefs[i] / (self.level * self.factor_scale))
        coefs_DWT = np.array(coefs_DWT)
    
        # Normalize the coefficients
        coefs_DWT_n, kr = normalize(coefs_DWT)
        

        # Adjust the quality target based on the selected metric
        if metric == "RMSE":
            quality_n = quality * 2 ** (-kr) / self.factor_scale
        elif metric == "SNR":
            quality_n = quality
        elif metric == "MSE":
            quality_n = quality * 2 ** (-2 * kr) / self.factor_scale
    
        # Encode the normalized coefficients using Khan encoding
        code_DWT, nb_sym_residual = self.get_code_res_Khan(
            coefs_DWT_n, metric, quality_n, nr, n_sym
        )
        coefs_rec_DWT_n = self.coefs_rec
    
        # Denormalize the reconstructed coefficients
        coefs_rec_DWT = coefs_rec_DWT_n * self.factor_scale * 2 ** (kr)
    
        # Reconstruct the residual signal using the inverse wavelet transform
        r_rec_DWT = self.get_x_rec_Khan(coefs_rec_DWT)
    
        # Handle case where all coefficients are zero
        if np.sum(np.abs(coefs_rec_DWT)) == 0:
            r_rec_DWT = np.zeros(self.N)
            code_DWT = []
    
        # If verbose mode is enabled, visualize the encoding process
        if self.verbose:
            # Plot original and reconstructed coefficients
            plt.figure(figsize=(8, 4), dpi=100)
            plt.plot(coefs_DWT_n, lw=2, label="Original coefficients")
            plt.plot(coefs_rec_DWT_n, lw=2, label="Reconstructed coefficients")
            plt.xlabel("Index")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.title(
                "DWT, Nb sym used / Nb sym max = {} / {}, kr = {}".format(
                    nb_sym_residual, 2 ** n_sym - 1, kr
                )
            )
            plt.grid(which="major", color="#666666", linestyle="-")
            plt.minorticks_on()
            plt.grid(which="minor", color="#999999", linestyle="-", alpha=0.2)
            plt.show()
    
            # Plot original and reconstructed residual signals
            plt.figure(figsize=(8, 4), dpi=100)
            plt.plot(r, lw=2, label="Original residual")
            plt.plot(r_rec_DWT, lw=2, label="Reconstructed residual")
            plt.xlabel("Index")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.title("DWT, SNR = {:.2f} dB".format(get_snr(r, r_rec_DWT)))
            plt.grid(which="major", color="#666666", linestyle="-")
            plt.minorticks_on()
            plt.grid(which="minor", color="#999999", linestyle="-", alpha=0.2)
            plt.show()
    
        # Return reconstructed signal, encoded bitstream, scaling factor, and symbol count
        return r_rec_DWT, code_DWT, kr, nb_sym_residual


class Residual_Decoder(Antonini_Decoder, Khan_Decoder):
    """
    A decoder that combines both DCT (Antonini) and DWT (Khan) decoding methods for residual signal reconstruction.
    
    Inherits:
    - Antonini_Decoder: For DCT-based encoding/decoding.
    - Khan_Decoder: For DWT-based encoding/decoding.
    """
    def __init__(self, N=128, factor_scale=4):
        """
        Initialize the Residual_Decoder with parameters for both Antonini and Khan decoders.

        Parameters:
        - N (int): Length of the signal to be decoded.
        - factor_scale (float): Scaling factor for the coefficients.
        """
        # Initialize Antonini decoder parameters
        M = 10
        initial_occurrence_first_Antonini = [1, 1]
        initial_occurrence_second_Antonini = [1, 1, 1, 1, 1]
        Antonini_Decoder.__init__(
            self,
            N=N,
            M=M,
            initial_occurrence_first=initial_occurrence_first_Antonini,
            initial_occurrence_second=initial_occurrence_second_Antonini,
            adaptive=True,
            verbose_AD=False
        )
        
        # Initialize wavelet parameters for Khan decoder
        self.wave_test = "coif5"  # Wavelet type for decomposition and reconstruction
        self.level = int(np.ceil(np.log2(N)))  # Decomposition level based on signal length
        self.mode = 'periodization'  # Signal extension mode
        
        initial_occurrence_first_Khan = [1, 1]
        initial_occurrence_second_Khan = [1, 1, 1, 1]
        Khan_Decoder.__init__(
            self,
            level=self.level,
            wave_test=self.wave_test,
            N=N,
            M=M,
            initial_occurrence_first_Khan=initial_occurrence_first_Khan,
            initial_occurrence_second_Khan=initial_occurrence_second_Khan,
            adaptive_Khan=True,
            verbose_KD=False
        )
        
        # Scaling factor for normalization
        self.factor_scale = factor_scale
    
    def get_r_DCT_BPC_tilde(self, code, nb_sym_residual, kr):
        """
        Reconstruct the residual signal from its encoded DCT-based bitstream.
        
        Parameters:
        - code: Encoded bitstream of the DCT-based residual.
        - nb_sym_residual: Number of significant symbols encoded in the residual.
        - kr: Normalization scaling factor applied during encoding.
        
        Returns:
        - r_rec: Reconstructed residual signal.
        - nr_rec: Number of bits used in the reconstruction.
        """
        # Decode the DCT coefficients from the encoded bitstream
        coefs_rec, nr_rec = self.get_coefs_rec_Antonini(code, nb_sym_residual)
        
        # Denormalize the coefficients
        coefs_rec *= self.factor_scale * 2 ** (kr)
        
        # Reconstruct the residual signal using inverse DCT
        r_rec = idct(coefs_rec) / 2
        
        return r_rec, nr_rec
    def get_r_DWT_BPC_tilde(self, code, nb_sym_residual, kr):
        """
        Reconstruct the residual signal from its encoded DWT-based bitstream.
        
        Parameters:
        - code: Encoded bitstream of the DWT-based residual.
        - nb_sym_residual: Number of significant symbols encoded in the residual.
        - kr: Normalization scaling factor applied during encoding.
        
        Returns:
        - r_rec: Reconstructed residual signal.
        - nr_rec: Number of bits used in the reconstruction.
        """
        # Decode the DWT coefficients from the encoded bitstream
        coefs_L, nr_rec = self.get_coefs_rec_Khan(code, nb_sym_residual)
        
        # Denormalize the coefficients
        coefs_L *= self.factor_scale * 2 ** (kr)
        
        # Reconstruct the residual signal using the inverse wavelet transform
        r_rec = self.get_x_rec_Khan(coefs_L)
        
        return r_rec, nr_rec

# Programme principal
if __name__ == "__main__":
    
    # Define the performance metric and target quality
    metric = "RMSE"  # Options: "SNR", "RMSE", "MSE"
    quality = 100  # For SNR: quality in -dB; for RMSE: in volts; for MSE: in volts squared
    
    # Determine the unit based on the metric
    if metric == "SNR":
        unity = "-dB"
    elif metric == "RMSE":
        unity = "V"
    elif metric == "MSE":
        unity = "V^2"
    
    nr=100 # rate constrain, priority on the quality constraint
    n_sym=10 #bits pour transmettre le nombre de symbols à décoder pour stoper le décodeur entropique 
   

    N=128 
    fn=50
    fs=6400
    

    t=np.linspace(0,(N-1)/fs,N)
    
    """
    x_test=np.array([-4.0809643 , -3.96240142, -3.73425528, -3.41351299, -3.02334315,
            -2.59063723, -2.14321518, -1.70702905, -1.30369448, -0.94863387,
            -0.65003431, -0.4087179 , -0.21890645, -0.06975024,  0.05260231,
             0.16268041,  0.27342325,  0.39426445,  0.52981029,  0.67926914,
             0.83668616,  0.99192939,  1.13227352,  1.24435112,  1.31619572,
             1.33909342,  1.30899047,  1.22726732,  1.10077829,  0.94115597,
             0.76347886,  0.5844851 ,  0.42057391,  0.28586227,  0.19055323,
             0.13982755,  0.13339713,  0.16576834,  0.22716765,  0.30499359,
             0.38559152,  0.45610737,  0.50617123,  0.52918936,  0.52308148,
             0.49037943,  0.43769302,  0.37463627,  0.31238053,  0.2620497 ,
             0.23319072,  0.23253769,  0.26324173,  0.32466938,  0.41278792,
             0.52107058,  0.64177805,  0.76741833,  0.89215984,  1.01297968,
             1.13036613,  1.24845933,  1.37459586,  1.51831065,  1.68993107,
             1.89896075,  2.15248632,  2.45384254,  2.80173975,  3.18999602,
             3.60793274,  4.04139851,  4.47429426,  4.89039675,  5.27522782,
             5.61770168,  5.91130466,  6.15461915,  6.35108983,  6.50803365,
             6.63500263,  6.74170541,  6.83576626,  6.92063805,  6.9939828 ,
             7.04678769,  7.06340208,  7.02256938,  6.89940263,  6.66812742,
             6.30530847,  5.79319938,  5.12282016,  4.29637919,  3.32871499,
             2.24753277,  1.09233963, -0.08787456, -1.23802241, -2.30106064,
            -3.22249015, -3.9547722 , -4.46122388, -4.71900897, -4.72093603,
            -4.47590115, -4.00795412, -3.35410751, -2.56113157, -1.68167119,
            -0.77007434,  0.12166975,  0.94751932,  1.67020415,  2.26309246,
             2.71102054,  3.0101173 ,  3.16675422,  3.19582494,  3.11860142,
             2.96042176,  2.74844196,  2.509635  ,  2.26915642,  2.04912545,
             1.86780678,  1.73912874,  1.67244538])*1000
    """
    
    x_test=np.array([-2.37775514e+00,  1.16706600e+00,  6.09402072e+00,  8.57629984e+00,
             8.45296710e+00,  7.39598358e+00,  4.39914901e+00,  2.09810931e+00,
             6.52261880e-01, -2.11283060e+00, -5.63644268e+00, -7.53064395e+00,
            -7.83295029e+00, -7.29569052e+00, -5.32152938e+00, -3.85188313e+00,
            -2.95883572e+00, -1.16656080e+00,  1.11819189e+00,  2.37364249e+00,
             2.62217004e+00,  2.30504572e+00,  1.03452568e+00,  8.20930245e-02,
            -5.17609822e-01, -1.73482010e+00, -3.31053393e+00, -4.57297101e+00,
            -5.57605197e+00, -6.46821806e+00, -7.09499451e+00, -8.11657164e+00,
            -9.55695485e+00, -1.09623649e+01, -1.24780176e+01, -1.36903112e+01,
            -1.45132069e+01, -1.52463627e+01, -1.57621195e+01, -1.61849066e+01,
            -1.65348904e+01, -1.66343584e+01, -1.65055794e+01, -1.65740223e+01,
            -1.68472193e+01, -1.71305121e+01, -1.75286287e+01, -1.77441931e+01,
            -1.77520248e+01, -1.77845991e+01, -1.77728173e+01, -1.78646658e+01,
            -1.81000736e+01, -1.83590242e+01, -1.86903036e+01, -1.90610260e+01,
            -1.94637746e+01, -1.99601532e+01, -2.05452847e+01, -2.02581418e+01,
            -1.90139761e+01, -1.72636760e+01, -1.47352986e+01, -1.19984785e+01,
            -9.07943289e+00, -5.44933089e+00, -1.21609890e+00,  1.66058263e+00,
             3.06470248e+00,  3.85851731e+00,  3.50949907e+00,  4.52595385e+00,
             7.09907649e+00,  9.54871047e+00,  1.24906336e+01,  1.85875747e+01,
             2.80019264e+01,  3.99910257e+01,  5.53468954e+01,  6.22179342e+01,
             5.96349341e+01, -3.37756049e+00, -1.35910032e+02, -1.81858436e+02,
            -1.44694555e+02, -1.15084233e+02, -6.09341372e+01, -3.38245944e+01,
            -4.63737767e+01, -5.12605780e+01, -6.38148554e+01, -6.48554906e+01,
            -4.84469913e+01, -3.67466459e+01, -2.39702853e+01, -1.29529651e+01,
            -6.63490229e+00,  9.01941745e-01,  8.05998859e+00,  1.46859915e+01,
             2.13791778e+01,  2.66712091e+01,  3.06361956e+01,  3.32784507e+01,
             3.42560356e+01,  3.35572446e+01,  3.09904739e+01,  3.00936931e+01,
             3.11027771e+01,  3.20800805e+01,  3.39451053e+01,  3.43545436e+01,
             3.30585021e+01,  3.19726317e+01,  3.05360384e+01,  2.84391194e+01,
             2.58564065e+01,  2.25153785e+01,  1.83748079e+01,  1.40153467e+01,
             9.40477101e+00,  4.21902890e+00, -1.43249998e+00, -5.15560619e+00,
            -6.73392254e+00, -7.21331730e+00, -5.90087098e+00, -4.27545461e+00])*1000
    
    
    
    
    l = Residual_Encoder(N, factor_scale=4)
    
    # Reconstruct residual using DCT (Antonini) encoder
    x_dec_enc_DCT, code_DCT, kr_DCT, nb_sym_DCT = l.get_r_DCT_BPC_tilde(x_test, metric, quality, nr, n_sym)
    
    # Print results for DCT encoding
    print("Nb bits used / Nb bits max = {} / {} bits DCT".format(len(code_DCT), nr),
          "{} = {} / {} {}".format(metric, get_quality(x_test, x_dec_enc_DCT, metric), quality, unity))
    
    # Calculate entropy for DCT
    occurrence_first = np.array(l.occurrence_first_true) - 1
    occurrence_second = np.array(l.occurrence_second_true) - 1
    
    # Entropy for first and second pass
    if np.sum(occurrence_first) != 0:
        p_first = occurrence_first / np.sum(occurrence_first)
        H_first = entropy(p_first)
    else:
        H_first = 0
    
    if np.sum(occurrence_second) != 0:
        p_second = occurrence_second / np.sum(occurrence_second)
        H_second = entropy(p_second)
    else:
        H_second = 0
    
    print("Occurrence of symbols for first pass (DCT):", occurrence_first)
    print("Occurrence of symbols for second pass (DCT):", occurrence_second)
    print("H first (DCT) = {:.2f} bits".format(H_first))
    print("H second (DCT) = {:.2f} bits".format(H_second))
    
    H_coefs_DCT = np.sum(occurrence_first) * H_first + np.sum(occurrence_second) * H_second
    print("H_coefs_DCT = {:.2f} bits".format(H_coefs_DCT))
    
    # Calculate entropy for DCT
    occurrence_first = np.array(l.occurrence_first_true) - 1
    occurrence_second = np.array(l.occurrence_second_true) - 1
    
    # Entropy for first and second pass
    if np.sum(occurrence_first) != 0:
        p_first = occurrence_first / np.sum(occurrence_first)
        H_first = entropy(p_first)
    else:
        H_first = 0
    
    if np.sum(occurrence_second) != 0:
        p_second = occurrence_second / np.sum(occurrence_second)
        H_second = entropy(p_second)
    else:
        H_second = 0
    
    print("Occurrence of symbols for first pass (DCT):", occurrence_first)
    print("Occurrence of symbols for second pass (DCT):", occurrence_second)
    print("H first (DCT) = {:.2f} bits".format(H_first))
    print("H second (DCT) = {:.2f} bits".format(H_second))
    
    H_coefs_DCT = np.sum(occurrence_first) * H_first + np.sum(occurrence_second) * H_second
    print("H_coefs_DCT = {:.2f} bits".format(H_coefs_DCT))
    
    
    # Reconstruct residual using DWT (Khan) encoder
    x_dec_enc_DWT, code_DWT, kr_DWT, nb_sym_DWT = l.get_r_DWT_BPC_tilde(x_test, metric, quality, nr, n_sym)
    print("Nb sym used / Nb sym max = {} / {} bits DWT".format(nb_sym_DWT, 2**n_sym - 1),
          "{} = {} / {} {}".format(metric, get_quality(x_test, x_dec_enc_DWT, metric), quality, unity))
    
    # Entropy for DWT
    occurrence_first_Khan = np.array(l.occurrence_first_true_Khan) - 1
    occurrence_second_Khan = np.array(l.occurrence_second_true_Khan) - 1
    
    # Entropy for first and second pass
    if np.sum(occurrence_first_Khan) != 0:
        p_first = occurrence_first_Khan / np.sum(occurrence_first_Khan)
        H_first = entropy(p_first)
    else:
        H_first = 0
    
    if np.sum(occurrence_second_Khan) != 0:
        p_second = occurrence_second_Khan / np.sum(occurrence_second_Khan)
        H_second = entropy(p_second)
    else:
        H_second = 0
    
    print("Occurrence of symbols for first pass (DWT):", occurrence_first_Khan)
    print("Occurrence of symbols for second pass (DWT):", occurrence_second_Khan)
    print("H first (DWT) = {:.2f} bits".format(H_first))
    print("H second (DWT) = {:.2f} bits".format(H_second))
    
    H_coefs_DWT = np.sum(occurrence_first_Khan) * H_first + np.sum(occurrence_second_Khan) * H_second
    print("H coefs DWT = {:.2f} bits".format(H_coefs_DWT))
    
    # Initialize residual decoder
    l_dec = Residual_Decoder(N, factor_scale=4)
    
    # Decode residuals using DCT and DWT decoders
    x_dec_DCT, nr_dec_DCT = l_dec.get_r_DCT_BPC_tilde(code_DCT, nb_sym_DCT, kr_DCT)
    x_dec_DWT, nr_dec_DWT = l_dec.get_r_DWT_BPC_tilde(code_DWT, nb_sym_DWT, kr_DWT)
    
    # Print results for decoding
    print("Nb sym used / Nb sym max = {} / {} bits (DCT)".format(nb_sym_DCT, int(2**n_sym - 1)),
          "{} = {} / {} {}".format(metric, get_quality(x_test, x_dec_DCT, metric), quality, unity))
    print("Nb sym used / Nb sym max = {} / {} bits (DWT)".format(nb_sym_DWT, int(2**n_sym - 1)),
          "{} = {} / {} {}".format(metric, get_quality(x_test, x_dec_DWT, metric), quality, unity))
    
    # Visualization for DCT reconstruction
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(t, x_test, lw=2, label='x')
    plt.plot(t, x_dec_enc_DCT, lw=2, label='x dec encoder, {}_enc = {:.5f}/ {} {}'.format(
        metric, get_quality(x_test, x_dec_enc_DCT, metric), quality, unity))
    plt.plot(t, x_dec_DCT, lw=2, label='x dec decoder, {}_dec = {:.5f} / {} {}'.format(
        metric, get_quality(x_test, x_dec_DCT, metric), quality, unity))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Reconstructed Residual DCT on nx={}, H coefs={:.1f}".format(len(code_DCT), H_coefs_DCT))
    plt.grid()
    plt.show()
    
    # Visualization for DWT reconstruction
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(t, x_test, lw=2, label='x')
    plt.plot(t, x_dec_enc_DWT, lw=2, label='x dec encoder, {}_enc = {:.5f}/ {} {}'.format(
        metric, get_quality(x_test, x_dec_enc_DWT, metric), quality, unity))
    plt.plot(t, x_dec_DWT, lw=2, label='x dec decoder, {}_dec = {:.5f} / {} {}'.format(
        metric, get_quality(x_test, x_dec_DWT, metric), quality, unity))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Reconstructed Residual DWT on nx={}, H coefs={:.1f} bits".format(len(code_DWT), H_coefs_DWT))
    plt.grid()
    plt.show()
