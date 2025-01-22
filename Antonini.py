# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 19:10:30 2023

@author: coren
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate
from scipy.fftpack import dct, idct
from Measures import get_quality, entropy
from Context_Arithmetic import Context_Aritmetic_Encoder, Context_Aritmetic_Decoder

class Antonini_Encoder(Context_Aritmetic_Encoder):
    """
    Antonini_Encoder is an adaptive arithmetic encoder tailored for specific
    signal compression tasks using threshold-based significance testing.
    """
    def __init__(self, M=9, initial_occurrence_first=[1, 1],
                 initial_occurrence_second=[1, 1, 1, 1, 1],
                 adaptive=True, verbose_AE=False):
        """
        Constructor for the Antonini_Encoder class.

        Parameters:
        - M: Context size for arithmetic encoding.
        - initial_occurrence_first: Initial probabilities for the first pass.
        - initial_occurrence_second: Initial probabilities for the second pass.
        - adaptive: Boolean, whether the arithmetic encoder is adaptive.
        - verbose_AE: Boolean, debug mode for verbose outputs.
        """
        self.initial_occurrence_first = initial_occurrence_first  # Initial probabilities for the first pass.
        self.initial_occurrence_second = initial_occurrence_second  # Initial probabilities for the second pass.
        self.verbose_AE = verbose_AE  # Debug mode for encoder.
        self.adaptive = adaptive  # Adaptive arithmetic encoder flag.

        # Alphabet definitions for both passes
        self.alphabet_first = ['R', 'S']  # 'R' (non-significant), 'S' (significant).
        self.alphabet_second = ['0', '1', '-', '+', 'E']  # '0', '1', '+' (sequence length), '-' (polarity), 'E' (end).

        # Initialize the parent class (Context_Aritmetic_Encoder)
        Context_Aritmetic_Encoder.__init__(self, M)

        # Save the current state of the arithmetic encoder
        self.memoire_l = self.l
        self.memoire_h = self.h
        self.memoire_follow = self.follow

    def get_x_rec_Antonini(self, coefs):
        """
        Reconstruct the signal from coefficients using the inverse DCT (IDCT).
        
        Parameters:
        - coefs: List of coefficients to reconstruct.
        
        Returns:
        - Reconstructed signal.
        """
        return idct(np.array(coefs)) / 2

    def reset_Antonini_Encoder(self, coefs, metric, quality, nr, n_sym):
        """
        Reset and initialize the encoder parameters for a new encoding session.

        Parameters:
        - coefs: Coefficients to be encoded.
        - metric: Metric for quality evaluation (e.g., "MSE", "SNR").
        - quality: Desired quality constraint.
        - nr: Maximum allowed number of residual bits.
        - n_sym: Number of symbols that can be encoded.
        """
        # Reset the parent class state
        self.reset_Context_Aritmetic_Encoder()

        # Signal and encoding parameters
        self.N = len(coefs)  # Length of the coefficients vector.
        self.code = []  # Encoded binary sequence.
        self.flag = np.zeros(self.N)  # Flags for significance testing (0: not significant, 1: significant).
        self.coefs = list(coefs)  # Coefficients to encode.
        self.res = list(self.coefs)  # Residuals (updated after encoding significant coefficients).
        self.coefs_rec = np.zeros(self.N)  # Reconstructed coefficients.
        self.threshold = 0.5  # Initial significance threshold (coefficients range from -1 to 1).

        # Symbol and probability tracking
        self.symbol = []  # List of coded symbols.
        self.occurrence_first = list(self.initial_occurrence_first)  # Occurrences for the first pass.
        self.occurrence_second = list(self.initial_occurrence_second)  # Occurrences for the second pass.
        self.cumulate_occurrence_first = list(accumulate(self.initial_occurrence_first))  # Cumulative probabilities (first pass).
        self.cumulate_occurrence_second = list(accumulate(self.initial_occurrence_second))  # Cumulative probabilities (second pass).
        self.nb_sym = 0  # Count of significant coefficients encoded.

        # Quality constraints and metric initialization
        self.metric = metric  # Metric for quality evaluation.
        self.quality_A = quality  # Quality constraint.
        self.occurrence_first_true = list(self.initial_occurrence_first)  # True occurrences (first pass).
        self.occurrence_second_true = list(self.initial_occurrence_second)  # True occurrences (second pass).

        # Maximum symbols and residuals to encode
        self.nb_sym_max = 2**n_sym - 1
        self.nr_max = nr

        # Initialize quality parameters based on the chosen metric
        if self.metric == "MSE":
            self.cst = 256
        elif self.metric == "RMSE":
            self.cst = 16 / 2  # Adjustment for RMSE scaling.
        elif self.metric == "SNR":
            x_test = self.get_x_rec_Antonini(coefs)
            MSE_x = get_quality(x_test, np.zeros(self.N), "MSE")
            self.quality_A = MSE_x / 10**(-quality / 10)  # Convert SNR to MSE equivalent.
            self.metric = "MSE"
            self.cst = 64
    
    
    def get_symbol_first(self):
        """
        Encodes coefficients during the first pass for the current bit plane.
    
        Returns:
        --------
        int:
            - Returns 1 if the entire bit plane has been encoded.
            - Returns 0 if the length of the code for the residual exceeds the bitrate constraint.
        """
        for i in range(self.N):  # Iterate through all flags
            if self.flag[i] != 0:  # Check if the coefficient is flagged as significant
                if self.res[i] >= 0:  # Positive residual
                    x = 1  # Symbol 'S' (significant)
                    code_first = self.encode_one_symbol(
                        x, self.occurrence_first, self.cumulate_occurrence_first
                    )
    
                    # Update variables if bitrate and symbol constraints are respected
                    if len(self.code) + len(code_first) + self.follow + 2 <= self.nr_max and self.nb_sym + 1 <= self.nb_sym_max:
                        # Add the symbol to the list and update reconstructed coefficients
                        self.symbol.append("S")
                        self.coefs_rec[i] += self.threshold / 2
                        self.res[i] -= self.threshold / 2
                        self.code.extend(code_first)
    
                        # Update adaptive occurrences if enabled
                        if self.adaptive:
                            self.occurrence_first[x] += 1
                            self.cumulate_occurrence_first[x] += 1
                        
                        self.occurrence_first_true[x] += 1
                        self.nb_sym += 1
    
                        # Save the current state of the encoder
                        self.memoire_l = self.l
                        self.memoire_h = self.h
                        self.memoire_follow = self.follow
    
    
                        # Evaluate quality
                        q = self.cst * get_quality(self.coefs, self.coefs_rec, self.metric)
                        if q <= self.quality_A:
                            return 0
                    else:
                        return 0
    
                else:  # Negative residual
                    x = 0  # Symbol 'R' (not significant)
                    code_first = self.encode_one_symbol(
                        x, self.occurrence_first, self.cumulate_occurrence_first
                    )
    
                    # Update variables if bitrate and symbol constraints are respected
                    if len(self.code) + len(code_first) + self.follow + 2 <= self.nr_max and self.nb_sym + 1 <= self.nb_sym_max:
                        # Add the symbol to the list and update reconstructed coefficients
                        self.symbol.append("R")
                        self.coefs_rec[i] -= self.threshold / 2
                        self.res[i] += self.threshold / 2
                        self.code.extend(code_first)
    
                        # Update adaptive occurrences if enabled
                        if self.adaptive:
                            self.occurrence_first[x] += 1
                            self.cumulate_occurrence_first[0] += 1
                            self.cumulate_occurrence_first[1] += 1
                        
                        self.occurrence_first_true[x] += 1
                        self.nb_sym += 1
    
                        # Save the current state of the encoder
                        self.memoire_l = self.l
                        self.memoire_h = self.h
                        self.memoire_follow = self.follow
    
          
    
                        # Evaluate quality
                        q = self.cst * get_quality(self.coefs, self.coefs_rec, self.metric)
                        if q <= self.quality_A:
                            return 0
                    else:
                        return 0
    
        # If the entire bit plane has been encoded successfully
        return 1
    


    def get_symbol_second(self):
        """
        Encodes coefficients during the second pass for the current bit plane.
    
        Returns:
        --------
        int:
            - Returns 1 if the entire bit plane has been successfully encoded.
            - Returns 0 if the residual code length exceeds the bitrate constraint.
        """
        count = 0  # Counter for consecutive "R" symbols (run length)
        
        for i in range(self.N):  # Iterate through all flags
            if self.flag[i] == 0:  # Coefficient is not flagged as significant
                if np.abs(self.res[i]) >= self.threshold:  # Significant symbol
                    # Encoding "R" symbols with run-length encoding
                    sym_R = []  # Symbols representing the run length
                    code_R = []  # Encoded run length bits
                    occurrence_second = list(self.occurrence_second)
                    occurrence_second_true = list(self.occurrence_second_true)
                    cumulate_occurrence_second = list(self.cumulate_occurrence_second)
    
                    if count > 0:  # Encode run-length count
                        count_bin = list(bin(count - 1)[2:])  # Convert count to binary
                        sym_R = count_bin[::-1]  # Reverse for encoding
                        for element in sym_R:
                            x = int(element)  # Symbol for run length
                            code_R_ = self.encode_one_symbol(x, occurrence_second, cumulate_occurrence_second)
    
                            # Update adaptive occurrences if enabled
                            if self.adaptive:
                                occurrence_second[x] += 1
                                for xx in range(x, 5):
                                    cumulate_occurrence_second[xx] += 1
                            
                            occurrence_second_true[x] += 1
                            code_R.extend(code_R_)
                        
                        count = 0  # Reset the run-length counter
    
                    # Encode significant symbol as '+' or '-'
                    x = int(2 + (np.sign(self.res[i]) + 1) / 2)  # '+' (index 3) or '-' (index 2)
                    code_second = self.encode_one_symbol(x, occurrence_second, cumulate_occurrence_second)
    
                    # Update variables if bitrate and symbol constraints are respected
                    if len(self.code) + len(code_R) + len(code_second) + self.follow + 2 <= self.nr_max and self.nb_sym + 1 <= self.nb_sym_max:
                        self.flag[i] = np.sign(self.res[i])  # Mark as significant
                        self.symbol.extend(sym_R)  # Add run-length symbols to the list
                        self.symbol.append("+" if self.flag[i] > 0 else "-")  # Add '+' or '-' to the symbol list
    
                        # Update reconstructed coefficients
                        self.coefs_rec[i] += self.flag[i] * (self.threshold + self.threshold / 2)
                        self.res[i] -= self.flag[i] * (self.threshold + self.threshold / 2)
    
                        # Append encoded symbols
                        self.code.extend(code_R + code_second)
    
                        # Update adaptive occurrences
                        self.occurrence_second = occurrence_second
                        self.cumulate_occurrence_second = cumulate_occurrence_second
                        self.occurrence_second_true = occurrence_second_true
                        
                        if self.adaptive:
                            self.occurrence_second[x] += 1
                            for xx in range(x, 5):
                                self.cumulate_occurrence_second[xx] += 1
                            self.occurrence_second_true[x] += 1
                        
                        self.nb_sym += 1
    
                        # Save the current state of the encoder
                        self.memoire_l = self.l
                        self.memoire_h = self.h
                        self.memoire_follow = self.follow
    
                        # Evaluate quality
                        q = self.cst * get_quality(self.coefs, self.coefs_rec, self.metric)
                        if q <= self.quality_A:
                            return 0  # Exit if quality constraint is satisfied
                    else:
                        return 0  # Exit if bitrate constraint is violated
                else:
                    count += 1  # Increment the run-length counter for "R"
    
        # Handle end-of-pass symbol "E"
        if count > 0:
            x = 4  # Index corresponding to "E"
            code_second = self.encode_one_symbol(x, self.occurrence_second, self.cumulate_occurrence_second)
    
            # Update variables if bitrate and symbol constraints are respected
            if len(self.code) + len(code_second) + self.follow + 2 <= self.nr_max and self.nb_sym + 1 <= self.nb_sym_max:
                self.symbol.append("E")  # Add "E" to the symbol list
                self.code.extend(code_second)
    
                if self.adaptive:
                    self.occurrence_second[x] += 1
                    self.cumulate_occurrence_second[x] += 1
                self.occurrence_second_true[x] += 1
    
                self.nb_sym += 1
    
                # Save the current state of the encoder
                self.memoire_l = self.l
                self.memoire_h = self.h
                self.memoire_follow = self.follow
    
                self.threshold /= 2  # Halve the threshold for the next pass
                return 1  # Entire bit plane encoded
            else:
                return 0  # Exit if bitrate constraint is violated
    
        self.threshold /= 2  # Halve the threshold for the next pass
        return 1  # Entire bit plane encoded
        
            
    
 

    def get_code_res_Antonini(self, coefs, metric, quality, nr, n_sym):
        """
        Generate the resulting code and the number of significant coefficients.
    
        This function encodes the given coefficients using the Antonini encoder
        by performing both the first and second passes iteratively until the encoding process is complete.
    
        Parameters:
        -----------
        coefs : list or np.ndarray
            Coefficients to be encoded.
        metric : str
            Metric used to evaluate quality (e.g., "MSE", "SNR").
        quality : float
            Quality constraint for the encoding process.
        nr : int
            Maximum allowed number of bits for the residual encoding.
        n_sym : int
            Maximum number of symbols that can be encoded.
    
        Returns:
        --------
        tuple:
            - self.code : list
                The resulting encoded bitstream.
            - self.nb_sym : int
                The total number of significant coefficients encoded.
        """
        # Reset the encoder for a new session
        self.reset_Antonini_Encoder(coefs, metric, quality, nr, n_sym)
    
        # Perform the first and second passes until encoding is complete
        while self.get_symbol_first() and self.get_symbol_second():
            pass  # Continue until both passes are complete
    
        # Finalize the encoding process and append the end-of-code symbols
        code_end = self.finish(self.memoire_l, self.memoire_follow)
        self.code.extend(code_end)
    
        # Return the resulting code and the number of significant coefficients
        return self.code, self.nb_sym
    
    
    



class Antonini_Decoder(Context_Aritmetic_Decoder):
    def __init__(self, N=128, M=9, initial_occurrence_first=[1, 1],
                 initial_occurrence_second=[1, 1, 1, 1, 1],
                 adaptive=True, verbose_AD=False):
        """
        Constructor for the Antonini_Decoder class.

        Parameters:
        - N: Length of the vector to decode.
        - M: Context size for arithmetic decoding.
        - initial_occurrence_first: Initial probabilities for the first pass.
        - initial_occurrence_second: Initial probabilities for the second pass.
        - adaptive: Enable or disable adaptive arithmetic decoding.
        - verbose_AD: Enable verbose mode for debugging.
        """
        self.N = N  # Length of the coefficient vector.
        self.initial_occurrence_first = initial_occurrence_first  # Initial probabilities for the first pass.
        self.initial_occurrence_second = initial_occurrence_second  # Initial probabilities for the second pass.
        self.verbose_AD = verbose_AD  # Verbose mode for debugging.
        self.adaptive = adaptive  # Enable or disable adaptive arithmetic decoding.

        # Define the alphabets for the first and second passes.
        self.alphabet_first = ['R', 'S']  # 'R' for "non-significant", 'S' for "significant".
        self.alphabet_second = ['0', '1', '-', '+', 'E']  # Symbols for the second pass.

        # Initialize the context arithmetic decoder by calling the parent class constructor.
        Context_Aritmetic_Decoder.__init__(self, M)

    def reset_Antonini_Decoder(self, code, nb_sym):
        """
        Reset the decoder's state for a new decoding session.

        Parameters:
        - code: The binary code to decode.
        - nb_sym: Number of symbols to decode.
        """
        self.reset_Context_Aritmetic_Decoder()  # Reset the parent class's state.

        # Initialize the state variables.
        self.flag = np.zeros(self.N)  # Flags to track significant coefficients (0: never significant, 1: significant).
        self.coefs_rec = np.zeros(self.N)  # Reconstructed coefficients.
        self.threshold = 0.5  # Initial threshold (coefficients are normalized between -1 and 1).
        self.symbol = []  # List of decoded symbols.

        # Initialize the occurrences and their cumulative sums for both passes.
        self.occurrence_first = list(self.initial_occurrence_first)
        self.occurrence_second = list(self.initial_occurrence_second)
        self.cumulate_occurrence_first = list(accumulate(self.initial_occurrence_first))
        self.cumulate_occurrence_second = list(accumulate(self.initial_occurrence_second))

        # Save the code and symbol count.
        self.code = code
        self.nb_sym = nb_sym  # Total number of symbols to decode.
        self.nb_sym_dec = 0  # Counter for decoded symbols.

    def get_symbol_first(self):
        """
        Decode a symbol from the first pass.

        Returns:
        - 0 if all symbols are decoded or if constraints are met.
        - 1 if there are more symbols to decode.
        """
        for i in range(self.N):
            if self.flag[i] != 0:  # Process only flagged coefficients.
                # Decode a symbol from the first pass alphabet.
                symbol_first = self.decode_one_symbol(
                    self.code, self.alphabet_first, self.occurrence_first, self.cumulate_occurrence_first
                )
                self.symbol.append(symbol_first)  # Store the decoded symbol.
                self.nb_sym_dec += 1  # Increment the decoded symbol count.

                # Update reconstructed coefficients and occurrences.
                if symbol_first == "S":  # Significant coefficient.
                    x = 1
                    if self.adaptive:
                        self.cumulate_occurrence_first[1] += 1
                    self.coefs_rec[i] += self.threshold / 2
                else:  # Non-significant coefficient.
                    x = 0
                    if self.adaptive:
                        self.cumulate_occurrence_first[0] += 1
                        self.cumulate_occurrence_first[1] += 1
                    self.coefs_rec[i] -= self.threshold / 2

                # Update occurrences for adaptive decoding.
                if self.adaptive:
                    self.occurrence_first[x] += 1

                # Stop decoding if all symbols are processed.
                if self.nb_sym_dec == self.nb_sym:
                    return 0

        return 1  # Continue decoding if not all symbols are processed.





    def get_symbol_second(self):
        """
        Decodes symbols during the second pass.
    
        Returns:
        - 1 if the current bit-plane decoding is complete but more decoding is required.
        - 0 if all symbols have been decoded or if constraints are met.
        """
        # Check if all coefficients have been processed.
        if np.sum(np.abs(self.flag)) == self.N:
            self.threshold /= 2  # Reduce the threshold for the next bit-plane.
            return 1
    
        count = 0  # Tracks the number of consecutive 'R' symbols (run length).
        pointer = 0  # Tracks the position of the current coefficient in the vector.
        cpt = 0  # Counter for consecutive binary digits of the run length.
    
        while True:
            # Decode a symbol from the second pass alphabet.
            symbol_second = self.decode_one_symbol(
                self.code, self.alphabet_second, self.occurrence_second, self.cumulate_occurrence_second
            )
    
            x = self.alphabet_second.index(symbol_second)  # Determine the index of the decoded symbol.
    
            if self.adaptive:
                # Update occurrence probabilities adaptively.
                self.occurrence_second[x] += 1
                for xx in range(x, len(self.alphabet_second)):
                    self.cumulate_occurrence_second[xx] += 1
    
            self.symbol.append(symbol_second)  # Append the decoded symbol.
    
            # Handle the decoded symbol.
            if symbol_second == '0':  # Binary 0 in run-length coding.
                cpt += 1
    
            elif symbol_second == '1':  # Binary 1 in run-length coding.
                count += 2**cpt
                cpt += 1
    
            elif symbol_second == '+' or symbol_second == '-':  # Significant coefficients.
                if cpt != 0:
                    count += 1  # Increment count if there was a preceding run-length code.
    
                # Process the coefficients for the decoded run length.
                for k in range(count + 1):
                    # Skip already processed coefficients.
                    while self.flag[pointer] != 0:
                        pointer += 1
    
                    # Handle the final coefficient in the run.
                    if k == count:
                        if symbol_second == '+':
                            self.coefs_rec[pointer] += self.threshold + self.threshold / 2
                            self.flag[pointer] = 1
                        else:  # symbol_second == '-'
                            self.coefs_rec[pointer] -= self.threshold + self.threshold / 2
                            self.flag[pointer] = -1
                    pointer += 1
    
                self.nb_sym_dec += 1  # Increment the number of decoded symbols.
                if self.nb_sym_dec == self.nb_sym:
                    return 0
    
                # Reset counters for the next sequence.
                cpt = 0
                count = 0
    
                # If all remaining coefficients have been processed, reduce the threshold.
                if np.sum(np.abs(self.flag[pointer - 1:])) == self.N - pointer + 1:
                    self.threshold /= 2
                    return 0 if self.nb_sym_dec == self.nb_sym else 1
    
            elif symbol_second == 'E':  # End of bit-plane encoding.
                self.threshold /= 2
                self.nb_sym_dec += 1
                return 0 if self.nb_sym_dec == self.nb_sym else 1
    
    
            

        
    def get_coefs_rec_Antonini(self, code, nb_sym):
        """
        Decodes the coefficients from the provided binary code sequence.
    
        Parameters:
        - code: list
            The binary sequence representing encoded coefficients.
        - nb_sym: int
            The number of symbols encoded in the sequence.
    
        Returns:
        - coefs_rec: list
            The reconstructed coefficients.
        - nr_dec: int
            The total number of bits processed during decoding.
        """
    
        # Reset the decoder with the provided code and symbol count.
        self.reset_Antonini_Decoder(code, nb_sym)
        
        # Initialize the decoding process by setting the codeword.
        self.ini_codeword(self.code)
    
        # Continue decoding symbols until all have been processed.
        while self.nb_sym_dec < self.nb_sym:
            # Decode symbols from the first pass.
            self.get_symbol_first()
            
            # If there are remaining symbols, proceed to the second pass.
            if self.nb_sym_dec < self.nb_sym:
                self.get_symbol_second()
    
        # Store the total number of bits processed during decoding.
        nr_dec = self.count
    
        # Return the reconstructed coefficients and the total number of bits used.
        return self.coefs_rec, nr_dec
    


# Main program
if __name__ == "__main__":

    from Models import Model_poly
    from Normalize import normalize
    # from Measures import curve_tex

    # Define the performance metric and target quality
    metric = "SNR"  # Options: "SNR", "RMSE", "MSE"
    quality = -30  # For SNR: quality in -dB; for RMSE: in volts; for MSE: in volts squared
    
    # Determine the unit based on the metric
    if metric == "SNR":
        unity = "-dB"
    elif metric == "RMSE":
        unity = "V"
    elif metric == "MSE":
        unity = "V^2"
    
    # Encoding parameters
    nr = 1000  # Bit rate constraint
    n_sym = 10  # Maximum number of symbols
    factor_scale = 4  # Scaling factor for coefficients
    M = 9  # Precision for arithmetic encoder
    adaptive = True  # Adaptive mode for arithmetic encoder
    verbose = False  # Verbose mode for debugging
    
    # Initial probabilities for the arithmetic encoder
    initial_occurrence_first = [1, 1]
    initial_occurrence_second = [1, 1, 1, 1, 1]

    # Signal parameters
    N = 128  # Number of samples
    fn = 50  # Nominal frequency (Hz)
    fs = 6400  # Sampling frequency (Hz)
    t = np.linspace(0, (N - 1) / fs, N)  # Time vector

    sigma = 1  # Standard deviation of noise in the test signal

    # Generate a test signal: Polynomial of order 'k'
    order = 8  # Polynomial order
    theta = np.random.uniform(-0.4, 0.4, order + 1)  # Random polynomial coefficients

    # Create a polynomial model and generate a noisy signal
    model_poly = Model_poly(fn, fs, N, verbose)
    x_test = model_poly.get_model_poly(t, *theta) + np.random.normal(0, sigma, N)

    # Normalize the test signal
    x_test, _ = normalize(x_test)

    # Compute the DCT coefficients and normalize them
    coefs = dct(x_test / N) / factor_scale
    coefs_n, kr = normalize(coefs)
   

    # Initialize the Antonini encoder
    AE = Antonini_Encoder(M, initial_occurrence_first, initial_occurrence_second, adaptive, verbose_AE=verbose)

    # Adjust the quality target based on the metric
    if metric == "RMSE":
        quality_n = quality * 2 ** (-kr) / factor_scale
    elif metric == "SNR":
        quality_n = quality
    elif metric == "MSE":
        quality_n = quality * 2 ** (-2 * kr) / factor_scale

    # Encode the coefficients using the Antonini encoder
    code, nb_sym = AE.get_code_res_Antonini(coefs_n, metric, quality_n, nr, n_sym)

    # Display encoding results
    print(f"Nb bits used / Nb bits max = {len(code)} / {nr} bits")
    print(f"Metric ({metric}) = {quality} / {unity}")
    print(f"Nb symbols encoded / Nb symbols max = {nb_sym} / {2**n_sym - 1}")
    print("Encoded symbols:", AE.symbol)

    # Analyze symbol occurrences and compute entropy
    occurrence_first = np.array(AE.occurrence_first_true) - 1
    occurrence_second = np.array(AE.occurrence_second_true) - 1

    if np.sum(occurrence_first) != 0:
        p_first = occurrence_first / np.sum(occurrence_first)
        H_first = entropy(p_first)
    else:
        p_first = occurrence_first
        H_first = 0

    if np.sum(occurrence_second) != 0:
        p_second = occurrence_second / np.sum(occurrence_second)
        H_second = entropy(p_second)
    else:
        p_second = occurrence_second
        H_second = 0

    H_tot = np.sum(occurrence_first) * H_first + np.sum(occurrence_second) * H_second
    print(f"H first = {H_first:.2f} bits")
    print(f"H second = {H_second:.2f} bits")
    print(f"H total = {H_tot:.2f} bits")

    # Reconstruct the signal from encoded coefficients
    coefs_rec_E_n = AE.coefs_rec
    coefs_rec_E = coefs_rec_E_n * 2**kr * factor_scale
    x_rec_E = AE.get_x_rec_Antonini(coefs_rec_E)

    # Plot the original and reconstructed signals
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(t, x_test, lw=2, label='x_test')
    plt.plot(t, x_rec_E, lw=2, label='x_rec_Enc')
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(f'Nb bits used / nb bits max = {len(code)} / {nr} bits, '
              f'H = {H_tot:.1f} bits, {metric} target = {get_quality(x_test, x_rec_E, metric):.5f} / {quality} {unity}')
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()


    
        
    
    #################### Decoder
    
    # Initialize the Antonini decoder
    AD = Antonini_Decoder(N, M, initial_occurrence_first, initial_occurrence_second, adaptive, verbose_AD=verbose)
    
    # Decode the encoded coefficients
    coefs_rec_D_n, nr_rec = AD.get_coefs_rec_Antonini(code, nb_sym)
    coefs_rec_D = coefs_rec_D_n * 2**kr * factor_scale  # Denormalize the decoded coefficients
    
    # Reconstruct the signal from the decoded coefficients
    x_rec_D = AE.get_x_rec_Antonini(coefs_rec_D)
    
    # Plot the original and reconstructed coefficients
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(coefs_n, lw=2, label='coefs')  # Original normalized coefficients
    plt.plot(AD.coefs_rec, lw=2, label='coefs rec')  # Reconstructed normalized coefficients
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Nb bits used / nb bits max = {} / {} bits, {} / {} target = {:.5f} / {} {}'.
              format(nr_rec, nr, metric, metric, get_quality(coefs, AD.coefs_rec, metric), quality, unity))
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    # Plot the original test signal, encoded reconstructed signal, and decoded reconstructed signal
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(t, x_test, lw=2, label='x_test')  # Original test signal
    plt.plot(t, x_rec_E, lw=2, label='x_rec_Enc, Nb bits used / nb bits max = {} / {}, {} / {} target = {:.5f} / {} {}'.
             format(len(code), nr, metric, metric, get_quality(x_test, x_rec_E, metric), quality, unity))  # Encoded reconstruction
    plt.plot(t, x_rec_D, lw=2, label='x_rec_Dec, Nb bits used / nb bits max = {} / {}, {} / {} target = {:.5f} / {} {}'.
             format(nr_rec, nr, metric, metric, get_quality(x_test, x_rec_D, metric), quality, unity))  # Decoded reconstruction
    plt.title("Nb symbols encoded / Nb symbols max = {} / {}".format(nb_sym, 2**n_sym - 1))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
        
       
        