# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 19:10:30 2023

@author: coren
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate
from Context_Arithmetic import Context_Aritmetic_Encoder, Context_Aritmetic_Decoder
from Measures import get_quality, entropy
import pywt


class Khan_Encoder(Context_Aritmetic_Encoder):
    def __init__(self, level, wave_test, M=9, initial_occurrence_first_Khan=[1, 1],
                 initial_occurrence_second_Khan=[1, 1, 1, 1],
                 adaptive_Khan=True, verbose_KE=False):
        # Constructor for the Khan_Encoder class
        
        # Initialize specific parameters for Khan_Encoder
        self.initial_occurrence_first_Khan = initial_occurrence_first_Khan  # Initial equivalent probabilities for the first pass.
        self.initial_occurrence_second_Khan = initial_occurrence_second_Khan  # Initial equivalent probabilities for the second pass.
        self.verbose_KE = verbose_KE  # Debugging mode for Khan_Encoder.
        self.adaptive_Khan = adaptive_Khan  # Whether the arithmetic encoder is adaptive (True) or not (False).
        
        # Other constants, initializations, etc.
        self.level = level
        self.wave_test = wave_test
        self.alphabet_first_Khan = ['RR', 'SS']  # Alphabet for the first pass: 'R' (non-significant), 'S' (significant).
        self.alphabet_second_Khan = ['-', '+', 'Z', 'R']  # Alphabet for the second pass: '+' and '-' (polarity), 'Z' (non-significant coefficient with significant children), 'R' (non-significant).
        
        # Initialize the adaptive context arithmetic encoder by calling the parent class constructor.
        Context_Aritmetic_Encoder.__init__(self, M)
        
        # Save the current state of the arithmetic encoder.
        self.memoire_l = self.l
        self.memoire_h = self.h
        self.memoire_follow = self.follow

    def get_x_rec_Khan(self, coefs):
        """
        Reconstruct the original signal from its wavelet coefficients.
        """
        x = [np.array(coefs[:int(self.N / 2 ** self.level)]) * self.level]
        start = int(self.N / 2 ** self.level)
        for i in range(self.level):
            # Append approximation and detail coefficients to the list with the same shape as x_test.
            end = 2 * start
            x.append(np.array(coefs[start:end]) * self.level)
            start = end
        
        return pywt.waverec(x, self.wave_test, mode='periodization')

    def reset_Khan_Encoder(self, coefs, metric, quality, nr, n_sym):
        """
        Reset parameters to begin a new encoding process.
        """
        self.reset_Context_Aritmetic_Encoder()  # Reset variables from the parent class Context_Aritmetic_Encoder.
        
        self.N = len(coefs)  # Determine the length of the vector to encode.
        self.code = []  # Initialize the binary sequence that will store the encoding result.
        self.flag = np.zeros(self.N)  # Flag indicating if a coefficient has already been considered significant. 0: never significant, 1: at least once significant.
        self.coefs = list(coefs)  # Initialize the list of coefficients to encode.
        self.res = list(self.coefs)  # Initialize the residue, which will be updated after encoding a significant coefficient.
        self.coefs_rec = np.zeros(self.N)  # Initialize the reconstructed coefficients.
        self.threshold = 0.5  # Initialize the threshold to 0.5. Since coefficients are between -1 and 1, the threshold starts at 0.5.
        self.symbol = []  # List to store the encoded symbols.
        self.occurrence_first_Khan = list(self.initial_occurrence_first_Khan)  # Initialize the occurrence list for the first pass, which may be adapted during encoding.
        self.occurrence_second_Khan = list(self.initial_occurrence_second_Khan)  # Initialize the occurrence list for the second pass, which may be adapted during encoding.
        self.cumulate_occurrence_first_Khan = list(accumulate(self.initial_occurrence_first_Khan))  # Calculate cumulative occurrences for the first and second passes.
        self.cumulate_occurrence_second_Khan = list(accumulate(self.initial_occurrence_second_Khan))
        self.nb_sym = 0  # Number of significant coefficients encoded.
        self.quality_K = quality  # Quality constraint.
        self.metric = metric  # Metric used for encoding.
        self.occurrence_first_true_Khan = list(self.initial_occurrence_first_Khan)  # Actual occurrences of symbols in the first pass.
        self.occurrence_second_true_Khan = list(self.initial_occurrence_second_Khan)  # Actual occurrences of symbols in the second pass.

        # Define the maximum number of encodable coefficients depending on `nr`.
        self.nb_sym_max = 2 ** n_sym - 1
        self.nr_max = nr
        
        # Flag array indicating if a coefficient has been considered significant. 0: never significant, 1: at least once significant.
        self.flag = np.zeros(self.N)
        
        # List of the same size as `coefs`, indicating the subband number for each coefficient.
        self.list_bande = np.zeros(self.N, dtype=int)
        start_bande = int(self.N / 2)
        end_bande = self.N
        for b in range(self.level):
            self.list_bande[start_bande:end_bande] = self.level - b
            end_bande = start_bande
            start_bande = int(start_bande / 2)
        
        # Flag array indicating if a child coefficient needs to be encoded.
        self.flag_tree = np.zeros(self.N)
        
        # Set constants based on the chosen metric.
        if self.metric == "MSE":
            self.cst = 4 * self.level ** 2
        elif self.metric == "RMSE":
            self.cst = self.level
        elif self.metric == "SNR":
            x_test = self.get_x_rec_Khan(coefs)
            MSE_x = get_quality(x_test, np.zeros(self.N), "MSE")
            
            self.quality_K = MSE_x / 10 ** (-quality / 10)
            self.metric = "MSE"
            self.cst = 64  # or self.level ** 2

    def get_ind_childs(self, ind_coef):
        """
        Function that returns all children of a given coefficient index (ind_coef).
        This function depends on the total number of coefficients (N) and the number of sub-bands (level).
    
        Parameters:
        ind_coef (int): Index of the coefficient whose children are to be retrieved.
    
        Returns:
        list: A list containing the indices of all child coefficients.
        """
        childs = []
    
        # Get the level of the coefficient index in the sub-band hierarchy
        level_ind_coef = self.list_bande[ind_coef]
        if level_ind_coef < self.level:  # Check if the coefficient belongs to a level below the maximum level
            if level_ind_coef == 0:  # If the coefficient is at the top level (root level)
                ind_coef += int(self.N / (2**self.level))  # Move to the first child level
                childs.extend([ind_coef])  # Add the index of the first child
                level_ind_coef += 1  # Increment the level
    
            # Iterate through levels below the current level to find all descendants
            for k in range(level_ind_coef, self.level):
                # Append the indices of all children at the current level
                childs.extend([u for u in range(ind_coef * 2**(k - level_ind_coef + 1), 
                                                 (ind_coef + 1) * 2**(k - level_ind_coef + 1))])
        return childs

    
    def get_symbol_first_Khan(self):
        """
        Function that encodes the coefficients during the first pass for the current bit plane.
    
        Returns:
            int: 1 or 0 if the codeword length for encoding the residue exceeds the `nr` constraint.
        """
        for i in range(self.N):  # Iterate over all flags
            if self.flag[i] != 0:  # If the coefficient has already been flagged
                if self.res[i] >= 0:
                    x = 1  # Corresponds to the index of "S" in the first pass dictionary
    
                    code_first = self.encode_one_symbol(x, self.occurrence_first_Khan, self.cumulate_occurrence_first_Khan)
    
                    #### Update variables
                    if len(self.code) + len(code_first) + self.follow + 2 <= self.nr_max and self.nb_sym + 1 <= self.nb_sym_max:
                        self.symbol.append("SS")  # Add the symbol to the list
                        self.coefs_rec[i] += self.threshold / 2  # Update the reconstructed coefficient
                        self.res[i] -= self.threshold / 2
                        self.code.extend(code_first)
    
                        if self.adaptive_Khan:
                            self.occurrence_first_Khan[x] += 1  # Update the dictionary if the bitrate constraint is respected
                            self.cumulate_occurrence_first_Khan[x] += 1  # Update cumulative occurrence
    
                        self.occurrence_first_true_Khan[x] += 1  # Update the true dictionary if the bitrate constraint is respected
    
                        self.nb_sym += 1
    
                        # Save the current state of the arithmetic encoder
                        self.memoire_l = self.l
                        self.memoire_h = self.h
                        self.memoire_follow = self.follow
    
                        # Check the quality constraint
                        q = self.cst * get_quality(self.coefs, self.coefs_rec, self.metric)
                        if q <= self.quality_K and self.nb_sym + 1 <= self.nb_sym_max:
                            return 0
                    else:
                        return 0
    
                else:
                    x = 0  # Corresponds to the index of "R" in the first pass dictionary
    
                    code_first = self.encode_one_symbol(x, self.occurrence_first_Khan, self.cumulate_occurrence_first_Khan)
    
                    #### Update variables
                    if len(self.code) + len(code_first) + self.follow + 2 <= self.nr_max and self.nb_sym + 1 <= self.nb_sym_max:
                        self.symbol.append("RR")  # Add the symbol to the list
                        self.coefs_rec[i] -= self.threshold / 2  # Update the reconstructed coefficient
                        self.res[i] += self.threshold / 2
                        self.code.extend(code_first)
    
                        if self.adaptive_Khan:
                            self.occurrence_first_Khan[x] += 1  # Update the dictionary if the bitrate constraint is respected
                            self.cumulate_occurrence_first_Khan[0] += 1  # Update cumulative occurrence
                            self.cumulate_occurrence_first_Khan[1] += 1  # Update cumulative occurrence
    
                        self.occurrence_first_true_Khan[x] += 1  # Update the true dictionary if the bitrate constraint is respected
    
                        self.nb_sym += 1
    
                        # Save the current state of the arithmetic encoder
                        self.memoire_l = self.l
                        self.memoire_h = self.h
                        self.memoire_follow = self.follow
    
                        # Check the quality constraint
                        q = self.cst * get_quality(self.coefs, self.coefs_rec, self.metric)
                        if q <= self.quality_K and self.nb_sym + 1 <= self.nb_sym_max:
                            return 0
                    else:
                        return 0
    
        return 1  # The bit plane has been entirely encoded

            
    def get_symbol_second_Khan(self):
        """
        Function that encodes coefficients during the second pass for the current bit-plane.
    
        Returns:
        -------
        bool:
            Returns 1 or 0 if the codeword length used to encode the residual exceeds the rate constraint (nr).
        """
        # Reset the tree flag at the start of each pass
        self.flag_tree = np.zeros(self.N)
    
        # Initialize local copies of occurrence variables
        occurrence_second_Khan = list(self.occurrence_second_Khan)
        cumulate_occurrence_second_Khan = list(self.cumulate_occurrence_second_Khan)
        occurrence_second_true_Khan = list(self.occurrence_second_true_Khan)
    
        code_ZR = []  # List to store binary codes for insignificant symbols
        symbol_ZR = []  # List to store symbolic representations for insignificant symbols
    
        for i in range(self.N):  # Iterate through all flags
            if self.flag[i] == 0 and self.flag_tree[i] == 0:  # Coefficient has never been significant
                # The coefficient belongs to the first band
                if np.abs(self.res[i]) >= self.threshold:  # Coefficient is significant
                    # Encode '+' or '-' depending on polarity
                    x = int((np.sign(self.res[i]) + 1) / 2)  # Map -1 -> 0 and +1 -> 1
    
                    code_second = self.encode_one_symbol(x, occurrence_second_Khan, cumulate_occurrence_second_Khan)
    
                    # Update variables
                    if len(self.code) + len(code_ZR) + len(code_second) + self.follow + 2 <= self.nr_max \
                            and self.nb_sym + 1 + len(symbol_ZR) <= self.nb_sym_max:
                        self.flag[i] = np.sign(self.res[i])
    
                        self.symbol.extend(symbol_ZR)  # Add 'R' and 'Z' symbols
                        self.symbol.append("+" if self.flag[i] > 0 else "-")  # Add '+' or '-' to symbols
                        self.coefs_rec[i] += self.flag[i] * (self.threshold + self.threshold / 2)  # Update reconstructed coefficients
                        self.res[i] -= self.flag[i] * (self.threshold + self.threshold / 2)  # Update residuals
    
                        self.code.extend(code_ZR)  # Append 'ZR' code
                        self.code.extend(code_second)  # Append '+' or '-' code
    
                        if self.adaptive_Khan:
                            occurrence_second_Khan[x] += 1
                            for xx in range(x, 4):
                                cumulate_occurrence_second_Khan[xx] += 1
                        occurrence_second_true_Khan[x] += 1
    
                        # Update dictionary variables
                        self.occurrence_second_Khan = occurrence_second_Khan
                        self.cumulate_occurrence_second_Khan = cumulate_occurrence_second_Khan
                        self.occurrence_second_true_Khan = occurrence_second_true_Khan
    
                        self.nb_sym += 1 + len(symbol_ZR)  # Update number of symbols
                        code_ZR = []
                        symbol_ZR = []
    
                        # Save encoder state
                        self.memoire_l = self.l
                        self.memoire_h = self.h
                        self.memoire_follow = self.follow
    
                        # Evaluate quality metric
                        q = self.cst * get_quality(self.coefs, self.coefs_rec, self.metric)
                        if q <= self.quality_K and self.nb_sym + 1 <= self.nb_sym_max:
                            return 0  # Stop encoding if quality constraint is met
                    else:
                        return 0  # Stop encoding if constraints are violated
                else:  # Coefficient is insignificant; check children
                    childs = self.get_ind_childs(i)
                    test = 0
                    for k in range(len(childs)):
                        if np.abs(self.res[childs[k]]) >= self.threshold:  # Child is significant
                            # Encode 'Z'
                            x = 2
                            code_ZR.extend(self.encode_one_symbol(x, occurrence_second_Khan, cumulate_occurrence_second_Khan))
                            if self.adaptive_Khan:
                                occurrence_second_Khan[x] += 1
                                for kk in range(x, 4):
                                    cumulate_occurrence_second_Khan[kk] += 1
                            occurrence_second_true_Khan[x] += 1
                            symbol_ZR.append('Z')
                            test = 1
                            break
                    if test == 0:  # None of the children are significant
                        # Encode 'R'
                        x = 3
                        code_ZR.extend(self.encode_one_symbol(x, occurrence_second_Khan, cumulate_occurrence_second_Khan))
                        if self.adaptive_Khan:
                            occurrence_second_Khan[x] += 1
                            for kk in range(x, 4):
                                cumulate_occurrence_second_Khan[kk] += 1
                        occurrence_second_true_Khan[x] += 1
                        symbol_ZR.append('R')
                        self.flag_tree[childs] = 1  # Mark that this coefficient has no significant descendants
    
        # Encode the end of the pass before moving to the next
        if len(self.code) + len(code_ZR) + self.follow + 2 <= self.nr_max and self.nb_sym + len(code_ZR) <= self.nb_sym_max:
            self.code.extend(code_ZR)
    
            # Update dictionary variables
            self.occurrence_second_Khan = occurrence_second_Khan
            self.cumulate_occurrence_second_Khan = cumulate_occurrence_second_Khan
            self.occurrence_second_true_Khan = occurrence_second_true_Khan
    
            self.symbol.extend(symbol_ZR)
            self.nb_sym += len(symbol_ZR)
    
            # Save encoder state
            self.memoire_l = self.l
            self.memoire_h = self.h
            self.memoire_follow = self.follow
    
            # Halve the threshold for the next pass
            self.threshold /= 2
            return 1  # Bit-plane encoding is complete
        else:
            return 0  # Stop encoding due to constraints
            
    
    
    def get_code_res_Khan(self, coefs, metric, quality, nr, n_sym):
        """
        Function to obtain the resulting code and the number of bits per coefficient.
    
        Parameters:
        ----------
        coefs : list
            List of coefficients to be encoded.
        metric : str
            Quality metric used for encoding (e.g., "MSE", "RMSE", "SNR").
        quality : float
            Target quality constraint.
        nr : int
            Maximum rate constraint (number of bits allowed for encoding).
        n_sym : int
            Maximum number of symbols that can be encoded.
    
        Returns:
        -------
        tuple:
            - code (list): The final encoded binary sequence.
            - nb_sym (int): The total number of significant symbols encoded.
        """
        # Reset the encoder state for a new encoding process
        self.reset_Khan_Encoder(coefs, metric, quality, nr, n_sym)
    
        # Perform encoding by processing first and second passes iteratively
        while self.get_symbol_first_Khan() and self.get_symbol_second_Khan():
            pass  # Continue encoding until bit-plane processing is complete
    
        # Finalize the encoding and append the termination sequence
        code_end = self.finish(self.memoire_l, self.memoire_follow)
        self.code.extend(code_end)
    
        # Return the final encoded sequence and the total number of symbols
        return self.code, self.nb_sym
    

class Khan_Decoder(Context_Aritmetic_Decoder):
    def __init__(self, level, wave_test, N=128, M=9, initial_occurrence_first_Khan=[1, 1],
                 initial_occurrence_second_Khan=[1, 1, 1, 1], adaptive_Khan=True, verbose_KD=False):
        """
        Initialize the Khan Decoder class, which is designed to decode encoded coefficients 
        using a multi-level wavelet decomposition approach.

        Parameters:
        ----------
        level : int
            Number of levels in the wavelet decomposition.
        wave_test : str
            Wavelet type used for decomposition and reconstruction.
        N : int, optional
            Length of the input signal (default is 128).
        M : int, optional
            Precision for arithmetic decoding (default is 9).
        initial_occurrence_first_Khan : list, optional
            Initial probabilities for the first pass (default is [1, 1]).
        initial_occurrence_second_Khan : list, optional
            Initial probabilities for the second pass (default is [1, 1, 1, 1]).
        adaptive_Khan : bool, optional
            Whether the decoder adapts to the input statistics (default is True).
        verbose_KD : bool, optional
            If True, enables debug mode (default is False).
        """
        self.N = N  # Length of the vector to decode.
        self.initial_occurrence_first_Khan = initial_occurrence_first_Khan  # Initial probabilities for the first pass.
        self.initial_occurrence_second_Khan = initial_occurrence_second_Khan  # Initial probabilities for the second pass.
        self.verbose_KD = verbose_KD  # Debug mode for Khan Decoder.
        self.adaptive_Khan = adaptive_Khan  # Whether the decoder is adaptive.

        self.level = level  # Number of levels in wavelet decomposition.
        self.wave_test = wave_test  # Wavelet type used.
        self.alphabet_first_Khan = ['RR', 'SS']  # Alphabet for the first pass: 'RR' (insignificant), 'SS' (significant).
        self.alphabet_second_Khan = ['-', '+', 'Z', 'R']  # Alphabet for the second pass.

        # Initialize the context arithmetic decoder (parent class).
        Context_Aritmetic_Decoder.__init__(self, M)

    def get_x_rec_Khan(self, coefs):
        """
        Reconstruct the signal from coefficients using wavelet decomposition.

        Parameters:
        ----------
        coefs : list
            Wavelet coefficients.

        Returns:
        -------
        ndarray:
            Reconstructed signal using inverse wavelet transform.
        """
        x = [np.array(coefs[:int(self.N / 2 ** self.level)]) * self.level]
        start = int(self.N / 2 ** self.level)
        for i in range(self.level):
            # Append approximation and detail coefficients for reconstruction.
            end = 2 * start
            x.append(np.array(coefs[start:end]) * self.level)
            start = end

        return pywt.waverec(x, self.wave_test, mode='periodization')

    def reset_Khan_Decoder(self, code, nb_sym):
        """
        Reset the decoder state for a new decoding process.

        Parameters:
        ----------
        code : list
            Encoded binary sequence.
        nb_sym : int
            Total number of symbols to decode.
        """
        self.reset_Context_Aritmetic_Decoder()  # Reset context arithmetic decoder variables.
        self.flag = np.zeros(self.N)  # Flag indicating if a coefficient has been significant before.
        self.coefs_rec = np.zeros(self.N)  # Reconstructed coefficients.
        self.threshold = 0.5  # Initial threshold (coefficients are between -1 and 1).
        self.symbol = []  # List to store decoded symbols.

        self.occurrence_first_Khan = list(self.initial_occurrence_first_Khan)  # Reset first pass probabilities.
        self.occurrence_second_Khan = list(self.initial_occurrence_second_Khan)  # Reset second pass probabilities.
        self.cumulate_occurrence_first_Khan = list(accumulate(self.initial_occurrence_first_Khan))  # Cumulative probabilities (first pass).
        self.cumulate_occurrence_second_Khan = list(accumulate(self.initial_occurrence_second_Khan))  # Cumulative probabilities (second pass).

        self.code = code  # Encoded binary sequence.
        self.nb_sym = nb_sym  # Total number of symbols to decode.
        self.nb_sym_dec = 0  # Counter for decoded symbols.

        # Assign each coefficient to its corresponding wavelet band.
        self.list_bande = np.zeros(self.N, dtype=int)
        start_bande = int(self.N / 2)
        end_bande = self.N
        for b in range(self.level):
            self.list_bande[start_bande:end_bande] = self.level - b
            end_bande = start_bande
            start_bande = int(start_bande / 2)

        self.flag_tree = np.zeros(self.N)  # Flag indicating if child coefficients need to be decoded.

    def get_ind_childs(self, ind_coef):
        """
        Get all child coefficients of a given coefficient index.

        Parameters:
        ----------
        ind_coef : int
            Index of the parent coefficient.

        Returns:
        -------
        list:
            Indices of all child coefficients.
        """
        childs = []
        level_ind_coef = self.list_bande[ind_coef]
        if level_ind_coef < self.level:
            if level_ind_coef == 0:
                ind_coef += int(self.N / (2 ** self.level))
                childs.extend([ind_coef])
                level_ind_coef += 1

            for k in range(level_ind_coef, self.level):
                childs.extend([u for u in range(ind_coef * 2 ** (k - level_ind_coef + 1),
                                                (ind_coef + 1) * 2 ** (k - level_ind_coef + 1))])
        return childs



    def get_symbol_first_Khan(self):
        """
        Decode a symbol from the first pass of the arithmetic decoder.
        Iterates through all coefficients and processes their significance.

        Returns:
        -------
        int:
            0 if all symbols have been decoded, 1 otherwise.
        """
        for i in range(self.N):
            if self.flag[i] != 0:  # Process only significant coefficients
                symbol_first = self.decode_one_symbol(
                    self.code, self.alphabet_first_Khan, self.occurrence_first_Khan, self.cumulate_occurrence_first_Khan
                )
                self.symbol.append(symbol_first)  # Append the decoded symbol
                self.nb_sym_dec += 1  # Increment decoded symbol counter

                if symbol_first == "SS":
                    x = 1
                    if self.adaptive_Khan:
                        self.cumulate_occurrence_first_Khan[1] += 1
                    self.coefs_rec[i] += self.threshold / 2  # Update reconstructed coefficient
                else:  # symbol_first == "RR"
                    x = 0
                    if self.adaptive_Khan:
                        self.cumulate_occurrence_first_Khan[0] += 1
                        self.cumulate_occurrence_first_Khan[1] += 1
                    self.coefs_rec[i] -= self.threshold / 2

                if self.adaptive_Khan:
                    self.occurrence_first_Khan[x] += 1  # Update occurrence statistics

                if self.nb_sym_dec == self.nb_sym:  # Stop if all symbols are decoded
                    return 0
        return 1  # Continue decoding

    def get_symbol_second_Khan(self):
        """
        Decode a symbol from the second pass of the arithmetic decoder.
        Iterates through all coefficients and processes significance, polarity, and child relationships.

        Returns:
        -------
        int:
            1 if the decoding process should continue, 0 if complete.
        """
        if np.sum(np.abs(self.flag)) == self.N:  # All coefficients are already significant
            self.threshold /= 2  # Reduce threshold for the next pass
            return 1

        pointeur = 0  # Pointer to the current coefficient

        while pointeur < self.N:
            if self.flag[pointeur] == 0 and self.flag_tree[pointeur] == 0:  # Process non-significant coefficients
                # Decode a symbol from the second pass
                symbol_second = self.decode_one_symbol(
                    self.code, self.alphabet_second_Khan, self.occurrence_second_Khan, self.cumulate_occurrence_second_Khan
                )

                x = self.alphabet_second_Khan.index(symbol_second)

                if self.adaptive_Khan:
                    self.occurrence_second_Khan[x] += 1
                    for xx in range(x, 4):  # Update cumulative occurrence
                        self.cumulate_occurrence_second_Khan[xx] += 1

                self.symbol.append(symbol_second)  # Append the decoded symbol

                if symbol_second == "+":
                    self.coefs_rec[pointeur] += self.threshold + self.threshold / 2
                    self.flag[pointeur] = 1  # Mark coefficient as significant
                elif symbol_second == "-":
                    self.coefs_rec[pointeur] -= self.threshold + self.threshold / 2
                    self.flag[pointeur] = -1  # Mark coefficient as significant with negative polarity
                elif symbol_second == "R":
                    childs = self.get_ind_childs(pointeur)  # Retrieve child coefficients
                    self.flag_tree[childs] = 1  # Mark children as already processed
                elif symbol_second == "Z":
                    pass  # No action for 'Z'

                pointeur += 1
                self.nb_sym_dec += 1
                if self.nb_sym_dec == self.nb_sym:  # Stop if all symbols are decoded
                    return 0
            else:
                pointeur += 1

        self.threshold /= 2  # Reduce threshold for the next pass
        self.flag_tree = np.zeros(self.N)  # Reset the child processing flag
        return 1

    def get_coefs_rec_Khan(self, code, nb_sym):
        """
        Retrieve reconstructed coefficients from the binary sequence.

        Parameters:
        ----------
        code : list
            Encoded binary sequence.
        nb_sym : int
            Number of symbols in the sequence.

        Returns:
        -------
        tuple:
            Reconstructed coefficients and the total number of decoded bits.
        """
        self.reset_Khan_Decoder(code, nb_sym)  # Reset the decoder state
        self.ini_codeword(self.code)  # Initialize decoding

        while self.nb_sym_dec < self.nb_sym:
            self.get_symbol_first_Khan()  # Decode first pass symbols
            if self.nb_sym_dec < self.nb_sym:
                self.get_symbol_second_Khan()  # Decode second pass symbols

        return self.coefs_rec, self.count  # Return reconstructed coefficients and bit count





# Main Program
if __name__ == "__main__":

    from Models import Model_poly
    from Normalize import normalize

    # Metric and quality settings
    metric = "RMSE"  # Can be "SNR", "RMSE", or "MSE"
    quality = 0.01  # Target quality; units depend on the metric

    # Determine the unit based on the selected metric
    if metric == "SNR":
        unity = "-dB"
    elif metric == "RMSE":
        unity = "V"
    elif metric == "MSE":
        unity = "V^2"

    # Encoder parameters
    nr = 100  # Maximum number of bits
    n_sym = 10  # Maximum number of symbols
    factor_scale = 4
    M = 9  # Arithmetic encoder's precision parameter
    adaptive_Khan = True  # Enables adaptive arithmetic coding
    verbose = False  # Debugging mode

    # Initialization of symbol probabilities
    initial_occurrence_first_Khan = [1, 1]
    initial_occurrence_second_Khan = [1, 1, 1, 1]

    # Signal and wavelet decomposition settings
    N = 128  # Signal length
    fn = 50  # Signal fundamental frequency (Hz)
    fs = 6400  # Sampling frequency (Hz)
    t = np.linspace(0, (N - 1) / fs, N)  # Time vector
    sigma = 1  # Standard deviation of noise added to the test signal

    # Polynomial-based test signal generation
    order = 8  # Polynomial order
    theta = np.random.uniform(-0.2, 0.2, order + 1)  # Random polynomial coefficients
    model_poly = Model_poly(fn, fs, N, verbose)
    x_test = model_poly.get_model_poly(t, *theta) + np.random.normal(0, sigma, N)
    x_test, _ = normalize(x_test)  # Normalize the signal

    # Wavelet settings
    wave_test = 'coif5'  # Wavelet type
    level = int(np.ceil(np.log2(N)))  # Decomposition levels
    mode = 'periodization'  # Wavelet mode

    # Initialize the Khan Encoder
    KE = Khan_Encoder(
        level, wave_test, M, initial_occurrence_first_Khan,
        initial_occurrence_second_Khan, adaptive_Khan, verbose_KE=verbose
    )

    # Perform wavelet decomposition
    coefs = pywt.wavedec(x_test, wave_test, mode=mode, level=level)

    # Flatten wavelet coefficients for encoding
    coefs_L = []
    for i in range(level + 1):
        coefs_L.extend(coefs[i] / level / factor_scale)
    coefs_L = np.array(coefs_L)

    # Normalize coefficients for encoding
    coefs_L_n, kr = normalize(coefs_L)

    # Adjust quality constraints based on metric
    if metric == "RMSE":
        quality_n = quality * 2 ** (-kr) / factor_scale
    elif metric == "SNR":
        quality_n = quality
    elif metric == "MSE":
        quality_n = quality * 2 ** (-2 * kr) / factor_scale

    # Encode the signal coefficients
    code, nb_sym = KE.get_code_res_Khan(coefs_L_n, metric, quality_n, nr, n_sym)

    # Decode the coefficients to reconstruct the signal
    coefs_L_rec_E_n = KE.coefs_rec
    coefs_L_rec_E = coefs_L_rec_E_n * factor_scale * 2 ** kr
    x_rec_E = KE.get_x_rec_Khan(coefs_L_rec_E)

    # Print encoding and quality results
    print("Nb bits used / Nb bits max = {} / {} bits".format(len(code), nr),
          "{} = {} / {} {}".format(metric, get_quality(x_test, x_rec_E, metric), quality, unity))
    print("Nb sym encoded / Nb sym max = {} / {}".format(nb_sym, 2 ** n_sym - 1))

    # Calculate symbol occurrences and entropy
    occurrence_first_Khan = np.array(KE.occurrence_first_true_Khan) - 1
    occurrence_second_Khan = np.array(KE.occurrence_second_true_Khan) - 1

    if np.sum(occurrence_first_Khan) != 0:
        p_first = occurrence_first_Khan / np.sum(occurrence_first_Khan)
        H_first = entropy(p_first)
    else:
        p_first = occurrence_first_Khan
        H_first = 0

    if np.sum(occurrence_second_Khan) != 0:
        p_second = occurrence_second_Khan / np.sum(occurrence_second_Khan)
        H_second = entropy(p_second)
    else:
        p_second = occurrence_second_Khan
        H_second = 0

    print("Occurrence of symbols in the first pass:", occurrence_first_Khan)
    print("Occurrence of symbols in the second pass:", occurrence_second_Khan)
    print("H first = {:.2f} bits".format(H_first))
    print("H second = {:.2f} bits".format(H_second))

    H_tot = np.sum(occurrence_first_Khan) * H_first + np.sum(occurrence_second_Khan) * H_second
    print("Total entropy H_tot = {:.2f} bits".format(H_tot))

    
        
    #################### Decoder ####################
    
    # Initialize the Khan Decoder
    KD = Khan_Decoder(
        level, wave_test, N, M, 
        initial_occurrence_first_Khan, initial_occurrence_second_Khan,
        adaptive_Khan, verbose_KD=verbose
    )
    
    # Decode the coefficients
    coefs_L_rec_D_n, nr_rec_D = KD.get_coefs_rec_Khan(code, nb_sym)
    
    # Denormalize the decoded coefficients
    coefs_L_rec_D = coefs_L_rec_D_n * factor_scale * 2 ** (kr)
    
    # Reconstruct the signal from the decoded coefficients
    x_rec_D = KE.get_x_rec_Khan(coefs_L_rec_D)
    
    # Plot the original and reconstructed coefficients
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(coefs_L_n, lw=2, label='Original coefficients')
    plt.plot(coefs_L_rec_D_n, lw=2, label='Reconstructed coefficients')
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Nb bits used / Nb bits max = {} / {}, H = {:.1f} bits'.format(len(code), nr, H_tot))
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    # Plot the original signal, encoded reconstructed signal, and decoded reconstructed signal
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(t, x_test, lw=2, label='Original signal (x_test)')
    plt.plot(t, x_rec_E, lw=2, label='Rec (Encoder), Nb bits used / Nb bits max = {} / {}, {} / {} target = {:.5f} / {} {}'.format(
        len(code), nr, metric, metric, get_quality(x_test, x_rec_E, metric), quality, unity))
    plt.plot(t, x_rec_D, lw=2, label='Rec (Decoder), Nb bits used / Nb bits max = {} / {}, {} / {} target = {:.5f} / {} {}'.format(
        nr_rec_D, nr, metric, metric, get_quality(x_test, x_rec_D, metric), quality, unity))
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title("Nb symbols encoded / Nb symbols max = {} / {}".format(nb_sym, 2 ** n_sym - 1))
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
        
        
        
        