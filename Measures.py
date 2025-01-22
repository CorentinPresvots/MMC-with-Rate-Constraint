# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:20:49 2023

@author: coren
"""

import numpy as np
from collections import Counter

# Function to calculate the entropy of a probability distribution
def entropy(p):
    """
    Calculate the Shannon entropy of a given probability distribution.

    Parameters:
    - p: List of probabilities for each symbol.

    Returns:
    - H: Entropy value in bits.
    """
    return -np.sum([p[i] * np.log2(p[i] + 1e-8) for i in range(len(p))])

# Function to compute the frequency of symbols in a list
def get_frequency(liste_symboles):
    """
    Compute the frequency of each unique symbol in a given list.

    Parameters:
    - liste_symboles: List of symbols (e.g., integers or strings).

    Returns:
    - alphabet: List of unique symbols.
    - frequency: List of normalized frequencies corresponding to each symbol in the alphabet.
    """
    size_liste_symbole = len(liste_symboles)
    compteur = Counter(liste_symboles)  # Count occurrences of each symbol
    alphabet = list(compteur.keys())  # Extract unique symbols
    frequency = [compteur[symbole] / size_liste_symbole for symbole in alphabet]
    return alphabet, frequency

# Function to calculate binary entropy
def calculate_entropy(S):
    """
    Calculate the binary entropy of a signal S.

    Parameters:
    - S: Binary signal (0s and 1s).

    Returns:
    - H: Binary entropy value in bits.
    """
    p = np.mean(S)  # Probability of 1s in the signal
    r = 1e-16  # Small value to avoid log(0)
    H = -p * np.log2(p + r) - (1 - p) * np.log2(1 - p + r)
    return H

# Function to calculate conditional entropy of a binary sequence
def calculate_conditional_entropy(S):
    """
    Calculate the conditional entropy of a binary sequence S.

    Parameters:
    - S: Binary sequence (list of 0s and 1s).

    Returns:
    - H1: Conditional entropy value in bits.
    """
    N = len(S)
    if N == 0:  # Handle empty input
        return 0

    # Prepend a starting value to the sequence
    S = [1] + S

    # Initialize counts for transitions between states
    n00 = n01 = n10 = n11 = 0
    r = 1e-8  # Small value to avoid log(0)

    # Count occurrences of transitions
    for i in range(1, len(S)):
        if S[i - 1] == 0 and S[i] == 0:
            n00 += 1
        elif S[i - 1] == 0 and S[i] == 1:
            n01 += 1
        elif S[i - 1] == 1 and S[i] == 0:
            n10 += 1
        elif S[i - 1] == 1 and S[i] == 1:
            n11 += 1

    # Normalize transition counts to obtain probabilities
    p00 = n00 / N
    p10 = n10 / N
    p11 = n11 / N
    p01 = n01 / N

    # Compute the conditional entropy
    H1 = -0.5 * (p00 * np.log2(r + p00) + p10 * np.log2(r + p10))
    H1 -= 0.5 * (p01 * np.log2(r + p01) + p11 * np.log2(r + p11))

    return H1

# Function to calculate the Signal-to-Noise Ratio (SNR) in decibels
def get_snr(signal, signal_rec):
    """
    Compute the Signal-to-Noise Ratio (SNR) between the original signal and its reconstruction.

    Parameters:
    - signal: Original signal (array or list).
    - signal_rec: Reconstructed signal (array or list).

    Returns:
    - snr_db: SNR value in decibels (dB).
    """
    # Calculate the power of the original signal
    signal_power = np.sum(np.square(signal))

    # Calculate the power of the noise (difference between original and reconstructed signals)
    noise_power = np.sum(np.square(signal - signal_rec))

    # Calculate the SNR in decibels
    snr_db = 10 * np.log10(signal_power / noise_power)

    return snr_db



# Function to compute Mean Squared Error (MSE) between two signals
def get_mse(signal, signal_rec):
    """
    Calculate the Mean Squared Error (MSE) between two signals.

    Parameters:
    - signal: Original signal (array or list).
    - signal_rec: Reconstructed signal (array or list).

    Returns:
    - mse: Mean Squared Error value.
    """
    return np.mean(np.square(signal - signal_rec))

# Function to compute Root Mean Squared Error (RMSE) between two signals
def get_rmse(signal, signal_rec):
    """
    Calculate the Root Mean Squared Error (RMSE) between two signals.

    Parameters:
    - signal: Original signal (array or list).
    - signal_rec: Reconstructed signal (array or list).

    Returns:
    - rmse: Root Mean Squared Error value.
    """
    return np.sqrt(get_mse(signal, signal_rec))

# Generalized function to calculate quality metrics (SNR, MSE, RMSE)
def get_quality(signal, signal_rec, metric):
    """
    Calculate the quality of reconstruction between an original signal and its reconstruction.

    Parameters:
    - signal: Original signal (array or list).
    - signal_rec: Reconstructed signal (array or list).
    - metric: Quality metric to compute ('SNR', 'MSE', 'RMSE').

    Returns:
    - value: Computed metric value.
    """
    if metric == "SNR":
        return -get_snr(signal, signal_rec)
    if metric == "MSE":
        return get_mse(signal, signal_rec)
    if metric == "RMSE":
        return get_rmse(signal, signal_rec)

# Function to convert an integer to binary with fixed number of bits
def my_bin(ind, b):
    """
    Convert an integer to a binary representation with fixed bit-length.

    Parameters:
    - ind: Integer to convert.
    - b: Number of bits in the binary representation.

    Returns:
    - code: List of 0s and 1s representing the integer in binary, 
            ordered from least significant to most significant bit.
    """
    q = -1
    code = [0] * b  # Initialize binary array with zeros
    i = 0
    while i < b:
        q = ind // 2  # Integer division
        r = ind % 2   # Remainder (binary digit)
        code[i] = int(r)
        ind = q
        i += 1    
    return code

# Function to convert binary representation back to an integer
def my_inv_bin(code):
    """
    Convert a binary representation back to its integer value.

    Parameters:
    - code: List of 0s and 1s (least significant to most significant bit).

    Returns:
    - val: Integer value represented by the binary code.
    """
    ind_pos = 0
    for k in range(len(code)):
        ind_pos += code[k] * 2**k  # Compute value as sum of weighted binary digits
    return ind_pos

# Function to encode an integer with variable-length binary representation
def encode_variable_length(ind):
    """
    Encode an integer using variable-length binary representation.

    Parameters:
    - ind: Integer to encode.

    Returns:
    - code: List of binary digits encoding the integer, including length prefix.
    """
    if ind < 0:
        raise ValueError("The integer must be non-negative.")

    print("ind", ind)
    len_ind = int(np.ceil(np.log2(ind + 1e-8)))  # Calculate the binary length of the integer
    print("Binary length of ind", len_ind)
    
    code_ind = my_bin(ind, len_ind)  # Binary representation of the integer
    print("Binary representation of ind", code_ind)
    
    code = [1] * (len_ind - 1) + [0] + code_ind  # Add prefix to indicate length
    return code

# Function to decode a variable-length binary representation
def decode_variable_length(code):
    """
    Decode an integer from its variable-length binary representation.

    Parameters:
    - code: List of binary digits encoding the integer.

    Returns:
    - ind: Decoded integer value.
    """
    len_ind = 1
    i = 0
    while code[i] != 0:  # Determine the length of the integer
        len_ind += 1
        i += 1
    print("Binary length of the integer used to decode ind", len_ind)
    
    # Decode the integer
    ind = my_inv_bin(code[i+1:i+1+len_ind])
    print("ind", ind)
    return ind

        
if __name__ == "__main__":
    
    # Import the source generation module
    from Source import Generation_Source
    
    ### Generate a random source
    N = 20  # Length of the generated source
    alphabet = ['AA', 'BB', 'CC']  # Possible symbols in the source
    probability = [0.6, 0.1, 0.3]  # Probabilities associated with each symbol
    
    # Initialize the source generator
    S = Generation_Source(alphabet, probability)
    
    # Generate a source of length N
    S.generate_source(N)
    
    # Retrieve the generated source
    x = S.source

    # Display the actual probabilities and the source's empirical probabilities
    for i in range(len(alphabet)):
        print("sym: {}, p real: {:.2f}, p source: {:.2f}".format(alphabet[i], probability[i], S.probability_source[i]))
    
    # Calculate and display the alphabet and probabilities from the generated source
    alphabet_source, probability_source = get_frequency(x)
    print("alphabet_source", alphabet_source)
    print("probability_source", probability_source)   
    
    # Compute and display the entropy of the source
    H = entropy(probability_source)
    print("entropy source: {:.2f}".format(H))
    
    # Test fixed-length binary encoding and decoding
    print("my_bin", my_bin(40, 150), "my_inv_bin", my_inv_bin(my_bin(40, 150)))
    
    ########## Test binary encoding and decoding for a large value
    b = 100  # Number of bits
    ind = 5.764607523034235e+18  # Large integer to encode
    a = my_bin(ind, b)
    print("a", a, len(a))  # Display the binary representation and its length
    ind_rec = my_inv_bin(a)
    print("ind_rec", ind_rec)  # Reconstruct the integer from its binary representation
    
    ##### Test variable-length encoding and decoding
    
    # Example: Encoding and decoding a number using variable-length representation
    num_to_encode = 50
    encoded = encode_variable_length(num_to_encode)  # Encode the integer
    decoded = decode_variable_length(encoded)  # Decode back to integer
    
    # Display results
    print(f"Original integer: {num_to_encode}")
    print(f"Encoded binary sequence: {encoded}")
    print(f"Decoded integer: {decoded}")

