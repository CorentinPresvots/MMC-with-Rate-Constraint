# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 08:53:36 2023

@author: coren
"""

import numpy as np
import matplotlib.pyplot as plt
from Measures import my_bin,my_inv_bin

class Quantizer: 
    """
    Class implementing both mid-tread and mid-rise quantizers.

    - A **mid-tread quantizer** has a central zero region, meaning small input values 
      around zero are mapped to zero. This type of quantizer is typically used when 
      zero values are more frequent and need better representation.

    - A **mid-rise quantizer** does not have a central zero region, meaning the 
      quantization levels start at nonzero values. This approach ensures that 
      every interval has a representative value, making it useful when preserving 
      small details in a signal.

    The class provides methods for:
    - Computing quantization indices (`get_ind` and `get_ind_u`).
    - Reconstructing quantized values (`get_q` and `get_q_u`).
    - Encoding quantized indices into binary format (`get_code` and `get_code_u`).
    - Decoding binary sequences back into quantization indices (`get_inv_code` and `get_inv_code_u`).
    """
    
    def __init__(self, verbose=False):
        """
        Constructor for the Quantizer class.
        Currently does nothing but can be used to set verbosity for debugging.
        """
        pass
     
    def get_ind(self, x, n, w, m): 
        """
        Computes the quantization index for a given input signal x.

        Parameters:
        - x: Input value to be quantized.
        - n: Number of quantization bits.
        - w: Quantization dynamic range.
        - m: Mean value (offset).

        Returns:
        - ind: Quantization index corresponding to the input value x.
        """
        if n < 2:  # If the number of bits is too low, return zero.
            return 0
            
        
        delta = w / (2**n - 1)  # Quantization step size.
        
        ind = np.round((x - m) / delta)  # Compute the index of the quantized value.

        # Ensure the index stays within valid limits.
        """
        ind_max = 2**(n - 1) - 1  # Maximum quantization index.
        if ind < -ind_max:
            ind = -ind_max
        if ind > ind_max:
            ind = ind_max
        """
        return ind   

    def get_q(self, ind, n, w, m): 
        """
        Computes the reconstructed quantized value from an index.

        Parameters:
        - ind: Quantization index.
        - n: Number of quantization bits.
        - w: Quantization dynamic range.
        - m: Mean value (offset).

        Returns:
        - The reconstructed value corresponding to the quantization index.
        """
        if n < 2:  # If the number of bits is too low, return mean value.
            return m
        delta = w / (2**n - 1)  # Quantization step size.
        return delta * ind + m  # Convert index back to quantized value.

    def get_code(self, ind, n):
        """
        Encodes a quantization index into a binary representation.

        Parameters:
        - ind: Quantization index.
        - n: Number of quantization bits.

        Returns:
        - code: A binary representation of the quantization index.
        """
        if n < 2:
            return []
        
        ind_pos = 2**(n - 1) - 1 + ind  # Convert index to positive space for encoding.
        
        # Convert index to binary using a helper function.
        code = my_bin(ind_pos, n)
    
        return code
     
    def get_inv_code(self, code, n):
        """
        Decodes a binary sequence back into a quantization index.

        Parameters:
        - code: Binary representation of the quantized index.
        - b: Number of quantization bits.

        Returns:
        - The decoded quantization index.
        """
        if n < 2:
            return 0

        ind_pos = my_inv_bin(code)  # Convert binary to integer.

        return ind_pos - 2**(n - 1) + 1  # Convert back to original index space.

    
    #################### Uniform Quantization
    
    def get_ind_u(self, x, n, w, m): 
        """
        Computes the quantization index for a given input value x 
        using uniform quantization.
    
        Parameters:
        - x: Input value to be quantized.
        - n: Number of quantization bits.
        - w: Quantization dynamic range.
        - m: Mean value (offset).
    
        Returns:
        - ind: Quantization index corresponding to the input value x.
        """
        if n == 0:  # If no bits are allocated, return zero.
            return 0
    
        delta = w / (2**n)  # Compute the quantization step size.
    
        # Compute the quantization index using floor rounding.
        ind = np.floor((x - m) / delta)
    
        # Ensure the index remains within valid bounds.
        ind = min(ind, round(w / (2 * delta)) - 1)
        ind = max(ind, round(-w / (2 * delta)))
    
        return ind
    
    
    def get_q_u(self, ind, b, w, m): 
        """
        Computes the reconstructed quantized value from an index.
    
        Parameters:
        - ind: Quantization index.
        - n: Number of quantization bits.
        - w: Quantization dynamic range.
        - m: Mean value (offset).
    
        Returns:
        - The reconstructed value corresponding to the quantization index.
        """
        if b == 0:  # If no bits are allocated, return mean value.
            return m
    
        delta = w / (2**b)  # Compute the quantization step size.
        return delta * (ind + 0.5) + m  # Convert index back to quantized value.
                       
    
    def get_code_u(self, ind, n):
        """
        Encodes a quantization index into a binary representation.
    
        Parameters:
        - ind: Quantization index.
        - n: Number of quantization bits.
    
        Returns:
        - code: A binary representation of the quantization index.
        """
        if n == 0:  # If no bits are allocated, return empty code.
            return []
    
        ind_pos = 2**(n - 1) + ind  # Convert index to positive space for encoding.
    
        # Convert index to binary using a helper function.
        code = my_bin(ind_pos, n)
    
        return code
         
    
    def get_inv_code_u(self, code, n):
        """
        Decodes a binary sequence back into a quantization index.
    
        Parameters:
        - code: Binary representation of the quantized index.
        - n: Number of quantization bits.
    
        Returns:
        - The decoded quantization index.
        """
        if n == 0:  # If no bits are allocated, return zero.
            return 0
    
        ind_pos = my_inv_bin(code)  # Convert binary to integer.
    
        return ind_pos - 2**(n - 1)  # Convert back to original index space.
    
    
# Main program execution
if __name__ == "__main__":
    n = 3  # Number of bits used for quantization
    w = 2  # Dynamic range of the quantizer
    m = 0  # Quantizer midpoint

    # Compute maximum index and quantization step size
    if n == 0:
        ind_max = 0
        delta = 0
    else:
        ind_max = 2**(n - 1) - 1  # Maximum quantization index
        delta = w / (2**n - 1)  # Quantization step size

    verbose = False  # Debugging flag

    # Generate test input values ranging from -w to w with fine resolution
    x = np.array([i / 100 + m for i in range(-int(w * 120), int(w * 120))])

    # Create an instance of the Quantizer class
    q_x = Quantizer(verbose)

    # Initialize arrays to store quantization indices and quantized values
    x_ind_q = np.zeros(len(x))
    x_q = np.zeros(len(x))

    # ==========================
    # Mid-Tread Quantization
    # ==========================

    delta = w / (2**n - 1)  # Compute quantization step size
    for i in range(len(x)):
        # Get quantization index
        x_ind_q[i] = q_x.get_ind(x[i], n, w, m)

        # Encode quantized index into binary representation
        code = q_x.get_code(x_ind_q[i], n)

        # Decode binary representation back to index
        ind_rec = q_x.get_inv_code(code, n)

        # Convert index back to quantized value
        x_q[i] = q_x.get_q(ind_rec, n, w, m)
        x_q[i] = q_x.get_q(x_ind_q[i], n, w, m)

    # ==========================
    # Plot results for mid-tread quantization
    # ==========================

    # Plot quantization indices
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(x, x_ind_q, lw=2, label="Quantization indices of x")
    plt.xlabel("Input values (x)")
    plt.ylabel("Quantization index")
    plt.legend()
    plt.title(f"n = {n} bits, levels: {2**n - 1}, Δ = {delta:.2f}, range: [{-w/2},{w/2}]")
    plt.grid( which="major", color="#666666", linestyle="-")
    plt.minorticks_on()
    plt.grid(which="minor", color="#999999", linestyle="-", alpha=0.2)
    plt.show()

    # Plot quantized values vs input values
    plt.figure(figsize=(8, 8), dpi=100)
    plt.plot(x, x, lw=2, label="Original signal (x)")
    plt.plot(x, x_q, lw=2, label="Quantized values of x")
    plt.xlabel("Input values (x)")
    plt.ylabel("Quantized values (x_q)")
    plt.axis("equal")
    plt.legend()
    plt.title(f"n = {n} bits, Δ = {delta:.2f}, range: [{-w/2},{w/2}]")
    plt.grid(which="major", color="#666666", linestyle="-")
    plt.minorticks_on()
    plt.grid(which="minor", color="#999999", linestyle="-", alpha=0.2)
    plt.show()

    # Plot quantization error
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(x, x - x_q, lw=2)
    plt.xlabel("Input values (x)")
    plt.ylabel("Quantization error")
    plt.title("Quantization error (x - x_q)")
    plt.grid(which="major", color="#666666", linestyle="-")
    plt.minorticks_on()
    plt.grid(which="minor", color="#999999", linestyle="-", alpha=0.2)
    plt.show()

    # ==========================
    # Mid-Rise Quantization
    # ==========================

    delta = w / (2**n)  # Compute new step size for mid-rise quantizer
    for i in range(len(x)):
        # Get quantization index
        x_ind_q[i] = q_x.get_ind_u(x[i], n, w, m)

        # Encode index into binary representation
        code = q_x.get_code_u(x_ind_q[i], n)

        # Decode binary representation back to index
        ind_rec = q_x.get_inv_code_u(code, n)

        # Convert index back to quantized value
        x_q[i] = q_x.get_q_u(ind_rec, n, w, m)
        x_q[i] = q_x.get_q_u(x_ind_q[i], n, w, m)

    # ==========================
    # Plot results for mid-rise quantization
    # ==========================

    # Plot quantization indices
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(x, x_ind_q, lw=2, label="Quantization indices of x")
    plt.xlabel("Input values (x)")
    plt.ylabel("Quantization index")
    plt.legend()
    plt.title(f"n = {n} bits, levels: {2**n + 1}, Δ = {delta:.2f}, range: [{-w/2},{w/2}]")
    plt.grid(which="major", color="#666666", linestyle="-")
    plt.minorticks_on()
    plt.grid(which="minor", color="#999999", linestyle="-", alpha=0.2)
    plt.show()

    # Plot quantized values vs input values
    plt.figure(figsize=(8, 8), dpi=100)
    plt.plot(x, x, lw=2, label="Original signal (x)")
    plt.plot(x, x_q, lw=2, label="Quantized values of x")
    plt.xlabel("Input values (x)")
    plt.ylabel("Quantized values (x_q)")
    plt.axis("equal")
    plt.legend()
    plt.title(f"n = {n} bits, Δ = {delta:.2f}, range: [{-w/2},{w/2}]")
    plt.grid(which="major", color="#666666", linestyle="-")
    plt.minorticks_on()
    plt.grid(which="minor", color="#999999", linestyle="-", alpha=0.2)
    plt.show()

    # Plot quantization error
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(x, x - x_q, lw=2)
    plt.xlabel("Input values (x)")
    plt.ylabel("Quantization error")
    plt.title("Quantization error (x - x_q)")
    plt.grid(which="major", color="#666666", linestyle="-")
    plt.minorticks_on()
    plt.grid(which="minor", color="#999999", linestyle="-", alpha=0.2)
    plt.show()
