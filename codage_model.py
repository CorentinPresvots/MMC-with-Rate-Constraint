# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 07:55:58 2023

@author: coren
"""

import numpy as np

from Quantization import Quantizer
from Bits_allocation import Allocation_sin, Allocation_poly, Allocation_pred_samples, Allocation_None
from Models import Model_sin, Model_poly, Model_pred_samples


class Model_Encoder(
    Model_sin, Model_poly, Model_pred_samples, Quantizer,
    Allocation_sin, Allocation_poly, Allocation_pred_samples, Allocation_None
):
    def __init__(self, fn=50, fs=6400, N=128, verbose=False):
        """
        Combines multiple models, allocation strategies, and quantization
        for encoding parameters in sinusoidal, polynomial, and predictive models.

        Parameters:
        - fn: Nominal frequency of the power system (Hz).
        - fs: Sampling frequency (Hz).
        - N: Number of samples in a signal window.
        - verbose: Enables detailed output for debugging.
        """
        self.verbose = verbose  # Verbosity flag
        self.t = np.linspace(0, (N - 1) / fs, N)  # Time vector

        # Initialize the models
        Model_sin.__init__(self, fn, fs, N)
        Model_poly.__init__(self, fn, fs, N)
        Model_pred_samples.__init__(self, fn, fs, N)

        # Initialize allocation strategies with no max_bits constraint
        max_bits = None
        Allocation_sin.__init__(self, fn, fs, N, max_bits)
        Allocation_poly.__init__(self, fn, fs, N, max_bits)
        Allocation_pred_samples.__init__(self, fn, fs, N, max_bits)
        Allocation_None.__init__(self, fn, fs, N, max_bits)

        # Initialize quantization utilities
        Quantizer.__init__(self)

    def get_theta_sin_tilde(self, theta_sin_hat, nx, m_theta_sin, w_theta_sin):
        """
        Quantize and encode the sinusoidal model parameters.

        Parameters:
        - theta_sin_hat: Estimated parameters [amplitude, frequency, phase].
        - nx: Total number of bits for allocation.
        - m_theta_sin: Mean values for the parameters.
        - w_theta_sin: Bounds width for the parameters.

        Returns:
        - theta_sin_tilde: Quantized parameters.
        - code_theta_sin_tilde: Encoded binary representation of the parameters.
        """
        # Optimal bit allocation for the sinusoidal model
        al_sin = self.get_nx_sin(nx, m_theta_sin, w_theta_sin)[0]

        # Quantization and encoding
        code_theta_sin_tilde = [0] * np.sum(al_sin)
        theta_sin_tilde = [0] * 3
        ptr = 0

        for i in range(3):  # Loop through amplitude, frequency, phase
            # Get quantized index for the parameter
            theta_sin_ind = self.get_ind_u(theta_sin_hat[i], al_sin[i], w_theta_sin[i], m_theta_sin[i])

            # Generate binary code for the quantized index
            code_theta_sin_tilde[ptr:ptr + al_sin[i]] = self.get_code_u(theta_sin_ind, al_sin[i])
            ptr += al_sin[i]

            # Reconstruct the quantized parameter
            theta_sin_tilde[i] = self.get_q_u(theta_sin_ind, al_sin[i], w_theta_sin[i], m_theta_sin[i])

        return theta_sin_tilde, code_theta_sin_tilde

    def get_theta_poly_tilde(self, theta_poly_hat, nx, m_theta_poly, w_theta_poly):
        """
        Quantize and encode the polynomial model parameters.

        Parameters:
        - theta_poly_hat: Estimated coefficients of the polynomial model.
        - nx: Total number of bits for allocation.
        - m_theta_poly: Mean values for the coefficients.
        - w_theta_poly: Bounds width for the coefficients.

        Returns:
        - theta_poly_tilde: Quantized coefficients.
        - code_theta_poly_tilde: Encoded binary representation of the coefficients.
        """
        order = len(theta_poly_hat) - 1  # Polynomial order
        al_poly = self.get_nx_poly(nx, w_theta_poly)[0]  # Optimal bit allocation

        # Quantization and encoding
        code_theta_poly_tilde = [0] * sum(al_poly)
        theta_poly_tilde = [0] * (order + 1)
        ptr = 0

        for i in range(order + 1):  # Loop through polynomial coefficients
            # Get quantized index for the coefficient
            theta_poly_ind = self.get_ind_u(theta_poly_hat[i], al_poly[i], w_theta_poly[i], m_theta_poly[i])

            # Generate binary code for the quantized index
            code_theta_poly_tilde[ptr:ptr + al_poly[i]] = self.get_code_u(theta_poly_ind, al_poly[i])
            ptr += al_poly[i]

            # Reconstruct the quantized coefficient
            theta_poly_tilde[i] = self.get_q_u(theta_poly_ind, al_poly[i], w_theta_poly[i], m_theta_poly[i])

        return theta_poly_tilde, code_theta_poly_tilde

    def get_theta_pred_samples_tilde(self, theta_pred_samples_hat, nx, m_theta_pred_samples, w_theta_pred_samples):
        """
        Quantize and encode the predictive model parameters.

        Parameters:
        - theta_pred_samples_hat: Estimated predictive coefficients.
        - nx: Total number of bits for allocation.
        - m_theta_pred_samples: Mean values for the coefficients.
        - w_theta_pred_samples: Bounds width for the coefficients.

        Returns:
        - theta_pred_samples_tilde: Quantized coefficients.
        - code_theta_pred_samples_tilde: Encoded binary representation of the coefficients.
        """
        order = len(theta_pred_samples_hat)  # Number of predictive coefficients
        al_pred_samples = self.get_nx_pred_samples(nx, w_theta_pred_samples, eta=0)[0]  # Optimal bit allocation

        # Quantization and encoding
        code_theta_pred_samples_tilde = [0] * np.sum(al_pred_samples)
        theta_pred_samples_tilde = [0] * order
        ptr = 0

        for i in range(order):  # Loop through predictive coefficients
            # Get quantized index for the coefficient
            theta_pred_samples_ind = self.get_ind_u(
                theta_pred_samples_hat[i], al_pred_samples[i],
                w_theta_pred_samples[i], m_theta_pred_samples[i]
            )

            # Generate binary code for the quantized index
            code_theta_pred_samples_tilde[ptr:ptr + al_pred_samples[i]] = self.get_code_u(theta_pred_samples_ind, al_pred_samples[i])
            ptr += al_pred_samples[i]

            # Reconstruct the quantized coefficient
            theta_pred_samples_tilde[i] = self.get_q_u(
                theta_pred_samples_ind, al_pred_samples[i],
                w_theta_pred_samples[i], m_theta_pred_samples[i]
            )

        return theta_pred_samples_tilde, code_theta_pred_samples_tilde

     
class Model_Decoder(
    Model_sin, Model_poly, Model_pred_samples,
    Allocation_sin, Allocation_poly, Allocation_pred_samples, Allocation_None, Quantizer
):
    def __init__(self, fn=50, fs=6400, N=128, verbose=False):
        """
        Combines multiple models, allocation strategies, and quantization for decoding
        parameters of sinusoidal, polynomial, and predictive models.

        Parameters:
        - fn: Nominal frequency of the power system (Hz).
        - fs: Sampling frequency (Hz).
        - N: Number of samples in a signal window.
        - verbose: Enables detailed output for debugging.
        """
        self.verbose = verbose  # Verbosity flag
        self.t = np.linspace(0, (N - 1) / fs, N)  # Time vector

        # Initialize models
        Model_sin.__init__(self, fn, fs, N)
        Model_poly.__init__(self, fn, fs, N)
        Model_pred_samples.__init__(self, fn, fs, N)

        # Initialize allocation strategies with no max_bits constraint
        max_bits = None
        Allocation_sin.__init__(self, fn, fs, N, max_bits)
        Allocation_poly.__init__(self, fn, fs, N, max_bits)
        Allocation_pred_samples.__init__(self, fn, fs, N, max_bits)
        Allocation_None.__init__(self, fn, fs, N, max_bits)

        # Initialize quantization utilities
        Quantizer.__init__(self)

    def get_theta_sin_tilde(self, code, nx, m_theta_sin, w_theta_sin):
        """
        Decodes and reconstructs quantized sinusoidal model parameters.

        Parameters:
        - code: Encoded binary representation of the parameters.
        - nx: Total number of bits for allocation.
        - m_theta_sin: Mean values for the parameters [amplitude, frequency, phase].
        - w_theta_sin: Bounds width for the parameters.

        Returns:
        - theta_sin_tilde: Decoded and reconstructed sinusoidal parameters.
        """
        al_sin = self.get_nx_sin(nx, m_theta_sin, w_theta_sin)[0]  # Optimal bit allocation
        theta_sin_tilde = [0] * 3
        ptr = 0

        for i in range(3):  # Loop through amplitude, frequency, and phase
            # Decode the quantized index from the binary code
            theta_sin_ind = self.get_inv_code_u(code[ptr:ptr + al_sin[i]], al_sin[i])
            ptr += al_sin[i]

            # Reconstruct the quantized parameter
            theta_sin_tilde[i] = self.get_q_u(theta_sin_ind, al_sin[i], w_theta_sin[i], m_theta_sin[i])

        return theta_sin_tilde

    def get_theta_poly_tilde(self, code, nx, m_theta_poly, w_theta_poly):
        """
        Decodes and reconstructs quantized polynomial model coefficients.

        Parameters:
        - code: Encoded binary representation of the coefficients.
        - nx: Total number of bits for allocation.
        - m_theta_poly: Mean values for the coefficients.
        - w_theta_poly: Bounds width for the coefficients.

        Returns:
        - theta_poly_tilde: Decoded and reconstructed polynomial coefficients.
        """
        order = len(w_theta_poly) - 1  # Polynomial order
        al_poly = self.get_nx_poly(nx, w_theta_poly)[0]  # Optimal bit allocation
        theta_poly_tilde = [0] * (order + 1)
        ptr = 0

        for i in range(order + 1):  # Loop through polynomial coefficients
            # Decode the quantized index from the binary code
            theta_poly_ind = self.get_inv_code_u(code[ptr:ptr + al_poly[i]], al_poly[i])
            ptr += al_poly[i]

            # Reconstruct the quantized coefficient
            theta_poly_tilde[i] = self.get_q_u(theta_poly_ind, al_poly[i], w_theta_poly[i], m_theta_poly[i])

        return theta_poly_tilde

    def get_theta_pred_samples_tilde(self, code, nx, m_theta_pred_samples, w_theta_pred_samples):
        """
        Decodes and reconstructs quantized predictive model coefficients.

        Parameters:
        - code: Encoded binary representation of the coefficients.
        - nx: Total number of bits for allocation.
        - m_theta_pred_samples: Mean values for the coefficients.
        - w_theta_pred_samples: Bounds width for the coefficients.

        Returns:
        - theta_pred_samples_tilde: Decoded and reconstructed predictive coefficients.
        """
        order = len(w_theta_pred_samples)  # Number of predictive coefficients
        al_pred_samples = self.get_nx_pred_samples(nx, w_theta_pred_samples, eta=0)[0]  # Optimal bit allocation
        theta_pred_samples_tilde = [0] * order
        ptr = 0

        for i in range(order):  # Loop through predictive coefficients
            # Decode the quantized index from the binary code
            theta_pred_samples_ind = self.get_inv_code_u(code[ptr:ptr + al_pred_samples[i]], al_pred_samples[i])
            ptr += al_pred_samples[i]

            # Reconstruct the quantized coefficient
            theta_pred_samples_tilde[i] = self.get_q_u(
                theta_pred_samples_ind, al_pred_samples[i],
                w_theta_pred_samples[i], m_theta_pred_samples[i]
            )

        return theta_pred_samples_tilde


# Main program
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Measures import get_snr  # Function to compute Signal-to-Noise Ratio (SNR)

    # Global parameters
    verbose = False
    N = 128          # Number of samples
    fn = 50          # Nominal frequency (Hz)
    fs = 6400        # Sampling frequency (Hz)
    nx = 30          # Total number of bits for allocation
    sigma = 0.01    # Standard deviation of the noise introduced in the test signal

    t = np.linspace(0, (N - 1) / fs, N)  # Time vector for the signal

    ####################### Initialize the Model_Encoder class
    m = Model_Encoder(fn=fn, fs=fs, N=N, verbose=verbose)

    ####################### Generate a noisy sinusoidal test signal
    m_theta_sin = [0.75, fn, 0]          # Mean parameters [amplitude, frequency, phase]
    w_theta_sin = [0.5, 0.2, 2 * np.pi]  # Bound widths for each parameter


    # -------------------------------------
    # Test 1: Sinusoidal Model
    # -------------------------------------
    # Randomly sample parameters within the defined bounds
    a = np.random.uniform(m_theta_sin[0] - 0.5 * w_theta_sin[0], m_theta_sin[0] + 0.5 * w_theta_sin[0])
    f = np.random.uniform(m_theta_sin[1] - 0.5 * w_theta_sin[1], m_theta_sin[1] + 0.5 * w_theta_sin[1])
    phi = np.random.uniform(m_theta_sin[2] - 0.5 * w_theta_sin[2], m_theta_sin[2] + 0.5 * w_theta_sin[2])
    theta_sin = [a, f, phi]  # Actual parameters

    print("theta sin: {:.2f}, {:.2f}, {:.2f}".format(*theta_sin))

    # Initialize the sinusoidal model
    model_sin = Model_sin(fn, fs, N)
    x_sin = model_sin.get_model_sin(t, *theta_sin) + np.random.normal(0, sigma, N)  # Generate noisy signal

    ##################### Encode x_sin using optimal allocation
    theta_sin_hat = m.get_theta_sin(x_sin, m_theta_sin, w_theta_sin)  # Estimate parameters
    print("theta sin hat: {:.2f}, {:.2f}, {:.2f}".format(*theta_sin_hat))

    theta_sin_tilde, _ = m.get_theta_sin_tilde(theta_sin_hat, nx, m_theta_sin, w_theta_sin)  # Quantize parameters
    print("theta sin tilde: {:.2f}, {:.2f}, {:.2f}".format(*theta_sin_tilde))

    x_sin_hat = m.get_model_sin(t, *theta_sin_hat)       # Reconstructed signal from estimated parameters
    x_sin_tilde = m.get_model_sin(t, *theta_sin_tilde)   # Reconstructed signal after quantization

    # Plot results
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(t, x_sin, lw=2, label='x (original)')
    plt.plot(t, x_sin_hat, lw=2, label='x hat, SNR={:.1f} dB'.format(get_snr(x_sin, x_sin_hat)))
    plt.plot(t, x_sin_tilde, lw=2, label='x tilde, SNR={:.1f} dB, nx={} bits'.format(get_snr(x_sin, x_sin_tilde), nx))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Sinusoidal Model")
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()

    # -------------------------------------
    # Test 2: Polynomial Model
    # -------------------------------------
    ####################### Generate a noisy polynomial test signal
    order = 4  # Polynomial order

    m_theta_poly = [0] * (order + 1)  # Mean parameters (centered polynomial)
    w_theta_poly = [2] * (order + 1)  # Bound widths for each coefficient

    # Initialize the polynomial model
    model_poly = Model_poly(fn, fs, N, order)

    # Randomly sample polynomial coefficients
    theta_poly = [np.random.uniform(-0.5 * w_theta_poly[k], 0.5 * w_theta_poly[k]) for k in range(order + 1)]
    print("theta poly:", ["{:.2f}".format(theta_poly[i]) for i in range(order + 1)])

    # Generate noisy polynomial signal
    x_poly = model_poly.get_model_poly(t, *theta_poly) + np.random.normal(0, sigma, N)

    ##################### Encode x_poly using optimal allocation
    theta_poly_hat = m.get_theta_poly(x_poly, m_theta_poly, w_theta_poly, order)  # Estimate coefficients
    print("theta poly hat:", ["{:.2f}".format(theta_poly_hat[i]) for i in range(order + 1)])

    theta_poly_tilde, _ = m.get_theta_poly_tilde(theta_poly_hat, nx, m_theta_poly, w_theta_poly)  # Quantize coefficients
    print("theta poly tilde:", ["{:.2f}".format(theta_poly_tilde[i]) for i in range(order + 1)])

    x_poly_hat = m.get_model_poly(t, *theta_poly_hat)       # Reconstructed signal from estimated coefficients
    x_poly_tilde = m.get_model_poly(t, *theta_poly_tilde)   # Reconstructed signal after quantization

    # Plot results
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(t, x_poly, lw=2, label='x (original)')
    plt.plot(t, x_poly_hat, lw=2, label='x hat, SNR={:.1f} dB'.format(get_snr(x_poly, x_poly_hat)))
    plt.plot(t, x_poly_tilde, lw=2, label='x tilde, SNR={:.1f} dB, nx={} bits'.format(get_snr(x_poly, x_poly_tilde), nx))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Polynomial Model")
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()


    # -------------------------------------
    # Test 3: Predictive Model
    # -------------------------------------
    ##################### Encode predictive samples with optimal allocation
    
    # Predictive model settings
    N_p = 5          # Order of the predictive model
    eta = 0          # Delay parameter
    #################### Generate a noisy sinusoidal test signal with harmonics
    
    # Parameters for the first sinusoidal component
    m_theta_sin = [0.75, fn, 0]                # Mean parameters [amplitude, frequency, phase]
    w_theta_sin = [0.5, 0., 2 * np.pi]         # Bound widths for each parameter
    
    # Randomly sample parameters for the first sinusoidal component
    a = np.random.uniform(m_theta_sin[0] - 0.5 * w_theta_sin[0], m_theta_sin[0] + 0.5 * w_theta_sin[0])
    f = np.random.uniform(m_theta_sin[1] - 0.5 * w_theta_sin[1], m_theta_sin[1] + 0.5 * w_theta_sin[1])
    phi = np.random.uniform(m_theta_sin[2] - 0.5 * w_theta_sin[2], m_theta_sin[2] + 0.5 * w_theta_sin[2])
    theta_sin = [a, f, phi]  # Actual parameters for the first sinusoidal component
    
    # Parameters for the second sinusoidal component (harmonic)
    m_theta_sin2 = [m_theta_sin[0] / 8, 3 * fn, 0]  # Mean parameters for the harmonic
    w_theta_sin2 = [w_theta_sin[0] / 8, w_theta_sin[1] / 8, w_theta_sin[2] / 8]  # Smaller bounds for the harmonic
    
    # Randomly sample parameters for the harmonic
    a2 = np.random.uniform(m_theta_sin2[0] - 0.5 * w_theta_sin2[0], m_theta_sin2[0] + 0.5 * w_theta_sin2[0])
    f2 = np.random.uniform(m_theta_sin2[1] - 0.5 * w_theta_sin2[1], m_theta_sin2[1] + 0.5 * w_theta_sin2[1])
    phi2 = np.random.uniform(m_theta_sin2[2] - 0.5 * w_theta_sin2[2], m_theta_sin2[2] + 0.5 * w_theta_sin2[2])
    theta_sin2 = [a2, f2, phi2]  # Actual parameters for the harmonic
    
    
    # Generate the composite signal with two components and noise
    t_pred_samples = np.linspace(0, (3 * N - 1) / fs, 3 * N)  # Extended time vector
    x_sin_H = (
        model_sin.get_model_sin(t_pred_samples, *theta_sin)  # First sinusoidal component
        + model_sin.get_model_sin(t_pred_samples, *theta_sin2)  # Harmonic
        + np.random.normal(0, sigma, 3 * N)  # Add noise
    )
    
    
    # Estimate mean parameters for predictive coefficients
    m_theta_pred_samples = m.get_m_theta_pred_samples(N_p, eta, sigma, [0] * N_p, [10] * N_p)
    w_theta_pred_samples = [1] * N_p  # Bounds for the predictive coefficients
    
    print("theta pred samples m:", ["{:.2f}".format(m_theta_pred_samples[i]) for i in range(N_p)])
    
    # Construct the regression matrix for prediction
    X = m.get_X(x_sin_H[0:2 * N], N_p, eta)
    
    # Estimate predictive coefficients
    theta_pred_samples_hat = m.get_theta_pred_samples(X, x_sin_H[2 * N:], m_theta_pred_samples, w_theta_pred_samples)
    print("theta pred samples hat:", ["{:.2f}".format(theta_pred_samples_hat[i]) for i in range(N_p)])
    
    # Quantize predictive coefficients
    theta_pred_samples_tilde, _ = m.get_theta_pred_samples_tilde(
        theta_pred_samples_hat, nx, m_theta_pred_samples, w_theta_pred_samples
    )
    print("theta pred samples tilde:", ["{:.2f}".format(theta_pred_samples_tilde[i]) for i in range(N_p)])
    
    # Reconstruct the predicted signal
    x_pred_samples_hat = m.get_model_pred_samples(X, *theta_pred_samples_hat)  # Reconstructed from estimated coefficients
    x_pred_samples_tilde = m.get_model_pred_samples(X, *theta_pred_samples_tilde)  # Reconstructed after quantization
    
    # Plot the generated composite signal
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(t_pred_samples[0:2 * N], x_sin_H[0:2 * N], lw=2, label='xp (past samples)')
    plt.plot(t_pred_samples[2 * N:], x_sin_H[2 * N:], lw=2, label='x (current samples)')
    plt.plot(
        t_pred_samples[2 * N:], x_pred_samples_tilde, lw=2,
        label='x tilde, SNR={:.1f} dB, nx={} bits'.format(get_snr(x_sin_H[2 * N:], x_pred_samples_tilde), nx)
    )
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Test Signal with Harmonics (Predictive Samples)")
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    # Plot the results
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(t, x_sin_H[2 * N:], lw=2, label='x (original)')
    plt.plot(t, x_pred_samples_hat, lw=2, label='x hat, SNR={:.1f} dB'.format(get_snr(x_sin_H[2 * N:], x_pred_samples_hat)))
    plt.plot(
        t, x_pred_samples_tilde, lw=2,
        label='x tilde, SNR={:.1f} dB, nx={} bits'.format(get_snr(x_sin_H[2 * N:], x_pred_samples_tilde), nx)
    )
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Predictive Model with Harmonics")
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
