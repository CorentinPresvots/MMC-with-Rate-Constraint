# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:48:58 2023

@author: presvotscor
"""

import numpy as np
from scipy.optimize import minimize
import math

class Models:
    def __init__(self, fn=50, fs=6400, N=128, verbose=False):
        """
        Base class for modeling electrical signals.

        Parameters:
        - fn: Nominal frequency of the power system (Hz).
        - fs: Sampling frequency (Hz).
        - N: Number of samples per window.
        - verbose: Flag to enable detailed output for debugging.
        """
        self.fn = fn  # Nominal frequency of the power system
        self.fs = fs  # Sampling frequency
        self.N = N    # Number of samples in the window
        self.verbose = verbose
        
        # Precomputed constants
        self.Ts = 1 / fs  # Sampling period
        self.T = N / fs   # Duration of one window
        self.Tn = 1 / fn  # Duration of one cycle
        self.t = np.linspace(0, (N - 1) * self.Ts, N)  # Time vector
        

    def norm(self, x, y):
        """
        Computes the squared error between two signals.

        Parameters:
        - x: Model output.
        - y: Observed signal.

        Returns:
        - error: Squared error between x and y.
        """
        error = np.sum(np.square(y - x))
        return error


class Model_sin(Models): 
    def __init__(self, fn=50, fs=6400, N=128, verbose=False):
        """
        Subclass for sinusoidal signal modeling.
        
        Inherits from the base Models class.
        """
        super().__init__(fn, fs, N, verbose)

    def get_theta_sin_ini(self, y):
        """
        Estimates initial parameters for the sinusoidal model.

        Parameters:
        - y: Observed signal.

        Returns:
        - Initial estimates [amplitude, frequency, phase].
        """
        a = np.std(y) * np.sqrt(2)  # Estimate amplitude

        # Find the time index of the maximum value in the signal
        index_max_y = list(y).index(np.max(y))
        delta = self.t[index_max_y]  # Time corresponding to the maximum value
        
        # Estimate the initial phase
        phi = -(delta * 2 * math.pi) / self.Tn
        
        # Normalize phase to be within [-π, π]
        while abs(phi) > np.pi:
            phi -= np.sign(phi) * 2 * np.pi
        
        if self.verbose:
            print("max y: {:.2f}".format(np.max(y)))
            print("index max y: {}".format(index_max_y))
            print("delta t: {:.2f}".format(delta))
            print("phi: {:.2f}".format(phi))
        
        return [a, self.fn, phi]

    def get_model_sin(self, t, *theta):
        """
        Generates a sinusoidal signal based on parameters.

        Parameters:
        - t: Time vector.
        - theta: Model parameters [amplitude, frequency, phase].

        Returns:
        - Sinusoidal signal.
        """
        return theta[0] * np.cos(2 * math.pi * theta[1] * t + theta[2])

    def cost_function_sin(self, theta, y):
        """
        Cost function for optimizing the sinusoidal model parameters.

        Parameters:
        - theta: Current parameter estimates.
        - y: Observed signal.

        Returns:
        - Squared error between the model output and the observed signal.
        """
        x = self.get_model_sin(self.t, *theta)
        return self.norm(x, y)

    def get_theta_sin(self, y, m, w):
        """
        Optimizes the sinusoidal model parameters using a bounded approach.

        Parameters:
        - y: Observed signal.
        - m: Initial parameter estimates [amplitude, frequency, phase].
        - w: Parameter bounds widths for [amplitude, frequency, phase].

        Returns:
        - Optimized parameter values [amplitude, frequency, phase].
        """
        # Define bounds as [center - width/2, center + width/2]
        bounds = [(m[i] - w[i] / 2, m[i] + w[i] / 2) for i in range(3)]

        # Optimize the parameters using the Sequential Least Squares Programming (SLSQP) method
        result = minimize(self.cost_function_sin, m, args=(y,), method='SLSQP', bounds=bounds)
        return [*result.x]



class Model_poly(Models):
    def __init__(self, fn=50, fs=6400, N=128, verbose=False):
        """
        Subclass for polynomial-based signal modeling using Chebyshev polynomials.

        Parameters:
        - fn: Nominal frequency of the power system (Hz).
        - fs: Sampling frequency (Hz).
        - N: Number of samples per window.
        - verbose: Flag to enable detailed output for debugging.
        """
        super().__init__(fn, fs, N, verbose)

        # Define Chebyshev polynomial time domain [-1, 1)
        self.t_cheby_p = np.linspace(-1, 1 - 1/self.N, self.N)

        # Generate Chebyshev polynomial basis up to a predefined degree
        degree = 16
        basis = [np.ones_like(self.t_cheby_p), self.t_cheby_p]  # T0(x) = 1, T1(x) = x
        for i in range(2, degree + 1):  # Generate higher-order polynomials
            basis.append(2 * self.t_cheby_p * basis[i - 1] - basis[i - 2])
        
        # Store the basis as a matrix (each column corresponds to a polynomial order)
        self.chebyshev_basis = np.vstack(basis).T

    def get_model_poly(self, t, *theta):
        """
        Reconstructs the signal using the Chebyshev polynomial model.

        Parameters:
        - t: Time vector (unused directly in this method).
        - theta: Coefficients for the Chebyshev polynomial model.

        Returns:
        - Reconstructed signal based on the polynomial coefficients.
        """
        # Evaluate the polynomial using the provided coefficients
        x_rec = np.polynomial.chebyshev.chebval(self.t_cheby_p, theta)
        return x_rec

    def cost_function_poly(self, theta, y):
        """
        Cost function for polynomial model optimization.

        Parameters:
        - theta: Current parameter estimates (polynomial coefficients).
        - y: Observed signal.

        Returns:
        - Squared error between the model output and the observed signal.
        """
        x = self.get_model_poly(self.t, *theta)
        return self.norm(x, y)

    def get_theta_poly(self, y, m, w, order):
        """
        Optimizes the polynomial model coefficients using least squares.

        Parameters:
        - y: Observed signal.
        - m: Mean values or central estimates for parameter bounds.
        - w: Widths for parameter bounds.
        - order: Polynomial order to fit.

        Returns:
        - Optimized polynomial coefficients clipped to the specified bounds.
        """
        # Solve the least squares problem to find the polynomial coefficients
        hat_theta, _, _, _ = np.linalg.lstsq(self.chebyshev_basis[:, :order + 1], y, rcond=None)

        # Clip coefficients to stay within specified bounds
        hat_theta = np.clip(
            hat_theta, 
            [m[i] - w[i] / 2 for i in range(len(m))], 
            [m[i] + w[i] / 2 for i in range(len(m))]
        )
        return hat_theta




class Model_pred_samples(Models): 
    def __init__(self, fn=50, fs=6400, N=128, verbose=False):
        """
        Subclass for predictive modeling based on past samples.

        Parameters:
        - fn: Nominal frequency of the power system (Hz).
        - fs: Sampling frequency (Hz).
        - N: Number of samples per window.
        - verbose: Flag to enable detailed output for debugging.
        """
        super().__init__(fn, fs, N, verbose)

    def get_m_theta_pred_samples(self, N_p, eta, sigma, m, w):
        """
        Generate synthetic data, construct the regression matrix, and compute optimized parameters.

        Parameters:
        - N_p: Number of prediction coefficients.
        - eta: Delay parameter for predictive modeling.
        - sigma: Standard deviation of noise.
        - m: Mean values for parameter bounds.
        - w: Widths for parameter bounds.

        Returns:
        - Optimized prediction coefficients.
        """
        # Generate synthetic signal with noise
        yp = (
            np.array([0.75 * np.cos(2 * np.pi * self.fn * k * (1 / self.fs)) for k in range(3 * self.N)])
            + np.random.normal(0, sigma, 3 * self.N)
        )

        # Construct the regression matrix using the past samples
        X = self.get_X(yp[:2 * self.N], N_p, eta)

        # Compute optimized prediction coefficients
        m_theta_pred_samples = self.get_theta_pred_samples(X, yp[2 * self.N:], m, w)
        return m_theta_pred_samples

    def get_model_pred_samples(self, X, *alpha):
        """
        Reconstructs the predicted signal using the regression matrix and coefficients.

        Parameters:
        - X: Regression matrix.
        - alpha: Prediction coefficients.

        Returns:
        - Predicted signal as a 1D array.
        """
        # Compute the reconstructed signal as X * alpha
        x_rec = X @ np.array(alpha).reshape((np.size(X, 1), 1))
        return x_rec.reshape(self.N)

    def get_X(self, y_p, N_p, eta):
        """
        Constructs the regression matrix based on past samples.

        Parameters:
        - y_p: Past samples used for prediction.
        - N_p: Number of prediction coefficients.
        - eta: Delay parameter for predictive modeling.

        Returns:
        - Regression matrix (X).
        """
        X = np.zeros((self.N, N_p))
        for i in range(self.N):
            # Select past samples for the current row
            A = np.array(y_p[self.N + i - eta - N_p + 1:self.N + i - eta + 1])
            X[i] = A[::-1]  # Reverse the order for consistency with prediction coefficients
        return X

    def cost_function_pred_samples(self, alpha, y, X):
        """
        Cost function for optimizing predictive model parameters.

        Parameters:
        - alpha: Current parameter estimates (prediction coefficients).
        - y: Observed signal to be predicted.
        - X: Regression matrix.

        Returns:
        - Squared error between the predicted and observed signals.
        """
        x = self.get_model_pred_samples(X, *alpha)
        return self.norm(x, y)

    def get_theta_pred_samples(self, X, y, m, w):
        """
        Computes the optimized prediction coefficients using least squares.

        Parameters:
        - X: Regression matrix.
        - y: Observed signal to be predicted.
        - m: Mean values for parameter bounds.
        - w: Widths for parameter bounds.

        Returns:
        - Optimized prediction coefficients clipped to specified bounds.
        """
        # Solve the least squares problem to find the prediction coefficients
        hat_alpha, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

        # Reshape and clip the coefficients to stay within bounds
        hat_alpha = hat_alpha.reshape(np.size(hat_alpha, 0))
        hat_alpha = np.clip(
            hat_alpha, 
            [m[i] - w[i] / 2 for i in range(len(m))], 
            [m[i] + w[i] / 2 for i in range(len(m))]
        )
        return hat_alpha



# Main program for testing the models
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Measures import get_snr

    # Global settings
    verbose = False
    N = 256  # Number of samples
    fn = 60  # Nominal frequency
    fs = 15384.6  # Sampling frequency
    t = np.linspace(0, (N - 1) / fs, N)  # Time vector

    # Noise standard deviation
    sigma = 0.01

    # -------------------------------------
    # Test 1: Sinusoidal Model
    # -------------------------------------
    model_sin = Model_sin(fn, fs, N, verbose)
    
    # Parameter ranges for sinusoidal model
    m = [0.75, fn, 0]  # Mean values [amplitude, frequency, phase]
    w = [0.5, 0.2, 2 * np.pi]  # Width of bounds for [amplitude, frequency, phase]

    # Generate random parameters within the bounds
    a = np.random.uniform(m[0] - w[0] / 2, m[0] + w[0] / 2)
    f = np.random.uniform(m[1] - w[1] / 2, m[1] + w[1] / 2)
    phi = np.random.uniform(m[2] - w[2] / 2, m[2] + w[2] / 2)
    theta = [a, f, phi]

    # Generate noisy sinusoidal signal
    x_sin = model_sin.get_model_sin(t, *theta) + np.random.normal(0, sigma, N)

    # Estimate parameters
    theta_sin_hat = model_sin.get_theta_sin(x_sin, m, w)

    # Reconstruct signals
    x_sin_hat = model_sin.get_model_sin(t, *theta_sin_hat)
    x_sin_ini = model_sin.get_model_sin(t, *model_sin.get_theta_sin_ini(x_sin))

    # Plot results
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(t, x_sin, lw=2, label='x (original)')
    plt.plot(t, x_sin_ini, lw=2, label='x ini, SNR={:.1f} dB'.format(get_snr(x_sin, x_sin_ini)))
    plt.plot(t, x_sin_hat, lw=2, label='x hat, SNR={:.1f} dB'.format(get_snr(x_sin, x_sin_hat)))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Sinusoidal Model, theta=[{:.2f}, {:.2f}, {:.2f}]".format(*theta_sin_hat))
    plt.grid()
    plt.show()

    # -------------------------------------
    # Test 2: Polynomial Model
    # -------------------------------------
    order = 3  # Polynomial order
    m = [0] * (order + 1)  # Mean values for coefficients
    w = [2] * (order + 1)  # Width of bounds
    theta = [np.random.uniform(m[i] - w[i] / 2, m[i] + w[i] / 2) for i in range(order + 1)]

    model_poly = Model_poly(fn, fs, N, verbose)

    # Generate noisy polynomial signal
    x_poly = model_poly.get_model_poly(t, *theta) + np.random.normal(0, sigma, N)

    # Estimate polynomial coefficients
    theta_poly_hat = model_poly.get_theta_poly(x_poly, m, w, order)
    x_poly_hat = model_poly.get_model_poly(t, *theta_poly_hat)

    # Plot results
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(t, x_poly, lw=2, label='x (original)')
    plt.plot(t, x_poly_hat, lw=2, label='x hat, SNR={:.1f} dB'.format(get_snr(x_poly, x_poly_hat)))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Polynomial Model (Order {}), theta={}".format(order, [round(100 * theta_poly_hat[i]) / 100 for i in range(order + 1)]))
    plt.grid()
    plt.show()

    # -------------------------------------
    # Test 3: Predictive Model
    # -------------------------------------
    eta = 0  # Shift parameter
    N_p = 4  # Predictor order

    pred = Model_pred_samples(fn, fs, N, verbose)

    # Generate past and current signals
    t_p = np.linspace(-(N * 2) / fs, 0 - 1 / fs, N * 2)
    t = np.linspace(0, (N - 1) / fs, N)
    
    # Create the first sinusoidal component
    m_theta_sin = [0.75, fn, 0]  # Mean parameters: [amplitude, frequency, phase]
    w_theta_sin = [0.5, 0.0, 2 * np.pi]  # Bounds width for [amplitude, frequency, phase]
    
    # Randomly sample the parameters for the first sinusoidal component within the bounds
    a = np.random.uniform(m_theta_sin[0] - 0.5 * w_theta_sin[0], m_theta_sin[0] + 0.5 * w_theta_sin[0])  # Amplitude
    f = np.random.uniform(m_theta_sin[1] - 0.5 * w_theta_sin[1], m_theta_sin[1] + 0.5 * w_theta_sin[1])  # Frequency
    phi = np.random.uniform(m_theta_sin[2] - 0.5 * w_theta_sin[2], m_theta_sin[2] + 0.5 * w_theta_sin[2])  # Phase
    
    theta = [a, f, phi]  # First set of parameters for the sinusoidal signal
    
    # Create the second sinusoidal component with scaled-down amplitude and a different frequency
    m_theta_sin2 = [m_theta_sin[0] / 8, 3 * fn, 0]  # Mean parameters: [amplitude, frequency, phase]
    w_theta_sin2 = [w_theta_sin[0] / 8, w_theta_sin[1] / 8, w_theta_sin[2] / 8]  # Bounds width for second component
    
    # Randomly sample the parameters for the second sinusoidal component within the bounds
    a2 = np.random.uniform(m_theta_sin2[0] - 0.5 * w_theta_sin2[0], m_theta_sin2[0] + 0.5 * w_theta_sin2[0])  # Amplitude
    f2 = np.random.uniform(m_theta_sin2[1] - 0.5 * w_theta_sin2[1], m_theta_sin2[1] + 0.5 * w_theta_sin2[1])  # Frequency
    phi2 = np.random.uniform(m_theta_sin2[2] - 0.5 * w_theta_sin2[2], m_theta_sin2[2] + 0.5 * w_theta_sin2[2])  # Phase
    
    theta2 = [a2, f2, phi2]  # Second set of parameters for the sinusoidal signal
    
    # Generate the composite signal for the past time window
    # Combine two sinusoidal components and add Gaussian noise
    x_p = (model_sin.get_model_sin(t_p, *theta)  # First sinusoidal component
           + model_sin.get_model_sin(t_p, *theta2)  # Second sinusoidal component
           + np.random.normal(0, sigma, 2 * N))  # Additive Gaussian noise
    
    # Generate the composite signal for the current time window
    # Combine two sinusoidal components and add Gaussian noise
    x = (model_sin.get_model_sin(t, *theta)  # First sinusoidal component
         + model_sin.get_model_sin(t, *theta2)  # Second sinusoidal component
         + np.random.normal(0, sigma, N))  # Additive Gaussian noise

    # Construct regression matrix and estimate coefficients
    X = pred.get_X(x_p, N_p, eta)
    alpha_hat = pred.get_theta_pred_samples(X, x, m=[0] * N_p, w=[2] * N_p)

    # Reconstruct the predicted signal
    x_rec = pred.get_model_pred_samples(X, *alpha_hat)
    SNR = get_snr(x, x_rec)

    # Plot results
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(t_p, x_p, lw=2, label="xp (past)")
    plt.plot(t, x, lw=2, label="x (original)")
    plt.plot(t, x_rec, lw=2, label="x rec, SNR={:.1f} dB".format(SNR))
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Predictive Model: eta={}, Np={}".format(eta, N_p))
    plt.grid()
    plt.show()

    
