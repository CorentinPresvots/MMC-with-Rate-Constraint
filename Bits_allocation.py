# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:53:45 2023

@author: coren
"""


import numpy as np


class Allocation:
    """Base class for handling allocation logic."""

    def __init__(self, verbose=False):
        """
        Initialize the Allocation class.

        Parameters:
        verbose (bool): If True, enables verbose logging.
        """
        self.verbose = verbose

    def round_allocation(self, L, nx, w):
        """
        Adjust allocation to match a specific total.

        Parameters:
        L (list or np.ndarray): Initial allocation levels.
        nx (int): Target total allocation.
        w (list or np.ndarray): Weights to determine adjustment priority.

        Returns:
        list: Adjusted allocation meeting the total target.
        """
        w_theta = np.array(list(w))
        ind_w_theta_sort = np.argsort(w_theta)

        # Ensure non-negative allocation levels and convert to integers
        root = np.maximum(np.array(L), 0)
        root = list(map(int, np.round(root)))

        # Adjust allocation levels iteratively
        if np.sum(root) > nx:
            # Reduce allocation if total exceeds the target
            i = 1
            while np.sum(root) != nx:
                if root[ind_w_theta_sort[i % len(L)]] > 0:
                    root[ind_w_theta_sort[i % len(L)]] -= 1
                i += 1
        elif np.sum(root) < nx:
            # Increase allocation if total falls short of the target
            i = -1
            while np.sum(root) != nx:
                root[ind_w_theta_sort[i % -len(L)]] += 1
                i -= 1

        return root


class Allocation_sin(Allocation):
    """Specialized allocation for sinusoidal models."""

    def __init__(self, fn, fs, N, verbose=False):
        """
        Initialize the Allocation_sin class.

        Parameters:
        fn (float): Fundamental frequency.
        fs (float): Sampling frequency.
        N (int): Number of samples.
        verbose (bool): If True, enables verbose logging.
        """
        self.N = N
        self.fs = fs
        self.fn = fn
        self.verbose = verbose
        #super().__init__(verbose)

    def gamma0_sin(self, x, m, w):
        """
        Compute the base gamma0 for sinusoidal models.

        Parameters:
        x (list or np.ndarray): Allocation level
        m (list or np.ndarray): mean of distribution
        w (list or np.ndarray): Width of distribution

        Returns:
        float: The computed gamma0 value.
        """
        return (w[0]**2 * (2**(-2*x[0])) +
                w[1]**2 * (2**(-2*x[1])) * (m[0]**2) * ((4*np.pi**2)/3) * (self.fn**(-2)) +
                w[2]**2 * (2**(-2*x[2])) * (m[0]**2)) / 24

    def gamma1_sin(self, x, m, w):
        """
        Compute the base gamma1 for sinusoidal models.

        Parameters:
        x (list or np.ndarray): Allocation level
        m (list or np.ndarray): mean of distribution
        w (list or np.ndarray): Width of distribution

        Returns:
        float: The computed gamma1 value.
        """
        return (1 - 1/self.N) * self.gamma0_sin(x, m, w)

    def get_nx_sin(self, nx, m, w):
        """
        Calculate allocation for a given total number of bits.

        Parameters:
        nx (int): Target number of bits to allocate.
        m (list or np.ndarray): mean of distribution
        w (list or np.ndarray): Width of distribution

        Returns:
        tuple: (allocation list (int), allocation (float))
        """
        Np = 3  # Number of parameters
        pw = np.array([w[0]**2, (m[0]**2)*(4*np.pi**2/3)*(self.fn**(-2))*w[1]**2, (m[0]**2)*w[2]**2])
        root_theo = [0, 0, 0]

        # Compute theoretical allocation
        for k in range(Np):
            root_theo[k] = 0.5 * (-(1/Np)*np.sum(np.log2(pw)) + 2*nx/Np + np.log2(pw[k]))

        # Round allocation to integers that sum to nx
        return self.round_allocation(root_theo, nx, w), root_theo

    def MSE_theoritical_sin(self, x, m, w, gamma0_em, gamma1_em):
        """
        Compute the theoretical mean squared error (MSE) for sinusoidal models.

        Parameters:
        x (list or np.ndarray): Allocation level
        m (list or np.ndarray): Parameters related to the model.
        w (list or np.ndarray): Width of distribution
        gamma0_em (float): Gamma0 from error model.
        gamma1_em (float): Gamma1 from error model.

        Returns:
        float: The computed theoretical MSE.
        """
        gamma0_q = self.gamma0_sin(x[:3], m, w)
        gamma1_q = self.gamma1_sin(x[:3], m, w)
        return ((gamma0_em**2 - gamma1_em**2) / gamma0_em + (gamma0_q**2 - gamma1_q**2) / gamma0_q) * 2**(-2*x[3])

    def get_nx_nr_constraint_bit_sin(self, m, w, error_model, n_tot):
        """
        Computes the bit allocation (nx) and the number of remaining bits (nr)
        that minimize the MSE for a given total number of available bits (n_tot).
        
        Parameters:
        - m: Prior mean values of the model parameters.
        - w: Prior width of the model parameters.
        - error_model: Model estimation error.
        - n_tot: Total available bits for allocation.
    
        Returns:
        - root: A list containing the bit allocation (nx) for each parameter and the remaining bits (nr).
        - self.MSE_theoritical_sin(root, m, w, gamma0_em, gamma1_em): The minimum achievable MSE for the optimal allocation.
        """
        # Compute empirical autocorrelation values from the error model
        gamma0_em = np.mean(error_model**2)
        gamma1_em = np.sum(error_model[0:self.N-1] * error_model[1:self.N]) / (self.N-1)
        
        # Initialize root (optimal allocation) and root_test (temporary allocation)
        root = [0, 0, 0, 0]
        root_test = [0, 0, 0, 0]
        MSE_memory = np.infty
        nx = 0
    
        # Iteratively test allocations, keeping track of the minimum MSE
        while nx <= n_tot:
            root_test[:3] = self.get_nx_sin(nx, m, w)[1]  # Get allocation for the first 3 parameters
            root_test[3] = (n_tot - nx) / self.N          # Remaining bits per sample
            
            # Calculate theoretical MSE for the current allocation
            MSE_test = self.MSE_theoritical_sin(root_test, m, w, gamma0_em, gamma1_em)
            
            # Update the best allocation if the MSE improves
            if MSE_test < MSE_memory:
                MSE_memory = MSE_test
                root = list(root_test)
            else:
                break
            
            nx += 1
        
        # Return the optimal allocation and the corresponding MSE
        return root, self.MSE_theoritical_sin(root, m, w, gamma0_em, gamma1_em)
    
    
    def get_nx_nr_constraint_MSE_sin(self, m, w, error_model, MSE_target):
        """
        Computes the bit allocation (nx) and the number of remaining bits (nr)
        that achieve a specified MSE target.
    
        Parameters:
        - m: Prior mean values of the model parameters.
        - w: Prior width of the model parameters.
        - error_model: Model estimation error.
        - MSE_target: The target MSE value.
    
        Returns:
        - root: A list containing the bit allocation (nx) for each parameter and the remaining bits (nr).
        - self.MSE_theoritical_sin(root, m, w, gamma0_em, gamma1_em): The actual MSE achieved with the optimal allocation.
        """
        # Compute empirical autocorrelation values from the error model
        gamma0_em = np.mean(error_model**2)
        gamma1_em = np.sum(error_model[0:self.N-1] * error_model[1:self.N]) / (self.N-1)
        
        # Initialize root (optimal allocation) and root_test (temporary allocation)
        root = [0, 0, 0, 0]
        root_test = [0, 0, 0, 0]
        n_tot_memory = np.infty
        nx = 0
    
        # Iteratively test allocations, stopping when the total number of bits reaches its minimum
        while True:
            root_test[:3] = self.get_nx_sin(nx, m, w)[1]  # Get allocation for the first 3 parameters
            gamma_0_q = self.gamma0_sin(root_test[:3], m, w)
            gamma_1_q = self.gamma1_sin(root_test[:3], m, w)
            
            # Calculate theoretical MSE for the current allocation
            MSE_test = (gamma0_em**2 - gamma1_em**2) / gamma0_em + (gamma_0_q**2 - gamma_1_q**2) / gamma_0_q
            
            # Determine the number of remaining bits (nr) required to meet the MSE target
            root_test[3] = -0.5 * np.log2(MSE_target / MSE_test)
            n_tot_test = nx + root_test[3] * self.N
    
            # Update the best allocation if it uses fewer total bits
            if n_tot_test < n_tot_memory:
                n_tot_memory = n_tot_test
                root = list(root_test)
            else:
                break
            
            nx += 1
        
        # Return the optimal allocation and the corresponding MSE
        return root, self.MSE_theoritical_sin(root, m, w, gamma0_em, gamma1_em)

    
class Allocation_None(Allocation):
    """
    A specialized Allocation class when the first stage is bypass
    """

    def __init__(self, fn, fs, N, verbose=False):
        """
        Initialize the Allocation_None class.

        Parameters:
        fn (float): Fundamental frequency of the signal (not used in this subclass).
        fs (float): Sampling frequency of the signal (not used in this subclass).
        N (int): Number of samples.
        verbose (bool): If True, enable verbose output.
        """
        self.N = N
        self.fs = fs
        self.fn = fn
        self.verbose = verbose
        #super().__init__(verbose)

    def MSE_theoritical_none(self, x, gamma0_em, gamma1_em):
        """
        Calculate the theoretical MSE .

        Parameters:
        x (float): Allocation level (e.g., bit allocation or signal precision).
        gamma0_em (float): Empirical zero-lag variance of the error model.
        gamma1_em (float): Empirical one-lag variance of the error model.

        Returns:
        float: Theoretical MSE value.
        """
        return ((gamma0_em**2 - gamma1_em**2) / gamma0_em) * 2**(-2 * x)

    def get_nx_nr_constraint_MSE_none(self, error_model, MSE_target):
        """
        Determine the allocation required to meet a given MSE target.

        Parameters:
        error_model (np.ndarray): Empirical error model samples.
        MSE_target (float): Desired MSE threshold.

        Returns:
        tuple:
            - root (float): Computed allocation value.
            - mse_theoretical (float): Theoretical MSE corresponding to the allocation.
        """
        gamma0_em = np.mean(error_model**2)
        gamma1_em = np.sum(error_model[0:self.N-1] * error_model[1:self.N]) / (self.N - 1)

        # Compute the initial MSE based on empirical gamma values
        MSE_test = (gamma0_em**2 - gamma1_em**2) / gamma0_em

        # Determine the allocation level needed to achieve the target MSE
        root = -0.5 * np.log2(MSE_target / MSE_test)

        return root, self.MSE_theoritical_none(root, gamma0_em, gamma1_em)

    def get_nx_nr_constraint_bit_none(self, error_model, n_tot):
        """
        Allocate bits to meet the total allocation constraint.

        Parameters:
        error_model (np.ndarray): Empirical error model samples.
        n_tot (int): Total bit allocation target.

        Returns:
        tuple:
            - root (int): Fixed allocation level, which is zero in this subclass.
            - mse_theoretical (float): Theoretical MSE corresponding to the allocation.
        """
        gamma0_em = np.mean(error_model**2)
        gamma1_em = np.sum(error_model[0:self.N-1] * error_model[1:self.N]) / (self.N - 1)


        # Return the zero allocation and corresponding theoretical MSE
        return 0, self.MSE_theoritical_none(n_tot / self.N, gamma0_em, gamma1_em)

        

    
class Allocation_poly(Allocation):
    """
    A specialized allocation class based on polynomial approximations.
    """

    def __init__(self, fn, fs, N, verbose=False):
        """
        Initialize the Allocation_poly class.

        Parameters:
        fn (float): Fundamental frequency of the signal.
        fs (float): Sampling frequency of the signal.
        N (int): Number of samples.
        verbose (bool): If True, enable verbose output.
        """
        self.verbose = verbose
        self.N = N
        self.fn = fn
        self.fs = fs
        self.NbyCycle = fs / fn  # Number of samples per cycle.

        # Chebyshev polynomials are generated over the range [-1, 1]
        self.t_cheby = np.linspace(0, self.N - 1, self.N)
        X = 2 * self.t_cheby / self.N - 1
        X1 = 2 * self.t_cheby[1:] / self.N - 1

        # Generate Chebyshev basis up to degree 15
        self.basis = [np.ones_like(X),
                      X,
                      2 * X**2 - 1,
                      4 * X**3 - 3 * X,
                      8 * X**4 - 8 * X**2 + 1,
                      16 * X**5 - 20 * X**3 + 5 * X,
                      32 * X**6 - 48 * X**4 + 18 * X**2 - 1,
                      64 * X**7 - 112 * X**5 + 56 * X**3 - 7 * X,
                      128 * X**8 - 256 * X**6 + 160 * X**4 - 32 * X**2 + 1,
                      256 * X**9 - 576 * X**7 + 432 * X**5 - 120 * X**3 + 9 * X,
                      512 * X**10 - 1280 * X**8 + 1120 * X**6 - 400 * X**4 + 50 * X**2 - 1,
                      1024 * X**11 - 2816 * X**9 + 2816 * X**7 - 1232 * X**5 + 220 * X**3 - 11 * X,
                      2048 * X**12 - 6144 * X**10 + 6912 * X**8 - 3584 * X**6 + 840 * X**4 - 72 * X**2 + 1,
                      4096 * X**13 - 13312 * X**11 + 16640 * X**9 - 9984 * X**7 + 2912 * X**5 - 364 * X**3 + 13 * X,
                      8192 * X**14 - 28672 * X**12 + 39424 * X**10 - 26880 * X**8 + 9408 * X**6 - 1568 * X**4 + 98 * X**2 - 1,
                      16384 * X**15 - 61440 * X**13 + 92160 * X**11 - 70400 * X**9 + 28800 * X**7 - 6048 * X**5 + 560 * X**3 - 15 * X]

        self.basis1 = [np.ones_like(X1),
                       X1,
                       2 * X1**2 - 1,
                       4 * X1**3 - 3 * X1,
                       8 * X1**4 - 8 * X1**2 + 1,
                       16 * X1**5 - 20 * X1**3 + 5 * X1,
                       32 * X1**6 - 48 * X1**4 + 18 * X1**2 - 1,
                       64 * X1**7 - 112 * X1**5 + 56 * X1**3 - 7 * X1,
                       128 * X1**8 - 256 * X1**6 + 160 * X1**4 - 32 * X1**2 + 1,
                       256 * X1**9 - 576 * X1**7 + 432 * X1**5 - 120 * X1**3 + 9 * X1,
                       512 * X1**10 - 1280 * X1**8 + 1120 * X1**6 - 400 * X1**4 + 50 * X1**2 - 1,
                       1024 * X1**11 - 2816 * X1**9 + 2816 * X1**7 - 1232 * X1**5 + 220 * X1**3 - 11 * X1,
                       2048 * X1**12 - 6144 * X1**10 + 6912 * X1**8 - 3584 * X1**6 + 840 * X1**4 - 72 * X1**2 + 1,
                       4096 * X1**13 - 13312 * X1**11 + 16640 * X1**9 - 9984 * X1**7 + 2912 * X1**5 - 364 * X1**3 + 13 * X1,
                       8192 * X1**14 - 28672 * X1**12 + 39424 * X1**10 - 26880 * X1**8 + 9408 * X1**6 - 1568 * X1**4 + 98 * X1**2 - 1,
                       16384 * X1**15 - 61440 * X1**13 + 92160 * X1**11 - 70400 * X1**9 + 28800 * X1**7 - 6048 * X1**5 + 560 * X1**3 - 15 * X1]

        # Precompute the norms and cross-correlations of the Chebyshev basis
        self.c = [np.mean(b**2) for b in self.basis]
        self.c1 = [np.sum(self.basis[i][:-1] * self.basis1[i]) / (self.N) for i in range(len(self.basis))]

        # Initialize the superclass
        #super().__init__(verbose)

    def gamma0_poly(self, x, w):
        """
        Compute gamma0 based on polynomial weights and allocations.

        Parameters:
        x (list): Allocation levels
        w (list): Width of distribution

        Returns:
        float: Gamma0 value.
        """
        return np.sum([self.c[i] * w[i]**2 * (2**(-2 * x[i])) for i in range(len(w))]) / 12

    def gamma1_poly(self, x, w):
        """
        Compute gamma1 based on polynomial weights and allocations.

        Parameters:
        x (list): Allocation levels
        w (list): Width of distribution

        Returns:
        float: Gamma1 value.
        """
        return np.sum([self.c1[i] * w[i]**2 * (2**(-2 * x[i])) for i in range(len(w))]) / 12

    def get_nx_poly(self, nx, w):
        """
        Compute polynomial allocation to match a given total number of bits.

        Parameters:
        nx (int): Total number of bits to allocate.
        w (list): width values.

        Returns:
        tuple:
            - rounded allocation (list)
            - theoretical allocation (list)
        """
        Np = len(w)
        pw = np.array([self.c[k] * w[k]**2 for k in range(Np)])
        root_theo = [0] * Np

        # Compute theoretical allocation
        for k in range(Np):
            root_theo[k] = 0.5 * (-(1 / Np) * np.sum(np.log2(pw)) + 2 * nx / Np + np.log2(pw[k]))

        # Round allocation to integers that sum to nx
        return self.round_allocation(root_theo, nx, w), root_theo

    def MSE_theoritical_poly(self, x, w, gamma0_em, gamma1_em):
        """
        Calculate theoretical MSE based on polynomial allocations.

        Parameters:
        x (list): Allocation levels.
        w (list): Width values.
        gamma0_em (float): Empirical zero-lag variance of the error model.
        gamma1_em (float): Empirical one-lag variance of the error model.

        Returns:
        float: Theoretical MSE value.
        """
        gamma0_q = self.gamma0_poly(x[:-1], w)
        gamma1_q = self.gamma1_poly(x[:-1], w)
        return ((gamma0_em**2 - gamma1_em**2) / gamma0_em + (gamma0_q**2 - gamma1_q**2) / gamma0_q) * 2**(-2 * x[-1])

    
    def get_nx_nr_constraint_bit_poly(self, w, error_model, n_tot):
        """
        Allocate bits under a total constraint and minimize the MSE.

        Parameters:
        w (list): width values.
        error_model (np.ndarray): Empirical error model samples.
        n_tot (int): Total number of bits to allocate.

        Returns:
        tuple:
            - root (list): Optimal allocation levels.
            - mse_theoretical (float): Theoretical MSE corresponding to the allocation.
        """
        gamma0_em = np.mean(error_model**2)
        gamma1_em = np.sum(error_model[0:self.N-1] * error_model[1:self.N]) / (self.N - 1)

        Np = len(w)
        root = [0] * (Np + 1)
        root_test = [0] * (Np + 1)
        MSE_memory = np.inf
        nx = 0

        while nx <= n_tot:
            # Calculate the allocation for the current total bits
            root_test[0:Np] = self.get_nx_poly(nx, w)[1]
            root_test[Np] = (n_tot - nx) / self.N
            MSE_test = self.MSE_theoritical_poly(root_test, w, gamma0_em, gamma1_em)

            # If the new allocation yields a smaller MSE, update the optimal allocation
            if MSE_test < MSE_memory:
                MSE_memory = MSE_test
                root = list(root_test)
            else:
                break  # Stop when MSE stops improving
            nx += 1

        return root, self.MSE_theoritical_poly(root, w, gamma0_em, gamma1_em)

    def get_nx_nr_constraint_MSE_poly(self, w, error_model, MSE_target):
        """
        Find the allocation that meets a given MSE target.

        Parameters:
        w (list): Width values.
        error_model (np.ndarray): Empirical error model samples.
        MSE_target (float): Desired MSE threshold.

        Returns:
        tuple:
            - root (list): Optimal allocation levels.
            - mse_theoretical (float): Theoretical MSE corresponding to the allocation.
        """
        Np = len(w)
        gamma0_em = np.mean(error_model**2)
        gamma1_em = np.sum(error_model[0:self.N-1] * error_model[1:self.N]) / (self.N - 1)

        root = [0] * (Np + 1)
        root_test = [0] * (Np + 1)
        n_tot_memory = np.inf
        nx = 0

        while True:
            # Calculate the allocation for the current total bits
            root_test[0:Np] = self.get_nx_poly(nx, w)[1]
            gamma_0_q = self.gamma0_poly(root_test[0:Np], w)
            gamma_1_q = self.gamma1_poly(root_test[0:Np], w)

            # Compute the MSE for the current allocation
            MSE_test = ((gamma0_em**2 - gamma1_em**2) / gamma0_em +
                        (gamma_0_q**2 - gamma_1_q**2) / gamma_0_q)

            # Determine the additional bits needed to meet the target MSE
            root_test[Np] = -0.5 * np.log2(MSE_target / MSE_test)

            # Total bits used by this allocation
            n_tot_test = nx + root_test[Np] * self.N

            # If the new allocation uses fewer bits while meeting the target MSE, update it
            if n_tot_test < n_tot_memory:
                n_tot_memory = n_tot_test
                root = list(root_test)
            else:
                break  # Stop when bit usage stops improving
            nx += 1

        return root, self.MSE_theoritical_poly(root, w, gamma0_em, gamma1_em)

        

class Allocation_pred_samples(Allocation):
    """
    Allocation class for predicted samples, which adjusts allocations
    based on weights and predicted variances (gamma0 and gamma1).
    """

    def __init__(self, fn, fs, N, verbose=False):
        """
        Initialize the Allocation_pred_samples class.

        Parameters:
        fn (float): Fundamental frequency of the signal.
        fs (float): Sampling frequency of the signal.
        N (int): Number of samples.
        verbose (bool): If True, enable verbose output.
        """
        self.N = N
        self.fn = fn
        self.fs = fs
        self.verbose = verbose
        #super().__init__(verbose)

    def gamma0_pred_samples(self, x, w, gamma0_q):
        """
        Compute gamma0 for predicted samples using given weights and allocation.

        Parameters:
        x (list): Allocation levels.
        w (list): Width values.
        gamma0_q (float): Variance of the predicted samples (zero lag).

        Returns:
        float: Computed gamma0 value.
        """
        return gamma0_q * np.sum([w[i]**2 * (2**(-2 * x[i])) for i in range(len(w))]) / 12

    def gamma1_pred_samples(self, x, w, gamma0_q, gamma1_q):
        """
        Compute gamma1 for predicted samples using given weights and allocation.

        Parameters:
        x (list): Allocation levels.
        w (list): Width values.
        gamma0_q (float): Variance of the predicted samples (zero lag).
        gamma1_q (float): Variance of the predicted samples (one lag).

        Returns:
        float: Computed gamma1 value.
        """
        return gamma1_q * np.sum([w[i]**2 * (2**(-2 * x[i])) for i in range(len(w))]) / 12

    def get_nx_pred_samples(self, nx, w, eta):
        """
        Calculate allocation for predicted samples under a total bit constraint.

        Parameters:
        nx (int): Target number of bits to allocate.
        w (list): Width values.
        eta (float): Unused parameter in this implementation.

        Returns:
        tuple:
            - rounded allocation (list)
            - theoretical allocation (list)
        """
        Np = len(w)
        pw = np.array([w[k]**2 for k in range(Np)])
        root_theo = [0] * Np

        for k in range(Np):
            root_theo[k] = 0.5 * (-(1 / Np) * np.sum(np.log2(pw)) + 2 * nx / Np + np.log2(pw[k]))

        return self.round_allocation(root_theo, nx, w), root_theo

    def MSE_theoritical_pred_samples(self, x, w, gamma0_em, gamma1_em, gamma0_q, gamma1_q):
        """
        Calculate theoretical MSE for predicted samples.

        Parameters:
        x (list): Allocation levels.
        w (list): Width values.
        gamma0_em (float): Empirical zero-lag variance of the error model.
        gamma1_em (float): Empirical one-lag variance of the error model.
        gamma0_q (float): Variance of the predicted samples (zero lag).
        gamma1_q (float): Variance of the predicted samples (one lag).

        Returns:
        float: Computed theoretical MSE.
        """
        gamma0_q_val = self.gamma0_pred_samples(x[:len(w)], w, gamma0_q)
        gamma1_q_val = self.gamma1_pred_samples(x[:len(w)], w, gamma0_q, gamma1_q)

        return ((gamma0_em**2 - gamma1_em**2) / gamma0_em +
                (gamma0_q_val**2 - gamma1_q_val**2) / gamma0_q_val) * 2**(-2 * x[-1])

    def get_nx_nr_constraint_bit_pred_samples(self, w, eta, xp, error_model, n_tot):
        """
        Find allocation under total bit constraint for predicted samples.

        Parameters:
        w (list): Weight values.
        eta (float): Unused parameter in this implementation.
        xp (np.ndarray): Previous decoded samples.
        error_model (np.ndarray): Empirical error model samples.
        n_tot (int): Total bit allocation target.

        Returns:
        tuple:
            - root (list): Optimal allocation levels.
            - mse_theoretical (float): Theoretical MSE corresponding to the allocation.
        """
        gamma0_em = np.mean(error_model**2)
        gamma1_em = np.sum(error_model[0:self.N-1] * error_model[1:self.N]) / (self.N - 1)
        gamma0_q = np.mean(xp[self.N:2*self.N]**2)
        gamma1_q = np.sum(xp[self.N:2*self.N-1] * xp[self.N+1:2*self.N]) / self.N

        Np = len(w)
        root = [0] * (Np + 1)
        root_test = [0] * (Np + 1)
        MSE_memory = np.inf
        nx = 0

        while nx <= n_tot:
            root_test[:Np] = self.get_nx_pred_samples(nx, w, eta)[1]
            root_test[Np] = (n_tot - nx) / self.N
            MSE_test = self.MSE_theoritical_pred_samples(root_test, w, gamma0_em, gamma1_em, gamma0_q, gamma1_q)
            
            if MSE_test < MSE_memory:
                MSE_memory = MSE_test
                root = list(root_test)
            else:
                break
            nx += 1

        return root, self.MSE_theoritical_pred_samples(root, w, gamma0_em, gamma1_em, gamma0_q, gamma1_q)

    def get_nx_nr_constraint_MSE_pred_samples(self, w, eta, xp, error_model, MSE_target):
        """
        Find allocation to meet a given MSE target for predicted samples.

        Parameters:
        w (list): Width values.
        eta (int): shift.
        xp (np.ndarray): Previous  decoded samples.
        error_model (np.ndarray): Empirical error model samples.
        MSE_target (float): Desired MSE threshold.

        Returns:
        tuple:
            - root (list): Optimal allocation levels.
            - mse_theoretical (float): Theoretical MSE corresponding to the allocation.
        """
        gamma0_em = np.mean(error_model**2)
        gamma1_em = np.sum(error_model[0:self.N-1] * error_model[1:self.N]) / (self.N - 1)
        gamma0_q = np.mean(xp[self.N:2*self.N]**2)
        gamma1_q = np.sum(xp[self.N:2*self.N-1] * xp[self.N+1:2*self.N]) / self.N

        Np = len(w)
        root = [0] * (Np + 1)
        root_test = [0] * (Np + 1)
        n_tot_memory = np.inf
        nx = 0

        while True:
            root_test[:Np] = self.get_nx_pred_samples(nx, w, eta)[1]
            gamma_0_q = self.gamma0_pred_samples(root_test[:Np], w, gamma0_q)
            gamma_1_q = self.gamma1_pred_samples(root_test[:Np], w, gamma0_q, gamma1_q)
            
            MSE_test = ((gamma0_em**2 - gamma1_em**2) / gamma0_em +
                        (gamma_0_q**2 - gamma_1_q**2) / gamma_0_q)
            
            root_test[Np] = -0.5 * np.log2(MSE_target / MSE_test)
            n_tot_test = nx + root_test[Np] * self.N
            
            if n_tot_test < n_tot_memory:
                n_tot_memory = n_tot_test
                root = list(root_test)
            else:
                break
            nx += 1

        return root, self.MSE_theoritical_pred_samples(root, w, gamma0_em, gamma1_em, gamma0_q, gamma1_q)


        
        

# Main program entry point
if __name__ == "__main__":
    # Import required modules
    from Models import Model_sin, Model_poly, Model_pred_samples
    import matplotlib.pyplot as plt
    from Measures import get_snr, get_mse
    from Quantization import Quantizer
    from Normalize import normalize
    from codage_residu import Residual_Encoder
    from get_test_signal import get_RTE_signal

    ##################################################
    # Load and preprocess the signal data
    ##################################################
    N = 128  # Number of samples
    fn = 50  # Fundamental frequency
    fs = 6400  # Sampling frequency
    n_tot = 128  # Total number of bits available for allocation
    nx_max = 64  # Maximum bits for quantization
    n_sym = 10  # Number of bits for encoding significant coefficients in Bit Plane Coding
    
    id_signal = 1  # Identifier for the signal
    id_phase = 0  # Identifier for the phase of the signal
    win = 22  # Window index to extract the current segment of the signal
    
    # Load the current segment of the signal
    signal = get_RTE_signal(id_signal)[id_phase][win * N:(win + 1) * N]
    
    # Load the previous signal segment for prediction purposes (if needed)
    signal_previous = np.zeros(2 * N)
    if win == 1:
        # Load one previous window
        signal_previous[N:2 * N] = get_RTE_signal(id_signal)[id_phase][(win - 1) * N:win * N]
    elif win >= 2:
        # Load two previous windows
        signal_previous = get_RTE_signal(id_signal)[id_phase][(win - 2) * N:win * N]
    
    # Generate a time vector for plotting
    t = np.linspace(0, (N - 1) / fs, N)
    
    # Plot the loaded signal segment
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(t, signal, lw=2, label=r'$\mathbf{{x}} {}$ p1'.format(win))
    plt.xlabel('t (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.title("Considered signals - id signal: {}, id phase: {}, window: {}".format(id_signal, id_phase, win))
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()

    ##################################################
    # Choose the model type (sin, poly, pred_samples)
    ##################################################
    sin = False  # Whether to use the sinusoidal model
    poly = True  # Whether to use the polynomial model
    deg = 5  # Degree of the polynomial
    
    pred_samples = False # Whether to use predicted samples
    order = 1  # Order for prediction
    eta = 0  # Parameter used for prediction
    
    # Determine the number of parameters (Np) based on the chosen model
    if sin:
        Np = 3
    elif poly:
        Np = deg + 1
    elif pred_samples:
        Np = order

    ##################################################
    # Initialize models and allocation objects
    ##################################################
    # Initialize sinusoidal model and allocation
    model_sin = Model_sin(fn, fs, N, verbose=False)
    allocation_sin = Allocation_sin(fn, fs, N, nx_max)
    
    # Initialize polynomial model and allocation
    model_poly = Model_poly(fn, fs, N, verbose=False)
    allocation_poly = Allocation_poly(fn, fs, N, nx_max)
    
    # Initialize predicted samples model and allocation
    model_pred_samples = Model_pred_samples(fn, fs, N, verbose=False)
    allocation_pred_samples = Allocation_pred_samples(fn, fs, N, nx_max)
    
    # Initialize quantizer and residual encoder
    quantizer = Quantizer(verbose=False)
    l = Residual_Encoder(N, factor_scale=4)

    # Initialization of MSE arrays
    # MSE between the signal and the model
    MSE_model = 0
    
    # MSE between the original model and the quantized model
    MSE_model_q = np.zeros(nx_max)
    MSE_reel_q = np.zeros(nx_max)
    
    # MSE between the residual and the reconstructed residual
    MSE_model_r = np.zeros(nx_max)
    MSE_reel_r = np.zeros(nx_max)
    
    # MSE between the signal and the model reconstructed with quantized parameters
    MSE_model_qm = np.zeros(nx_max)
    MSE_reel_qm = np.zeros(nx_max)
    
    # Total reconstruction MSE
    MSE_model_tot = np.zeros(nx_max)
    MSE_reel_tot = np.zeros(nx_max)
    
    # Optimal bit allocation
    al_float = np.zeros((nx_max, Np))
    al_int = np.zeros((nx_max, Np))
    
    # Autocorrelation metrics
    gamma0_model = np.zeros(nx_max)
    gamma0_reel = np.zeros(nx_max)
    gamma1_model = np.zeros(nx_max)
    gamma1_reel = np.zeros(nx_max)
    
    # Normalize the signal and previous signal
    signal_test, k = normalize(signal)
    signal_previous *= 2 ** (-k)
    
    if sin:
        # Prior distribution for sinusoidal model parameters
        m_theta_test = [0.75, fn, 0]
        w_theta_test = [0.5, 0.4, 2 * np.pi]
        
        # Estimate parameters using the sinusoidal model
        theta_test_hat = model_sin.get_theta_sin(signal_test, m_theta_test, w_theta_test)
        signal_test_hat = model_sin.get_model_sin(t, *theta_test_hat)
        
        # Allocate bits and calculate MSE
        for nx in range(nx_max):
            # Compute optimal bit allocation and corresponding MSE
            al_int[nx], al_float[nx] = allocation_sin.get_nx_sin(nx, m_theta_test, w_theta_test)
            MSE_model_q[nx] = allocation_sin.gamma0_sin(al_float[nx], m_theta_test, w_theta_test)
            
            # Store autocorrelation values
            gamma0_model[nx] = MSE_model_q[nx]
            gamma1_model[nx] = allocation_sin.gamma1_sin(al_float[nx], m_theta_test, w_theta_test)
    
    elif poly:
        # Prior distribution for polynomial model parameters
        m_theta_test = np.zeros(deg + 1)
        sigma_poly = [
            [0.215],
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
        w_theta_test = [min(1.55, sigma_poly[deg][j] * 7) for j in range(deg + 1)]
        
        # Estimate parameters using the polynomial model
        theta_test_hat = model_poly.get_theta_poly(signal_test, m_theta_test, w_theta_test, deg)
        signal_test_hat = model_poly.get_model_poly(t, *theta_test_hat)
        
        # Allocate bits and calculate MSE
        for nx in range(nx_max):
            # Compute optimal bit allocation and corresponding MSE
            al_int[nx], al_float[nx] = allocation_poly.get_nx_poly(nx, w_theta_test)
            MSE_model_q[nx] = allocation_poly.gamma0_poly(al_float[nx], w_theta_test)
            
            # Store autocorrelation values
            gamma0_model[nx] = MSE_model_q[nx]
            gamma1_model[nx] = allocation_poly.gamma1_poly(al_float[nx], w_theta_test)

    elif pred_samples:
        # Obtain prior mean and variance for the predicted samples
        m_theta_test = model_pred_samples.get_m_theta_pred_samples(order, eta, 0, order * [0], [10] * order)
        w_theta_test = [2] * order
    
        # Extract the features (X) from the previous signal segment
        X = model_pred_samples.get_X(signal_previous, order, eta)
    
        # Estimate the model parameters for the predicted samples
        theta_test_hat = model_pred_samples.get_theta_pred_samples(X, signal_test, m_theta_test, w_theta_test)
        signal_test_hat = model_pred_samples.get_model_pred_samples(X, *theta_test_hat)
    
        # Allocate bits and calculate MSE for each bit budget
        for nx in range(nx_max):
            al_int[nx], al_float[nx] = allocation_pred_samples.get_nx_pred_samples(nx, w_theta_test, eta)
    
            # Calculate autocorrelation-based metrics
            gamma0_q = np.mean(signal_test**2)
            gamma1_q = np.sum(signal_test[0:N-1] * signal_test[1:N]) / N
            MSE_model_q[nx] = allocation_pred_samples.gamma0_pred_samples(al_float[nx], w_theta_test, gamma0_q)
    
            gamma0_model[nx] = MSE_model_q[nx]
            gamma1_model[nx] = allocation_pred_samples.gamma1_pred_samples(al_float[nx], w_theta_test, gamma0_q, gamma1_q)
    
    # Compute the model error (difference between original signal and model estimate)
    error_model = signal_test - signal_test_hat
    
    # Calculate empirical autocorrelation metrics
    gamma0_em = np.mean(error_model**2)
    gamma1_em = np.sum(error_model[0:N-1] * error_model[1:N]) / (N - 1)
    
    # Calculate the overall MSE between the original signal and the estimated signal
    MSE_model = get_mse(signal_test, signal_test_hat)
    
    # Initialize quantized parameter array
    theta_test_tilde = [0] * Np
    
    for nx in range(nx_max):
        # Quantize the estimated parameters
        for i in range(Np):
            theta_test_tilde[i] = quantizer.get_q_u(
                quantizer.get_ind_u(theta_test_hat[i], al_int[nx, i], w_theta_test[i], m_theta_test[i]),
                al_int[nx, i],
                w_theta_test[i],
                m_theta_test[i]
            )
    
        if sin:
            # Reconstruct the signal from the quantized parameters using the sinusoidal model
            signal_test_tilde = model_sin.get_model_sin(t, *theta_test_tilde)
            # Compute the theoretical MSE for the quantized sinusoidal model
            MSE_model_tot[nx] = allocation_sin.MSE_theoritical_sin(
                [*al_float[nx]] + [(n_tot - nx - n_sym) / N],
                m_theta_test, w_theta_test, gamma0_em, gamma1_em
            )
            
        elif poly:
            # Reconstruct the signal from the quantized parameters using the polynomial model
            signal_test_tilde = model_poly.get_model_poly(t, *theta_test_tilde)
            # Compute the theoretical MSE for the quantized polynomial model
            MSE_model_tot[nx] = allocation_poly.MSE_theoritical_poly(
                [*al_float[nx]] + [(n_tot - nx - n_sym) / N],
                w_theta_test, gamma0_em, gamma1_em
            )
            
        elif pred_samples:
            # Reconstruct the signal from the quantized parameters using the predicted samples model
            signal_test_tilde = model_pred_samples.get_model_pred_samples(X, *theta_test_tilde)
            # Compute the theoretical MSE for the quantized predicted samples model
            MSE_model_tot[nx] = allocation_pred_samples.MSE_theoritical_pred_samples(
                [*al_float[nx]] + [(n_tot - nx - n_sym) / N],
                w_theta_test, gamma0_em, gamma1_em, gamma0_q, gamma1_q
            )
    
    
                
        
        # Compute the MSE between the estimated model and the quantized reconstructed model
        MSE_reel_q[nx] = get_mse(signal_test_hat, signal_test_tilde)
        
        # Compute the MSE between the original signal and the quantized reconstructed model
        MSE_reel_qm[nx] = get_mse(signal_test, signal_test_tilde)
        
        # Compute the residual (difference between original signal and quantized reconstructed signal)
        residual_test = signal_test - signal_test_tilde
        
        # Define the remaining bits available for residual encoding
        nr = n_tot - nx
        
        # Encode and decode the residual using two different methods, choosing the one with the lower MSE
        residual_test_tilde, code, _, _ = l.get_r_DCT_BPC_tilde(residual_test, "RMSE", -np.inf, nr, n_sym)
        residual_test_tilde_2, code_2, _, _ = l.get_r_DWT_BPC_tilde(residual_test, "RMSE", -np.inf, nr, n_sym)
        
        # Use the better method based on MSE
        if get_mse(residual_test, residual_test_tilde_2) < get_mse(residual_test, residual_test_tilde):
            residual_test_tilde = residual_test_tilde_2
        
        # Compute the MSE between the original residual and the reconstructed residual
        MSE_reel_r[nx] = get_mse(residual_test, residual_test_tilde)
        
        # Combine the quantized reconstructed model and the residual to get the final reconstructed signal
        signal_test_rec = signal_test_tilde + residual_test_tilde
        
        # Compute the MSE between the original signal and the final reconstructed signal
        MSE_reel_tot[nx] = get_mse(signal_test, signal_test_rec)
        
        # Compute the quantization error and its autocorrelation
        error_q = signal_test_hat - signal_test_tilde
        gamma0_eq = np.mean(error_q**2)
        gamma1_eq = np.sum(error_q[0:N-1] * error_q[1:N]) / (N-1)
        
        # Store the autocorrelation metrics
        gamma0_reel[nx] = gamma0_eq
        gamma1_reel[nx] = gamma1_eq
        
        # Visualization of the impact of quantization on the model and the reconstructed signal
        if nx in [10]:
            # Plot the original, estimated, and quantized models
            plt.figure(figsize=(8, 4), dpi=100)
            plt.plot(t, signal_test, lw=1, label="real")
            plt.plot(t, signal_test_hat, lw=1, label="model est SNR={:.2f} dB".format(get_snr(signal_test, signal_test_hat)))
            plt.plot(t, signal_test_tilde, lw=1, label="quantized model nx={}, SNR={:.2f} dB".format(nx, get_snr(signal_test, signal_test_tilde)))
            plt.ylabel('Amplitude')
            plt.xlabel('t (s)')
            plt.legend()
            plt.title("Modeling Error")
            plt.grid(which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show()
        
            # Plot the modeling error (difference between real and estimated signal)
            plt.figure(figsize=(8, 4), dpi=100)
            plt.plot(t, signal_test - signal_test_hat, lw=2, label='em')
            plt.ylabel('Amplitude')
            plt.xlabel('t (s)')
            plt.legend()
            plt.title("Modeling Error")
            plt.grid(which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show()
        
            # Plot the quantization error (difference between estimated and quantized models)
            plt.figure(figsize=(8, 4), dpi=100)
            plt.plot(t, signal_test_hat - signal_test_tilde, lw=2, label="eq")
            plt.xlabel('t (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.title("Quantization Error nx={} bits".format(nx))
            plt.grid(which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show()
        
            # Plot the residuals before and after reconstruction
            plt.figure(figsize=(8, 4), dpi=100)
            plt.plot(t, residual_test, lw=2, label="r")
            plt.plot(t, residual_test_tilde, lw=2, label="reconstructed r SNR={:.2f} dB".format(get_snr(residual_test, residual_test_tilde)))
            plt.xlabel('t (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.title("Residual and Reconstructed Residual nx={}, nr={} bits".format(nx, nr))
            plt.grid(which='major', color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.show()
        
    
     
    # Verification curves to see if the model performs well
    SNR_em_eq_model = 10 * np.log10(np.mean(signal_test**2) / (MSE_model_q + MSE_model))
    SNR_em_eq_real = 10 * np.log10(np.mean(signal_test**2) / MSE_reel_qm)
    
    SNR_em_eq_res_real = 10 * np.log10(np.mean(signal_test**2) / MSE_reel_r)
    SNR_em_eq_res_model = 10 * np.log10(np.mean(signal_test**2) / MSE_model_tot)
    
    SNR_est = 10 * np.log10(np.mean(signal_test**2) / (np.ones(nx_max) * MSE_model))
    
    # Plotting SNR curves to show the contributions of the first and second stage
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot([i for i in range(nx_max)], SNR_em_eq_model, lw=2, label="first stage model")
    plt.plot([i for i in range(nx_max)], SNR_em_eq_real, '*', lw=2, label="first stage real")
    plt.plot([i for i in range(nx_max)], SNR_em_eq_res_model - SNR_em_eq_model, lw=2, label='second stage model')
    plt.plot([i for i in range(nx_max)], SNR_em_eq_res_real - SNR_em_eq_real, '*', lw=2, label='second stage real')
    plt.xlabel('nx (bits)')
    plt.ylabel('SNR (dB)')
    plt.legend()
    plt.title("Contribution of the first and second stage")
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    # Compare the model's rate-distortion performance to the real rate-distortion curve
    if sin:
        root, MSE_model_tot_min = allocation_sin.get_nx_nr_constraint_bit_sin(
            m_theta_test, w_theta_test, signal_test - signal_test_hat, n_tot - n_sym
        )
        MSE_target = MSE_model_tot_min
        nx_nr_find, MSE_model_tot_min_find = allocation_sin.get_nx_nr_constraint_MSE_sin(
            m_theta_test, w_theta_test, signal_test - signal_test_hat, MSE_target
        )
    elif poly:
        root, MSE_model_tot_min = allocation_poly.get_nx_nr_constraint_bit_poly(
            w_theta_test, signal_test - signal_test_hat, n_tot - n_sym
        )
        MSE_target = MSE_model_tot_min
        nx_nr_find, MSE_model_tot_min_find = allocation_poly.get_nx_nr_constraint_MSE_poly(
            w_theta_test, signal_test - signal_test_hat, MSE_target
        )
    elif pred_samples:
        root, MSE_model_tot_min = allocation_pred_samples.get_nx_nr_constraint_bit_pred_samples(
            w_theta_test, eta, signal_previous, signal_test - signal_test_hat, n_tot - n_sym
        )
        MSE_target = MSE_model_tot_min
        nx_nr_find, MSE_model_tot_min_find = allocation_pred_samples.get_nx_nr_constraint_MSE_pred_samples(
            w_theta_test, eta, signal_previous, signal_test - signal_test_hat, MSE_target
        )
    
    # Print information about the achieved SNR and bit allocation
    print("SNR target={} dB".format(10 * np.log10(np.mean(signal_test**2) / MSE_target)))
    print("SNR find={} dB".format(10 * np.log10(np.mean(signal_test**2) / MSE_model_tot_min_find)))
    print(
        "nx find ={} b, nr find={} b".format(
            np.sum(nx_nr_find[:len(w_theta_test)]), N * nx_nr_find[-1]
        ),
        "nx target={} b, nr target={} b".format(np.sum(root[:Np]), N * root[-1])
    )
    
    # Plot SNR and MSE curves
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(
        [i for i in range(nx_max)],
        10 * np.log10(np.mean(signal_test**2) / MSE_model_tot),
        lw=2,
        label='Model'
    )
    plt.plot(
        [i for i in range(nx_max)],
        10 * np.log10(np.mean(signal_test**2) / MSE_reel_tot),
        '-*',
        lw=2,
        label='real'
    )
    plt.plot(
        np.sum(root[:Np]),
        10 * np.log10(np.mean(signal_test**2) / MSE_model_tot_min),
        'o',
        lw=4,
        label="Theoretical point find"
    )
    plt.xlabel('nx (bits)')
    plt.ylabel('SNR (dB)')
    plt.legend()
    plt.title("SNR two stage")
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(
        [i for i in range(nx_max)],
        MSE_model_tot * 2**(2 * k),
        lw=2,
        label='model'
    )
    plt.plot(
        [i for i in range(nx_max)],
        MSE_reel_tot * 2**(2 * k),
        '-*',
        lw=2,
        label='real'
    )
    plt.plot(
        np.sum(root[:Np]),
        MSE_model_tot_min * 2**(2 * k),
        'o',
        lw=4,
        label="Theoretical point"
    )
    plt.xlabel('nx (bits)')
    plt.ylabel('MSE (V)')
    plt.legend()
    plt.yscale("log")
    plt.title("MSE with sin depending on nx, nx_max={} b".format(nx_max))
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    # Plot gamma model and real gamma
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(
        [i for i in range(nx_max)],
        10 * np.log10(gamma0_model),
        lw=2,
        label='gamma0 model'
    )
    plt.plot(
        [i for i in range(nx_max)],
        10 * np.log10(gamma1_model),
        lw=2,
        label='gamma1 model'
    )
    plt.plot(
        [i for i in range(nx_max)],
        10 * np.log10(gamma0_reel),
        '-*',
        lw=2,
        label='gamma0 real'
    )
    plt.plot(
        [i for i in range(nx_max)],
        10 * np.log10(gamma1_reel),
        '-*',
        lw=2,
        label='gamma1 real'
    )
    plt.xlabel('nx (bits)')
    plt.ylabel('gamma (dB)')
    plt.legend()
    plt.title("Gamma0 model and real")
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(
        [i for i in range(nx_max)],
        10 * np.log10(gamma0_model) - 10 * np.log10(gamma0_reel),
        "-*",
        lw=2,
        label=r'gamma0 model / gamma0 real, mean={:.2f}'.format(
            np.mean(10 * np.log10(gamma0_model) - 10 * np.log10(gamma0_reel))
        )
    )
    plt.plot(
        [i for i in range(nx_max)],
        10 * np.log10(gamma1_model) - 10 * np.log10(gamma1_reel),
        "-*",
        lw=2,
        label=r'gamma1 model / gamma1 real, mean={:.2f}'.format(
            np.mean(10 * np.log10(gamma1_model) - 10 * np.log10(gamma1_reel))
        )
    )
    plt.xlabel('nx (bits)')
    plt.ylabel('gamma model / real (dB)')
    plt.legend()
    plt.title("gamma model - gamma real")
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
