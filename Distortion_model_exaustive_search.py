# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:41:41 2024

@author: presvotscor
"""
import numpy as np
import copy

from Normalize import normalize
from codage_model import Model_Encoder, Model_Decoder
from codage_residu import Residual_Encoder, Residual_Decoder
from Measures import get_quality, my_bin, my_inv_bin

class Encode_one_window(Model_Encoder, Residual_Encoder):
    def __init__(self, fn=50, fs=6400, N=128, Model_used={}, Residual_used={}):
        """
        Constructor for the Encode_one_window class, which combines model-based encoding 
        and residual encoding techniques.

        Parameters:
        - fn: Fundamental frequency (default: 50 Hz).
        - fs: Sampling frequency (default: 6400 Hz).
        - N: Number of samples in the window (default: 128).
        - Model_used: Dictionary of model configurations.
        - Residual_used: Dictionary of residual encoding configurations.
        """
        self.Model_used = Model_used  # Set of models used for encoding.
        self.Residual_used = Residual_used  # Set of residual encoding techniques used.

        # Initialize parent classes for model and residual encoding.
        Model_Encoder.__init__(self, fn, fs, N, False) 
        Residual_Encoder.__init__(self, N)   

        ##################### Initialization of header parameters
        
        self.nm = np.max([1, int(np.ceil(np.log2(len(Model_used))))])  # Number of bits to encode the model ID.
        self.nl = np.max([1, int(np.ceil(np.log2(len(Residual_used))))])  # Number of bits to encode the residual ID.
        self.n_kx = 5  # Number of bits to encode the normalization factor `kx`.
        self.n_kr = 5  # Number of bits to encode the normalization factor `kr`, or 0 if no residual encoding is used.

        self.nb_min_bit_theta = 2  # Minimum number of bits per parameter.
        self.nb_max_bit_theta = 10  # Maximum number of bits per parameter.
        self.n_sym = 10  # Maximum number of bits to reconstruct the residual.
        
        
        self.delta_nx=4 # number of tested value for nx
        self.delta_M=3  # number of tested model
        
        self.max_size_Nx = 0 # Maximum size of the search interval for nx; determines when the Golden Section search stops.

        
    def ini_MMC_enc(self):
        """
        Initialize the best model and residual encoding configurations.
        This creates a deep copy of the provided `Model_used` and `Residual_used`.
        """
        self.best_Model_used = copy.deepcopy(self.Model_used)
        self.best_Residual_used = copy.deepcopy(self.Residual_used)

    def get_header(self, x, x_p):
        """
        Calculate the header information, including normalization factors (`kx` and `kx_p`)
        and initialize the parameters for each model and residual method.

        Parameters:
        - x: Current signal window.
        - x_p: Previous signal window (used for predictive models).

        Returns:
        - kx: Normalization factor for the current window.
        - kx_p: Normalization factor for the previous window.
        """
        # Normalize the current window and calculate its normalization factor `kx`.
        _, kx = normalize(x)
        if kx >= 2**self.n_kx:
            kx = 2**self.n_kx - 1
        if kx < 0:
            kx = 0

        # Normalize the previous window and calculate its normalization factor `kx_p`.
        _, kx_p = normalize(x_p[self.N:])
        if kx_p >= 2**self.n_kx:
            kx_p = 2**self.n_kx - 1
        if kx_p < 0:
            kx_p = 0

        # Loop through all models and set their header parameters.
        for id_model in self.Model_used:
            model = self.best_Model_used[id_model]
            if model["family"] == "sin":  # Sine model
                model["n nx"] = int(np.ceil(np.log2(3 * self.nb_max_bit_theta)))
                model["n kx"] = self.n_kx
                model["kx"] = kx
                model["xn"] = x * 2**(-kx)

            elif model["family"] == "poly":  # Polynomial model
                model["n nx"] = int(np.ceil(np.log2((model["order"] + 1) * self.nb_max_bit_theta)))
                model["n kx"] = self.n_kx
                model["kx"] = kx
                model["xn"] = x * 2**(-kx)

            elif model["family"] == "pred samples":  # Predictive model based on samples
                model["n nx"] = int(np.ceil(np.log2(model["order"] * self.nb_max_bit_theta)))
                model["n kx"] = self.n_kx
                model["kx"] = kx
                model["xn"] = x * 2**(-kx)
                model["xn previous"] = x_p * 2**(-kx)

            elif model["family"] == "pred para":  # Predictive model based on parameters
                id_previous_model = model["model used"]
                model["n kx"] = self.n_kx
                model["kx"] = kx
                model["xn"] = x * 2**(-kx)
                model["xn previous"] = x_p * 2**(-kx)
                if self.Model_used[id_previous_model]["family"] == "sin":
                    model["n nx"] = int(np.ceil(np.log2(3 * self.nb_max_bit_theta)))
                elif self.Model_used[id_previous_model]["family"] == "poly":
                    model["n nx"] = int(np.ceil(np.log2((self.best_Model_used[id_previous_model]["order"] + 1) * self.nb_max_bit_theta)))
                elif self.Model_used[id_previous_model]["family"] == "pred samples":
                    model["n nx"] = int(np.ceil(np.log2(self.best_Model_used[id_previous_model]["order"] * self.nb_max_bit_theta)))
                elif self.Model_used[id_previous_model]["family"] == "none":
                    model["n nx"] = 0

            elif model["family"] == "none":  # No model
                model["n nx"] = 0
                model["n kx"] = self.n_kx
                model["kx"] = kx
                model["xn"] = x * 2**(-kx)

        # Loop through all residual methods and set their header parameters.
        for id_residual in self.Residual_used:
            residual = self.best_Residual_used[id_residual]
            if residual["name"] == "DCT+BPC":
                residual["n kr"] = self.n_kr
                residual["n nr"] = self.n_sym
            elif residual["name"] == "DWT+BPC":
                residual["n kr"] = self.n_kr
                residual["n nr"] = self.n_sym
            elif residual["name"] == "none":
                residual["n kr"] = 0
                residual["n nr"] = 0
            else:
                print(f"Error: The method {id_residual} does not exist.")

        return kx, kx_p             
                   
    
    def get_m_nx_nr(self,n_max):
        """
        Determine the best models and allocate bit budgets (`nx`) for the models under a given bit constraint (`n_max`).
        
        Parameters:
        - n_max: The maximum number of bits available for encoding.
        
        Returns:
        - id_models: Sorted list of the best model indices based on total SNR performance.
        """
        
        # Compute the mean squared error (MSE) of the normalized signal.
        
        MSE_S=np.mean(self.best_Model_used[0]["xn"]**2)
        
        # Initialize lists to store total SNR values and model indices.
        SNR_tot_model=[]
        id_models=[]
        c=6  #number of bits per coefs (estimation)
        
        # Iterate through all available models.
        for id_model in self.Model_used:

            # Handle polynomial models.
            if self.best_Model_used[id_model]["family"]=="poly" :
                # Compute the model's estimated signal and residual error.
                x_model_hat=self.get_model_poly(self.t,*self.best_Model_used[id_model]["theta hat"])
                error_model=self.best_Model_used[id_model]["xn"]-x_model_hat
                
                # Calculate the header size required for the model and residual.
                n_h=self.nm+self.best_Model_used[id_model]["n kx"]+self.best_Model_used[id_model]["n nx"]+self.nl
                
                # Check if enough bits are available for the residual and model parameters.
                if n_h+self.n_kr+self.n_sym+c*len(self.best_Model_used[id_model]["m theta"])<n_max:
                    n_h+=self.n_kr+self.n_sym
                    al_pred,MSE_pred=self.get_nx_nr_constraint_bit_poly(self.best_Model_used[id_model]["w theta"],error_model,n_max-n_h)
                    self.best_Model_used[id_model]["nx"]=int(np.ceil(np.sum(al_pred[0:len(self.best_Model_used[id_model]["m theta"])])))
                        
                else :
                    al_pred,MSE_pred=self.get_nx_nr_constraint_bit_poly(self.best_Model_used[id_model]["w theta"],error_model,n_max-n_h)
                    self.best_Model_used[id_model]["nx"]=np.min([n_max-n_h-self.delta_nx,2**self.best_Model_used[id_model]["n nx"]-1-self.delta_nx])
                   
            # Handle sinusoidal models.        
            elif self.best_Model_used[id_model]["family"]=="sin":    
                x_model_hat=self.get_model_sin(self.t,*self.best_Model_used[id_model]["theta hat"])
                error_model=self.best_Model_used[id_model]["xn"]-x_model_hat
                n_h=self.nm+self.best_Model_used[id_model]["n kx"]+self.best_Model_used[id_model]["n nx"]+self.nl
                
                if n_h+self.n_kr+self.n_sym+c*len(self.best_Model_used[id_model]["m theta"])<n_max:
                    n_h+=self.n_kr+self.n_sym
                    al_pred,MSE_pred=self.get_nx_nr_constraint_bit_sin(self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"],error_model,n_max-n_h)
                    self.best_Model_used[id_model]["nx"]=int(np.ceil(np.sum(al_pred[0:len(self.best_Model_used[id_model]["m theta"])])))
                else :
                    al_pred,MSE_pred=self.get_nx_nr_constraint_bit_sin(self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"],error_model,n_max-n_h)
                    self.best_Model_used[id_model]["nx"]=np.min([n_max-n_h-self.delta_nx,2**self.best_Model_used[id_model]["n nx"]-1-self.delta_nx])
            
                                
            # Handle prediction-based models using previous samples.
            elif self.best_Model_used[id_model]["family"]=="pred samples":    
                x_model_hat=self.get_model_pred_samples(self.best_Model_used[id_model]["X"],*self.best_Model_used[id_model]["theta hat"])
                error_model=self.best_Model_used[id_model]["xn"]-x_model_hat
                n_h=self.nm+self.best_Model_used[id_model]["n kx"]+self.best_Model_used[id_model]["n nx"]+self.nl
                
                if n_h+self.n_kr+self.n_sym+c*len(self.best_Model_used[id_model]["m theta"])<n_max:
                    n_h+=self.n_kr+self.n_sym
                    al_pred,MSE_pred=self.get_nx_nr_constraint_bit_pred_samples(self.best_Model_used[id_model]["w theta"],self.best_Model_used[id_model]["eta"],self.best_Model_used[id_model]["xn previous"],error_model,n_max-n_h)
                    self.best_Model_used[id_model]["nx"]=int(np.ceil(np.sum(al_pred[0:len(self.best_Model_used[id_model]["m theta"])])))
                else :
                    al_pred,MSE_pred=self.get_nx_nr_constraint_bit_pred_samples(self.best_Model_used[id_model]["w theta"],self.best_Model_used[id_model]["eta"],self.best_Model_used[id_model]["xn previous"],error_model,n_max-n_h)
                    self.best_Model_used[id_model]["nx"]=np.min([n_max-n_h-self.delta_nx,2**self.best_Model_used[id_model]["n nx"]-1-self.delta_nx])
     
            # Handle prediction-based models using parameters from previous models.        
            elif self.best_Model_used[id_model]["family"]=="pred para":
                id_previous_model=self.best_Model_used[id_model]["model used"]
                
                if self.best_Model_used[id_previous_model]["family"]=="sin":    
                    x_model_hat=self.get_model_sin(self.t,*self.best_Model_used[id_model]["theta hat"])
                    error_model=self.best_Model_used[id_model]["xn"]-x_model_hat
                    n_h=self.nm+self.best_Model_used[id_model]["n kx"]+self.best_Model_used[id_model]["n nx"]+self.nl
                    
                    if n_h+self.n_kr+self.n_sym+c*len(self.best_Model_used[id_model]["m theta"])<n_max:
                        n_h+=self.n_kr+self.n_sym
                        al_pred,MSE_pred=self.get_nx_nr_constraint_bit_sin(self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"],error_model,n_max-n_h)
                        self.best_Model_used[id_model]["nx"]=int(np.ceil(np.sum(al_pred[0:len(self.best_Model_used[id_model]["m theta"])])))
                    else :
                        al_pred,MSE_pred=self.get_nx_nr_constraint_bit_sin(self.best_Model_used[id_model]["m theta"],self.best_Model_used[id_model]["w theta"],error_model,n_max-n_h)
                        self.best_Model_used[id_model]["nx"]=np.min([n_max-n_h-self.delta_nx,2**self.best_Model_used[id_model]["n nx"]-1-self.delta_nx])
         
                        
                   
                elif self.best_Model_used[id_previous_model]["family"]=="poly":
                    x_model_hat=self.get_model_poly(self.t,*self.best_Model_used[id_model]["theta hat"])
                    error_model=self.best_Model_used[id_model]["xn"]-x_model_hat
                    n_h=self.nm+self.best_Model_used[id_model]["n kx"]+self.best_Model_used[id_model]["n nx"]+self.nl
                    
                    if n_h+self.n_kr+self.n_sym+c*len(self.best_Model_used[id_model]["m theta"])<n_max:
                        n_h+=self.n_kr+self.n_sym
                        al_pred,MSE_pred=self.get_nx_nr_constraint_bit_poly(self.best_Model_used[id_model]["w theta"],error_model,n_max-n_h)
                        self.best_Model_used[id_model]["nx"]=int(np.ceil(np.sum(al_pred[0:len(self.best_Model_used[id_model]["m theta"])])))
                    else :
                        al_pred,MSE_pred=self.get_nx_nr_constraint_bit_poly(self.best_Model_used[id_model]["w theta"],error_model,n_max-n_h)
                        self.best_Model_used[id_model]["nx"]=np.min([n_max-n_h-self.delta_nx,2**self.best_Model_used[id_model]["n nx"]-1-self.delta_nx])
         
                        
                elif self.best_Model_used[id_previous_model]["family"]=="pred samples":    
                    x_model_hat=self.get_model_pred_samples(self.best_Model_used[id_previous_model]["X"],*self.best_Model_used[id_model]["theta hat"])
                    error_model=self.best_Model_used[id_model]["xn"]-x_model_hat
                    n_h=self.nm+self.best_Model_used[id_model]["n kx"]+self.best_Model_used[id_model]["n nx"]+self.nl
                    
                    if n_h+self.n_kr+self.n_sym+c*len(self.best_Model_used[id_model]["m theta"])<n_max:
                        n_h+=self.n_kr+self.n_sym
                        al_pred,MSE_pred=self.get_nx_nr_constraint_bit_pred_samples(self.best_Model_used[id_model]["w theta"],self.best_Model_used[id_previous_model]["eta"],self.best_Model_used[id_model]["xn previous"],error_model,n_max-n_h)
                        self.best_Model_used[id_model]["nx"]=int(np.ceil(np.sum(al_pred[0:len(self.best_Model_used[id_model]["m theta"])])))
                    else :
                        al_pred,MSE_pred=self.get_nx_nr_constraint_bit_pred_samples(self.best_Model_used[id_model]["w theta"],self.best_Model_used[id_previous_model]["eta"],self.best_Model_used[id_model]["xn previous"],error_model,n_max-n_h)
                        self.best_Model_used[id_model]["nx"]=np.min([n_max-n_h-self.delta_nx,2**self.best_Model_used[id_model]["n nx"]-1-self.delta_nx])
         
                     
                elif self.best_Model_used[id_previous_model]["family"]=="none":
                    x_model_hat=np.zeros(self.N)
                    error_model=self.best_Model_used[id_model]["xn"]-x_model_hat
                    n_h=self.nm+self.best_Model_used[id_model]["n kx"]+self.best_Model_used[id_model]["n nx"]+self.nl
                    
                    if n_h+self.n_kr+self.n_sym+c*len(self.best_Model_used[id_model]["m theta"])<n_max:
                        n_h+=self.n_kr+self.n_sym
                        al_pred,MSE_pred=self.get_nx_nr_constraint_bit_none(error_model,n_max-n_h)
                        al_pred=[0,al_pred]
                        self.best_Model_used[id_model]["nx"]=0
                    else :
                        al_pred,MSE_pred=self.get_nx_nr_constraint_bit_none(error_model,n_max-n_h)
                        al_pred=[0,al_pred]
                        self.best_Model_used[id_model]["nx"]=0
            # Handle cases where no model is used (default to zeros)             
            elif self.best_Model_used[id_model]["family"]=="none":
                x_model_hat=np.zeros(self.N)
                error_model=self.best_Model_used[id_model]["xn"]-x_model_hat
                n_h=self.nm+self.best_Model_used[id_model]["n kx"]+self.best_Model_used[id_model]["n nx"]+self.nl
                
                if n_h+self.n_kr+self.n_sym+5*len(self.best_Model_used[id_model]["m theta"])<n_max:
                    n_h+=self.n_kr+self.n_sym
                    al_pred,MSE_pred=self.get_nx_nr_constraint_bit_none(error_model,n_max-n_h)
                    al_pred=[0,al_pred]
                    self.best_Model_used[id_model]["nx"]=0
                else :
                    MSE_pred=MSE_S
                    self.best_Model_used[id_model]["nx"]=0
                    
            # Store models that meet the bit constraints and calculate SNR.
            if self.best_Model_used[id_model]["nx"]>=self.nb_min_bit_theta*len(self.best_Model_used[id_model]["m theta"]) and self.best_Model_used[id_model]["nx"]<=2**self.best_Model_used[id_model]["n nx"]-1:#+self.n_kx+self.nm<=n_max  :# or al_pred[-1]<=0 :
             
                SNR_tot_model.append(-10*np.log(MSE_S/MSE_pred))
                id_models.append(id_model)   
      
        # Sort models by total SNR in descending order.        
        SNR_tot_model=np.array(SNR_tot_model)
        id_models=np.array(id_models)
        sorted_indices = np.argsort(SNR_tot_model)
        
        ## plot best model candidates
        #print([[self.best_Model_used[id_models[sorted_indices[k]]]["name"],np.round(-10*SNR_tot_model[sorted_indices[k]])/10,self.best_Model_used[id_models[sorted_indices[k]]]["nx"]] for k in range(np.min([len(SNR_tot_model),self.delta_M]))])
      
        return  id_models[sorted_indices[0:np.max([0,self.delta_M])]]

    

    def get_theta(self):
        """
        Compute the estimated parameters (theta) for all models in `Model_used`.
    
        The method iterates over each model, determines its family type (e.g., sine, polynomial, predictive), 
        and calculates the appropriate theta values. The results are stored in the `theta hat` field 
        of each model configuration.
    
        Raises:
        - Prints an error message if a model family does not exist.
        """
        for id_model in self.Model_used:
            if self.best_Model_used[id_model]["family"] == "sin":
                # Calculate estimated parameters for the sine model
                theta_sin_hat = self.get_theta_sin(
                    self.best_Model_used[id_model]["xn"], 
                    self.best_Model_used[id_model]["m theta"], 
                    self.best_Model_used[id_model]["w theta"]
                )
                self.best_Model_used[id_model]["theta hat"] = theta_sin_hat
    
            elif self.best_Model_used[id_model]["family"] == "poly":
                # Calculate estimated parameters for the polynomial model
                theta_poly_hat = self.get_theta_poly(
                    self.best_Model_used[id_model]["xn"], 
                    self.best_Model_used[id_model]["m theta"], 
                    self.best_Model_used[id_model]["w theta"], 
                    self.best_Model_used[id_model]["order"]
                )
                self.best_Model_used[id_model]["theta hat"] = theta_poly_hat
    
            elif self.best_Model_used[id_model]["family"] == "pred samples":
                # Predictive model based on samples
                m_theta_pred_samples = self.get_m_theta_pred_samples(
                    self.best_Model_used[id_model]["order"], 
                    self.best_Model_used[id_model]["eta"], 
                    0, 
                    [0] * self.best_Model_used[id_model]["order"], 
                    [10] * self.best_Model_used[id_model]["order"]
                )
                self.best_Model_used[id_model]["m theta"] = m_theta_pred_samples
                
                X_pred_samples = self.get_X(
                    self.best_Model_used[id_model]["xn previous"], 
                    self.best_Model_used[id_model]["order"], 
                    self.best_Model_used[id_model]["eta"]
                )
                theta_pred_samples_hat = self.get_theta_pred_samples(
                    X_pred_samples, 
                    self.best_Model_used[id_model]["xn"], 
                    self.best_Model_used[id_model]["m theta"], 
                    self.best_Model_used[id_model]["w theta"]
                )
                
                self.best_Model_used[id_model]["X"] = X_pred_samples
                self.best_Model_used[id_model]["theta hat"] = theta_pred_samples_hat
    
            elif self.best_Model_used[id_model]["family"] == "pred para":
                # Predictive model based on parameters from a previous model
                id_previous_model = self.best_Model_used[id_model]["model used"]
                if self.Model_used[id_previous_model]["family"] == "sin":
                    theta_sin_hat = self.get_theta_sin(
                        self.best_Model_used[id_model]["xn"], 
                        self.best_Model_used[id_model]["m theta"], 
                        self.best_Model_used[id_model]["w theta"]
                    )
                    self.best_Model_used[id_model]["theta hat"] = theta_sin_hat
    
                elif self.Model_used[id_previous_model]["family"] == "poly":
                    theta_poly_hat = self.get_theta_poly(
                        self.best_Model_used[id_model]["xn"], 
                        self.best_Model_used[id_model]["m theta"], 
                        self.best_Model_used[id_model]["w theta"], 
                        self.best_Model_used[id_previous_model]["order"]
                    )
                    self.best_Model_used[id_model]["theta hat"] = theta_poly_hat 
    
                elif self.Model_used[id_previous_model]["family"] == "pred samples":
                    X_pred_samples = self.get_X(
                        self.best_Model_used[id_model]["xn previous"], 
                        self.best_Model_used[id_previous_model]["order"], 
                        self.best_Model_used[id_previous_model]["eta"]
                    )
                    theta_pred_samples_hat = self.get_theta_pred_samples(
                        X_pred_samples, 
                        self.best_Model_used[id_model]["xn"], 
                        self.best_Model_used[id_model]["m theta"], 
                        self.best_Model_used[id_model]["w theta"]
                    )
                    self.best_Model_used[id_model]["X"] = self.best_Model_used[id_previous_model]["X"]
                    self.best_Model_used[id_model]["theta hat"] = theta_pred_samples_hat
    
                elif self.best_Model_used[id_previous_model]["family"] == "none":
                    self.best_Model_used[id_model]["theta hat"] = []
    
            elif self.best_Model_used[id_model]["family"] == "none":
                # No model used
                self.best_Model_used[id_model]["theta hat"] = []
    
            else:
                # Handle unknown model family
                print(f"Error: The model {id_model} does not exist.")
        
   
    def enc_model(self, id_model, nx):
        """
        Encode the parameters (theta) and reconstruct the signal for a specific model.
    
        Parameters:
        - id_model: Identifier for the model to be encoded.
        - nx: Number of bits allocated for encoding the model parameters.
    
        Returns:
        - theta_tilde: Encoded parameters.
        - code_theta_tilde: Binary representation of the encoded parameters.
        - x_rec: Reconstructed signal from the model.
        """
        if self.best_Model_used[id_model]["family"] == "pred samples":
            # Encode predictive model based on samples
            theta_tilde, code_theta_tilde = self.get_theta_pred_samples_tilde(
                self.best_Model_used[id_model]["theta hat"], 
                nx, 
                self.best_Model_used[id_model]["m theta"], 
                self.best_Model_used[id_model]["w theta"]
            )
            x_rec = self.get_model_pred_samples(self.best_Model_used[id_model]["X"], *theta_tilde)
    
        elif self.best_Model_used[id_model]["family"] == "pred para":
            # Encode predictive model based on parameters of a previous model
            id_previous_model = self.best_Model_used[id_model]["model used"]
            if self.Model_used[id_previous_model]["family"] == "sin":
                theta_tilde, code_theta_tilde = self.get_theta_sin_tilde(
                    self.best_Model_used[id_model]["theta hat"], 
                    nx, 
                    self.best_Model_used[id_model]["m theta"], 
                    self.best_Model_used[id_model]["w theta"]
                )
                x_rec = self.get_model_sin(self.t, *theta_tilde)
    
            elif self.Model_used[id_previous_model]["family"] == "pred samples":
                theta_tilde, code_theta_tilde = self.get_theta_pred_samples_tilde(
                    self.best_Model_used[id_model]["theta hat"], 
                    nx, 
                    self.best_Model_used[id_model]["m theta"], 
                    self.best_Model_used[id_model]["w theta"]
                )
                x_rec = self.get_model_pred_samples(self.best_Model_used[id_model]["X"], *theta_tilde)
    
            elif self.Model_used[id_previous_model]["family"] == "poly":
                theta_tilde, code_theta_tilde = self.get_theta_poly_tilde(
                    self.best_Model_used[id_model]["theta hat"], 
                    nx, 
                    self.best_Model_used[id_model]["m theta"], 
                    self.best_Model_used[id_model]["w theta"]
                )
                x_rec = self.get_model_poly(self.t, *theta_tilde)
    
            elif self.best_Model_used[id_previous_model]["family"] == "none":
                theta_tilde = []
                code_theta_tilde = []
                x_rec = np.zeros(self.N)
    
        elif self.best_Model_used[id_model]["family"] == "sin":
            # Encode sine model
            theta_tilde, code_theta_tilde = self.get_theta_sin_tilde(
                self.best_Model_used[id_model]["theta hat"], 
                nx, 
                self.best_Model_used[id_model]["m theta"], 
                self.best_Model_used[id_model]["w theta"]
            )
            x_rec = self.get_model_sin(self.t, *theta_tilde)
    
        elif self.best_Model_used[id_model]["family"] == "poly":
            # Encode polynomial model
            theta_tilde, code_theta_tilde = self.get_theta_poly_tilde(
                self.best_Model_used[id_model]["theta hat"], 
                nx, 
                self.best_Model_used[id_model]["m theta"], 
                self.best_Model_used[id_model]["w theta"]
            )
            x_rec = self.get_model_poly(self.t, *theta_tilde)
    
        elif self.best_Model_used[id_model]["family"] == "none":
            # No model used
            theta_tilde = []
            code_theta_tilde = []
            x_rec = np.zeros(self.N)
    
        else:
            print(f"Error: The model {id_model} does not exist.")
    
        return theta_tilde, code_theta_tilde, x_rec
            
    
    def enc_residual(self, id_residual, r, n_r):
        """
        Encode the residual signal using a specified method.
    
        Parameters:
        - id_residual: Identifier for the residual encoding method.
        - r: The residual signal to be encoded.
        - n_r: Number of bits allocated for encoding the residual.
    
        Returns:
        - r_rec: Reconstructed residual signal.
        - code_r: Binary representation of the encoded residual.
        - kr: Normalization factor for the residual.
        - nb_sym: Number of symbols used for encoding the residual.
        """
        quality_r = 0  # Quality target for the residual encoding
        if self.best_Residual_used[id_residual]["name"] == "DCT+BPC":
            # Discrete Cosine Transform + Binary Plane Coding
            r_rec, code_r, kr, nb_sym = self.get_r_DCT_BPC_tilde(r, "MSE", quality_r, n_r, self.n_sym)
    
        elif self.best_Residual_used[id_residual]["name"] == "DWT+BPC":
            # Discrete Wavelet Transform + Binary Plane Coding
            r_rec, code_r, kr, nb_sym = self.get_r_DWT_BPC_tilde(r, "MSE", quality_r, n_r, self.n_sym)
    
        elif self.best_Residual_used[id_residual]["name"] == "none":
            # No residual encoding
            r_rec = np.zeros(self.N)
            code_r = []
            kr = 0
            nb_sym = 0
    
        else:
            # Error handling for unknown residual methods
            print(f"Error: The method {id_residual} does not exist.")
    
        return r_rec, code_r, kr, nb_sym

    
    def f(self, id_model, nx, n_max):
        """
        Given a model and a specified number of bits (nx) for encoding the model parameters,
        this function quantizes the model parameters, computes the residual signal, and identifies
        the best compression method for the residual using the remaining bits.
        
        Parameters:
        - id_model: Identifier of the current model to be encoded.
        - nx: Number of bits allocated for quantizing the model parameters.
        - n_max: Total available bits for encoding both the model parameters and the residual.
    
        Returns:
        - SNR_model: Signal-to-Noise Ratio (SNR) of the quantized model.
        - SNR_residual: SNR of the best-compressed residual.
        - theta_tilde: Quantized model parameters.
        - code_model: Binary representation of the quantized model parameters.
        - x_model: Signal reconstructed from the quantized model parameters.
        - id_residual: Identifier of the selected residual compression method.
        - kr: Scaling factor or key parameter used in the residual compression.
        - nb_sym: Number of symbols used in the residual compression.
        - code_residual: Compressed representation of the residual.
        - x_residual: Reconstructed residual signal.
        """
        # Step 1: Encode and quantize the model parameters using nx bits
        theta_tilde, code_model, x_model = self.enc_model(id_model, nx)
        
        # Step 2: Compute the SNR of the model
        SNR_model = -get_quality(self.best_Model_used[id_model]["xn"], x_model, "SNR")
        
        # Step 3: Compute the residual (difference between the original signal and the model's approximation)
        r = self.best_Model_used[id_model]["xn"] - x_model
        
        # Initialize variables for tracking the best residual compression method
        id_residual = 0
        x_residual = []
        code_residual = []
        kr = 0
        nb_sym = 0
        SNR_residual = -np.inf
        
        # Step 4: Test all available residual compression methods
        for id_residual_test in self.best_Residual_used:
            
            # Calculate the number of bits remaining for encoding the residual
            nr = n_max - self.nm - self.best_Model_used[id_model]["n kx"] - \
                 self.best_Model_used[id_model]["n nx"] - nx - self.nl - \
                 self.best_Residual_used[id_residual_test]["n kr"] - \
                 self.best_Residual_used[id_residual_test]["n nr"]
            
            if nr >= 0:
                # Encode the residual using the current compression method
                x_residual_test, code_residual_test, kr_test, nb_sym_test = self.enc_residual(id_residual_test, r, nr)
                
                # Compute the SNR for the residual
                SNR_residual_test = -get_quality(r, x_residual_test, "SNR")
                
                # If this method gives a higher SNR, update the best method
                if SNR_residual_test > SNR_residual:
                    id_residual = id_residual_test
                    x_residual = copy.copy(x_residual_test)
                    code_residual = copy.copy(code_residual_test)
                    kr = kr_test
                    nb_sym = nb_sym_test
                    SNR_residual = SNR_residual_test
    
        # Step 5: Increment the counter for completed transformations (e.g., encoding and residual compression)
        self.max_size_Nx += 2
        
        # Return the model's SNR, the best residual's SNR, and the associated encoding information
        return SNR_model, SNR_residual, theta_tilde, code_model, x_model, id_residual, kr, nb_sym, code_residual, x_residual

    
    def MMC_enc(self, x, x_previous, n_max):
        """
        Perform Multi-Model Compression (MMC) encoding. Distortion model folowed by exaustive search
    
        Parameters:
        - x: Current signal to encode.
        - x_previous: Previous signal used for prediction in certain models.
        - n_max: Maximum number of bits available for encoding.
    
        Returns:
        - Updates the best model and residual configurations and stores the encoded signal.
        """
        
        self.ini_MMC_enc()  # Initialize the encoding process with default values for models and residuals.
    
        # Retrieve normalization factors for the current and previous signals.
        kx, kx_previous = self.get_header(x, x_previous)
           
        self.get_theta()  # Compute parameters (theta) for all candidate models.
        
        # Get a list of candidate models and associated bit constraints for `nx` and `nr`.
        id_models = self.get_m_nx_nr(n_max)
    
        SNR_tot = -np.infty  # Initialize the total signal-to-noise ratio (SNR) as a negative infinity.
    
        for id_model in id_models:
            # Optimal `nx` value for the current model.
            n_x_opt = self.best_Model_used[id_model]["nx"]
    
            # Define a range of `nx` values to explore around the optimal value (`n_x_opt`).
            fact = 2  # Reduce computational cost by skipping every other bit value.
            
            min_rate_per_parameter=np.max([n_x_opt - fact*self.delta_nx,self.nb_min_bit_theta*len(self.best_Model_used[id_model]["w theta"])])
            max_rate_per_parameter=np.min([n_x_opt + fact*self.delta_nx+fact,1+np.min([self.nb_max_bit_theta*len(self.best_Model_used[id_model]["w theta"]),2**self.best_Model_used[id_model]["n nx"]-1])])
            
            Nx = [i for i in range(min_rate_per_parameter, max_rate_per_parameter, fact)]
            
            """
            print("Nx",Nx,n_x_opt,"len(Nx)",len(Nx),"self.delta_nx",self.delta_nx)
            print("n_x_opt - fact*self.delta_nx",n_x_opt - fact*self.delta_nx)
            print("n_x_opt + fact*self.delta_nx",n_x_opt + fact*self.delta_nx)
            print("max_bit",np.min([self.nb_max_bit_theta*len(self.best_Model_used[id_model]["w theta"]),2**self.best_Model_used[id_model]["n nx"]-1]))
            """
            
            for n_x in Nx:
                ## First Stage: Model Encoding ##
                
                SNR_model, SNR_residual, theta_tilde, code_model, x_model, id_residual, \
                kr, nb_sym, code_residual, x_residual = self.f(id_model, n_x, n_max)
    
    
                # Update the overall best configuration if the total SNR (model + residual) improves.
                if SNR_model + SNR_residual > SNR_tot:
                    SNR_tot = SNR_model + SNR_residual
                    self.best_Model_used[id_model].update({
                        "theta tilde": theta_tilde,
                        "code model": code_model,
                        "x model": x_model,
                        "quality model": SNR_model,
                        "nx": n_x,
                        "id residual": id_residual,
                        "name residual": self.best_Residual_used[id_residual]['name'],
                        "x residual": x_residual,
                        "nb sym residual": nb_sym,
                        "nr": len(code_residual),
                        "code residual": code_residual
                    })

                    self.best_Residual_used[id_residual]["kr"] = kr

                    # Record the identifiers for the best model and residual methods.
                    self.best_model_used = id_model
                    self.best_residual_used = id_residual


            
        # If the best model is not "none", update dependent models with the current best model
        if self.best_Model_used[self.best_model_used]["name"] != "none":
            for id_model in self.Model_used:
                # If the model uses predictive samples
                if self.Model_used[id_model]["family"] == "pred samples":
                    # Update the model used if the best model is not a "predictive parameter"
                    if self.best_Model_used[self.best_model_used]["family"] != "pred para":
                        self.Model_used[id_model]["model used"] = self.best_model_used
        
                # If the model uses predictive parameters
                elif self.Model_used[id_model]["family"] == "pred para":
                    if self.best_Model_used[self.best_model_used]["family"] != "pred para":
                        self.Model_used[id_model]["model used"] = self.best_model_used
                        # Update the model's parameters
                        self.Model_used[id_model]["m theta"] = self.best_Model_used[self.best_model_used]["theta tilde"]
                        factor = self.Model_used[id_model]["factor"]
                        self.Model_used[id_model]["w theta"] = [
                            self.Model_used[self.best_model_used]["w theta"][i] / factor
                            for i in range(len(self.best_Model_used[self.best_model_used]["w theta"]))
                        ]
                        self.Model_used[id_model]["n nx"] = self.best_Model_used[self.best_model_used]["n nx"]
        
        # Encode the signal and store encoding metadata
        self.id_model_enc = self.best_model_used  # Identifier for the encoded model
        self.id_residual_enc = self.best_residual_used  # Identifier for the encoded residual
        
        # Metadata for encoding
        self.nm_enc = self.nm  # Number of bits to encode the model index
        self.nl_enc = self.nl  # Number of bits to encode the residual index
        
        # Encoded model and residual names
        self.m_enc = self.best_Model_used[self.best_model_used]["name"]
        self.l_enc = self.best_Model_used[self.best_model_used]["name residual"]
        
        # Encoding parameters for the model and residual
        self.n_nx_enc = self.best_Model_used[self.best_model_used]["n nx"]
        self.nx_enc = self.best_Model_used[self.best_model_used]["nx"]
        
        self.n_nr_enc = self.best_Residual_used[self.best_residual_used]["n nr"]
        self.nb_sym_residual_enc = self.best_Model_used[self.best_model_used]["nb sym residual"]
        self.nr_enc = self.best_Model_used[self.best_model_used]["nr"]
        
        self.n_kx_enc = self.best_Model_used[self.best_model_used]["n kx"]
        self.n_kr_enc = self.best_Residual_used[self.best_residual_used]["n kr"]
        
        # Normalization factors for encoding
        self.kx_enc = self.best_Model_used[self.best_model_used]["kx"]
        self.kr_enc = self.best_Residual_used[self.best_residual_used]["kr"]
        
        # Reconstructed model and residual signals after encoding
        self.x_model_enc = self.best_Model_used[self.best_model_used]["x model"] * 2 ** (self.kx_enc)
        self.x_residual_enc = self.best_Model_used[self.best_model_used]["x residual"] * 2 ** (self.kx_enc)
        self.x_rec_enc = self.x_model_enc + self.x_residual_enc  # Final reconstructed signal
        
        # First stage encoding: encode the model identifier, normalization, and parameters
        code_m = my_bin(self.best_model_used, self.nm)  # Binary representation of the model index
        code_kx = my_bin(self.kx_enc, self.n_kx_enc)  # Binary representation of kx
        code_nx = my_bin(self.nx_enc, self.n_nx_enc)  # Binary representation of nx
        
        # Second stage encoding: encode residual-related parameters
        code_l = my_bin(self.best_Model_used[self.best_model_used]["id residual"], self.nl)  # Binary for residual index
        code_kr = my_bin(-self.kr_enc, self.n_kr_enc)  # Binary representation of kr
        code_nr = my_bin(self.nb_sym_residual_enc, self.n_nr_enc)  # Binary for residual symbols count
        
        # Combine all encoding stages to produce the final encoded signal
        code = (
            code_m
            + code_kx
            + code_nx
            + self.best_Model_used[self.best_model_used]["code model"]
            + code_l
            + code_kr
            + code_nr
            + self.best_Model_used[self.best_model_used]["code residual"]
        )
        
        # Store the final encoded signal
        self.code = code
        return code  # Return the complete encoded binary representation


class Decode_one_window(Model_Decoder, Residual_Decoder):
    """
    Class to decode a single window of data, implementing both model and residual decoding.
    """

    def __init__(self, fn=50, fs=6400, N=128, Model_used={}, Residual_used={}):
        """
        Initialize the decoder with model and residual configurations.
        
        Parameters:
        - fn: Base frequency of the signal.
        - fs: Sampling frequency of the signal.
        - N: Length of the signal window.
        - Model_used: Dictionary defining the set of models to be used in decoding.
        - Residual_used: Dictionary defining the set of residuals to be used in decoding.
        """
        self.Model_used = Model_used  # Set of models used for decoding.
        self.Residual_used = Residual_used  # Set of residuals used for decoding.

        # Initialize parent classes for model and residual decoding.
        Model_Decoder.__init__(self, fn, fs, N, False)
        Residual_Decoder.__init__(self, N)

        # Header parameter initialization for decoding.
        self.nm = np.max([1, int(np.ceil(np.log2(len(Model_used))))])  # Bits for model encoding.
        self.nl = np.max([1, int(np.ceil(np.log2(len(Residual_used))))])  # Bits for residual encoding.
        self.n_kx = 5  # Bits for kx encoding.
        self.n_kr = 5  # Bits for kr encoding.
        self.nb_max_bit_theta = 10  # Maximum number of bits per parameter for model coefficients.
        self.n_sym = 10  # Bits for residual symbols.

    def ini_MMC_dec(self):
        """
        Initialize the decoding process by creating deep copies of model and residual configurations.
        """
        self.best_Model_used = copy.deepcopy(self.Model_used)  # Deep copy of model configurations.
        self.best_Residual_used = copy.deepcopy(self.Residual_used)  # Deep copy of residual configurations.

    def dec_header(self):
        """
        Decode the header of the encoded signal to initialize model and residual configurations.
        """
        for id_model in self.Model_used:  # Iterate through each model in the used models set.

            if self.best_Model_used[id_model]["family"] == "pred samples":
                # Calculate the number of bits required for the model based on its order.
                self.best_Model_used[id_model]["n nx"] = int(np.ceil(np.log2(
                    self.best_Model_used[id_model]["order"] * self.nb_max_bit_theta)))
                self.best_Model_used[id_model]["n kx"] = self.n_kx

            elif self.best_Model_used[id_model]["family"] == "pred para":
                # If the model is a predictive parameter, calculate bits based on its previous model.
                id_previous_model = self.Model_used[id_model]["model used"]
                self.best_Model_used[id_model]["n kx"] = self.n_kx

                if self.Model_used[id_previous_model]["family"] == "sin":
                    self.best_Model_used[id_model]["n nx"] = int(np.ceil(np.log2(3 * self.nb_max_bit_theta)))

                elif self.Model_used[id_previous_model]["family"] == "pred samples":
                    self.best_Model_used[id_model]["n nx"] = int(np.ceil(np.log2(
                        self.Model_used[id_previous_model]["order"] * self.nb_max_bit_theta)))

                elif self.Model_used[id_previous_model]["family"] == "poly":
                    self.best_Model_used[id_model]["n nx"] = int(np.ceil(np.log2(
                        (self.Model_used[id_previous_model]["order"] + 1) * self.nb_max_bit_theta)))

                elif self.Model_used[id_previous_model]["family"] == "none":
                    self.best_Model_used[id_model]["n nx"] = 0

            elif self.best_Model_used[id_model]["family"] == "sin":
                # For sinusoidal models, calculate based on the maximum bits per parameter.
                self.best_Model_used[id_model]["n nx"] = int(np.ceil(np.log2(3 * self.nb_max_bit_theta)))
                self.best_Model_used[id_model]["n kx"] = self.n_kx

            elif self.best_Model_used[id_model]["family"] == "poly":
                # For polynomial models, calculate based on the order of the polynomial.
                self.best_Model_used[id_model]["n nx"] = int(np.ceil(np.log2(
                    (self.best_Model_used[id_model]["order"] + 1) * self.nb_max_bit_theta)))
                self.best_Model_used[id_model]["n kx"] = self.n_kx

            elif self.best_Model_used[id_model]["family"] == "none":
                # For models that don't involve encoding, set nx to 0.
                self.best_Model_used[id_model]["n nx"] = 0
                self.best_Model_used[id_model]["n kx"] = self.n_kx

        # Configure residual properties for each residual type.
        for id_residual in self.Residual_used:
            if self.best_Residual_used[id_residual]["name"] == "DCT+BPC":
                self.best_Residual_used[id_residual]["n kr"] = self.n_kr
                self.best_Residual_used[id_residual]["n nr"] = self.n_sym

            elif self.best_Residual_used[id_residual]["name"] == "DWT+BPC":
                self.best_Residual_used[id_residual]["n kr"] = self.n_kr
                self.best_Residual_used[id_residual]["n nr"] = self.n_sym

            elif self.best_Residual_used[id_residual]["name"] == "none":
                # If no residual is used, set kr and nr to 0.
                self.best_Residual_used[id_residual]["n kr"] = 0
                self.best_Residual_used[id_residual]["n nr"] = 0

            else:
                # Handle invalid or unknown residual types.
                print("error: the method {} does not exist".format(id_residual))


    def dec_model(self, id_model, code_m, x_previous_n):
        """
        Decode the model parameters and reconstruct the model signal.
    
        Parameters:
        - id_model: Identifier of the model to be decoded.
        - code_m: Encoded model parameters as a binary string.
        - x_previous_n: Normalized version of the previous signal (for predictive models).
        """
        n_x = len(code_m)  # Length of the encoded model parameters.
    
        if self.best_Model_used[id_model]["family"] == "pred samples":
            # Predictive model using sample values
            self.best_Model_used[id_model]["m theta"] = self.get_m_theta_pred_samples(
                self.best_Model_used[id_model]["order"],
                self.best_Model_used[id_model]["eta"],
                0,
                [0] * self.best_Model_used[id_model]["order"],
                [10] * self.best_Model_used[id_model]["order"]
            )
            self.best_Model_used[id_model]["X"] = self.get_X(
                x_previous_n[:2 * self.N],
                self.best_Model_used[id_model]["order"],
                self.best_Model_used[id_model]["eta"]
            )
            self.best_Model_used[id_model]["theta tilde"] = self.get_theta_pred_samples_tilde(
                code_m,
                n_x,
                self.best_Model_used[id_model]["m theta"],
                self.best_Model_used[id_model]["w theta"]
            )
            self.best_Model_used[id_model]["x model"] = self.get_model_pred_samples(
                self.best_Model_used[id_model]["X"],
                *self.best_Model_used[id_model]["theta tilde"]
            ) * 2**self.kx_dec
    
        elif self.best_Model_used[id_model]["family"] == "pred para":
            # Predictive model based on parameters
            id_previous_model = self.Model_used[id_model]["model used"]
    
            if self.Model_used[id_previous_model]["family"] == "sin":
                self.best_Model_used[id_model]["theta tilde"] = self.get_theta_sin_tilde(
                    code_m, n_x,
                    self.best_Model_used[id_model]["m theta"],
                    self.best_Model_used[id_model]["w theta"]
                )
                self.best_Model_used[id_model]["x model"] = self.get_model_sin(
                    self.t,
                    *self.best_Model_used[id_model]["theta tilde"]
                ) * 2**self.kx_dec
    
            elif self.Model_used[id_previous_model]["family"] == "pred samples":
                self.best_Model_used[id_model]["X"] = self.get_X(
                    x_previous_n[:2 * self.N],
                    self.Model_used[id_previous_model]["order"],
                    self.Model_used[id_previous_model]["eta"]
                )
                self.best_Model_used[id_model]["theta tilde"] = self.get_theta_pred_samples_tilde(
                    code_m, n_x,
                    self.best_Model_used[id_model]["m theta"],
                    self.best_Model_used[id_model]["w theta"]
                )
                self.best_Model_used[id_model]["x model"] = self.get_model_pred_samples(
                    self.best_Model_used[id_model]["X"],
                    *self.best_Model_used[id_model]["theta tilde"]
                ) * 2**self.kx_dec
    
            elif self.Model_used[id_previous_model]["family"] == "poly":
                self.best_Model_used[id_model]["theta tilde"] = self.get_theta_poly_tilde(
                    code_m, n_x,
                    self.best_Model_used[id_model]["m theta"],
                    self.best_Model_used[id_model]["w theta"]
                )
                self.best_Model_used[id_model]["x model"] = self.get_model_poly(
                    self.t,
                    *self.best_Model_used[id_model]["theta tilde"]
                ) * 2**self.kx_dec
    
            elif self.Model_used[id_previous_model]["family"] == "none":
                self.best_Model_used[id_model]["theta tilde"] = []
                self.best_Model_used[id_model]["x model"] = np.zeros(self.N)
    
        elif self.best_Model_used[id_model]["family"] == "sin":
            # Sinusoidal model
            self.best_Model_used[id_model]["theta tilde"] = self.get_theta_sin_tilde(
                code_m, n_x,
                self.best_Model_used[id_model]["m theta"],
                self.best_Model_used[id_model]["w theta"]
            )
            self.best_Model_used[id_model]["x model"] = self.get_model_sin(
                self.t,
                *self.best_Model_used[id_model]["theta tilde"]
            ) * 2**self.kx_dec
    
        elif self.best_Model_used[id_model]["family"] == "poly":
            # Polynomial model
            self.best_Model_used[id_model]["theta tilde"] = self.get_theta_poly_tilde(
                code_m, n_x,
                self.best_Model_used[id_model]["m theta"],
                self.best_Model_used[id_model]["w theta"]
            )
            self.best_Model_used[id_model]["x model"] = self.get_model_poly(
                self.t,
                *self.best_Model_used[id_model]["theta tilde"]
            ) * 2**self.kx_dec
    
        elif self.best_Model_used[id_model]["family"] == "none":
            # No model
            self.best_Model_used[id_model]["theta tilde"] = []
            self.best_Model_used[id_model]["x model"] = np.zeros(self.N)
    
    def dec_residual(self, id_residual, code_r, nb_sym_residual):
        """
        Decode the residual signal using the specified residual method.
    
        Parameters:
        - id_residual: Identifier of the residual decoding method.
        - code_r: Encoded residual as a binary string.
        - nb_sym_residual: Number of symbols used for the residual encoding.
    
        Returns:
        - Decoded residual signal.
        - Number of bits used in the residual decoding.
        """
        if self.best_Residual_used[id_residual]["name"] == "DCT+BPC":
            x_residual, nr_dec = self.get_r_DCT_BPC_tilde(code_r, nb_sym_residual, self.kr_dec)
            return x_residual * 2**(self.kx_dec), nr_dec
    
        elif self.best_Residual_used[id_residual]["name"] == "DWT+BPC":
            x_residual, nr_dec = self.get_r_DWT_BPC_tilde(code_r, nb_sym_residual, self.kr_dec)
            return x_residual * 2**(self.kx_dec), nr_dec
    
        elif self.best_Residual_used[id_residual]["name"] == "none":
            return np.zeros(self.N), 0
    
        else:
            print("Error in residual decoding: method not recognized.")
    
    
    def MMC_dec(self, code, x_previous):
        """
        Perform Multi-Model Coding (MMC) decoding.
    
        Parameters:
        - code: Encoded signal as a binary string.
        - x_previous: Previous signal used for prediction in some models.
        """
        self.ini_MMC_dec()  # Initialize model and residual configurations.
        self.dec_header()  # Decode header information.
    
        ptr = 0  # Pointer for traversing the binary code.
    
        # Decode model information.
        self.id_model_dec = int(my_inv_bin(code[ptr:ptr + self.nm]))
        ptr += self.nm
    
        self.kx_dec = int(my_inv_bin(code[ptr:ptr + self.n_kx]))
        ptr += self.n_kx
    
        self.n_nx_dec = self.best_Model_used[self.id_model_dec]["n nx"]
        self.nx_dec = int(my_inv_bin(code[ptr:ptr + self.n_nx_dec]))
        ptr += self.n_nx_dec
    
        self.dec_model(self.id_model_dec, code[ptr:ptr + self.nx_dec], x_previous * 2**(-self.kx_dec))
        ptr += self.nx_dec
    
        # Decode residual information.
        self.id_residual_dec = int(my_inv_bin(code[ptr:ptr + self.nl]))
        ptr += self.nl
    
        self.kr_dec = -int(my_inv_bin(code[ptr:ptr + self.best_Residual_used[self.id_residual_dec]["n kr"]]))
        ptr += self.best_Residual_used[self.id_residual_dec]["n kr"]
    
        self.n_nr_dec = self.best_Residual_used[self.id_residual_dec]["n nr"]
        self.nb_sym_residual = int(my_inv_bin(code[ptr:ptr + self.n_nr_dec]))
        ptr += self.n_nr_dec
    
        self.best_Model_used[self.id_model_dec]["x residual"], self.nr_dec = self.dec_residual(
            self.id_residual_dec, code[ptr:], self.nb_sym_residual)
    
        # Combine model and residual to reconstruct the signal.
        self.x_rec_dec = self.best_Model_used[self.id_model_dec]["x model"] + self.best_Model_used[self.id_model_dec]["x residual"]
    
        # Update models for predictive decoding.
        if self.best_Model_used[self.id_model_dec]["name"] != "none":
            for name in self.Model_used:
                if self.Model_used[name]["family"] == "pred samples" and self.best_Model_used[self.id_model_dec]["family"] != "pred para":
                    self.Model_used[name]["model used"] = self.id_model_dec
    
                elif self.Model_used[name]["family"] == "pred para" and self.best_Model_used[self.id_model_dec]["family"] != "pred para":
                    self.Model_used[name]["model used"] = self.id_model_dec
                    self.Model_used[name]["m theta"] = self.best_Model_used[self.id_model_dec]["theta tilde"]
                    factor = self.Model_used[name]["factor"]
                    self.Model_used[name]["w theta"] = [
                        self.Model_used[self.id_model_dec]["w theta"][i] / factor
                        for i in range(len(self.best_Model_used[self.id_model_dec]["w theta"]))
                    ]
                    self.Model_used[name]["n nx"] = self.best_Model_used[self.id_model_dec]["n nx"]
