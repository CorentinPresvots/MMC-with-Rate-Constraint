# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:22:19 2023

@author: presvotscor
"""
import numpy as np


def dynamic_subsample(signal, threshold):
    # Calcul de la dérivée du signal
    derivatives = np.gradient(signal)

    # Initialisation des indices des points à conserver
    indices_to_keep = [0]
    
    derivative_memoire=derivatives[0]
    # Parcours des dérivées pour identifier les segments monotones
    for i in range(1, len(derivatives)-1):
        if abs(derivatives[i]-derivative_memoire) > threshold:
            indices_to_keep.append(i)
            derivative_memoire=derivatives[i]
            

    # Création du signal sous-échantillonné en fonction des indices à conserver
    #subsampled_signal = signal[indices_to_keep]

    return indices_to_keep+[len(signal)-1]
"""


def dynamic_subsample(signal, threshold):


    # Initialisation des indices des points à conserver
    indices_to_keep = [0]
    
    signal_memoire=signal[0]
    # Parcours des dérivées pour identifier les segments monotones
    for i in range(1, len(signal)-1):
        if abs(signal[i]-signal_memoire) > threshold:
            indices_to_keep.append(i)
            signal_memoire=signal[i]
            

    # Création du signal sous-échantillonné en fonction des indices à conserver
    #subsampled_signal = signal[indices_to_keep]

    return indices_to_keep+[len(signal)-1]

"""

"""
class KalmanFilter:
    def __init__(self, initial_state_mean, initial_state_covariance, transition_matrix, observation_matrix, observation_covariance, process_covariance):
        self.state_mean = initial_state_mean
        self.state_covariance = initial_state_covariance
        self.transition_matrix = transition_matrix
        self.observation_matrix = observation_matrix
        self.observation_covariance = observation_covariance
        self.process_covariance = process_covariance

    def predict(self):
        # Prediction step
        predicted_state_mean = np.dot(self.transition_matrix, self.state_mean)
        predicted_state_covariance = np.dot(np.dot(self.transition_matrix, self.state_covariance), self.transition_matrix.T) + self.process_covariance

        return predicted_state_mean, predicted_state_covariance

    def update(self, observation):
        # Update step
        prediction_mean, prediction_covariance = self.predict()

        kalman_gain = np.dot(np.dot(prediction_covariance, self.observation_matrix.T), np.linalg.inv(np.dot(np.dot(self.observation_matrix, prediction_covariance), self.observation_matrix.T) + self.observation_covariance))

        self.state_mean = prediction_mean + np.dot(kalman_gain, observation - np.dot(self.observation_matrix, prediction_mean))
        self.state_covariance = prediction_covariance - np.dot(np.dot(kalman_gain, self.observation_matrix), prediction_covariance)

        return self.state_mean , self.state_covariance 

def dynamic_subsample(signal, threshold):
    # Paramètres du filtre de Kalman
    initial_state_mean = np.array([0.0, 0.0])  # État initial
    initial_state_covariance = np.eye(2)  # Matrice de covariance de l'état initial
    transition_matrix = np.array([[1.0, 1.0], [0.0, 1.0]])  # Matrice de transition
    observation_matrix = np.array([[1.0, 0.0]])  # Matrice d'observation
    observation_covariance = np.array([[1.0]])  # Matrice de covariance de l'observation
    process_covariance = np.eye(2)  # Matrice de covariance du processus

    # Création du filtre de Kalman
    kf = KalmanFilter(initial_state_mean, initial_state_covariance, transition_matrix, observation_matrix, observation_covariance, process_covariance)
    
    indices_to_keep = [0]

    for i in range(1,len(signal)-1):
        # Observation
        observation = np.array([signal[i]])
    
        # Prédiction et mise à jour
        predicted_state_mean, _ = kf.predict()
        
        
        #print("signal {:.2f}".format(signal[i]))
        #print("État prédit {:.2f}".format(predicted_state_mean[0]))

        if abs(predicted_state_mean[0] - signal[i]) > threshold:
            indices_to_keep.append(i)
            print("signal {:.2f}".format(signal[i]))
            print("État prédit {:.2f}".format(predicted_state_mean[0]))
       
            updated_state_mean, _ = kf.update(observation)
                

    
    indices_to_keep.append(len(signal)-1)
    return indices_to_keep


"""

# Exemple d'utilisation
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    
    # Génération d'un signal de test (par exemple, un signal sinusoïdal avec du bruit)
    N=128 # number of sample
    t = np.linspace(0, 0.02, N)
    signal = np.sin(2*np.pi*50*t) + 0.00 * np.random.randn(N)

    # Seuil de pente pour déterminer la monotonie
    threshold = 0.01
    # Sous-échantillonnage dynamique
    indices_to_keep= dynamic_subsample(signal, threshold)

    # Affichage du signal original et du signal sous-échantillonné

    plt.figure(figsize=(10, 5))
    plt.plot(t, signal, label="Signal original")
    plt.plot(t[indices_to_keep], signal[indices_to_keep],'-+', label="sunsample, ind keep={}".format(len(indices_to_keep)))
    plt.legend()
    plt.xlabel("t (s)")
    plt.ylabel("Amplitude")
    plt.show()
    
