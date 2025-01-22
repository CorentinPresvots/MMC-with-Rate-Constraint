# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:04:56 2023

@author: presvotscor
"""
# Load and visualize signals from the RTE Digital Fault Recording database
# Note: This script is designed to load only 12 signals. For access to more signals,
# download the full database from the official repository:
# https://github.com/rte-france/digital-fault-recording-database

import numpy as np
import matplotlib.pyplot as plt

def get_RTE_signal(signal_index):
    """
    Load and scale a specific signal from the RTE database.

    Parameters:
        signal_index (int): Index of the signal to load (0 to 11 for this example).

    Returns:
        tuple: Scaled voltage and current signals (v1, v2, v3, i1, i2, i3).
    """
    # Scaling factors for voltage and current
    Delta_u = 18.31055  # Voltage scaling factor
    Delta_i = 4.314     # Current scaling factor

    # Load the pre-saved signal dataset
    loaded = np.load('DATA_S2.npz') 
    DATA_S2 = loaded['DATA_S2']

    # Extract and scale voltage (v1, v2, v3) and current (i1, i2, i3) signals
    v1 = DATA_S2[signal_index][0] * Delta_u
    v2 = DATA_S2[signal_index][1] * Delta_u
    v3 = DATA_S2[signal_index][2] * Delta_u
    i1 = DATA_S2[signal_index][3] * Delta_i
    i2 = DATA_S2[signal_index][4] * Delta_i
    i3 = DATA_S2[signal_index][5] * Delta_i

    return v1, v2, v3, i1, i2, i3

# Main program
if __name__ == "__main__":
    # Select the signal ID to analyze
    signal_id = 1  # Example signal index (adjust as needed)

    # Retrieve the selected signal's voltage and current data
    v1, v2, v3, i1, i2, i3 = get_RTE_signal(signal_id)

    # Define the sampling frequency and time vector
    fs = 6400  # Sampling frequency in Hz
    t = np.linspace(0, (len(v1) - 1) * (1 / fs), len(v1))  # Time vector

    # Plot voltage signals
    fig = plt.figure(figsize=(8, 5), dpi=100)
    plt.plot(t[:12800], v1[:12800], lw=2, label='v1')
    plt.plot(t[:12800], v2[:12800], lw=2, label='v2')
    plt.plot(t[:12800], v3[:12800], lw=2, label='v3')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage Signals')
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.legend()
    plt.minorticks_on()
    plt.show()

    # Plot current signals
    fig = plt.figure(figsize=(8, 5), dpi=100)
    plt.plot(t[:12800], i1[:12800], lw=2, label='i1')
    plt.plot(t[:12800], i2[:12800], lw=2, label='i2')
    plt.plot(t[:12800], i3[:12800], lw=2, label='i3')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.title('Current Signals')
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.legend()
    plt.minorticks_on()
    plt.show()
