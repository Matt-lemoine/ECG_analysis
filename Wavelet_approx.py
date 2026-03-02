"""
Notes on the code here. The coeffs is what I want to be spit out by the def get_wave_eq. But I'm still not sure what this is telling us.
    I also do not understand why we are seeing so many different coefficients in the plt.show() stuff.
    But the fact that we are getting coefficients and plots is amazing. 
    These cna be used to get a function with which we can plot the SWE stuff.
"""

import pywt
import numpy as np
import matplotlib.pyplot as pyplt
import wfdb
import matplotlib as plt

import Visualize_EKG as vekg
import Normalize_EKG as nekg

"""
For the following function plot_wavelet:
    The ptf is the path to file, the lead_s are the lead_s you are interested in.
    The wavelet for us should always be either db3 or db4.
    level_decomp should be about 4.
    sample_num should be greater than or equal to the len(signal).
    naming
    we also normalize our signals inside each of these functions.
"""

def plot_wavelet(ptf, lead_s, wavelet, level_decomp, naming_things, sample_num = None):

    wavelet = pywt.Wavelet(wavelet)

    if len(lead_s) == 1:
        signal = nekg.trim_EKG(ptf, lead_s)
    else:
        error = 'You can only import one lead at a time for plot_wavelet.'
        return print(error)
    
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level_decomp)

    coeffs_approx = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[0:]]

    signal_approx = pywt.waverec(coeffs_approx, wavelet=wavelet)

    phi, psi, x = wavelet.wavefun(level=10)

    def phi_eval(x_query):
        return np.interp(x_query, x, phi)

    def signal_approx_eval(t, cJ):
        val = 0.0
        for k, ck in enumerate(cJ):
            val += ck * phi_eval(t-k)
        return val

    if sample_num is None:
        add_how_much = 1
    else:
        add_how_much = len(signal)/sample_num

    cJ = coeffs_approx[0]
    values = []
    i = 0
    while i <= len(signal):
        values.append((signal_approx_eval(i,cJ)/level_decomp))
        i += add_how_much

    pyplt.plot(values)
    return pyplt.show() 

"""
The following function wavelet_coeffs gets the coefficients of the wavelet approximation.
I want a function that tells me the coefficients of the approximation and the detail coefficients.

Fill in stuff here.
"""

def wavelet_coeffs(ptf, lead_s, wavelet, level_decomp):

    wavelet = pywt.Wavelet(wavelet)

    if len(lead_s) == 1:
        signal = nekg.trim_EKG(ptf, lead_s)
    else:
        error = 'You can only import one lead at a time for wavelet_coeffs.'
        return print(error)
    
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level_decomp)

    return coeffs

"""
The following function takes in the same as the functions before it, but returns the coefficients as 
    part of the graphs with the original signal and the first approximation of that signal.
"""

def plot_wavelet_w_coeffs(ptf, lead_s, wavelet, level_decomp):
    
    wavelet = pywt.Wavelet(wavelet)

    if len(lead_s) == 1:
        signal = nekg.trim_EKG(ptf, lead_s)
    else:
        error = 'You can only import one lead at a time for wavelet_coeffs.'
        return print(error)

    t = np.arange(0,len(signal), 1)

    coeffs = pywt.wavedec(signal, wavelet, level = level_decomp)

    l = level_decomp + 2

    # Plot the original signal and wavelet coefficients
    pyplt.figure(figsize=(10, 8))
    pyplt.subplot(l, 1, 1) # This is the first of l plots in the first column
    pyplt.plot(t, signal)
    pyplt.title("Original Signal")

    for i, coeff in enumerate(coeffs):
        pyplt.subplot(l, 1, i + 2) # (num of plots, how many per column, cycling through the coeff.)
        pyplt.plot(coeff)
        pyplt.title(f"Wavelet Coefficients - Level {i}")

    pyplt.tight_layout()
    return pyplt.show()

"""
Now I want a function that will be our function we use in the SWE.
This function takes in
The path to file, the leads you care about, the wavelet you're using to approximate with, and the time at which you'd like to approximate.
    note on the time: It has to be within the range of the signal so if your ekg has readings at every 100th and it lasts for 12 seconds,
        you cannot input 1201 because that exceeds the bounds of the original wavelet.
This function will return the value at the inputed time.
"""

def wavelet(ptf, lead_s, wavelet, level_decomp, t):

    wavelet = pywt.Wavelet(wavelet)

    if len(lead_s) == 1:
        signal = nekg.trim_EKG(ptf, lead_s)
    else:
        error = 'You can only import one lead at a time for wavelet.'
        return print(error)
    
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level_decomp)

    coeffs_approx = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[0:]]

    signal_approx = pywt.waverec(coeffs_approx, wavelet=wavelet)

    phi, psi, x = wavelet.wavefun(level=10)

    def phi_eval(x_query):
        return np.interp(x_query, x, phi)

    def signal_approx_eval(t, cJ):
        val = 0.0
        for k, ck in enumerate(cJ):
            val += ck * phi_eval(t-k)
        return val

    cJ = coeffs_approx[0]

    if t <= len(signal) and t >= 0:
        signal_at_t = signal_approx_eval(t,cJ)/level_decomp
    elif t > len(signal) or t < 0:
        error_message = "Your t exceed the bounds of the signal."
        return error_message

    return signal_at_t