import numpy as np
import matplotlib as plt
import wfdb
import matplotlib.pyplot as pyplt

import Visualize_EKG as vekg
import Normalize_EKG as nekg
import Wavelet_approx as wa

"""
Note the first 2 functions are strictly for plotting.
The next 3 functions will fix the sample of points taken to be the length of the signal. (for a signal of about 1000 points it will take about 100 seconds.)
There is a function below them, that doesn't have a fixed sample points taken place. Use this to find a suitable tau.
"""

def plot_3d(projected_points):

    fig = pyplt.figure()
    ax = fig.add_subplot(projection='3d')

    for x,y,z in projected_points:
        ax.scatter(x,y,z)

    return pyplt.show()

def plot_2d(projected_points):

    fig = pyplt.figure()
    ax = fig.add_subplot(projection='2d')

    for x,y in projected_points:
        ax.scatter(x,y)

    return pyplt.show()

def save_plot_3d(projected_points, naming_things):
    
    fig = pyplt.figure()
    ax = fig.add_subplot(projection='3d')

    for x,y,z in projected_points:
        ax.scatter(x,y,z)

    # Save the plot as an image file with the specified output file name
    pyplt.savefig(naming_things)
    
    # Print a confirmation message that the persistence diagram was saved
    print(f"Projected points saved as {naming_things}.png")


def save_plot_2d(projected_points, naming_things):
    
    fig = pyplt.figure()
    ax = fig.add_subplot(projection='2d')

    for x,y,z in projected_points:
        ax.scatter(x,y)

    # Save the plot as an image file with the specified output file name
    pyplt.savefig(naming_things)
    
    # Print a confirmation message that the persistence diagram was saved
    print(f"Projected points saved as {naming_things}")


"""
The following function will retrieve the desired points in whatever dimensional projection space you want.
M tells you which dimension to push to. (e.g. M = 2  ==> R^3)
tau tells you about what's happening in the periodicity of your signal. 
The breakup_interval_more gives you the option to breakup your signal into smaller pieces (note that this will add computation time.)
"""

def SWE_get_points_nd(ptf, lead_s, wavelet, level_decomp, tau, M, breakup_interval_more = None):

    signal = nekg.trim_EKG(ptf, lead_s)

    if breakup_interval_more is None:
        bim = len(signal)
    else:
        bim = breakup_interval_more

    projected_points = []
    points_in_interval = np.linspace(0, len(signal), bim)

    i = 0
    while i < bim:
        p = points_in_interval[i]
        point = []
        j = 0
        while j <= M: # This takes care of values outside of our interval.
            # if p+j*tau > len(signal): # This piece of the code says that if your sliding window extends beyond the scope of the signal, say it's 0.
            #     k = 0
            if p+j*tau > len(signal): # This piece of the code says that if your sliding window extends beyond the scope of the signal, loop back around to the beginning and use that.
                x = p+j*tau - len(signal)
                k = wa.wavelet(ptf, lead_s, wavelet, level_decomp, x)[0]
            else:
                k = wa.wavelet(ptf, lead_s, wavelet, level_decomp, p + j*tau)[0]
            point.append(k)
            j = j+1
        projected_points.append(point)
        i = i+1
    
    return projected_points

"""
The following code will return a 3d plot of the projected wavelet. Here we fix M = 2.
"""
def SWE_plot_3d(ptf, lead_s, wavelet, level_decomp, tau, naming_things):
    
    M = 2

    projected_points = get_points_nd(ptf, lead_s, wavelet, level_decomp, tau, M)
    
    return plot_3d(projected_points)

"""
The following code will return a 3d plot of the projected wavelet. Here we fix M = 1.
"""
def SWE_plot_2d(ptf, lead_s, wavelet, level_decomp, tau, naming_things):
    
    M = 1

    projected_points = get_points_nd(ptf, lead_s, wavelet, level_decomp, tau, M)
    
    return plot_2d(projected_points)


"""
The following function is to test and find a suitable tau.
After this function, you may want to plot them, if your M = 1,2.
Use the plot_2d or plot_3d respectively.
"""

def SWE_get_points_nd_TESTING(ptf, lead_s, wavelet, level_decomp, tau, M, sample_points, breakup_interval_more = None):

    signal = nekg.trim_EKG(ptf, lead_s)

    if breakup_interval_more is None:
        bim = len(signal)
    else:
        bim = breakup_interval_more

    projected_points = []
    points_in_interval = np.linspace(0, len(signal), bim)

    i = 0
    while i < sample_points:
        p = points_in_interval[i]
        point = []
        j = 0
        while j <= M: # This takes care of values outside of our interval.
            # if p+j*tau > len(signal): # This piece of the code says that if your sliding window extends beyond the scope of the signal, say it's 0.
            #     k = 0
            if p+j*tau > len(signal): # This piece of the code says that if your sliding window extends beyond the scope of the signal, loop back around to the beginning and use that.
                x = p+j*tau - len(signal)
                k = wa.wavelet(ptf, lead_s, wavelet, level_decomp, x)[0]
            else:
                k = wa.wavelet(ptf, lead_s, wavelet, level_decomp, p + j*tau)[0]
            point.append(k)
            j = j+1
        projected_points.append(point)
        i = i+1

    return projected_points
