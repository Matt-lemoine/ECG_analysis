import numpy as np
import matplotlib as plt
import pandas as pd
import wfdb
import matplotlib.pyplot as pyplt

import Visualize_EKG as vekg
import Normalize_EKG as nekg
import Spline_approx as sapp

"""
Note the first functions are strictly for plotting.
"""

def plot_3d(projected_points):

    fig = pyplt.figure()
    ax = fig.add_subplot(projection='3d')

    for x,y,z in projected_points:
        ax.scatter(x,y,z)

    return pyplt.show()

def plot_2d(projected_points):

    fig = pyplt.figure()

    for x,y in projected_points:
        pyplt.scatter(x,y)

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

    for x,y in projected_points:
        pyplt.scatter(x,y)

    # Save the plot as an image file with the specified output file name
    pyplt.savefig(naming_things)
    
    # Print a confirmation message that the persistence diagram was saved
    print(f"Projected points saved as {naming_things}")

"""
The following function performs the sliding window embedding of the ECGs by using the trimmed signal outlined in the "All_Trimming_info.csv" file.
"""

def SWE_no_approx_with_CSV(ptf, lead, M, lb, ub): # for this function tau is not defined because you just pick the next point in the ECG signal. So it is 1 or 0.01 sec.

    # The default for this function is to break up the interval in which the trimmed EKG into 1 for each point 
    #   in the original signal. So, if your trimmed signal has 802 points, then this will break up the interval
    #   [0, 802] into 802 one unit pieces.

    trimmed_signal = nekg.standardize_EKG(ptf, lead, lb, ub)

    n = len(trimmed_signal)

    projected_points = []

    i = 0
    while i < n:
        x = i
        point = []
        j = 0
        while j <= M:
            x_pro = x+j
            if x_pro > n-1:
                point = []
                i = n
                break
            else:
                k = trimmed_signal[x_pro]
                point.append(k)
                j = j+1
        
        if point != []:
            projected_points.append(point)
            i = i+1
        else:
            break

    return projected_points


"""
The following definition performs the Spline approximation so that all the trimmed EKGs have the same number of points in the SWE.
"""

def SWE_w_Splines(ptf, lead, tau, M):

    fun, n = sapp.get_cs_trimmed_signal(ptf, lead)

    print(n)

    tau = tau

    num_samp = n - tau* M # This gives you enough room to make the 615 samples in each dimension and to sample a few different tau values.

    number_of_swe_points = 615 # Chosen from the average length of the trimmed signals being 615.45

    sliding_by = num_samp/number_of_swe_points # This number tells you how much you need to slide over along the interval so that you have 615 points total.

    projected_points = []

    i = 0
    while i < number_of_swe_points:
        x = i * sliding_by
        point = []
        j = 0
        while j <= M:
            x_pro = x+(j*tau)
            if x_pro >= n:
                point = []
                i = number_of_swe_points
                break
            else:
                k = fun(x_pro)
                point.append(k)
                j = j+1
        
        if point != []:
            projected_points.append(point)
            i = i+1
        else:
            break

    return projected_points
