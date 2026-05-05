import numpy as np
import matplotlib as plt
import wfdb
import matplotlib.pyplot as pyplt

import Visualize_EKG as vekg
import Normalize_EKG as nekg
import Wavelet_approx as wa

"""
Note the first functions are strictly for plotting.
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


"""
The following piece of code is to use the function defended by the cubic splines to give you the sliding window
    embedding. I want to use as many points as possible, so the loops in the upstairs space are prominant and 
    the noise acts as its own cluster.
"""

import Spline_approx as sa

def SWE_w_spline(ptf, lead_s, tau, M):

    # The default for this function is to break up the interval in which the trimmed EKG into 1 for each point 
    #   in the original signal. So, if your trimmed signal has 802 points, then this will break up the interval
    #   [0, 802] into 802 one unit pieces.

    fun, n = sa.get_trimmed_cs_int(ptf, lead_s)

    # I want there to be the same number of points in the SWEs, so I will set the n as 1000.

    num_points = 2500

    step_size = n/num_points

    # tau = n/6 # this is the period of the ekg.

    projected_points = []

    i = 0
    while i < num_points:
        x = step_size*i
        point = []
        j = 0
        while j <= M:
            x_pro = x+(j*tau)
            if x_pro > n-1:
                point = []
                i = num_points
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

def SWE_no_approx(ptf, lead, M): # for this function tau is not defined because you just pick the next point in the ECG signal. So it is 1 or 0.01 sec.

    # The default for this function is to break up the interval in which the trimmed EKG into 1 for each point 
    #   in the original signal. So, if your trimmed signal has 802 points, then this will break up the interval
    #   [0, 802] into 802 one unit pieces.

    trimmed_signal = nekg.trim_EKG(ptf, lead)

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




# tau = 1.25

# ptf = "Brugada_dataset/files/188981/188981"
# lead_s = ['V2']
# # pro_points = np.array(SWE_w_spline(ptf, lead_s, tau, 2))

# pro_points = np.array(SWE_no_approx(ptf, lead_s, 2))

# plot_2d(pro_points)


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# points = pro_points

# # Build edges (you actually don't even need this list)
# edges = [(i, (i+1) % len(points)) for i in range(len(points))]

# fig, ax = plt.subplots()

# # Plot all points (static)
# ax.scatter(points[:, 0], points[:, 1], color='blue', s = 1)

# # Line that will grow over time
# line, = ax.plot([], [], color='red', linewidth=1)

# def update(frame):
#     # Take all points up to current frame
#     current_points = points[:frame]

#     # Close the loop if at the end
#     if frame == len(points):
#         current_points = np.vstack([current_points, points[0]])

#     line.set_data(current_points[:, 0], current_points[:, 1])

#     return line,

# ani = FuncAnimation(fig, update, frames=len(points), interval=30)

# # ani.save("animation_251972_2500.gif", writer='pillow', fps=20)

# plt.show()





# for 3d






# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# points = pro_points

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# # Static points
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', s=1)

# points = np.squeeze(pro_points)

# line, = ax.plot([], [], [], color='red', linewidth=2)

# def update(frame):
#     current_points = points[:frame+1]

#     if frame == len(points) - 1:
#         current_points = np.vstack([current_points, points[0]])

#     line.set_data(current_points[:, 0], current_points[:, 1])
#     line.set_3d_properties(current_points[:, 2])

#     return line,

# ani = FuncAnimation(fig, update, frames=len(points), interval=30)

# plt.show()
