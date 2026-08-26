# Spline approximation of ECGs.

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import wfdb

import Visualize_EKG as vekg
import Normalize_EKG as nekg

# Function to get the cubic spline interpolation function. Only takes in one lead at a time.

# cs = cubic spline

def get_cs_trimmed_signal(ptf, lead):

    p_signal = nekg.trim_by_CSV(ptf, lead)

    xs = np.arange(0, len(p_signal))

    f_lead = CubicSpline(xs, p_signal, bc_type = 'natural')

    return f_lead, len(p_signal)