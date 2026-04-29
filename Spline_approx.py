# Spline approximation of ECGs.

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import wfdb

import Visualize_EKG as ve
import Normalize_EKG as ne

# Function to get the cubic spline interpolation function. Only takes in one lead at a time.

# cs = cubic spline

def get_cs_int(ptf, lead):
    p_signal = ve.get_EKG_leads(ptf, lead)

    xs = np.arange(0, len(p_signal))

    f_lead = CubicSpline(xs, p_signal, bc_type = 'natural')

    return f_lead

def get_trimmed_cs_int(ptf, lead):

    p_signal = ne.trim_EKG(ptf, lead)

    xs = np.arange(0, len(p_signal))

    f_lead = CubicSpline(xs, p_signal, bc_type = 'natural')

    return f_lead, len(p_signal)