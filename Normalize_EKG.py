"""
This file is to normalize the EKG signal to:
60 bpm,
and extract about 8 peaks.
"""

import numpy as np
import matplotlib as plt
import wfdb
import csv

import Visualize_EKG as vekg

# You will want an input to be the normalization frequency.
# Big question: How do you stretch out or shrink down a signal to fit in xx bpm?

def find_peaks(ptf, lead_s)

def trim_EKG(ptf, lead_s, num_peaks)

def normalize(ptf, lead_s, bpm)