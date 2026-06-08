# ECG Analysis

This GitHub repository is for the Python files used to analyze ECG data provided https://doi.org/10.13026/9njx-6322.

## The Data

The data was obtained through the open source data platform PhysioNet. This particular data set deals with patients having Brugada Syndrome. This disease effects certain aspects of the heartbeat which can be detected through Electrocardiography readings (ECG).

In `Visualize_EKG.py`, the data can be visualized with all the leads that are of interest.

The data was processed and trimmed down to only include the middle most 6 peaks and valleys. This was performed in `Normalize_EKG.py`.

When the data was processed and normalized, it was passed to the Sliding Window Embedding https://arxiv.org/abs/1307.6188, in `SWE.py`. The data was then projected into a higher dimension where persistent homology can be calculated using `construct_complex.py`.

This is the README.md for the EKG_analysis pipeline. I have outlined below all the different files in this repo, the functions in these files, what they do, and how to use them. There is also more documentation in each of the files outlining how the functions call and work.

In this repo we will have a pipeline that takes in EKG readings and outputs
    Persistent homology of the SWE of the EKG.
