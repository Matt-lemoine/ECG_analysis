# ECG Analysis

This GitHub repository is for the Python files used to analyze ECG data provided https://doi.org/10.13026/9njx-6322.

## The Pipeline

The data was obtained through the open source data platform PhysioNet. This particular data set deals with patients having Brugada Syndrome. This disease effects certain aspects of the heartbeat which can be detected through Electrocardiography readings (ECG).

In `Visualize_EKG.py`, the data can be visualized with all the leads that are of interest.

The data was processed and trimmed down to only include the middle most 6 peaks and valleys. This was performed in `Normalize_EKG.py`.

When the data was processed and normalized, it was passed to the Sliding Window Embedding https://arxiv.org/abs/1307.6188, in `SWE.py`. The data was then projected into a higher dimension where persistent homology can be calculated using `construct_complex.py`.

In `main_EKG.py`, all of the pieces listed above were put together to get an analysis of the ECGs. The output of this python file is a persistence diagram and a CSV file outlining all of the points in the persistence diagram for each of the patients and each of the leads of interest. 

These persistence diagrams were then given to `Vectorization.py` where they were turned into vectors using two methods: Betti Vectors and Cumulative Persistence Vectors. These resulting vectors were then used to train an XGBoost model to predict the prescence of Brugada Syndrome in a given ECG.

## Notes

Both `Spline_approx.py` and `Wavelet_approx.py` were used initially to approximate the ECG signals. These were later not used in the computatino of the persistence diagrams nor in the training of the XGBoost. 

Additionally, some of the ECG readings were difficult to pick out the 6 middle most peaks and valleys, so `messy_main.py` was used to address these ECGs. In this file, the ECGs were trimmed by hand and then the persistent homology was calculated.

## Necessary Libraries

The necessary libraries for this code includes the following.

1. numpy
2. matplotlib
3. matplotlib.pyplot
4. csv
5. wfdb (This library allows for visualization of the ECGs and to be able to use the leads as an array.)
6. os
7. pathlib
    8. Path
9. time
10. gudhi (This library is used for computing the persistent homology.)
11. pandas
12. sklearn.metrics
    * accuracy_score, confusion_matrix
14. sklearn.model_selection
    * train_test_split
16. xgboost





This is the README.md for the EKG_analysis pipeline. I have outlined below all the different files in this repo, the functions in these files, what they do, and how to use them. There is also more documentation in each of the files outlining how the functions call and work.

In this repo we will have a pipeline that takes in EKG readings and outputs
    Persistent homology of the SWE of the EKG.
