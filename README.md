# ECG Analysis

This GitHub repository is for the Python files used to analyze ECG data provided at https://doi.org/10.13026/9gvk-m592.

## The Pipeline

The data is obtained through the open source data platform PhysioNet. This particular data set deals with patients having Brugada Syndrome. This disease effects certain aspects of the heartbeat which can be detected through Electrocardiography readings (ECG).

In `Visualize_EKG.py`, the data can be visualized with all the leads that are of interest.

The data is processed and trimmed down to only include the middle most 6 sinus rhythm periods. This is performed in `Normalize_EKG.py`. The `All_Trimming_info.csv` file the bounds that are used to trim the ECG's to the middle most 6 sinus periods. The `Spline_approx.py` is then used to approximate the ECG reading before passing to the Sliding Window Embedding. This ensures that the ECG readings (and therefore the embeddings) are as smooth as possible.

When the data is processed and normalized, it is passed to the Sliding Window Embedding https://arxiv.org/abs/1307.6188, in `SWE_Dynamic_T.py`. The data is then projected into a higher dimension where persistent homology can be calculated using `construct_complex.py`.

In `main_EKG.py`, all of the pieces listed above are put together to get the persistence diagram of the point cloud formed from the sliding window embedding. The output of this python file is a persistence diagram and a CSV file outlining all of the points in the persistence diagram for each of the patients and each of the leads of interest. 

These persistence diagrams are then given to `Vectorization.py` where they are turned into vectors using two methods: Betti Vectors and Cumulative Persistence Vectors. These resulting vectors are then used to train the variety of machine learning algorithms to predict the prescence of Brugada Syndrome in a given ECG. The output of the `Vectorization.py` file is a CSV with vectors for each of the patients. There is a new CSV file for each vectorization method, lead, and embedding dimension. They are all saved to a folder called `Vectorization`.

In each of the machin learning algorithms used here, the optimization comes from the Optuna package in Python.

## Installation

In this project all of the packages and versions are listed in the requirements.txt file. The following command will install these packages.

```
pip install -r requirements.txt
```

# Notes on the Nested cross-validation: five classifiers

Keep `nested_cv_common.py` in the same directory as all five model-specific scripts. These are separate entry points for the same evaluation procedure; the shared file prevents inconsistencies in splits and CSV output.

## Running

Place your `Vectorization` folder beside the scripts, or change `INPUT_DIR` in `nested_cv_common.py`. Each model writes to its own `<Model>_Nested_Optuna_Results` folder, with `Table_2x7.csv`, `Nested_CV_Summary.csv`, `Outer_Fold_Results.csv`, `Outer_Predictions.csv`, `All_Trial_Candidate_Results.csv`, and `Final_Selected_Configurations.csv`.

## Study design

- 7 lead selections x 2 families = 14 comparisons per algorithm.
- Candidate vectorizations: Betti with R2/R3/R4 and N=1/5/10/100; CPV with R2/R3/R4 and five variants.
- Outer stratified five-fold held-out assessment; inner stratified five-fold Optuna selection (same patient folds across methods).
- One study per lead/family and outer training set; each trial evaluates its hyperparameters across every candidate vectorization of the family, choosing the best **only using inner validation**.
- Accuracy maximized; AUC recorded, with AUC tie-break. `Table_2x7.csv` contains outer-fold average accuracies, not tuned inner values.
- Optionally refit a final configuration on all available patients; this is not a separate test score.

## Special notes

- SVM matches the original `LinearSVC` model, with fold-local standardization and tuning of C and tolerance. ROC-AUC uses decision scores.
- AdaBoost tunes learning rate, estimator count and decision-tree base depth.
- CatBoost tunes depth, iterations, learning rate and L2 leaf regularization.
- PyTorch network retains the 512→64→8→1 architecture and tunes learning rate, batch size, epochs, weight decay. Each fold trains a *new* network, fixes your original issue of reusing weights across folds, and evaluates AUC/accuracy from the **same** trained model. Scaling is fit only on training rows.
- NeoLSSVM uses internally optimized kernel/regularization parameters; Optuna evaluates standardized versus unstandardized training, while representation choice is searched in the inner CV. Its reported probability-based AUC uses `predict_proba` where available (otherwise raw predicted values). Depending on the installed NeoLSSVM version, check the probability and prediction conventions.
- Exact patient order **and labels** are validated across input files. Non-finite features rejected.
- This scheme assumes the vectorizations were produced per patient independently, without fitting preprocessing on all patients in advance.
- With N_TRIALS=10 and 5 outer/inner folds, the full search is expensive. Start with reduced candidate lists, and a small trial count, then increase.

## Verification

Syntax checks passed for all six scripts. The model-specific SVM, AdaBoost, CatBoost and neural-network training/evaluation functions were tested on synthetic data, plus a miniature end-to-end nested-CV run for SVM. The `neo-ls-svm` package was not installed in the verification environment, so the LSSVM runtime path was not executed here.

Code created by Matthew Lemoine, for more information about the project and the results see https://matthewlemoine.com/Research.html. If you have questions, please email mlemoine559@gmail.com.
