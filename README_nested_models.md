# Nested cross-validation: five classifiers

Keep `nested_cv_common.py` in the same directory as all five model-specific scripts. These are separate entry points for the same evaluation procedure; the shared file prevents inconsistencies in splits and CSV output.

## Installation

```bash
pip install numpy pandas scikit-learn optuna catboost torch neo-ls-svm
```

## Running

```bash
python SVM_nested_optuna.py
python AdaBoost_nested_optuna.py
python CatBoost_nested_optuna.py
python LSSVM_nested_optuna.py
python Neural_Network_nested_optuna.py
```

Place your `Vectorizations_10_16_26` folder beside the scripts, or change `INPUT_DIR` in `nested_cv_common.py`. Each model writes to its own `<Model>_Nested_Optuna_Results` folder, with `Table_2x7.csv`, `Nested_CV_Summary.csv`, `Outer_Fold_Results.csv`, `Outer_Predictions.csv`, `All_Trial_Candidate_Results.csv`, and `Final_Selected_Configurations.csv`.

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
