"""Nested Optuna evaluation for LSSVM; same patient folds as XGBoost.
Keep nested_cv_common.py beside this file.
"""
import nested_cv_common as experiment

# Edit configuration here or edit the defaults in nested_cv_common.py.
experiment.MODEL = 'LSSVM'
experiment.OUTPUT_DIR = experiment.Path('LSSVM_Nested_Optuna_Results')
experiment.N_TRIALS = 10
experiment.N_OUTER_SPLITS = 5
experiment.N_INNER_SPLITS = 5

if __name__ == '__main__':
    experiment.main()
