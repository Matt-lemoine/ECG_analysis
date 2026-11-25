"""Nested CV comparison of Betti versus cumulative persistence vectorizations.

Run: pip install optuna xgboost scikit-learn pandas numpy
     python my_xgboost_nested_optuna.py

For each of seven lead combinations and two representation families, an
INNER Optuna search chooses XGBoost hyperparameters AND the best candidate
representation (embedding R, Betti N, or CPV variant). The OUTER folds then
measure performance on patients excluded from the search. All data sets must
have rows in the same patient order and identical binary labels.

IMPORTANT: Input CSVs and any data-derived feature construction must not use
information from held-out patients (e.g. fitted scalers, feature selection).
If vectorizations are computed independently for each recording, they may be
precomputed. Otherwise perform feature learning within the CV folds.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

# ---------------------------- SETTINGS ---------------------------------
INPUT_DIR = Path('Vectorization')
OUTPUT_DIR = Path('XGBoost_Nested_Optuna_Results')
N_TRIALS = 10             # Increase after testing; can be VERY computationally expensive
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
SEED = 42
MODEL_N_JOBS = 1        # Prevent CPU oversubscription
Rns = ['R2', 'R3', 'R4']
Ns = [1, 5, 10, 100]
ending_leads = ['_all', '_V1', '_V2', '_V3', '_V1_V2', '_V1_V3', '_V2_V3']
CPV_VARIANTS = [
    'Pers_Vec_Vectorization',
    'Pers_Vec_Vectorization_nooverlap',
    'Pers_Vec_Vectorization_nocount',
    'Pers_Vec_Vectorization_nocount_nooverlap',
    'Pers_Vec_Vectorization_noarea',
]
FAMILIES = ('Betti', 'CPV')


def specifications():
    for lead in ending_leads:
        for r in Rns:
            for n in Ns:
                variant = 'Betti_Vectorization'
                yield dict(Lead=lead, Family='Betti', Variant=variant, R_n=r, N=n,
                           Path=INPUT_DIR / f'{variant}{lead}_{n}_{r}.csv')
            for variant in CPV_VARIANTS:
                yield dict(Lead=lead, Family='CPV', Variant=variant, R_n=r, N=None,
                           Path=INPUT_DIR / f'{variant}{lead}_{r}.csv')


def load_datasets():
    specs = list(specifications())
    missing = [str(s['Path']) for s in specs if not s['Path'].is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} files are missing; examples:\n' +
                                '\n'.join(missing[:15]))
    datasets = {(lead, family): [] for lead in ending_leads for family in FAMILIES}
    reference_labels = None
    for spec in specs:
        # Matches ORIGINAL loader: Betti files have no header; CPV files do.
        df = pd.read_csv(spec['Path'], header=None if spec['Family'] == 'Betti' else 0)
        if df.shape[1] < 3:
            raise ValueError(f'Expected index, features, and label in {spec["Path"]}')
        # Original code drops column 0; last remaining column is the label.
        X = df.iloc[:, 1:-1].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
        y = df.iloc[:, -1].to_numpy()
        if not np.isfinite(X).all() or pd.isna(y).any():
            raise ValueError(f'Non-finite features or missing labels: {spec["Path"]}')
        if reference_labels is None:
            reference_labels = y.copy()
            classes, counts = np.unique(y, return_counts=True)
            if len(classes) != 2:
                raise ValueError('Binary classification requires exactly two target classes.')
            if counts.min() < max(N_OUTER_SPLITS, N_INNER_SPLITS):
                raise ValueError('Not enough patients of each class for the chosen CV splits.')
        elif not np.array_equal(reference_labels, y):
            raise ValueError(f'Patient count, order, or labels differ in {spec["Path"]}')
        datasets[(spec['Lead'], spec['Family'])].append({**spec, 'X': X})
    # Align labels and folds for ALL representations / lead combinations.
    classes = np.unique(reference_labels)
    encoded_y = (reference_labels == classes[1]).astype(int)
    return datasets, encoded_y, [str(c) for c in classes]


def make_model(params, seed=SEED):
    return XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        tree_method='hist', random_state=seed, n_jobs=MODEL_N_JOBS,
        **params,
    )


def suggest_params(trial):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10, log=True),
    }


def cv_metrics(X, y, train_indices, params, inner_folds):
    accuracies, aucs = [], []
    for fold_train, fold_val in inner_folds:
        tr, va = train_indices[fold_train], train_indices[fold_val]
        model = make_model(params)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        prob = model.predict_proba(X[va])[:, 1]
        accuracies.append(accuracy_score(y[va], pred))
        aucs.append(roc_auc_score(y[va], prob))
    return float(np.mean(accuracies)), float(np.mean(aucs))


def optimize_cell(candidates, y, selection_indices, outer_fold_id, lead, family,
                  trial_records):
    """Choose params AND representation using only selection_indices."""
    inner = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True,
                            random_state=SEED + 100 + outer_fold_id)
    local_splits = list(inner.split(np.zeros(len(selection_indices)), y[selection_indices]))

    def objective(trial):
        params = suggest_params(trial)
        best = None
        for index, candidate in enumerate(candidates):
            accuracy, auc = cv_metrics(candidate['X'], y, selection_indices,
                                       params, local_splits)
            row = {'Lead': lead, 'Family': family, 'Outer_Fold': outer_fold_id,
                   'Trial': trial.number, 'Candidate_Index': index,
                   'Variant': candidate['Variant'], 'R_n': candidate['R_n'],
                   'N': candidate['N'], 'Inner_Accuracy': accuracy, 'Inner_AUC': auc,
                   **params}
            trial_records.append(row)
            # Accuracy is the primary metric; AUC breaks exact ties.
            if best is None or (accuracy, auc) > (best[0], best[1]):
                best = (accuracy, auc, index)
        trial.set_user_attr('best_candidate', best[2])
        trial.set_user_attr('best_candidate_auc', best[1])
        return best[0]

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED + outer_fold_id))
    study.optimize(objective, n_trials=N_TRIALS)
    # If trials tie on accuracy, prefer the one with higher AUC.
    winning_trial = max(study.trials, key=lambda t:
                        (t.value, t.user_attrs['best_candidate_auc']))
    return candidates[winning_trial.user_attrs['best_candidate']], winning_trial


def candidate_metadata(candidate):
    return {k: (str(candidate[k]) if k == 'Path' else candidate[k])
            for k in ('Variant', 'R_n', 'N', 'Path')}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets, y, label_classes = load_datasets()
    outer_cv = StratifiedKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=SEED)
    outer_folds = list(outer_cv.split(np.zeros(len(y)), y))
    # Stronger validation: each outer training set must permit inner stratification.
    for train_idx, _ in outer_folds:
        if np.bincount(y[train_idx], minlength=2).min() < N_INNER_SPLITS:
            raise ValueError('Not enough samples per class in an outer training fold '
                             'for inner CV. Reduce N_INNER_SPLITS.')
    trial_records, fold_records, prediction_records, final_records = [], [], [], []
    summary = []

    for lead in ending_leads:
        for family in FAMILIES:
            candidates = datasets[(lead, family)]
            print(f'\n=== {lead}: {family} ({len(candidates)} representations) ===', flush=True)
            cell_folds = []
            for outer_id, (train_idx, test_idx) in enumerate(outer_folds, start=1):
                selected, best_trial = optimize_cell(
                    candidates, y, train_idx, outer_id, lead, family, trial_records)
                params = dict(best_trial.params)
                model = make_model(params)
                model.fit(selected['X'][train_idx], y[train_idx])
                prediction = model.predict(selected['X'][test_idx])
                probability = model.predict_proba(selected['X'][test_idx])[:, 1]
                row = {'Lead': lead, 'Family': family, 'Outer_Fold': outer_id,
                       'Accuracy': float(accuracy_score(y[test_idx], prediction)),
                       'AUC': float(roc_auc_score(y[test_idx], probability)),
                       'Inner_Selected_Accuracy': float(best_trial.value),
                       'Inner_Selected_AUC': best_trial.user_attrs['best_candidate_auc'],
                       **candidate_metadata(selected), **params}
                cell_folds.append(row)
                fold_records.append(row)
                for idx, predicted, prob in zip(test_idx, prediction, probability):
                    prediction_records.append({'Lead': lead, 'Family': family,
                                               'Outer_Fold': outer_id,
                                               'Patient_Row': int(idx),
                                               'True_Label_Encoded': int(y[idx]),
                                               'Prediction_Encoded': int(predicted),
                                               'Positive_Class_Probability': float(prob)})
                print(f'  Outer fold {outer_id}: accuracy={row["Accuracy"]:.3f}, '
                      f'AUC={row["AUC"]:.3f}, {row["Variant"]}, R={row["R_n"]}, '
                      f'N={row["N"]}', flush=True)
                # Save incremental results in case a long run is interrupted.
                pd.DataFrame(fold_records).to_csv(OUTPUT_DIR / 'Outer_Fold_Results.csv', index=False)
                pd.DataFrame(prediction_records).to_csv(OUTPUT_DIR / 'Outer_Predictions.csv', index=False)
                pd.DataFrame(trial_records).to_csv(OUTPUT_DIR / 'All_Trial_Candidate_Results.csv', index=False)

            accuracies = np.array([r['Accuracy'] for r in cell_folds])
            aucs = np.array([r['AUC'] for r in cell_folds])
            summary.append({'Lead': lead, 'Family': family,
                            'Mean_Outer_Accuracy': float(accuracies.mean()),
                            'SD_Outer_Accuracy': float(accuracies.std(ddof=1)),
                            'Mean_Outer_AUC': float(aucs.mean()),
                            'SD_Outer_AUC': float(aucs.std(ddof=1))})
            pd.DataFrame(summary).to_csv(OUTPUT_DIR / 'Nested_CV_Summary.csv', index=False)

    # Format 2-by-7 results for a manuscript; the values are OUTER-fold scores.
    by_key = {(r['Lead'], r['Family']): r for r in summary}
    table = []
    for family in FAMILIES:
        row = {'Vectorization': 'Betti Vector' if family == 'Betti' else 'CPV'}
        for lead in ending_leads:
            s = by_key[(lead, family)]
            row[lead.removeprefix('_')] = f'{100*s["Mean_Outer_Accuracy"]:.1f}%'
        table.append(row)
    pd.DataFrame(table).to_csv(OUTPUT_DIR / 'Table_2x7.csv', index=False)

    # Separately, tune each cell on ALL patients to choose one final model for
    # later use. These are NOT independently assessed test-set accuracies.
    all_idx = np.arange(len(y))
    for lead in ending_leads:
        for family in FAMILIES:
            selected, trial = optimize_cell(datasets[(lead, family)], y, all_idx,
                                            0, lead, family, trial_records)
            final_records.append({'Lead': lead, 'Family': family,
                                  'Selection_CV_Accuracy': float(trial.value),
                                  'Selection_CV_AUC': trial.user_attrs['best_candidate_auc'],
                                  **candidate_metadata(selected), **trial.params})
            pd.DataFrame(final_records).to_csv(OUTPUT_DIR / 'Final_Selected_Configurations.csv', index=False)
    pd.DataFrame(trial_records).to_csv(OUTPUT_DIR / 'All_Trial_Candidate_Results.csv', index=False)
    (OUTPUT_DIR / 'Class_Label_Mapping.json').write_text(
        json.dumps({'encoded_0': label_classes[0], 'encoded_1': label_classes[1]}, indent=2))
    print('\nDone. Outputs saved to:', OUTPUT_DIR.resolve())
    print(pd.DataFrame(table).to_string(index=False))


if __name__ == '__main__':
    main()
