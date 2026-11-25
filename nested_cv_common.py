"""Nested CV comparison of Betti versus cumulative persistence vectorizations.

Requires optuna, scikit-learn, pandas, numpy and selected model library
     python my_xgboost_nested_optuna.py

For each of seven lead combinations and two representation families, an
INNER Optuna search chooses model hyperparameters AND the best candidate
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

# ---------------------------- SETTINGS ---------------------------------
MODEL = 'SVM'  # overridden by each model-specific launcher
INPUT_DIR = Path('Vectorization')
OUTPUT_DIR = Path(f'{MODEL}_Nested_Optuna_Results')
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


def suggest_params(trial):
    if MODEL == 'SVM':
        return {'C': trial.suggest_float('C', 1e-3, 100, log=True),
                'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True)}
    if MODEL == 'AdaBoost':
        return {'n_estimators': trial.suggest_int('n_estimators', 50, 400, step=25),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 2, log=True),
                'tree_max_depth': trial.suggest_int('tree_max_depth', 1, 3)}
    if MODEL == 'CatBoost':
        return {'iterations': trial.suggest_int('iterations', 100, 500, step=50),
                'depth': trial.suggest_int('depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10, log=True)}
    if MODEL == 'LSSVM':
        # NeoLSSVM selects regularization/kernel scales internally.
        # Vary a legitimate training-only preprocessing choice instead.
        return {'standardize': trial.suggest_categorical('standardize', [True, False])}
    if MODEL == 'Neural_Network':
        return {'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'epochs': trial.suggest_categorical('epochs', [5, 10, 20]),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)}
    raise ValueError(f'Unsupported model {MODEL}')


def fit_and_score(X, y, tr, va, params, seed=SEED):
    """Fresh estimator and training-only scaling on EVERY inner/outer fold."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, roc_auc_score
    if MODEL == 'Neural_Network':
        return neural_fit_and_score(X, y, tr, va, params, seed)
    if MODEL == 'SVM':
        from sklearn.svm import LinearSVC
        estimator = make_pipeline(StandardScaler(), LinearSVC(
            C=params['C'], tol=params['tol'], max_iter=10000,
            dual='auto', random_state=seed))
    elif MODEL == 'CatBoost':
        from catboost import CatBoostClassifier
        estimator = CatBoostClassifier(
            iterations=params['iterations'], depth=params['depth'],
            learning_rate=params['learning_rate'], l2_leaf_reg=params['l2_leaf_reg'],
            verbose=False, random_seed=seed, thread_count=1,
            allow_writing_files=False)
    elif MODEL == 'AdaBoost':
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        stump = DecisionTreeClassifier(max_depth=params['tree_max_depth'], random_state=seed)
        estimator = AdaBoostClassifier(estimator=stump,
            n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
            random_state=seed)
    elif MODEL == 'LSSVM':
        from neo_ls_svm import NeoLSSVM
        estimator = (make_pipeline(StandardScaler(), NeoLSSVM())
                     if params['standardize'] else NeoLSSVM())
    else:
        raise ValueError(MODEL)
    estimator.fit(X[tr], y[tr])
    if MODEL == 'SVM':
        score = np.asarray(estimator.decision_function(X[va])).ravel()
        pred = np.asarray(estimator.predict(X[va])).ravel()
    elif MODEL == 'LSSVM':
        # The model may expose numerical predictions rather than sklearn's
        # classifier metadata. Prefer predicted probabilities, if supported.
        if hasattr(estimator, 'predict_proba'):
            proba = np.asarray(estimator.predict_proba(X[va]))
            score = proba[:, -1] if proba.ndim == 2 else proba.ravel()
            pred = (score >= .5).astype(int)
        else:
            score = np.asarray(estimator.predict(X[va])).ravel()
            pred = (score >= .5).astype(int)
    else:
        pred = np.asarray(estimator.predict(X[va])).ravel().astype(int)
        score = np.asarray(estimator.predict_proba(X[va]))[:, 1]
    return (float(accuracy_score(y[va], pred)),
            float(roc_auc_score(y[va], score)), pred, score)


def neural_fit_and_score(X, y, tr, va, params, seed):
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = StandardScaler().fit(X[tr])
    xtr = torch.tensor(scaler.transform(X[tr]), dtype=torch.float32)
    xva = torch.tensor(scaler.transform(X[va]), dtype=torch.float32)
    ytr = torch.tensor(y[tr].reshape(-1, 1), dtype=torch.float32)
    # Recreate your original architecture for EACH fold (no learned weights shared).
    net = nn.Sequential(nn.Linear(X.shape[1], 512), nn.ReLU(),
                        nn.Linear(512, 64), nn.ReLU(),
                        nn.Linear(64, 8), nn.ReLU(), nn.Linear(8, 1)).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=params['batch_size'],
                        shuffle=True, generator=generator)
    net.train()
    for _ in range(params['epochs']):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimizer.step()
    net.eval()
    with torch.no_grad():
        score = torch.sigmoid(net(xva.to(device))).cpu().numpy().ravel()
    pred = (score >= .5).astype(int)
    return (float(accuracy_score(y[va], pred)),
            float(roc_auc_score(y[va], score)), pred, score)

def cv_metrics(X, y, train_indices, params, inner_folds):
    accuracies, aucs = [], []
    for k, (fold_train, fold_val) in enumerate(inner_folds):
        tr, va = train_indices[fold_train], train_indices[fold_val]
        accuracy, auc, _, _ = fit_and_score(X, y, tr, va, params, seed=SEED+k)
        accuracies.append(accuracy)
        aucs.append(auc)
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
                _, _, prediction, probability = fit_and_score(
                    selected['X'], y, train_idx, test_idx, params,
                    seed=SEED + 1000 + outer_id)
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
