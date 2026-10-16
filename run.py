#!/usr/bin/env python3
"""Command-line entry point for lung histogram normative modelling.

Use `extract` to convert histograms to q01-q99 curves, `fit` to train regional
PCA/ST1 models and estimate residual covariance, or `score` to apply saved
models and reconstruct conditional central quantile curves. Fitting uses only
normal_train subjects, with patient-grouped folds for marginal cross-fitting.
Scoring returns regional R4/W2 scores, joint R20 scores, and reconstructions.

Example (run from this directory):
    python run.py score --model models/ctrate --input examples/ctrate --output output/ctrate

Python handles quantile/PCA operations and scoring; gamlss.R handles ST1
fitting and prediction. Outputs must be written to a new or empty directory.
See README.md for dependencies, input formats, and fitting requirements.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from qpca import REGIONS, PROBABILITIES, fit_pca, load_pca, project_quantile_vector
from quantiles import load_inputs
from scores import fit_covariances, residual_cube, calculate_scores

HERE = Path(__file__).resolve().parent


def score_frame(metadata, scores):
    required = ['id', 'age', 'sex', 'KVP', 'manufacturer']
    if not set(required).issubset(metadata):
        raise ValueError(f'Metadata requires {required}')
    if metadata[required].isna().any().any():
        raise ValueError('Missing required metadata')
    for col in ['age', 'KVP']:
        if not np.isfinite(pd.to_numeric(metadata[col], errors='raise')).all():
            raise ValueError(f'Nonfinite {col}')
    parts = []
    for r, region in enumerate(REGIONS):
        for k in range(4):
            part = metadata.copy()
            part['region'], part['component'], part['observed_score'] = region, k + 1, scores[:, r, k]
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def run_r(mode, rscript='Rscript', **kwargs):
    args = [rscript, str(HERE / 'gamlss.R'), mode]
    for k, v in kwargs.items():
        args.extend(['--' + k, str(v)])
    subprocess.run(args, check=True)


def fit(metadata, quantiles, output, rscript='Rscript'):
    """Fit a fresh model using normal_train only; supplied samples cannot fit it."""
    if not {'split', 'patient_id'}.issubset(metadata):
        raise ValueError('Fitting requires split and patient_id metadata')
    if metadata.patient_id.isna().any():
        raise ValueError('Missing patient_id')
    if metadata.groupby('patient_id')['split'].nunique().max() > 1:
        raise ValueError('A patient occurs in multiple splits')
    train = metadata.split.eq('normal_train').to_numpy()
    metadata, quantiles = metadata.loc[train].reset_index(drop=True), quantiles[train]
    if metadata.patient_id.nunique() < 5:
        raise ValueError('Five-fold fitting requires at least five healthy training patients')
    folds = np.empty(len(metadata), dtype=int)
    for k, (_, held_out) in enumerate(GroupKFold(5).split(metadata, groups=metadata.patient_id)):
        folds[held_out] = k
    metadata = metadata.copy()
    metadata['fold'] = folds
    fit_pca(quantiles, output)
    frame = score_frame(metadata, project_quantile_vector(load_pca(output), quantiles))
    # Private training intermediates are temporary and never saved into models/.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frame.to_csv(tmp / 'training.csv', index=False)
        run_r('fit', rscript, input=tmp / 'training.csv', output=tmp)
        oof = pd.read_csv(tmp / 'oof_predictions.csv', dtype={'id': str})
        fit_covariances(residual_cube(oof, metadata.id.tolist()), output)
        (tmp / 'gamlss.rds').replace(Path(output) / 'gamlss.rds')


def score(metadata, quantiles, model, output, rscript='Rscript'):
    pca = load_pca(model)
    frame = score_frame(metadata, project_quantile_vector(pca, quantiles))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'scores.csv'
        frame.to_csv(path, index=False)
        run_r('score', rscript, input=path, models=Path(model) / 'gamlss.rds',
              output=Path(output) / 'component_predictions.csv')
    predictions = pd.read_csv(Path(output) / 'component_predictions.csv', dtype={'id': str})
    calculate_scores(metadata, quantiles, pca, predictions, model, output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['extract', 'fit', 'score'])
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--model', type=Path)
    p.add_argument('--histograms', action='store_true', help='Derive quantiles from histogram bins')
    p.add_argument('--rscript', default='Rscript')
    args = p.parse_args()
    if args.command == 'score' and args.model is None:
        p.error('score requires --model')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Output directory must be empty (prevents replacing models or examples)')
    metadata, quantiles = load_inputs(args.input, histogram=args.histograms or args.command == 'extract')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'extract':
        metadata.to_csv(args.output / 'metadata.csv', index=False)
        np.savez_compressed(args.output / 'quantiles.npz', ids=metadata.id.to_numpy(str),
                            regions=np.array(REGIONS), probabilities=PROBABILITIES, quantiles=quantiles)
    elif args.command == 'fit':
        fit(metadata, quantiles, args.output, args.rscript)
    else:
        score(metadata, quantiles, args.model, args.output, args.rscript)
    print(f'{args.command} complete: {args.output}')


if __name__ == '__main__':
    main()
