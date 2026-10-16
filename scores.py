"""Estimate residual dependence and calculate lung abnormality scores.

R4 combines four PIT-normal residuals within one lobe; R20 combines all
twenty residuals across five lobes. Both are quadratic scores z.T @ P @ z,
where P is a zero-centred Ledoit-Wolf precision matrix estimated from
out-of-fold healthy-training residuals. Covariance diagonals are not
normalized to one, and regional covariances are estimated separately.

W2 compares observed q01-q99 curves with inverse-PCA reconstructions of the
conditional ST1 component medians. PAVA repairs any crossings before computing
the root mean squared quantile difference in HU. This central-grid measure
approximates a distributional distance over the retained probability range.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from qpca import REGIONS, reconstruct_quantile_function


def pava(x):
    """Project a curve onto nondecreasing values using pooled adjacent blocks."""
    levels, counts = [], []
    for z in np.asarray(x, float):
        levels.append(float(z))
        counts.append(1)
        while len(levels) > 1 and levels[-2] > levels[-1]:
            n = counts[-2] + counts[-1]
            levels[-2:] = [(levels[-2] * counts[-2] + levels[-1] * counts[-1]) / n]
            counts[-2:] = [n]
    return np.repeat(np.asarray(levels, float), counts)


def residual_cube(predictions, ids):
    columns = pd.MultiIndex.from_product([REGIONS, range(1, 5)])
    wide = predictions.pivot(index='id', columns=['region', 'component'], values='z')
    return wide.reindex(index=ids, columns=columns).to_numpy(float).reshape(-1, 5, 4)


def fit_covariances(z, model_dir):
    """OOF healthy residuals only; separate complete cases for each estimator."""
    z = np.asarray(z, float)
    if z.ndim != 3 or z.shape[1:] != (5, 4):
        raise ValueError('Expected (subjects, 5, 4) residuals')
    estimates = []
    for x in [z.reshape(-1, 20), *[z[:, r] for r in range(5)]]:
        x = x[np.isfinite(x).all(axis=1)]
        if len(x) < 2:
            raise ValueError('Insufficient complete OOF reference residuals')
        estimates.append(LedoitWolf(assume_centered=True).fit(x))
    np.savez_compressed(Path(model_dir) / 'covariance.npz', regions=np.array(REGIONS),
                        joint_covariance=estimates[0].covariance_,
                        joint_precision=estimates[0].precision_,
                        regional_covariance=np.stack([e.covariance_ for e in estimates[1:]]),
                        regional_precision=np.stack([e.precision_ for e in estimates[1:]]))


def calculate_scores(metadata, quantiles, pca, predictions, model_dir, output):
    ids = metadata.id.tolist()
    z = residual_cube(predictions, ids)
    if not np.isfinite(z).all():
        raise ValueError('Cannot score incomplete/nonfinite residual vectors')
    with np.load(Path(model_dir) / 'covariance.npz', allow_pickle=False) as f:
        if tuple(f['regions']) != REGIONS:
            raise ValueError('Covariance lobe order mismatch')
        regional = np.einsum('nri,rij,nrj->nr', z, f['regional_precision'], z)
        joint = np.einsum('ni,ij,nj->n', z.reshape(-1, 20), f['joint_precision'], z.reshape(-1, 20))
    columns = pd.MultiIndex.from_product([REGIONS, range(1, 5)])
    medians = predictions.pivot(index='id', columns=['region', 'component'], values='median')
    medians = medians.reindex(index=ids, columns=columns).to_numpy(float).reshape(-1, 5, 4)
    raw = reconstruct_quantile_function(pca, medians)
    crossing = (np.diff(raw, axis=-1) < 0).any(axis=-1)
    expected = raw.copy()
    for i, r in zip(*np.where(crossing)):
        expected[i, r] = pava(raw[i, r])
    w2 = np.sqrt(np.square(quantiles - expected).mean(axis=-1))
    if not all(np.isfinite(x).all() for x in [joint, regional, w2, expected]):
        raise ValueError('Nonfinite score or reconstruction')
    output = Path(output)
    pd.DataFrame({'id': ids, 'R20': joint}).to_csv(output / 'subject_scores.csv', index=False)
    pd.DataFrame({'id': np.repeat(ids, 5), 'region': np.tile(REGIONS, len(ids)),
                  'R4': regional.ravel(), 'W2': w2.ravel(),
                  'reconstruction_crossing': crossing.ravel()}).to_csv(output / 'regional_scores.csv', index=False)
    np.savez_compressed(output / 'reconstruction.npz', ids=np.array(ids), regions=np.array(REGIONS),
                        probabilities=pca['probabilities'], quantiles=expected)
