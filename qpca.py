"""Fit, apply, and invert principal component models of lung quantile curves.

Each subject has 99 HU quantiles (q01-q99) for each of five lung lobes.
PCA centres the curves using healthy training means and retains four
components per lobe, without standardizing individual quantile coordinates.
Projection yields arrays of shape (subjects, 5, 4); reconstruction returns
(subjects, 5, 99). Lobe order and probability grid are defined below and saved
with the means and loadings in pca.npz. Crossing reconstructions are repaired
separately when calculating distributional scores.
"""
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

REGIONS = ('left_upper_lobe', 'left_lower_lobe', 'right_upper_lobe',
           'right_middle_lobe', 'right_lower_lobe')
PROBABILITIES = np.arange(1, 100) / 100


def validate_quantiles(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (5, 99):
        raise ValueError('Expected quantiles of shape (subjects, 5 lobes, 99)')
    if not np.isfinite(values).all() or (np.diff(values, axis=-1) < 0).any():
        raise ValueError('Quantiles must be finite and nondecreasing')
    return values


def fit_pca(quantiles, model_dir):
    """Fit on healthy training subjects only; retain four unscaled PCs per lobe."""
    quantiles = validate_quantiles(quantiles)
    if len(quantiles) < 5:
        raise ValueError('At least five training subjects required for four PCs')
    models = [PCA(n_components=4, svd_solver='full').fit(quantiles[:, r])
              for r in range(5)]
    np.savez_compressed(Path(model_dir) / 'pca.npz', regions=np.array(REGIONS),
                        probabilities=PROBABILITIES,
                        training_mean=np.stack([m.mean_ for m in models]),
                        components=np.stack([m.components_ for m in models]))


def load_pca(model_dir):
    with np.load(Path(model_dir) / 'pca.npz', allow_pickle=False) as f:
        model = {k: f[k] for k in f.files}
    if tuple(model['regions']) != REGIONS or not np.array_equal(model['probabilities'], PROBABILITIES):
        raise ValueError('Unsupported lobe ordering or probability grid')
    if model['components'].shape != (5, 4, 99) or model['training_mean'].shape != (5, 99):
        raise ValueError('Invalid PCA dimensions')
    return model


def project_quantile_vector(model, quantiles):
    values = validate_quantiles(quantiles)
    return np.einsum('nri,rki->nrk', values - model['training_mean'], model['components'])


def reconstruct_quantile_function(model, scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 3 or scores.shape[1:] != (5, 4) or not np.isfinite(scores).all():
        raise ValueError('Expected finite scores of shape (subjects, 5, 4)')
    return np.einsum('nrk,rki->nri', scores, model['components']) + model['training_mean']
