"""Extract lung HU quantiles and load the prototype's input arrays.

Quantiles at probabilities 0.01 through 0.99 can be computed directly from
finite CT voxels within existing segmentation masks or from histogram counts.
Histogram extraction uses bin centres and accounts for underflow/overflow
counts when finding ranks; unknown tail values cause an explicit error.
Binning can change quantiles compared with direct voxel extraction.

Input directories pair metadata.csv with quantiles.npz or histograms.npz.
Loaded quantiles have shape (subjects, 5 lobes, 99), with array IDs required
to match metadata row order. See README.md for the complete input schema.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from qpca import REGIONS, PROBABILITIES, validate_quantiles


def quantiles_from_voxels(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError('No finite voxels')
    return np.quantile(values, PROBABILITIES)


def quantiles_from_nifti(image_path, segmentation_path, region_labels):
    """Already aligned CT and integer TotalSegmentator label map; no segmentation."""
    import nibabel as nib
    image, mask = nib.load(image_path), nib.load(segmentation_path)
    if image.shape != mask.shape or not np.allclose(image.affine, mask.affine):
        raise ValueError('CT and segmentation must have matching shapes and affines')
    image, mask = image.get_fdata(), mask.get_fdata()
    return np.stack([quantiles_from_voxels(image[np.isin(mask, region_labels[r])])
                     for r in REGIONS])


def quantiles_from_histogram(counts, hu, below=0, above=0):
    """Linear empirical quantiles of bin centres, without expanding voxel counts.

    Under/overflow counts contribute to ranks. Fail if a requested quantile
    needs an unknown tail value. Binning loses within-bin HU information.
    """
    counts, hu = np.asarray(counts), np.asarray(hu, dtype=float)
    if (counts.ndim != 1 or hu.shape != counts.shape or
            not np.isfinite(counts).all() or not np.isfinite(hu).all() or
            (counts < 0).any() or (counts != np.floor(counts)).any() or
            (np.diff(hu) <= 0).any()):
        raise ValueError('Expected nonnegative integer counts and increasing finite HU centres')
    if not np.isfinite([below, above]).all() or min(below, above) < 0 or int(below) != below or int(above) != above:
        raise ValueError('Invalid tail counts')
    total = counts.sum() + below + above
    if total < 1:
        raise ValueError('Empty histogram')
    ranks = (total - 1) * PROBABILITIES
    lo, hi = np.floor(ranks).astype(np.int64), np.ceil(ranks).astype(np.int64)
    if (lo < below).any() or (hi >= total - above).any():
        raise ValueError('Requested quantiles reach unobserved histogram tails')
    cumulative = np.cumsum(counts, dtype=np.int64)
    left = hu[np.searchsorted(cumulative, lo - below, side='right')]
    right = hu[np.searchsorted(cumulative, hi - below, side='right')]
    return left + (ranks - lo) * (right - left)


def load_inputs(directory, histogram=False):
    directory = Path(directory)
    metadata = pd.read_csv(directory / 'metadata.csv', dtype={'id': str})
    if metadata.id.isna().any() or metadata.id.duplicated().any():
        raise ValueError('Metadata requires unique nonmissing IDs')
    filename = 'histograms.npz' if histogram else 'quantiles.npz'
    with np.load(directory / filename, allow_pickle=False) as f:
        if tuple(f['regions']) != REGIONS or list(f['ids'].astype(str)) != metadata.id.tolist():
            raise ValueError('Metadata and array IDs/lobe ordering do not match')
        if histogram:
            counts, below, above = f['counts'], f['below'], f['above']
            if counts.ndim != 3 or counts.shape[:2] != (len(metadata), 5) or below.shape != counts.shape[:2] or above.shape != below.shape:
                raise ValueError('Invalid histogram dimensions')
            values = np.stack([[quantiles_from_histogram(counts[i, r], f['hu'], below[i, r], above[i, r])
                                for r in range(5)] for i in range(len(metadata))])
        else:
            if not np.array_equal(f['probabilities'], PROBABILITIES):
                raise ValueError('Expected q01 through q99')
            values = f['quantiles']
    if len(values) != len(metadata):
        raise ValueError('Metadata and quantile row counts do not match')
    return metadata, validate_quantiles(values)
