#!/usr/bin/env python3
"""Verify the distributed models and both five-subject example datasets.

Run `python verify.py` from this directory. Checks cover artifact checksums,
parameter-only RDS contents, histogram quantile extraction, and inference
from both quantile and histogram inputs. Predictions, R4/R20 scores, W2, and
reconstructed curves are compared with the supplied reference outputs using
the numerical tolerances recorded in VALIDATION.md.

Scoring runs in fresh R processes and writes to temporary directories that
are removed on exit. Verification requires the same dependencies as inference;
it does not fit models or require access to training data.
"""
from pathlib import Path
import subprocess
import tempfile
import json
import hashlib
import numpy as np
import pandas as pd
from qpca import REGIONS
from quantiles import load_inputs, quantiles_from_histogram
from run import score

HERE = Path(__file__).resolve().parent


def compare(actual, expected, keys, columns, atol):
    a = pd.read_csv(actual).set_index(keys).sort_index()
    b = pd.read_csv(expected).set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise AssertionError('Reference keys differ')
    errors = {}
    for col in columns:
        np.testing.assert_allclose(a[col], b[col], atol=atol, rtol=1e-10)
        errors[col] = float(abs(a[col] - b[col]).max())
    return errors


def main():
    manifest = json.loads((HERE / 'ARTIFACTS.json').read_text())
    for relative, spec in manifest['files'].items():
        data = (HERE / relative).read_bytes()
        if len(data) != spec['bytes'] or hashlib.sha256(data).hexdigest() != spec['sha256']:
            raise AssertionError(f'Artifact checksum mismatch: {relative}')
    # Inspect nested values too: full fitted models and environments are rejected.
    audit = r'''
      args <- commandArgs(TRUE)
      allowed <- c('region','component','scaling','mu_formula','sigma_formula',
                   'levels','contrasts','coefficients','links')
      check <- function(x) {
        if (is.environment(x) || is.language(x) || is.function(x) || isS4(x)) stop('Unsafe object type')
        if (is.list(x)) { for(v in x) check(v) } else if(!is.atomic(x) && !is.null(x)) stop('Unexpected object type')
        if (!is.null(attributes(x))) for(v in attributes(x)) check(v)
      }
      for (p in args) {
        models <- readRDS(p)
        stopifnot(length(models)==20L)
        for (m in models) {
          stopifnot(identical(sort(names(m)),sort(allowed)))
          check(m)
        }
      }
    '''
    subprocess.run(['Rscript', '-e', audit, *[str(HERE / 'models' / n / 'gamlss.rds')
                                            for n in ['ctrate', 'inspect']]], check=True)
    # Histogram adapter must match numpy's linear empirical quantile convention.
    counts = np.array([4, 1, 6, 3])
    centres = np.array([-1000, -800, -700, -500])
    np.testing.assert_allclose(quantiles_from_histogram(counts, centres),
                               np.quantile(np.repeat(centres, counts), np.arange(1, 100) / 100))
    try:
        quantiles_from_histogram(counts, centres, below=100)
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown histogram tails must be rejected')
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in ['ctrate', 'inspect']:
            example = HERE / 'examples' / name
            metadata, q = load_inputs(example)
            assert len(metadata) == 5
            assert set(metadata) == {'id', 'age', 'sex', 'KVP', 'manufacturer', 'site'}
            out = Path(tmp) / name
            out.mkdir()
            score(metadata, q, HERE / 'models' / name, out)
            results[name] = {
                'components': compare(out / 'component_predictions.csv', example / 'reference_components.csv',
                                      ['id','region','component'],
                                      ['observed_score','mu','sigma','nu','tau','pit','z'], 1e-8),
                'subject': compare(out / 'subject_scores.csv', example / 'reference_subject.csv',
                                   ['id'], ['R20'], 1e-8),
                'regional': {
                    **compare(out / 'regional_scores.csv', example / 'reference_regional.csv',
                              ['id','region'], ['R4'], 1e-8),
                    **compare(out / 'regional_scores.csv', example / 'reference_regional.csv',
                              ['id','region'], ['W2'], 1e-4),
                },
            }
            with np.load(out / 'reconstruction.npz') as a, np.load(example / 'reference_reconstruction.npz') as b:
                np.testing.assert_allclose(a['quantiles'], b['quantiles'], atol=1e-4, rtol=0)
                results[name]['reconstruction_max_abs_HU'] = float(abs(a['quantiles'] - b['quantiles']).max())
                assert (np.diff(a['quantiles'], axis=-1) >= 0).all()
            _, hq = load_inputs(example, histogram=True)
            results[name]['histogram_quantile_max_abs_HU'] = float(abs(q - hq).max())
            assert abs(q - hq).max() <= .500001
            hout = Path(tmp) / (name + '_histogram')
            hout.mkdir()
            score(metadata, hq, HERE / 'models' / name, hout)
            for file in ['subject_scores.csv', 'regional_scores.csv']:
                frame = pd.read_csv(hout / file)
                assert np.isfinite(frame.select_dtypes(include='number')).all().all()
    print(json.dumps(results, indent=2))
    print('PASS: both datasets, quantile and histogram inference, release agreement, and parameter-only RDS audit')


if __name__ == '__main__':
    main()
