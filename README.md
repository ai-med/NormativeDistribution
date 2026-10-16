# NormativeDistribution

Normative modelling of distributions using **Wasserstein Quantile PCA
(W-qPCA)** and GAMLSS. W-qPCA represents each distribution through its quantile
function and embeds it in a low-dimensional space. Conditional models of the
embedding describe expected variation with demographic and acquisition
covariates, supporting abnormality scoring and reconstruction of expected
distributions.

The approach is general and can be applied to distributions from different
anatomical regions or measurement domains. The provided applications are
noncontrast and contrast-enhanced lung models, with five held-out examples
from CT-RATE and five from INSPECT.

## Install

Python 3.9+ and R with `gamlss` and `gamlss.dist` are required:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Rscript -e 'install.packages(c("gamlss", "gamlss.dist"), repos="https://cloud.r-project.org")'
```

Run the following commands from this directory. Outputs must be new or empty
directories; existing outputs are never silently overwritten.

## Method

Each distribution is represented by its quantiles at probabilities 0.01 through
0.99. W-qPCA centres the quantile functions using the healthy reference mean,
without standardizing individual quantile coordinates. On this equally spaced
probability grid, root mean squared differences between quantile vectors
approximate Wasserstein-2 distances between distributions. The retained components provide
an embedding that can also be inverted to reconstruct quantile functions.

ST1 GAMLSS models describe the conditional distribution of each W-qPCA
coordinate. The supplied models use:

```text
mu    ~ age + sex + KVP + manufacturer + site
sigma ~ site
nu    ~ 1
tau   ~ 1
```

PIT probabilities are clipped to [1e-12, 1-1e-12] before the standard-normal
inverse CDF. Zero-centred Ledoit-Wolf covariance is estimated from five-fold
out-of-fold healthy-training residuals. Abnormality scores combine these
residuals as `z.T @ precision @ z`, without subtracting an empirical residual
mean or normalizing the covariance diagonal to one.

Expected distributions are reconstructed by mapping the conditional ST1
coordinate medians through inverse W-qPCA. PAVA repairs crossings in the
reconstructed quantile curves. The distributional discrepancy W2 is computed
as `sqrt(mean((observed - reconstructed)**2))` on the 99 central quantiles.
This approximates Wasserstein-2 distance over q01-q99 rather than integrating
over the full probability range. Conditional medians are calculated with R's
`qST1()`.

## Provided models and examples

The repository includes two fitted applications:

| Application | Model directory | Examples |
| --- | --- | --- |
| Noncontrast lung model | `models/ctrate/` | Five CT-RATE subjects in `examples/ctrate/` |
| Contrast-enhanced lung model | `models/inspect/` | Five INSPECT subjects in `examples/inspect/` |

Each application uses four W-qPCA components per lung lobe. R4 scores use a
separately estimated 4-by-4 covariance for each lobe. R20 uses one 20-by-20
covariance across all five lobes. Choose the corresponding model for each
example set and acquisition domain.

The current scripts and saved artifacts are configured for this five-region,
four-component application. Applying the method to other regions or domains
requires adapting the region configuration and array dimensions, then fitting
suitable reference models.

Run both example sets:

```bash
python run.py score --model models/ctrate --input examples/ctrate --output output/ctrate
python run.py score --model models/inspect --input examples/inspect --output output/inspect
```

Each command scores five subjects and all five lobes. Outputs are:

- `component_predictions.csv`: observed W-qPCA coordinates, ST1 parameters, PIT,
  normal residuals, and conditional coordinate medians. Parameters are in the
  model's scaled coordinate units; `observed_score` and `median` are in original
  W-qPCA units.
- `regional_scores.csv`: R4 quadratic scores and central-grid W2 per lobe,
  plus an indicator that the reconstructed curve required monotonic repair.
- `subject_scores.csv`: the joint R20 quadratic score.
- `reconstruction.npz`: conditional central quantile curves, IDs, lobe order,
  and the probability grid.

### Starting from binned distributions

The examples provide both quantile functions and binned HU distributions.
The sample quantiles were extracted directly from CT voxels before binning.
To derive quantiles from the supplied binned distributions instead:

```bash
python run.py extract --input examples/ctrate --output output/ctrate_binned_quantiles
python run.py score --model models/ctrate --input output/ctrate_binned_quantiles --output output/ctrate_binned_scores
```

Alternatively pass `--histograms` to `score`. This option reads the binned
representation stored in `histograms.npz`. Quantiles derived from bin counts
can differ from direct voxel quantiles and consequently change the scores.
The provided distributions use nearest-integer HU bin centres from -1024
through 1024 inclusive, with separate underflow/overflow counts. Extraction
fails if a requested quantile needs unknown tail values.

## Fitted artifacts

`models/{ctrate,inspect}/` contains:

- `pca.npz`: W-qPCA population means, loadings, probability grid, and lobe order,
  supporting both embedding and reconstruction.
- `gamlss.rds`: a list of coefficients, formula strings, factor levels,
  contrasts, links, and scaling constants.
- `covariance.npz`: joint and regional covariance/precision matrices and lobe order.

These files contain no individual training rows, responses, fitted values,
residuals, training IDs, calls, or attached training environments. They are
prediction parameters, not a formal guarantee of anonymization. Calibration
residuals are omitted; the code reports raw scores, without empirical p-values
or diagnostic thresholds. Unknown factor levels fail explicitly.

## Input format

An input directory contains `metadata.csv` and `quantiles.npz`, or
`histograms.npz` for binned distributions. Both example sets demonstrate the
complete schema. All NPZ arrays load with `allow_pickle=False`.

`metadata.csv` has one row per scan: `id`, `age` (years), `sex`, `KVP`,
`manufacturer`, and `site` for pooled models. Factor strings must match the
trained model exactly (including case). The supplied examples retain the
covariates required for predictions. IDs are new example labels; no source-ID
mapping, scan paths, patient IDs, dates, or disease labels are included.
These are real patient-derived distributions and covariates, not synthetic
or guaranteed anonymous records. Both example sets come from held-out
`normal_test` subjects. They are not representative evaluation cohorts and
should not be used to estimate clinical performance.

The supplied application uses this fixed region order:

```text
left_upper_lobe, left_lower_lobe, right_upper_lobe,
right_middle_lobe, right_lower_lobe
```

`quantiles.npz` keys: `ids` (N strings), `regions` (5 strings), `probabilities`
(99 floats, 0.01 through 0.99), `quantiles` (N x 5 x 99 floats).

`histograms.npz` keys: `ids`, `regions`, `hu` (B increasing bin centres),
`counts` (N x 5 x B nonnegative integer counts), `below`, `above` (N x 5
integer tail counts). Array IDs must match metadata row order exactly.

For already aligned CT and TotalSegmentator label maps, use
`quantiles.quantiles_from_nifti(image_path, segmentation_path, region_labels)`;
`region_labels` maps each region name to its integer label list.
This adapter checks geometry and extracts unmodified finite voxel HU quantiles.
No CT scans or segmentation software are included. The command-line `extract`
operation accepts binned distribution inputs.

## Fit on your own reference data

The included ten examples are for inference and cannot support fitting.
Prepare the same input schema with additional metadata columns `patient_id`
and `split`. Healthy reference training subjects must have `split=normal_train`.
Use a sufficiently large reference cohort with all categorical levels supported
in each of five patient-grouped folds. Repeated scans of a patient must remain
in the same split. Other splits are excluded from fitting.

```bash
python run.py fit --input /path/to/reference --output output/my_model
python run.py score --model output/my_model --input /path/to/cases --output output/my_scores
```

The supplied fitter retains four W-qPCA components per region, fits full-data
ST1 marginals, and refits them in five patient-grouped folds to estimate
residual dependence. The W-qPCA basis is fixed across marginal cross-fitting.
Constant factor terms are omitted for single-level training data; sigma uses
site only when multiple sites are present. W-qPCA coordinates are scaled by
their training SD for numerical fitting; the saved scaling is reversed during
reconstruction. Failed convergence or unsupported fold factor levels stop
the fit explicitly.

Fitting uses temporary private intermediates that are removed on exit. The
resulting model directory contains parameter artifacts only. No calibration,
hyperparameter search, cohort filtering, or disease evaluation is performed.

## License

The code is licensed under the Apache License, Version 2.0; see `LICENSE`.
This code license does not establish redistribution rights for the underlying
source datasets or override their applicable terms.
