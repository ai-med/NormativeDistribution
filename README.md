# Lung histogram normative modelling

A compact inference and fitting prototype accompanying the article. It includes
the fitted CT-RATE+TUM noncontrast and INSPECT+TUM contrast-enhanced lung models,
and five held-out examples from each source cohort. It demonstrates the method;
it does not reproduce the article's experiments or performance estimates.

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

## Run the ten examples

```bash
python run.py score --model models/ctrate --input examples/ctrate --output output/ctrate
python run.py score --model models/inspect --input examples/inspect --output output/inspect
python verify.py
```

Each command scores five subjects and all five lobes. Outputs are:

- `component_predictions.csv`: observed PC scores, ST1 parameters, PIT, normal
  residuals, and conditional component medians. Parameters are in the model's
  scaled score units; `observed_score` and `median` are in original PCA units.
- `regional_scores.csv`: R4 quadratic scores and central-grid W2 per lobe,
  plus an indicator that the reconstructed curve required monotonic repair.
- `subject_scores.csv`: the joint R20 quadratic score.
- `reconstruction.npz`: conditional central quantile curves, IDs, lobe order,
  and the probability grid.

The sample quantiles were extracted from CT voxels before histogram rounding.
To run from the supplied histograms instead:

```bash
python run.py extract --input examples/ctrate --output output/ctrate_hist_quantiles
python run.py score --model models/ctrate --input output/ctrate_hist_quantiles --output output/ctrate_hist_scores
```

Alternatively pass `--histograms` to `score`. Binned-histogram quantiles can
differ from the original voxel quantiles and consequently change the scores.
Histograms use nearest-integer HU bin centres from -1024 through 1024 inclusive,
with separate underflow/overflow counts. Extraction fails if q01 or q99 needs
unknown tail values, rather than inventing values outside the stored range.

## Method and fitted artifacts

Each lobe is represented by q01 through q99. Four PCA components are fitted per
lobe on healthy training subjects, centring the quantiles without scaling them.
The supplied PCA means and loadings support both embedding and reconstruction.

The supplied ST1 GAMLSS models use:

```text
mu    ~ age + sex + KVP + manufacturer + site
sigma ~ site
nu    ~ 1
tau   ~ 1
```

PIT probabilities are clipped to [1e-12, 1-1e-12] before the standard-normal
inverse CDF. Zero-centred Ledoit-Wolf covariance is estimated from five-fold
out-of-fold healthy-training residuals. R4 uses a separately estimated 4-by-4
covariance for each lobe. R20 uses one 20-by-20 covariance across all five lobes.
Scores are `z.T @ precision @ z`, without subtracting an empirical residual mean.
Despite the R4/R20 naming, these release artifacts are covariances, without
normalizing their diagonal to one.

W2 reconstructs the four conditional ST1 medians through inverse PCA, repairs
crossings using PAVA, then computes `sqrt(mean((observed - reconstructed)**2))`
on the 99 central quantiles. It is an approximation over q01-q99, in HU, not a
full-tail Wasserstein integral or the predictive histogram sampling used for
the article's lesion-removal analyses. Medians use R's `qST1`; the research
Python implementation used a numerical inverse-CDF grid. `VALIDATION.md`
records the measured numerical difference.

`models/{ctrate,inspect}/` contains only:

- `pca.npz`: population means, loadings, grid, and lobe order.
- `gamlss.rds`: an ordinary list of coefficients, formula strings, factor levels,
  contrasts, links, and scaling constants. These are not full `gamlss` objects.
- `covariance.npz`: joint and regional covariance/precision matrices and lobe order.

These files contain no individual training rows, responses, fitted values,
residuals, training IDs, calls, or attached training environments. They are
prediction parameters, not a formal guarantee of anonymization. Calibration
residuals are omitted; the prototype reports raw scores, without empirical
p-values or diagnostic thresholds. Unknown factor levels fail explicitly.
The CT-RATE and INSPECT models are specific to their respective acquisition
domains; choose the corresponding model for the included examples.

## Input format

An input directory contains `metadata.csv` and `quantiles.npz`, or
`histograms.npz` when using histogram extraction. The two examples demonstrate
the complete schema. All NPZ arrays load with `allow_pickle=False`.

`metadata.csv` has one row per scan: `id`, `age` (years), `sex`, `KVP`,
`manufacturer`, and `site` for pooled models. Factor strings must match the
trained model exactly (including case). The supplied examples retain the
covariates required for predictions. IDs are new example labels; no source-ID
mapping, scan paths, patient IDs, dates, or disease labels are included.
These are real patient-derived distributions and covariates, not synthetic
or guaranteed anonymous records. Both example sets come from held-out
`normal_test` subjects, not the fitting data. They are not representative
evaluation cohorts and should not be used to estimate clinical performance.

Arrays have this fixed lobe order:

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
`region_labels` maps each of the five names above to its integer label list.
This adapter checks geometry and extracts unmodified finite voxel HU quantiles.
No CT scans, segmentation software, or scanner-specific path discovery are
included. The command-line `extract` operation accepts histogram inputs.

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

The fitter retains four PCs per lobe, fits full-data ST1 marginals, and refits
them in five patient-grouped folds to estimate residual dependence. As in the
release, the PCA basis is fixed across marginal cross-fitting. Constant factor
terms are omitted for single-level training data; sigma uses site only when
multiple sites are present. PCA scores are scaled by their training SD for
numerical fitting; the saved scaling is reversed during reconstruction.
Failed convergence or unsupported fold factor levels stop the fit explicitly.
Fitting uses temporary private intermediates that are removed on exit. The
resulting model directory contains parameter artifacts only. No calibration,
hyperparameter search, cohort filtering, or disease evaluation is performed.

## Code provenance and verification

The implementation is reduced and adapted from the research repository:

| Prototype | Research sources |
| --- | --- |
| `quantiles.py` | `data/quantiles/extract_region_quantiles.py` (voxel extraction); histogram rank adapter added here |
| `qpca.py` | `data/quantiles/fit_quantile_pca.py` (training-only PCA, projection, inverse transform) |
| `gamlss.R` | `models/gamlss/fit_quantile_pca_st1.R`, `score_tum_normative_st1.R`, and `score_staged_pooled_abdominal.R` |
| `scores.py` | `data/audits/evaluate_tum_normative_outputs.py`, `export_pooled_final_evaluation.py`, and `evaluate_pooled_distributional.py` |

The entry point and standalone input adapters are small additions. Research
cohort management, plots, performance analyses, and private data paths have
been removed. `verify.py` reruns the supplied examples against reference outputs;
`VALIDATION.md` records agreement with the original release and the artifact
content checks. Code licensing should be selected by the authors before public
publication; no license to the underlying source datasets is implied.
