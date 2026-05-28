# Cross-Validation Methodology Report

**Project:** TCN-based seizure detection from rodent EEG (UNIQURE / UCD)
**Document:** Stratified mouse-disjoint 5-fold cross-validation design,
training protocol, and test-fold evaluation protocol
**Companion scripts:** `create_5fold_cv_splits.py`,
`train_5fold_cv_multiscaleTCN.py`
**Companion outputs:** `data_splits_5fold_cv.json`, per-fold model
weights and metrics, `cv_summary.{csv,json}`

---

## Abstract

This report documents the design, justification, implementation, and
validation of a stratified mouse-disjoint 5-fold cross-validation (CV)
scheme constructed from a training cohort of 71 rodent subjects, and
the accompanying training and evaluation protocol used to compare
candidate architectures across the five folds. The 71 mice are
partitioned into six mutually exclusive groups: five rotate as held-out
test sets across the five CV folds, and the sixth is a fixed set used
only as the early-stopping criterion during model training. Subjects
are assigned to partitions using `StratifiedKFold` with a composite
stratum defined by the Cartesian product of prevalence-median bins and
volume-median bins, balancing both the ictal-to-non-ictal class
proportion and the absolute number of segments contributed by each
partition. The 10-subject hyperparameter tuning set and the 20-subject
independent test set from the upstream manifest are preserved unchanged
and excluded from CV. For each fold, the MultiScaleTCN architecture is
trained for up to 100 epochs with the HPT-best hyperparameters; the
fixed early-stopping partition supplies the per-epoch validation loss
and macro F1 (macro F1 alone drives early stopping at patience 15);
and the held-out CV partition is evaluated exactly once, with raw
probabilities at a fixed segmentation threshold of 0.5, deliberately
omitting all post-processing. Eight segment-level metrics (accuracy,
precision, recall, specificity, F1, MCC, AUROC, PRAUC) are reported
per fold and aggregated as mean ± standard deviation. `precision`,
`recall`, and `F1` are macro-averaged across the two classes; the
bare column labels are retained for compatibility with downstream
tooling. Reproducibility is
ensured by a fixed random seed (42), recorded software versions, and
a persistent log. We additionally document the limitations inherent
in the design (fold-size variance, training-set shrinkage, seed
sensitivity) and provide a ready-to-cite Methods paragraph.

---

## 1. Introduction

### 1.1 Motivation

Standard random K-fold cross-validation assumes that samples are
independent and identically distributed (i.i.d.). For rodent EEG
recordings, this assumption is violated: segments from the same
animal share subject-specific spectral characteristics arising from
electrode placement, anatomical variation, sleep-wake cycles, and
strain-level neurophysiology. Allowing segments from the same mouse
to appear in both training and validation partitions enables a model
to learn subject-identity features rather than seizure morphology,
producing optimistic and non-generalisable performance estimates
(Saeb et al., 2017; Roberts et al., 2017).

The mouse-disjoint constraint, also termed *subject-disjoint* or
*group-aware* cross-validation, is therefore standard practice in
biomedical machine learning whenever multiple samples are drawn from
a single subject (Esteva et al., 2017; Varma & Simon, 2006).

### 1.2 Project context

This CV scheme sits at the validation stage of a four-phase pipeline:

1. Preprocessing of raw 24-hour EEG recordings into fixed-length
   5-second segments (`preprocessing_binary.py`).
2. Proximity-aware downsampling of the non-ictal majority class,
   retaining hard negatives within +/- 60 s of seizure events and
   sampling 5% of the distant background (`create_balanced_splits.py`).
3. Hyperparameter tuning on a held-out 10-mouse validation set,
   conducted prior to and independently of this CV procedure.
4. Five-fold cross-validation on the remaining 71 training mice for
   model selection between architectural variants (TCN, MultiScaleTCN,
   MultiScaleTCNAttention, TCNTemporalAttention). Final performance
   is reported on a held-out 20-mouse independent test set.

The 5-fold CV described here therefore serves three distinct purposes:
(a) to provide a statistically defensible estimate of inter-subject
generalisation variance; (b) to enable paired statistical comparison
of competing architectures across matched folds; and (c) to detect
overfitting to particular subjects that a single fixed train-val split
might mask.

### 1.3 Design objectives

The CV partitioning scheme is required to satisfy six properties:

1. **Mouse-disjoint.** No mouse appears in more than one partition.
2. **Prevalence-balanced.** Each partition has an ictal-to-non-ictal
   ratio close to the overall cohort average (29.7% ictal).
3. **Volume-balanced.** Each partition contributes a comparable
   absolute number of segments, mitigating the influence of the
   ~70-fold range in per-mouse recording length.
4. **Reproducible.** A fixed random seed produces identical
   assignments across re-runs.
5. **Auditable.** Per-mouse and per-fold statistics are logged to
   diagnostic CSVs and a persistent text log.
6. **Robust.** A documented fallback strategy is in place should the
   joint stratification become infeasible due to sparse strata.

---

## 2. Data

### 2.1 Source manifest

The CV procedure consumes the chronology-enriched binary classification
manifest produced by `enrich_manifest.py`:

```
/home/people/22206468/scratch/INPUT_DATA/data_splits_outputs/
    data_splits_nonictal_sampled_filtered_enriched.json
```

This manifest is the cumulative product of three upstream filtering
stages: (i) initial train/val/test split construction
(`generate_data_splits.py`); (ii) proximity-aware non-ictal
downsampling and amplitude filtering of train segments
(`create_balanced_splits.py`); and (iii) amplitude filtering of val
and test segments (`apply_val_test_filter.py`). The 5-fold CV
procedure introduces no further filtering and inherits all upstream
quality-control decisions.

### 2.2 Cohort description

The training partition comprises 71 unique rodent subjects, denoted
`m2`, `m4`, `m7`, ..., `m375`. Subject m254 was excluded upstream
because it contained no annotated ictal segments and could not
participate in any stratified procedure. Across the 71 subjects, the
training corpus contains:

| Metric | Value |
|---|---|
| Total segments | 220,653 |
| Ictal segments | 65,510 |
| Non-ictal segments | 155,143 |
| Overall ictal prevalence | 29.69 % |
| Subjects | 71 |
| Per-subject prevalence range | 6.56 % (m235) - 45.32 % (m282) |
| Per-subject segment count range | 162 (m285) - 12,754 (m294) |

The hyperparameter tuning set (10 subjects) and independent test set
(20 subjects) are disjoint from the 71 training subjects and from
each other. They are preserved in the CV manifest under the
`val_holdout` and `test` keys for downstream evaluation scripts but
play no role in fold construction.

### 2.3 Joint distribution of stratification variables

Within the 71 training subjects, ictal prevalence and total segment
volume exhibit a positive correlation. The smallest-volume subjects
(m235, m303, m359, m265) cluster among the lowest-prevalence subjects
(6-10%), and the largest-volume subjects (m294, m321, m271, m345)
fall in the mid-to-high prevalence range (25-37%). This correlation
informs the choice of stratification design described in Section 3.2.

---

## 3. Methods

### 3.1 Cross-validation design

The 71 training subjects are partitioned into six mutually exclusive
groups using a single application of `sklearn.model_selection.
StratifiedKFold` with `n_splits = 6`, `shuffle = True`, and
`random_state = 42`. The six groups are labelled `fold_0` through
`fold_4` and `early_stop`. For each of the five CV folds *k*:

- The **held-out test set** comprises the subjects in `fold_k`
  (approximately 11-12 subjects). In the manifest these subjects are
  stored under `fold_definitions[k].test_mice` (schema v2; renamed
  from `val_mice` in v1).
- The **training set** comprises the subjects in the four CV folds
  other than *k* (approximately 47-48 subjects).
- The **early-stopping criterion** is evaluated on the subjects in
  `early_stop` (approximately 11-12 subjects), identically across
  all five folds.

The separation of the CV test partition from the early-stopping set
is intentional. Standard K-fold CV uses the same held-out partition
both for early stopping and for reporting the fold metric, which can
introduce subtle selection bias because the stopping decision adapts
to the partition that subsequently scores the model. By holding the
early-stopping set fixed and disjoint from every CV test partition,
the reported fold-level metrics are computed on truly unseen subjects
that influenced no training decision.

### 3.2 Stratification strategy

Each subject is assigned to a stratum defined as the Cartesian
product of two binary bins:

- **Prevalence bin:** `low` if the subject's ictal prevalence is at
  or below the cohort median (approximately 27.7%), `high` otherwise.
- **Volume bin:** `small` if the subject's total segment count is at
  or below the cohort median, `big` otherwise.

The resulting four strata (`low_small`, `low_big`, `high_small`,
`high_big`) capture the joint distribution of class balance and
sample volume across subjects. `StratifiedKFold` then balances the
proportion of each stratum across the six output partitions, ensuring
that no partition is disproportionately composed of low-prevalence or
small-volume subjects.

The choice of a 2x2 crossed scheme, rather than higher-resolution
quartile or tertile bins on either axis, was made for two reasons.
First, `StratifiedKFold` requires each stratum to contain at least
`n_splits` members; with 71 subjects and `n_splits = 6`, only the
2x2 scheme reliably satisfies this constraint after accounting for
the prevalence-volume correlation noted in Section 2.3. Second, a
coarser binning is more robust to small perturbations in the cohort
(e.g., the addition or removal of a single subject), avoiding
fragile assignments at the bin boundaries.

### 3.3 Fallback mechanism

In the unlikely event that one or more of the four 2x2 cells falls
below the `MIN_CELL_SIZE = 6` threshold required by `StratifiedKFold`,
the script automatically falls back to a one-dimensional stratification
using prevalence tertiles (low / mid / high). The fallback condition
is logged as a `WARNING` and recorded in the output manifest under
`metadata.stratification.fallback_used = true`, ensuring full
transparency in downstream reporting.

### 3.4 Subject-to-fold assignment

The first five groups returned by `StratifiedKFold.split()` are
labelled `fold_0` through `fold_4`; the sixth group is labelled
`early_stop`. This labelling is deterministic given the seed.
Because `StratifiedKFold` constructs disjoint test indices that
together cover the full dataset exactly once, the six partitions are
guaranteed mutually exclusive and collectively exhaustive over the
71 subjects.

### 3.5 Output schema

The CV manifest is written to:

```
/home/people/22206468/scratch/INPUT_CV_PROJECT/manifest/
    data_splits_5fold_cv.json
```

It uses a single flat records list combined with per-fold mouse-ID
membership definitions, rather than duplicating training segments
across folds. This reduces output size by approximately a factor of
five and makes the per-fold composition fully transparent. The
top-level keys are:

| Key | Purpose |
|---|---|
| `metadata` | Provenance, seed, stratification parameters, per-partition and per-fold summary statistics |
| `mouse_partitions` | Dictionary `mouse_id -> partition_label` for all 71 train subjects |
| `fold_definitions` | List of 5 dictionaries; each contains `fold`, `train_mice`, `test_mice` (manifest schema v2; renamed from `val_mice` in v1 to match the training-protocol terminology — the held-out CV partition is the per-fold *test* set, not a validation set) |
| `early_stop_mice` | Sorted list of `mouse_id` strings for the fixed early-stop partition |
| `records` | Flat list of all training segment records, each with `filepath`, `label`, `mouse_id`, `filename` |
| `val_holdout` | The 10-subject HPT validation set, passed through unchanged |
| `test` | The 20-subject independent test set, passed through unchanged |

The downstream training scripts expand the records list to fold-specific
(filepath, label) pairs by filtering on `mouse_id` membership.

### 3.6 Diagnostic outputs

In addition to the manifest, the script writes four diagnostic files
under `INPUT_CV_PROJECT/diagnostics/`:

- `prevalence_volume_bins.csv` - one row per stratum, listing the
  number of subjects and the mean, minimum, and maximum prevalence
  and volume within the stratum.
- `fold_mouse_assignment.csv` - one row per subject, listing the
  per-subject ictal and non-ictal counts, prevalence, both binning
  variables, the composite stratum, and the assigned partition.
- `fold_mouse_assignment.md` - the same data as the CSV, rendered as
  one markdown sub-table per partition (`early_stop` first, then
  `fold_0..fold_4`), each preceded by a one-line summary of mouse
  count, segment count, and ictal prevalence. Intended for direct
  paste into this report or a manuscript supplement.
- `fold_stratification_report.csv` - two-section summary; the first
  section contains the per-partition aggregate (mouse count, ictal
  count, prevalence) for all six partitions; the second contains the
  per-fold train and test aggregates for the five CV folds, with
  columns `n_train_mice`, `n_test_mice`, `n_train_segments`,
  `n_test_segments`, `n_train_ictal`, `n_test_ictal`,
  `train_prevalence_pct`, and `test_prevalence_pct` (manifest
  schema v2; the `n_val_*` and `val_prevalence_pct` columns of v1
  are renamed accordingly).

These files are intended both for in-line inspection during the
project and for direct inclusion as supplementary tables in the
manuscript.

---

## 4. Training Protocol

### 4.0 Terminology note

The held-out CV partition of each fold is called the **test fold** or
**held-out test partition** throughout this document and in the
training script. It is touched exactly once per fold, after the
training loop has terminated, and never used to drive any
model-selection decision.

Manifest schema v2 (the version produced by the current
`create_5fold_cv_splits.py`) names the corresponding dict key
`fold_definitions[k].test_mice` and reports per-fold prevalences
under `test_prevalence_pct`. Earlier prose in Section 3 refers to
this same partition as the **validation set** in places where the
discussion is about the split-construction algorithm
(`StratifiedKFold` calls it the test set of a fold, and earlier
drafts of the manifest used the key `val_mice`); the two names
denote the same set of subjects. The upstream `val_holdout` key
(the 10-subject HPT cohort) is a completely separate set and is
never aliased.

### 4.1 Inherited protocol

The training script (`train_5fold_cv_multiscaleTCN.py`) reuses the
training protocol from the parent project (`DL_WITH_SSL_GA`, deployed
as `TCN_SSL_GA` on the cluster) verbatim, importing the model class,
the per-epoch training loop, the DataLoader factory, and the
random-seed helper from `tcn_utils.py`. No architectural or
optimisation parameter is tuned during CV; the hyperparameters are
loaded once from `best_multiscale_params.json`, the JSON artefact
produced by the upstream Optuna study (`tune_multiscale_tcn.py`).

| Component | Setting |
|---|---|
| Model | `MultiScaleTCN(num_filters, kernel_size, dropout, branch1_dilations=[1, 2, 4], branch2_dilations=[8, 16, 32], branch3_dilations=[32, 64, 128], fusion)` with HPT values for `num_filters`, `kernel_size`, `dropout`, `fusion` |
| Optimiser | `AdamW(lr, weight_decay)` with HPT values |
| Scheduler | `CosineAnnealingLR(T_max = 100, eta_min = lr * 0.01)` |
| Loss | `BCEWithLogitsLoss()` (unweighted; the input manifest is already proximity-aware non-ictal-downsampled by `create_balanced_splits.py`) |
| Mixed precision | `torch.amp` autocast forward + `GradScaler` on CUDA |
| Gradient clipping | `clip_grad_norm_(max_norm = 1.0)` |
| Batch size | from HPT |
| Maximum epochs | 100 |
| Early stopping | patience = 15, **no warm-up** (may fire from epoch 1) |
| Random seed | 42 (re-set at the start of every fold) |
| Device | CUDA when available, else CPU |

### 4.2 Per-epoch validation monitor

In every epoch the entire `early_stop_mice` partition (the fixed sixth
partition, identical across all folds) is passed through the model in
inference mode. Two quantities are computed:

- **`val_loss`** — mean per-batch `BCEWithLogitsLoss`. Logged for
  diagnostics; not used for any selection decision.
- **`val_macro_F1`** — `sklearn.metrics.f1_score(..., average="macro",
  zero_division=0)` of the binarised predictions at fixed threshold
  0.5. This is the sole signal driving early stopping and best-weight
  retention.

The decoupling of `val_loss` (logged) from `val_macro_F1` (selection)
follows the parent project's convention and ensures the documented
selection criterion is exactly one quantity.

### 4.3 Why a fixed early-stopping partition

A conventional K-fold protocol uses the same held-out partition both
for early stopping and for reporting the fold metric. This couples the
stopping decision to the partition that subsequently scores the model,
introducing a subtle optimistic bias because the network weights are
implicitly tuned to the held-out distribution. The fixed
`early_stop_mice` partition is disjoint from every CV test partition
*and* from each fold's training set, so the reported per-fold test
metrics are computed on subjects that influenced no training decision.

### 4.4 Why no warm-up

The parent training script gates early stopping behind a 40-epoch
warm-up (`MIN_EPOCHS_BEFORE_ES = 40`), which suits a long single
training run where the optimiser may temporarily fall into and out of
suboptimal basins early on. The CV protocol prioritises wall-clock
budget across five separate trainings and reports a generalisation
estimate rather than a single best model, so early stopping is allowed
to fire from epoch 1. This trades a modest risk of premature stopping
against substantial compute savings on folds that converge quickly.

### 4.5 Best-weight retention and restoration

At every epoch in which `val_macro_F1` strictly exceeds the running
best, the full `state_dict` of the model is detached, copied to host
memory, and atomically written to `fold{k}/best_weights.pt`.
Immediately after the training loop terminates (either by early stop
or by reaching epoch 100), the in-memory best weights are restored
into the live model object prior to test-fold evaluation. This
guarantees that the test-fold evaluation uses the same parameter
configuration that produced the best monitor metric, not the last
training epoch.

---

## 5. Test-Fold Evaluation Protocol

### 5.1 Single evaluation per fold

After the training loop terminates, the checkpoint at the epoch of
highest `val_macro_F1` is restored and the model is evaluated **exactly
once** on the held-out CV partition (`fold_definitions[k].test_mice`).
No further parameter is selected, no threshold is swept, and the test
set is not consulted again. This single-shot constraint is the
mechanism by which the per-fold test metric earns its interpretation
as a generalisation estimate rather than a tuning signal.

### 5.2 Raw predictions only; no post-processing

The test-fold evaluation deliberately omits the post-processing
pipeline used in the parent project's `evaluate_event_level` function.
Specifically:

- **No probability smoothing.** Predictions are read directly from
  `sigmoid(logits)` segment-by-segment. The parent project applies a
  three-segment uniform moving average prior to event extraction; that
  smoothing window is a hyperparameter and would, if tuned against
  per-fold test metrics, contaminate the generalisation estimate.
- **No event-level metrics.** Event extraction, refractory-period
  merging, and minimum-event-duration filtering each introduce
  hyperparameters (`SMOOTHING_WIN = 3`, `REFRACTORY_SEC = 30 s`,
  `MIN_EVENT_SEC = 10 s`) inherited from the parent project that have
  not been re-validated on the CV cohort. Event-level evaluation is
  deferred to the subsequent full-cohort retraining stage, where the
  20-subject independent test set provides an unbiased reference.
- **Fixed segmentation threshold.** A single decision threshold of
  0.5 is applied to the raw probabilities to produce binarised
  predictions. No Youden-J or F1-optimal threshold is computed on the
  test fold (or on any other partition during CV).

This deliberate minimality is the CV protocol's defining choice: it
answers the question *"does the fixed architecture, with fixed HPT
hyperparameters, generalise robustly to held-out subject cohorts?"*
and nothing else.

### 5.3 Reported metrics

For each fold, eight segment-level metrics are computed from the raw
predictions on the held-out CV partition. Three of them
(`precision`, `recall`, and `f1`) are **macro-averaged across the two
classes, not binary positive-class scores**; see the sklearn-call
column for the exact invocation and the *Naming convention* note
below the table for the rationale:

| Metric | sklearn call | Definition |
|---|---|---|
| `accuracy` | `accuracy_score` | `(TP + TN) / (TP + TN + FP + FN)` |
| `precision` | `precision_score(..., average="macro")` | Mean of class-0 and class-1 precision (i.e. mean of negative and positive predictive value) |
| `recall` | `recall_score(..., average="macro")` | Mean of class-0 and class-1 recall (i.e. mean of specificity and sensitivity) |
| `specificity` | `confusion_matrix` | `TN / (TN + FP)`; class-0 (non-ictal) recall, reported separately because the macro-averaged `recall` above does not expose it |
| `f1` | `f1_score(..., average="macro")` | Mean of class-0 and class-1 F1 |
| `mcc` | `matthews_corrcoef` | Matthews correlation coefficient; a single-number summary of the binary confusion matrix that is robust under class imbalance |
| `auroc` | `roc_auc_score` | Area under the receiver-operating-characteristic curve |
| `prauc` | `average_precision_score` | Area under the precision-recall curve (average precision) |

`accuracy`, `precision`, `recall`, `specificity`, `f1`, and `mcc` are
computed at the fixed 0.5 threshold. `auroc` and `prauc` are
threshold-free and are computed from the raw probabilities, providing
a threshold-invariant view of discriminative quality. All threshold-
based scikit-learn metrics are called with `zero_division = 0`.

**Naming convention (important).** `precision`, `recall`, and `f1`
are **macro-averaged, not binary**. They are computed with sklearn's
`average="macro"` argument, i.e. the unweighted mean of the class-0
and class-1 scores; they are **not** the positive-class scores
returned by sklearn's default `average="binary"`. The column labels
deliberately omit the `_macro` suffix so that downstream tooling and
figure scripts can read a stable schema. The convention is documented
here once rather than embedded in every column header, and it applies
uniformly across `cv_summary.csv`, `cv_summary.json`, every
`fold{k}/test_metrics.json`, and the docstrings of
`train_5fold_cv_multiscaleTCN.py`. Any future plotting or
table-rendering code that consumes these files must label its axes
and captions accordingly (e.g. "Precision (macro)") to avoid being
misread as positive-class precision in publications.

### 5.4 Cross-fold aggregation

Each of the eight metrics is reported as **mean ± standard deviation
(sample std, `ddof = 1`)** over the five folds, in addition to the
raw per-fold values. The aggregated values inherit the macro-averaging
convention from §5.3: the `mean` and `std` rows for `precision`,
`recall`, and `f1` are means and standard deviations of the per-fold
*macro* scores, not of the binary positive-class scores. The
aggregation step writes two artefacts:

- `cv_summary.csv` — per-fold metric rows followed by a `mean` row and
  a `std` row, with one column per metric;
- `cv_summary.json` — the same content in nested form, suitable for
  downstream plotting or paired comparison across architectures.

The standard deviation across the five folds is the primary
quantitative summary of inter-subject generalisation variance, and
should be the figure reported when describing the variability of the
CV estimate in the manuscript.

### 5.5 Per-fold artefacts

For each fold *k*, the following are persisted under `fold{k}/`:

| File | Contents |
|---|---|
| `best_weights.pt` | `state_dict` of the model at the epoch of highest `val_macro_F1` |
| `training_history.csv` | Per-epoch `train_loss`, `val_loss`, `val_f1`, `lr` |
| `test_predictions.npz` | `y_true`, `y_prob`, `y_pred`, `test_mice`, `threshold` for the held-out partition |
| `test_metrics.json` | The eight scalar metrics plus `best_epoch`, `best_val_macro_f1`, `test_loss`, segment and mouse counts |
| `train.log` | Per-epoch log lines for this fold |

These artefacts permit re-computation of any per-fold quantity without
re-training, and are used by the downstream architecture-comparison
and plotting scripts.

---

## 6. Validation and Sanity Checks

Five invariants are enforced programmatically before the manifest is
written. Failure of any invariant aborts the script with an
`AssertionError` and a diagnostic message; no partial output is
produced.

### 6.1 Complete assignment

Every one of the 71 training subjects must be assigned to exactly one
of the six partitions. This is verified after `StratifiedKFold.split()`
by counting `NaN` entries in the partition column.

### 6.2 Pairwise disjointness of training partitions

All six partitions are checked for pairwise empty intersection. This
is theoretically guaranteed by the K-fold algorithm but is verified
defensively to catch any future bug introduced by post-processing of
the partition labels.

### 6.3 Disjointness between train cohort and upstream val_holdout / test

The 71 training subjects are checked against the `val_holdout` and
`test` mouse-ID sets from the source manifest to confirm that no
subject crosses the boundary between the CV cohort and the upstream
held-out sets. This mirrors the `run_leakage_check` function in
`generate_data_splits.py`.

### 6.4 Disjointness between upstream val_holdout and test

The upstream `val_holdout` (10 HPT subjects) and `test` (20
independent-test subjects) sets are themselves checked for empty
intersection. The upstream `generate_data_splits.run_leakage_check`
already guarantees this for the source manifest, but re-asserting it
here makes the CV manifest self-contained and catches any regression
introduced upstream between manifest builds.

### 6.5 Empirical validation criteria

The following criteria, evaluated from `fold_stratification_report.csv`
after the first run, are recommended as a post-hoc empirical check:

| Criterion | Acceptable range |
|---|---|
| Per-partition prevalence spread (max - min) | <= 5 percentage points |
| Per-partition volume ratio (max / min) | <= 1.5 |
| Per-partition mouse count | 11 - 13 subjects |
| Fallback flag | `false` |

Deviation from any criterion does not automatically invalidate the
split but should be documented and discussed in the manuscript.

---

### 6.6 Training-protocol invariants

Three additional invariants are enforced at training time and on the
contents of `best_multiscale_params.json`.

- The CV manifest must contain a non-empty `early_stop_mice` list and
  exactly five entries in `fold_definitions`, indexed 0 through 4.
- For each fold *k*, the three mouse sets (train, early-stop, test)
  must be pairwise disjoint.
- `best_multiscale_params.json` must contain a `hyperparameters`
  dictionary with all of `num_filters`, `kernel_size`, `dropout`,
  `fusion`, `learning_rate`, `weight_decay`, and `batch_size`; missing
  keys abort training before any GPU work is done.

---

## 7. Reproducibility

### 7.1 Fixed random state

The single source of randomness is the `random_state = 42` argument to
`StratifiedKFold`. Given an unchanged source manifest, repeated
invocation of the script produces byte-identical output JSON. This
seed is recorded in `metadata.seed` and should be reported in the
Methods section of any publication.

### 7.2 Software environment

The script depends on:

- Python 3.10 (cluster environment `torch_v100_py310`)
- `numpy`
- `pandas`
- `scikit-learn`

No GPU is required. The script runs in under one minute on a single
CPU core.

### 7.3 Provenance trail

The output manifest's `metadata` block records the timestamp of
creation, the absolute path of the source manifest, the seed, the
stratification parameters, and the per-partition and per-fold
summary statistics. This is sufficient to reconstruct the procedure
from the manifest alone without consulting source code.

### 7.4 Persistent log

A persistent append-mode log file is written to:

```
/home/people/22206468/scratch/INPUT_CV_PROJECT/logs/
    create_5fold_cv_splits.log
```

The log preserves the history of every invocation, including any
fallback warnings or assertion failures.

---

## 8. Limitations and Caveats

### 8.1 Fold-size variance

Although the 2x2 stratification balances per-partition volume far
better than prevalence-only stratification, residual variance remains.
Because subjects are indivisible and the per-subject volume range
spans nearly two orders of magnitude, the fold that happens to draw
the single largest subject (m294: 12,754 segments) will contain
approximately 12% more validation segments than a fold drawing only
median-volume subjects. The empirical magnitude of this effect is
reported in `fold_stratification_report.csv` and should be presented
in the manuscript so that fold-level metric variation can be
interpreted in light of partition size.

### 8.2 Training-set shrinkage relative to final model

Each CV fold trains on approximately 47 of the 71 available training
subjects (66% of the training cohort). The CV-estimated metric is
therefore a systematically conservative estimate of the performance
of a final model trained on all 71 subjects. This is intrinsic to
all K-fold cross-validation and is consistent with the role of CV as
a model-selection and variance-estimation tool rather than a primary
performance metric (Kohavi, 1995; Varma & Simon, 2006). The held-out
20-subject independent test set is reserved for the final performance
estimate.

### 8.3 Seed sensitivity

With 71 subjects distributed across six partitions, each partition
contains only 11-12 subjects. Re-running the procedure with a
different random seed will move two to three subjects between
partitions and produce slightly different per-fold metrics. A single
fixed seed is reported here; consumers requiring tighter confidence
bounds on the CV estimate may run repeated CV with multiple seeds and
pool the results, at the cost of proportionally more compute.

### 8.4 Stratification on subject-level summaries

The stratification variables (prevalence and volume) are subject-level
summaries computed after upstream proximity-aware downsampling. This
means that the prevalence used for stratification is the prevalence
of the *training corpus* contributed by each subject, not the
prevalence in the underlying raw recording. This is the relevant
quantity for balancing the training-time class distribution but
should be reported as such in the manuscript.

---

## 9. Methods-Section Text for Publication

The following paragraph is suitable for direct inclusion (with minor
adjustment) in the Methods section of a manuscript.

> The 71 training subjects were partitioned into six mutually exclusive
> groups using stratified K-fold cross-validation
> (`sklearn.model_selection.StratifiedKFold`, `n_splits = 6`,
> `shuffle = True`, `random_state = 42`). The stratification target was
> a composite categorical variable defined as the Cartesian product of
> two binary bins: the subject's ictal prevalence dichotomised at the
> cohort median (approximately 27.7%) and the subject's total segment
> count dichotomised at the cohort median. This joint stratification
> balanced both the class distribution and the absolute segment volume
> across the six output groups, mitigating the influence of the
> approximately 70-fold range in per-subject recording length and the
> positive correlation between volume and prevalence observed in the
> cohort. Five of the six groups were rotated as held-out test sets in
> a five-fold cross-validation procedure (training set: approximately
> 47 subjects; held-out test set per fold: approximately 12 subjects).
> The sixth group (approximately 12 subjects) was held out from all
> five folds and used exclusively as the early-stopping criterion
> during training, ensuring that the reported fold-level test metrics
> were computed on subjects that influenced no training decision.
> Subject-disjointness between all six groups, and between the
> cross-validation cohort and the upstream 10-subject
> hyperparameter-tuning set and 20-subject independent test set, was
> verified by exhaustive pairwise intersection.
>
> For each fold, the MultiScaleTCN architecture was trained using the
> hyperparameters previously identified by an Optuna study on the
> 10-subject hyperparameter-tuning set; no further tuning was performed
> during cross-validation. The optimiser was AdamW with the tuned
> learning rate and weight decay, paired with a cosine learning-rate
> schedule (`T_max = 100`, `eta_min = lr * 0.01`). The loss was
> unweighted binary cross-entropy with logits, applied to a training
> manifest already proximity-aware non-ictal-downsampled upstream.
> Mixed-precision training (`torch.amp` autocast plus `GradScaler`) and
> gradient-norm clipping at 1.0 were used on CUDA. Training proceeded
> for up to 100 epochs with an early-stopping patience of 15 epochs on
> the validation macro F1 measured on the fixed early-stopping
> partition; no warm-up period was imposed. The model parameters at
> the epoch of highest validation macro F1 were retained for
> evaluation. The held-out fold partition was then evaluated exactly
> once, with raw sigmoid probabilities and a fixed segmentation
> threshold of 0.5; no probability smoothing, event-level
> post-processing, or threshold sweep was performed. Per-fold
> performance was summarised by eight segment-level metrics
> (accuracy; macro-averaged precision, recall, and F1; specificity;
> Matthews correlation coefficient; AUROC; and average precision)
> and aggregated across the five folds as mean ± sample standard
> deviation. The cross-validation manifest was generated
> once offline by `create_5fold_cv_splits.py` and consumed unchanged
> by all subsequent training runs; per-fold training and evaluation
> were performed by `train_5fold_cv_multiscaleTCN.py`.

---

## 10. Pipeline Position

```
+-------------------------------+
| generate_data_splits.py        |    train + val + test partitions (subject-disjoint)
+--------------+----------------+
               |
               v
+-------------------------------+
| create_balanced_splits.py      |    proximity-aware non-ictal downsampling
+--------------+----------------+
               |
               v
+-------------------------------+
| apply_val_test_filter.py       |    val/test amplitude filtering
+--------------+----------------+
               |
               v
+-------------------------------+
| enrich_manifest.py             |    val/test chronology fields
+--------------+----------------+
               |
               v
+-------------------------------+
| create_5fold_cv_splits.py      |    <-- splits design
+--------------+----------------+
               |
               v
+-------------------------------+
| train_5fold_cv_multiscaleTCN.py|    <-- training + evaluation
| (and one analogue per variant) |        protocol (this report)
+--------------+----------------+
               |
               v
+-------------------------------+
| Architecture selection         |    paired comparison of per-fold
| (offline analysis)             |    metrics across architectures
+--------------+----------------+
               |
               v
+-------------------------------+
| Full-cohort retraining +       |    final, single training on all
| event-level evaluation         |    71 mice; 20-subject independent
+-------------------------------+    test set with post-processing
```

---

## 11. References

Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009).
Curriculum learning. *Proceedings of the 26th International
Conference on Machine Learning*, 41-48.

Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M.,
Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification
of skin cancer with deep neural networks. *Nature*, 542(7639), 115-118.

Kohavi, R. (1995). A study of cross-validation and bootstrap for
accuracy estimation and model selection. *Proceedings of the 14th
International Joint Conference on Artificial Intelligence*, 2(12),
1137-1143.

Litt, B., & Echauz, J. (2002). Prediction of epileptic seizures.
*The Lancet Neurology*, 1(1), 22-30.

Luttjohann, A., Fabene, P. F., & van Luijtelaar, G. (2009). A
revised Racine's scale for PTZ-induced seizures in rats.
*Physiology & Behavior*, 98(5), 579-586.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B.,
Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V.,
Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M.,
& Duchesnay, E. (2011). Scikit-learn: Machine learning in Python.
*Journal of Machine Learning Research*, 12, 2825-2830.

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J.,
Guillera-Arroita, G., ... & Dormann, C. F. (2017). Cross-validation
strategies for data with temporal, spatial, hierarchical, or
phylogenetic structure. *Ecography*, 40(8), 913-929.

Roy, S., Kiral-Kornek, I., & Bhattacharya, S. (2019). ChronoNet:
A deep recurrent neural network for abnormal EEG identification.
*Artificial Intelligence in Medicine*, 103, 101789.

Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P.
(2017). The need to approximate the use-case in clinical machine
learning. *GigaScience*, 6(5), 1-9.

Shrivastava, A., Gupta, A., & Girshick, R. (2016). Training
region-based object detectors with online hard example mining.
*Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition*, 761-769.

Varma, S., & Simon, R. (2006). Bias in error estimation when using
cross-validation for model selection. *BMC Bioinformatics*, 7(1), 91.

---

## Appendix A. Output File Inventory

### A.1 Splits-construction outputs (`create_5fold_cv_splits.py`)

| Path | Purpose |
|---|---|
| `INPUT_CV_PROJECT/manifest/data_splits_5fold_cv.json` | CV manifest consumed by all downstream training scripts |
| `INPUT_CV_PROJECT/diagnostics/prevalence_volume_bins.csv` | Per-stratum subject counts and bin statistics |
| `INPUT_CV_PROJECT/diagnostics/fold_mouse_assignment.csv` | Per-subject assignment with binning variables and partition label |
| `INPUT_CV_PROJECT/diagnostics/fold_mouse_assignment.md` | Same content as the CSV but rendered as markdown sub-tables, one per partition (`early_stop` first, then `fold_0..fold_4`), each preceded by a one-line summary of mouse count, segment count, and ictal prevalence. Suitable for direct paste into this report or a manuscript supplement. |
| `INPUT_CV_PROJECT/diagnostics/fold_stratification_report.csv` | Per-partition and per-fold summary statistics |
| `INPUT_CV_PROJECT/logs/create_5fold_cv_splits.log` | Persistent append-mode invocation log |

### A.2 Training + evaluation outputs (per-architecture)

Per-architecture output roots under
`/home/people/22206468/scratch/OUTPUT_CV_PROJECT/`:

| Architecture | Script | Output root |
|---|---|---|
| MultiScaleTCN (M3) | `train_5fold_cv_multiscaleTCN.py` | `MODEL_3_MTCN/` |
| MultiScaleTCNWithAttention (M4) | `train_5fold_cv_multiscaleTCNwithATTN.py` | `MODEL_4_MTCN_ATTN/` |

The per-fold artefact layout is identical across architectures:

| Path | Purpose |
|---|---|
| `fold{k}/best_weights.pt` | Model `state_dict` at the epoch of highest `val_macro_F1` for fold *k* |
| `fold{k}/training_history.csv` | Per-epoch `train_loss`, `val_loss`, `val_f1`, `lr` for fold *k* |
| `fold{k}/test_predictions.npz` | `y_true`, `y_prob`, `y_pred`, `mouse_id` (per-segment), `test_mice` (unique, numeric-sorted), `threshold` on the held-out CV partition for fold *k* |
| `fold{k}/test_metrics.json` | Eight scalar metrics plus `best_epoch`, `best_val_macro_f1`, `test_loss`, segment and mouse counts for fold *k* |
| `fold{k}/train.log` | Per-epoch log for fold *k* |
| `cv_summary.csv` | Per-fold metric rows + `mean` row + `std` row |
| `cv_summary.json` | Same content as `cv_summary.csv` in nested form |
| `cv_summary.log` | End-to-end run log spanning all folds |

### A.2.1 M4-specific protocol deltas vs M3

The M4 script is otherwise byte-for-byte similar to the M3 script
(same optimiser, scheduler, loss, gradient clipping, AMP regime
during training, early-stop policy, and aggregation). Two deltas:

1. **Two HP JSONs**. M4 loads
   `best_multiscale_params.json` (backbone HPs, held fixed during the
   M4 study) and
   `best_multiscale_attn_params.json` (attention HPs +
   re-tuned `learning_rate` / `weight_decay` / `batch_size`). The
   loader asserts that the attn JSON's embedded
   `backbone_hyperparameters` snapshot agrees with the live backbone
   JSON, and aborts on disagreement to prevent training on a stale
   snapshot.

2. **FP32 forward for the final test-fold pass**. The per-epoch
   early-stop monitor still uses AMP, but the single test-fold pass
   that produces the reported CV metrics runs in pure FP32. This
   mirrors the parent `eval_utils.py` "Layer 3" NaN protection: the
   attention softmax can overflow in FP16 on adversarial inputs, and
   any silent NaN in attention weights would corrupt the
   cross-architecture comparison with M3. Training and per-epoch
   monitoring on AMP, final eval on FP32, is the same regime the
   parent driver `MultiScaleTCNAttention.py` uses.

## Appendix B. Configuration Parameters

### B.1 Splits-construction constants (`create_5fold_cv_splits.py`)

| Constant | Value | Description |
|---|---|---|
| `N_CV_FOLDS` | 5 | Number of cross-validation folds |
| `N_PARTITIONS` | 6 | CV folds plus the fixed early-stopping group |
| `EARLY_STOP_GROUP` | 5 | Index of the partition designated as early-stop |
| `SEED` | 42 | Random seed for `StratifiedKFold.shuffle` |
| `MIN_CELL_SIZE` | 6 | Minimum subjects per stratum (equal to `n_splits`) |
| `N_PREV_BINS` | 2 | Number of prevalence bins (median split) |
| `N_VOL_BINS` | 2 | Number of volume bins (median split) |
| `FALLBACK_N_BINS` | 3 | Number of prevalence tertiles used if 2x2 fails |
| `EXPECTED_N_TRAIN_MICE` | 71 | Documented assumption about cohort size |

### B.2 Training-protocol constants (`train_5fold_cv_multiscaleTCN.py`)

| Constant | Value | Description |
|---|---|---|
| `MODEL_NAME` | `"MultiScaleTCN"` | Architecture under evaluation |
| `N_FOLDS` | 5 | Number of cross-validation folds |
| `MAX_EPOCHS` | 100 | Hard upper bound on training epochs per fold |
| `ES_PATIENCE` | 15 | Epochs without `val_macro_F1` improvement before early stop |
| `GRAD_CLIP` | 1.0 | Maximum gradient L2 norm |
| `THRESHOLD` | 0.5 | Fixed segmentation decision threshold |
| `SEED` | 42 | Random seed re-set at the start of every fold |
