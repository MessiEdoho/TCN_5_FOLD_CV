# Cross-Validation Methodology Report

**Project:** TCN-based seizure detection from rodent EEG (UNIQURE / UCD)
**Document:** Stratified mouse-disjoint 5-fold cross-validation design
**Companion script:** `create_5fold_cv_splits.py`
**Companion output:** `data_splits_5fold_cv.json`

---

## Abstract

This report documents the design, justification, implementation, and
validation of a stratified mouse-disjoint 5-fold cross-validation (CV)
scheme constructed from a training cohort of 71 rodent subjects. The
71 mice are partitioned into six mutually exclusive groups: five rotate
as held-out validation sets across the five CV folds, and the sixth is
a fixed set used only as the early-stopping criterion during model
training. Subjects are assigned to partitions using `StratifiedKFold`
with a composite stratum defined by the Cartesian product of
prevalence-median bins and volume-median bins, balancing both the
ictal-to-non-ictal class proportion and the absolute number of
segments contributed by each partition. The 10-subject hyperparameter
tuning set and the 20-subject independent test set from the upstream
manifest are preserved unchanged and excluded from CV. Reproducibility
is ensured by a fixed random seed (42), recorded software versions,
and a persistent log. We additionally document the limitations
inherent in the design (fold-size variance, training-set shrinkage,
seed sensitivity) and provide a ready-to-cite Methods paragraph.

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

- The **validation set** comprises the subjects in `fold_k`
  (approximately 11-12 subjects).
- The **training set** comprises the subjects in the four CV folds
  other than *k* (approximately 47-48 subjects).
- The **early-stopping criterion** is evaluated on the subjects in
  `early_stop` (approximately 11-12 subjects), identically across
  all five folds.

The separation of the CV validation set from the early-stopping set
is intentional. Standard K-fold CV uses the same held-out partition
both for early stopping and for reporting the fold metric, which can
introduce subtle selection bias because the stopping decision adapts
to the partition that subsequently scores the model. By holding the
early-stopping set fixed and disjoint from every CV val set, the
reported fold-level metrics are computed on truly unseen subjects
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
| `fold_definitions` | List of 5 dictionaries; each contains `fold`, `train_mice`, `val_mice` |
| `early_stop_mice` | Sorted list of `mouse_id` strings for the fixed early-stop partition |
| `records` | Flat list of all training segment records, each with `filepath`, `label`, `mouse_id`, `filename` |
| `val_holdout` | The 10-subject HPT validation set, passed through unchanged |
| `test` | The 20-subject independent test set, passed through unchanged |

The downstream training scripts expand the records list to fold-specific
(filepath, label) pairs by filtering on `mouse_id` membership.

### 3.6 Diagnostic outputs

In addition to the manifest, the script writes three diagnostic CSV
files under `INPUT_CV_PROJECT/diagnostics/`:

- `prevalence_volume_bins.csv` - one row per stratum, listing the
  number of subjects and the mean, minimum, and maximum prevalence
  and volume within the stratum.
- `fold_mouse_assignment.csv` - one row per subject, listing the
  per-subject ictal and non-ictal counts, prevalence, both binning
  variables, the composite stratum, and the assigned partition.
- `fold_stratification_report.csv` - two-section summary; the first
  section contains the per-partition aggregate (mouse count, ictal
  count, prevalence) for all six partitions; the second contains the
  per-fold train and val aggregates for the five CV folds.

These files are intended both for in-line inspection during the
project and for direct inclusion as supplementary tables in the
manuscript.

---

## 4. Validation and Sanity Checks

Three invariants are enforced programmatically before the manifest is
written. Failure of any invariant aborts the script with an
`AssertionError` and a diagnostic message; no partial output is
produced.

### 4.1 Complete assignment

Every one of the 71 training subjects must be assigned to exactly one
of the six partitions. This is verified after `StratifiedKFold.split()`
by counting `NaN` entries in the partition column.

### 4.2 Pairwise disjointness of training partitions

All six partitions are checked for pairwise empty intersection. This
is theoretically guaranteed by the K-fold algorithm but is verified
defensively to catch any future bug introduced by post-processing of
the partition labels.

### 4.3 Disjointness from upstream val and test

The 71 training subjects are checked against the `val_holdout` and
`test` mouse-ID sets from the source manifest to confirm that no
subject crosses the boundary between the CV cohort and the upstream
held-out sets. This mirrors the `run_leakage_check` function in
`generate_data_splits.py`.

### 4.4 Empirical validation criteria

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

## 5. Reproducibility

### 5.1 Fixed random state

The single source of randomness is the `random_state = 42` argument to
`StratifiedKFold`. Given an unchanged source manifest, repeated
invocation of the script produces byte-identical output JSON. This
seed is recorded in `metadata.seed` and should be reported in the
Methods section of any publication.

### 5.2 Software environment

The script depends on:

- Python 3.10 (cluster environment `torch_v100_py310`)
- `numpy`
- `pandas`
- `scikit-learn`

No GPU is required. The script runs in under one minute on a single
CPU core.

### 5.3 Provenance trail

The output manifest's `metadata` block records the timestamp of
creation, the absolute path of the source manifest, the seed, the
stratification parameters, and the per-partition and per-fold
summary statistics. This is sufficient to reconstruct the procedure
from the manifest alone without consulting source code.

### 5.4 Persistent log

A persistent append-mode log file is written to:

```
/home/people/22206468/scratch/INPUT_CV_PROJECT/logs/
    create_5fold_cv_splits.log
```

The log preserves the history of every invocation, including any
fallback warnings or assertion failures.

---

## 6. Limitations and Caveats

### 6.1 Fold-size variance

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

### 6.2 Training-set shrinkage relative to final model

Each CV fold trains on approximately 47 of the 71 available training
subjects (66% of the training cohort). The CV-estimated metric is
therefore a systematically conservative estimate of the performance
of a final model trained on all 71 subjects. This is intrinsic to
all K-fold cross-validation and is consistent with the role of CV as
a model-selection and variance-estimation tool rather than a primary
performance metric (Kohavi, 1995; Varma & Simon, 2006). The held-out
20-subject independent test set is reserved for the final performance
estimate.

### 6.3 Seed sensitivity

With 71 subjects distributed across six partitions, each partition
contains only 11-12 subjects. Re-running the procedure with a
different random seed will move two to three subjects between
partitions and produce slightly different per-fold metrics. A single
fixed seed is reported here; consumers requiring tighter confidence
bounds on the CV estimate may run repeated CV with multiple seeds and
pool the results, at the cost of proportionally more compute.

### 6.4 Stratification on subject-level summaries

The stratification variables (prevalence and volume) are subject-level
summaries computed after upstream proximity-aware downsampling. This
means that the prevalence used for stratification is the prevalence
of the *training corpus* contributed by each subject, not the
prevalence in the underlying raw recording. This is the relevant
quantity for balancing the training-time class distribution but
should be reported as such in the manuscript.

---

## 7. Methods-Section Text for Publication

The following paragraph is suitable for direct inclusion (with minor
adjustment) in the Methods section of a manuscript.

> The 71 training subjects were partitioned into six mutually exclusive
> groups using stratified K-fold cross-validation
> (`sklearn.model_selection.StratifiedKFold`, `n_splits = 6`,
> `shuffle = True`, `random_state = 42`). The stratification target
> was a composite categorical variable defined as the Cartesian
> product of two binary bins: the subject's ictal prevalence
> dichotomised at the cohort median (approximately 27.7%) and the
> subject's total segment count dichotomised at the cohort median.
> This joint stratification balanced both the class distribution and
> the absolute segment volume across the six output groups, mitigating
> the influence of the approximately 70-fold range in per-subject
> recording length and the positive correlation between volume and
> prevalence observed in the cohort. Five of the six groups were
> rotated as held-out validation sets in a conventional five-fold
> cross-validation procedure (training set: approximately 47 subjects;
> validation set: approximately 12 subjects). The sixth group
> (approximately 12 subjects) was held out from all five folds and
> used exclusively as the early-stopping criterion during training,
> ensuring that the reported fold-level metrics were computed on
> subjects that influenced no training decision. Subject-disjointness
> between all six groups, and between the cross-validation cohort and
> the upstream 10-subject hyperparameter-tuning set and 20-subject
> independent test set, was verified by exhaustive pairwise
> intersection. The cross-validation manifest was generated once
> offline by `create_5fold_cv_splits.py` and consumed unchanged by
> all subsequent training runs.

---

## 8. Pipeline Position

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
| create_5fold_cv_splits.py      |    <-- THIS REPORT
| (this script)                  |
+--------------+----------------+
               |
               v
+-------------------------------+
| 5-fold CV training scripts     |    TCN, MultiScaleTCN,
| (one per architectural variant)|    MultiScaleTCNAttention,
+--------------+----------------+    TCNTemporalAttention
               |
               v
+-------------------------------+
| Final evaluation               |    20-subject independent test set
+-------------------------------+
```

---

## 9. References

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

| Path | Purpose |
|---|---|
| `INPUT_CV_PROJECT/manifest/data_splits_5fold_cv.json` | CV manifest consumed by all downstream training scripts |
| `INPUT_CV_PROJECT/diagnostics/prevalence_volume_bins.csv` | Per-stratum subject counts and bin statistics |
| `INPUT_CV_PROJECT/diagnostics/fold_mouse_assignment.csv` | Per-subject assignment with binning variables and partition label |
| `INPUT_CV_PROJECT/diagnostics/fold_stratification_report.csv` | Per-partition and per-fold summary statistics |
| `INPUT_CV_PROJECT/logs/create_5fold_cv_splits.log` | Persistent append-mode invocation log |

## Appendix B. Configuration Parameters

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
