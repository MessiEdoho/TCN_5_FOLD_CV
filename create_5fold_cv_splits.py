"""
create_5fold_cv_splits.py
=========================
Builds data_splits_5fold_cv.json -- the 5-fold cross-validation manifest
for the TCN seizure-detection project.

THREE GUARANTEES ENFORCED BY THIS SCRIPT
----------------------------------------
1. MOUSE-DISJOINT. Each of the 71 training mice is assigned to
   exactly one of 6 partitions; no mouse appears in two partitions,
   and no train mouse appears in either the upstream val_holdout
   (10 mice) or test (20 mice) sets. Guaranteed by construction by
   sklearn.StratifiedKFold and re-verified defensively by
   assert_partitions_disjoint() *before* any output is written.
   See partition_mice() and assert_partitions_disjoint().

2. PREVALENCE + VOLUME STRATIFIED. Mice are stratified via a 2x2
   crossed categorical (median split on ictal prevalence x median
   split on segment count). sklearn.StratifiedKFold then balances
   the proportion of the 4 strata across the 6 output partitions,
   so each partition contains a comparable mix of low/high-prevalence
   and small/big-volume mice. This produces partition-level ictal
   percentages that are close (but not exactly equal) to the cohort
   mean. A 1D prevalence-tertile fallback activates automatically
   if any 2x2 cell falls below the n_splits = 6 floor required by
   StratifiedKFold. See assign_crossed_strata() and the
   "Why two stratification axes" note further down this docstring.

3. SEEDED FOR REPRODUCIBILITY. SEED = 42 is the SOLE source of
   randomness; it is passed as random_state to StratifiedKFold(...,
   shuffle=True, random_state=SEED), logged at startup, and
   recorded in the output manifest's metadata.seed block. An
   unchanged source manifest therefore produces byte-identical
   output JSON across re-runs. This seed is to be reported in the
   Methods section of any publication that consumes the CV manifest.

THE 6 PARTITIONS
----------------
    fold_0 .. fold_4   -- rotate as the held-out CV val (= per-fold
                          test) set across the 5 cross-validation runs
    early_stop         -- fixed across all 5 folds, used only for the
                          training-time early-stopping criterion

The 10 HPT val mice and 20 independent test mice from the source
manifest are passed through unchanged (val_holdout, test).

Why two stratification axes
---------------------------
Mouse-level segment counts span ~70x (m294: 12,754 segments vs m235:
183), and prevalence is positively correlated with volume in this
cohort (small mice tend to also be low-prevalence). Stratifying on
prevalence alone produces folds with similar ictal % but very uneven
segment counts; stratifying on volume alone produces equal fold sizes
but uneven class proportions. A crossed 2x2 stratum
(low/high prevalence x small/big volume) balances both axes at once
and keeps every cell comfortably above the n_splits=6 minimum that
sklearn.StratifiedKFold requires.

Pipeline position
-----------------
After  : enrich_manifest.py  -> data_splits_nonictal_sampled_filtered_enriched.json
Before : 5-fold training scripts (TCN.py / MultiScaleTCN.py / etc.,
         in their CV variants), which consume data_splits_5fold_cv.json

Output schema v2 (single flat records list, avoids 5x duplication)
------------------------------------------------------------------
v2 renames the per-fold held-out key from `val_mice` to `test_mice`
to match the training-protocol terminology in [train_5fold_cv_
multiscaleTCN.py] and CV_METHODOLOGY_REPORT.md: the held-out CV
partition is the per-fold TEST set, not a validation set. The
`val_holdout` top-level key still refers to the upstream 10-subject
HPT validation set, which is a distinct cohort.

{
  "metadata": {schema_version: "2", ...},
  "mouse_partitions":  {mouse_id -> "fold_0" | ... | "fold_4" | "early_stop"},
  "fold_definitions":  [{fold, train_mice, test_mice}, ...]  # 5 entries
  "early_stop_mice":   [mouse_id, ...]                       # fixed
  "records":           [train segment records, each with mouse_id]
  "val_holdout":       [upstream HPT val passthrough]
  "test":              [upstream independent test passthrough]
}

Downstream loading pattern (for new CV training scripts):

  import json
  splits = json.load(open(MANIFEST_PATH))
  records         = splits["records"]
  early_stop_mice = set(splits["early_stop_mice"])
  fold_def        = splits["fold_definitions"][fold_idx]
  train_mice      = set(fold_def["train_mice"])
  test_mice       = set(fold_def["test_mice"])

  train_pairs      = [(r["filepath"], r["label"]) for r in records
                      if r["mouse_id"] in train_mice]
  test_pairs       = [(r["filepath"], r["label"]) for r in records
                      if r["mouse_id"] in test_mice]
  early_stop_pairs = [(r["filepath"], r["label"]) for r in records
                      if r["mouse_id"] in early_stop_mice]

Inputs
------
  /home/people/22206468/scratch/INPUT_DATA/data_splits_outputs/
      data_splits_nonictal_sampled_filtered_enriched.json

Outputs (all under /home/people/22206468/scratch/INPUT_CV_PROJECT/)
-------------------------------------------------------------------
  manifest/data_splits_5fold_cv.json
  diagnostics/prevalence_volume_bins.csv
  diagnostics/fold_mouse_assignment.csv
  diagnostics/fold_mouse_assignment.md
  diagnostics/fold_stratification_report.csv
  logs/create_5fold_cv_splits.log

Usage
-----
python create_5fold_cv_splits.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
SOURCE_MANIFEST = Path(
    "/home/people/22206468/scratch/INPUT_DATA/data_splits_outputs/"
    "data_splits_nonictal_sampled_filtered_enriched.json"
)
OUTPUT_BASE  = Path("/home/people/22206468/scratch/INPUT_CV_PROJECT")
MANIFEST_DIR = OUTPUT_BASE / "manifest"
DIAG_DIR     = OUTPUT_BASE / "diagnostics"
LOGS_DIR     = OUTPUT_BASE / "logs"

OUTPUT_JSON           = MANIFEST_DIR / "data_splits_5fold_cv.json"
BINS_CSV              = DIAG_DIR / "prevalence_volume_bins.csv"
MOUSE_ASSIGN_CSV      = DIAG_DIR / "fold_mouse_assignment.csv"
MOUSE_ASSIGN_MD       = DIAG_DIR / "fold_mouse_assignment.md"
STRATIFICATION_CSV    = DIAG_DIR / "fold_stratification_report.csv"
LOG_PATH              = LOGS_DIR / "create_5fold_cv_splits.log"

# CV design
N_CV_FOLDS       = 5          # number of cross-validation folds
N_PARTITIONS     = 6          # 5 CV folds + 1 fixed early-stop set
EARLY_STOP_GROUP = 5          # which of the 6 StratifiedKFold groups becomes early-stop
SEED             = 42         # fix StratifiedKFold shuffle for reproducibility

# Stratification
MIN_CELL_SIZE     = N_PARTITIONS    # sklearn requires >= n_splits per stratum
N_PREV_BINS       = 2               # low/high prevalence (median split)
N_VOL_BINS        = 2               # small/big volume    (median split)
FALLBACK_N_BINS   = 3               # prevalence tertiles, used if 2x2 fails

# Sanity check
EXPECTED_N_TRAIN_MICE = 71


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_path):
    """Configure a dual-handler logger (stdout + append-mode file).

    Matches the formatting convention used across the sibling repo
    DL_WITH_SSL_GA so log lines are grep-compatible across scripts.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("create_5fold_cv_splits")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Per-mouse aggregation
# ---------------------------------------------------------------------------
def aggregate_per_mouse(train_records, logger):
    """Reduce the train segment list to a per-mouse summary DataFrame.

    Returns
    -------
    pandas.DataFrame indexed by mouse_id with columns:
        n_ictal, n_nonictal, n_total, prevalence_pct
    """
    rows = {}
    for rec in train_records:
        # mouse_id is already attached by build_pairs() in
        # generate_data_splits.py and preserved by enrich_manifest.py
        # for train records. No filename parsing required.
        mid = rec["mouse_id"]
        d = rows.setdefault(mid, {"n_ictal": 0, "n_nonictal": 0})
        if int(rec["label"]) == 1:
            d["n_ictal"] += 1
        else:
            d["n_nonictal"] += 1

    df = pd.DataFrame.from_dict(rows, orient="index")
    df["n_total"] = df["n_ictal"] + df["n_nonictal"]
    df["prevalence_pct"] = 100.0 * df["n_ictal"] / df["n_total"]
    df.index.name = "mouse_id"
    df = df.sort_index()

    logger.info("Per-mouse aggregation: %d unique mice", len(df))
    logger.info("  Total segments     : %d", int(df["n_total"].sum()))
    logger.info("  Total ictal        : %d", int(df["n_ictal"].sum()))
    logger.info("  Total non-ictal    : %d", int(df["n_nonictal"].sum()))
    logger.info("  Overall prevalence : %.2f %%",
                100.0 * df["n_ictal"].sum() / df["n_total"].sum())
    logger.info("  Prevalence range   : %.2f %% .. %.2f %%",
                df["prevalence_pct"].min(), df["prevalence_pct"].max())
    logger.info("  Volume range       : %d .. %d segments",
                int(df["n_total"].min()), int(df["n_total"].max()))
    return df


# ---------------------------------------------------------------------------
# Stratum assignment: 2x2 crossed bins, with graceful fallback
# ---------------------------------------------------------------------------
def assign_crossed_strata(df, logger):
    """Add a `stratum` column to df via 2x2 crossed (prevalence x volume).

    If any of the 4 cells has fewer than MIN_CELL_SIZE mice, fall back
    to a 1D prevalence-tertile stratum so StratifiedKFold remains valid.

    Returns
    -------
    df : DataFrame with new columns: prev_bin, vol_bin, stratum
    info : dict with keys
        binning: "2x2_crossed_median" or "prevalence_tertiles_fallback"
        prev_thresholds, vol_thresholds, cell_counts
        fallback_used (bool), fallback_reason (str or None)
    """
    df = df.copy()
    qcut_failure = None
    cell_counts  = {}
    too_small    = []
    prev_edges   = None
    vol_edges    = None

    # -- Attempt 2x2 crossed binning via the median of each axis ------------
    # pd.qcut with explicit labels raises ValueError if duplicates='drop'
    # collapses a quantile edge (e.g. when many mice tie at the median, so
    # only one unique edge remains for q=2 but two labels were requested).
    # B2: catch that and fall through to the prevalence-tertile branch
    # instead of crashing.
    # B7: capture the actual quantile edges via retbins=True so the
    # metadata records the *real* split point used, not an independently-
    # computed Series.median() that can differ by interpolation rounding.
    try:
        prev_bins, prev_edges = pd.qcut(
            df["prevalence_pct"], q=N_PREV_BINS,
            labels=["low", "high"], duplicates="drop", retbins=True,
        )
        vol_bins, vol_edges = pd.qcut(
            df["n_total"], q=N_VOL_BINS,
            labels=["small", "big"], duplicates="drop", retbins=True,
        )
        df["prev_bin"]  = prev_bins.astype(str)
        df["vol_bin"]   = vol_bins.astype(str)
        df["stratum"]   = df["prev_bin"] + "_" + df["vol_bin"]

        # B1: explicit int() cast so json.dump doesn't choke on numpy.int64
        # returned by Series.value_counts().to_dict() on some pandas versions.
        cell_counts = {str(k): int(v) for k, v in
                       df["stratum"].value_counts().to_dict().items()}
        logger.info("2x2 crossed cell populations:")
        for cell, n in sorted(cell_counts.items()):
            logger.info("  %-12s : %d mice", cell, n)
        too_small = [c for c, n in cell_counts.items() if n < MIN_CELL_SIZE]

    except ValueError as e:
        qcut_failure = str(e)
        logger.warning(
            "2x2 crossed pd.qcut raised ValueError (likely median ties "
            "with duplicates='drop'): %s. Falling back to prevalence-"
            "tertile stratification.", qcut_failure)
        # Force the fallback path; clear any partial state on df so the
        # fallback branch repopulates prev_bin / vol_bin / stratum cleanly.
        too_small = ["<2x2_qcut_ValueError>"]
        for col in ("prev_bin", "vol_bin", "stratum"):
            if col in df.columns:
                df = df.drop(columns=[col])

    if too_small:
        # -- Fall back to 1D prevalence tertiles --------------------------
        # Reason: any cell < n_splits=6 would make StratifiedKFold error
        # out on that stratum. Prevalence-only is the conservative default
        # we agreed on as v1 if the joint binning is infeasible.
        if qcut_failure is None:
            logger.warning(
                "Cell(s) below MIN_CELL_SIZE=%d: %s. "
                "Falling back to prevalence-tertile stratification.",
                MIN_CELL_SIZE, too_small)

        tertile_bins, tertile_edges = pd.qcut(
            df["prevalence_pct"], q=FALLBACK_N_BINS,
            labels=["low", "mid", "high"], duplicates="drop",
            retbins=True,
        )
        df["prev_bin"] = tertile_bins.astype(str)
        df["vol_bin"]  = "n/a"
        df["stratum"]  = df["prev_bin"]

        # B1: same int() cast as the 2x2 branch above.
        cell_counts = {str(k): int(v) for k, v in
                       df["stratum"].value_counts().to_dict().items()}
        logger.info("Fallback prevalence-tertile cell populations:")
        for cell, n in sorted(cell_counts.items()):
            logger.info("  %-12s : %d mice", cell, n)

        # N2: unified key names. Median-based thresholds don't apply in the
        # tertile branch, so prev_threshold / vol_threshold are explicit
        # nulls; the tertile edges themselves are recorded as a sibling
        # field so the actual cuts are still auditable.
        info = {
            "binning":           "prevalence_tertiles_fallback",
            "prev_threshold":    None,
            "vol_threshold":     None,
            "prev_tertile_edges": [float(x) for x in tertile_edges],
            "cell_counts":       cell_counts,
            "fallback_used":     True,
            "fallback_reason":   qcut_failure
                                 or "2x2 crossed produced cell < MIN_CELL_SIZE",
            "small_cells":       too_small,
        }
    else:
        # B7: prev_threshold / vol_threshold are taken from the actual
        # qcut quantile edges (prev_edges[1] is the single mid-edge for
        # q=2), so the recorded value is exactly the split that grouped
        # mice in df["prev_bin"] / df["vol_bin"].
        prev_threshold = float(prev_edges[1])
        vol_threshold  = float(vol_edges[1])
        info = {
            "binning":          "2x2_crossed_median",
            "prev_threshold":   prev_threshold,
            "vol_threshold":    vol_threshold,
            "cell_counts":      cell_counts,
            "fallback_used":    False,
            "fallback_reason":  None,
        }
        logger.info("2x2 crossed binning OK (all cells >= %d).", MIN_CELL_SIZE)
        logger.info("  prev_threshold (qcut split) : %.4f %%", prev_threshold)
        logger.info("  vol_threshold  (qcut split) : %.1f segments", vol_threshold)

    return df, info


# ---------------------------------------------------------------------------
# Partition assignment via StratifiedKFold
# ---------------------------------------------------------------------------
def partition_mice(df, logger):
    """Assign each mouse to one of 6 partitions: fold_0..fold_4, early_stop.

    StratifiedKFold(n_splits=6) produces 6 disjoint test-index groups
    that together cover all mice exactly once. We label the first 5
    as cv_fold groups and the 6th as the fixed early-stop set.

    Returns
    -------
    df : DataFrame with new column `partition`
    """
    df = df.copy()
    skf = StratifiedKFold(n_splits=N_PARTITIONS, shuffle=True, random_state=SEED)

    # X is a dummy index array; only y (the stratum) drives the split.
    X = np.arange(len(df))
    y = df["stratum"].to_numpy()

    partition_labels = np.empty(len(df), dtype=object)
    for group_idx, (_, test_idx) in enumerate(skf.split(X, y)):
        if group_idx == EARLY_STOP_GROUP:
            label = "early_stop"
        else:
            label = "fold_%d" % group_idx
        partition_labels[test_idx] = label

    df["partition"] = partition_labels

    # -- Verify mouse-disjoint -- every mouse has exactly one partition --
    n_unassigned = (df["partition"].isna()).sum()
    assert n_unassigned == 0, (
        "StratifiedKFold left %d mice unassigned" % n_unassigned)

    # -- Verify partition coverage matches N_PARTITIONS -------------------
    partitions_seen = set(df["partition"].unique())
    expected = {"fold_%d" % i for i in range(N_CV_FOLDS)} | {"early_stop"}
    assert partitions_seen == expected, (
        "Partition label mismatch. Got %s, expected %s"
        % (sorted(partitions_seen), sorted(expected)))

    logger.info("Partition assignment complete (seed=%d):", SEED)
    summary = df.groupby("partition").agg(
        n_mice=("partition", "size"),
        n_ictal=("n_ictal", "sum"),
        n_nonictal=("n_nonictal", "sum"),
        n_total=("n_total", "sum"),
    )
    summary["prevalence_pct"] = (
        100.0 * summary["n_ictal"] / summary["n_total"]).round(2)
    # N4: emit one timestamped log line per partition row so the persistent
    # log stays grep-friendly (the previous multi-line DataFrame dump left
    # the body lines without the asctime/level prefix).
    for part, srow in summary.iterrows():
        logger.info(
            "  %-12s : n_mice=%2d  n_ictal=%6d  n_nonictal=%6d  "
            "n_total=%6d  prev=%5.2f%%",
            part, int(srow["n_mice"]), int(srow["n_ictal"]),
            int(srow["n_nonictal"]), int(srow["n_total"]),
            float(srow["prevalence_pct"]),
        )

    return df


# ---------------------------------------------------------------------------
# Disjointness check
# ---------------------------------------------------------------------------
def assert_partitions_disjoint(df, val_holdout, test, logger):
    """Verify the 6 partitions are pairwise disjoint and don't touch
    the upstream val_holdout / test mice. This is the same invariant
    as run_leakage_check() in generate_data_splits.py.
    """
    partition_mice = {}
    for part in sorted(df["partition"].unique()):
        partition_mice[part] = set(df.index[df["partition"] == part])

    parts = list(partition_mice.keys())
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            shared = partition_mice[a] & partition_mice[b]
            assert not shared, (
                "LEAKAGE: mice in BOTH %s and %s: %s" % (a, b, sorted(shared))
            )

    train_mice    = set(df.index)
    holdout_mice  = {r["mouse_id"] for r in val_holdout}
    upstream_test = {r["mouse_id"] for r in test}

    cross_holdout = train_mice & holdout_mice
    cross_test    = train_mice & upstream_test
    # B3: also check val_holdout intersect test. The upstream
    # generate_data_splits.run_leakage_check already guarantees this for
    # the source manifest, but re-asserting here makes the CV manifest
    # self-contained and catches any regression introduced upstream.
    cross_holdout_test = holdout_mice & upstream_test

    assert not cross_holdout, (
        "LEAKAGE: train mice also appear in val_holdout: %s"
        % sorted(cross_holdout))
    assert not cross_test, (
        "LEAKAGE: train mice also appear in test: %s"
        % sorted(cross_test))
    assert not cross_holdout_test, (
        "LEAKAGE: val_holdout mice also appear in test: %s"
        % sorted(cross_holdout_test))

    logger.info(
        "Disjointness check PASS: 6 train partitions pairwise disjoint; "
        "no overlap between train (%d mice), val_holdout (%d mice), or "
        "test (%d mice).",
        len(train_mice), len(holdout_mice), len(upstream_test),
    )


# ---------------------------------------------------------------------------
# Build fold_definitions (train/val mouse-id lists per CV fold)
# ---------------------------------------------------------------------------
def build_fold_definitions(df):
    """Construct the per-fold train_mice / test_mice lists.

    For fold k: test_mice  = mice in partition fold_k (the held-out CV
                             test set for this fold);
                train_mice = mice in any of the other 4 CV folds
                             (early_stop is excluded from train).

    Schema v2: this dict key was named `val_mice` in v1. It is renamed
    to `test_mice` so the manifest matches the training-protocol
    terminology (the held-out CV partition is the per-fold *test* set;
    `val_holdout` denotes a distinct upstream HPT cohort).
    """
    fold_defs = []
    for k in range(N_CV_FOLDS):
        test_mice = sorted(df.index[df["partition"] == "fold_%d" % k])
        # train = the OTHER cv folds; early_stop is NOT in train
        train_mask = (
            df["partition"].str.startswith("fold_")
            & (df["partition"] != "fold_%d" % k)
        )
        train_mice = sorted(df.index[train_mask])
        fold_defs.append({
            "fold":        k,
            "train_mice":  train_mice,
            "test_mice":   test_mice,
        })
    return fold_defs


# ---------------------------------------------------------------------------
# Per-fold summary statistics (for metadata + diagnostic CSV)
# ---------------------------------------------------------------------------
def compute_fold_summary(df, fold_defs):
    rows = []
    for fd in fold_defs:
        train = df.loc[fd["train_mice"]]
        test  = df.loc[fd["test_mice"]]
        rows.append({
            "fold":                  fd["fold"],
            "n_train_mice":          len(train),
            "n_test_mice":           len(test),
            "n_train_segments":      int(train["n_total"].sum()),
            "n_test_segments":       int(test["n_total"].sum()),
            "n_train_ictal":         int(train["n_ictal"].sum()),
            "n_test_ictal":          int(test["n_ictal"].sum()),
            "train_prevalence_pct":  round(
                100.0 * train["n_ictal"].sum() / train["n_total"].sum(), 4),
            "test_prevalence_pct":   round(
                100.0 * test["n_ictal"].sum() / test["n_total"].sum(), 4),
        })
    return rows


def compute_partition_summary(df):
    rows = []
    for part in sorted(df["partition"].unique()):
        sub = df[df["partition"] == part]
        rows.append({
            "partition":          part,
            "n_mice":             len(sub),
            "n_ictal":            int(sub["n_ictal"].sum()),
            "n_nonictal":         int(sub["n_nonictal"].sum()),
            "n_total":            int(sub["n_total"].sum()),
            "prevalence_pct":     round(
                100.0 * sub["n_ictal"].sum() / sub["n_total"].sum(), 4),
            "mean_mouse_volume":  round(float(sub["n_total"].mean()), 1),
            "mean_mouse_prevalence_pct": round(
                float(sub["prevalence_pct"].mean()), 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Diagnostic CSV writers
# ---------------------------------------------------------------------------
def write_diag_csv(path, rows, fieldnames, logger):
    """Write a list-of-dicts to CSV with explicit column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("Wrote diagnostic CSV: %s (%d rows)", path, len(rows))


def write_mouse_assignment_csv(df, logger):
    rows = []
    for mid, row in df.iterrows():
        rows.append({
            "mouse_id":       mid,
            "n_ictal":        int(row["n_ictal"]),
            "n_nonictal":     int(row["n_nonictal"]),
            "n_total":        int(row["n_total"]),
            "prevalence_pct": round(float(row["prevalence_pct"]), 4),
            "prev_bin":       row["prev_bin"],
            "vol_bin":        row["vol_bin"],
            "stratum":        row["stratum"],
            "partition":      row["partition"],
        })
    write_diag_csv(
        MOUSE_ASSIGN_CSV, rows,
        fieldnames=["mouse_id", "n_ictal", "n_nonictal", "n_total",
                    "prevalence_pct", "prev_bin", "vol_bin",
                    "stratum", "partition"],
        logger=logger)


def _mouse_sort_key(mid):
    """Numeric sort: 'm10' after 'm2', not before.

    Falls back to lexicographic order for any id that does not match
    the conventional 'm<digits>' pattern.
    """
    rest = str(mid)[1:] if str(mid).startswith("m") else str(mid)
    try:
        return (0, int(rest))
    except ValueError:
        return (1, str(mid))


# Partition display order: early_stop first (fixed monitor set), then the
# 5 rotating CV folds in numeric order. Used by both the markdown writer
# and the log pretty-printer for consistency.
PARTITION_ORDER = ["early_stop"] + ["fold_%d" % i for i in range(N_CV_FOLDS)]


def write_mouse_assignment_md(df, partition_summary, source_manifest,
                              stratification_info, logger):
    """Markdown rendering of the mouse-to-partition assignment.

    Same data as fold_mouse_assignment.csv, but organised as one
    sub-table per partition (early_stop first, then fold_0..fold_4)
    with a one-line summary above each table. Intended for direct
    paste into CV_METHODOLOGY_REPORT.md or any manuscript supplement.
    """
    MOUSE_ASSIGN_MD.parent.mkdir(parents=True, exist_ok=True)
    summary_by_part = {r["partition"]: r for r in partition_summary}

    lines = []
    lines.append("# Mouse-to-Partition Assignment")
    lines.append("")
    lines.append(
        "Generated by `create_5fold_cv_splits.py` on "
        "%s." % datetime.now().isoformat(timespec="seconds")
    )
    lines.append("")
    lines.append("- **Source manifest**: `%s`" % source_manifest)
    lines.append("- **Seed**: %d (StratifiedKFold `random_state`)" % SEED)
    lines.append(
        "- **Stratification**: %s (fallback used: %s)"
        % (stratification_info["binning"],
           stratification_info["fallback_used"])
    )
    lines.append("- **Partition display order**: `early_stop` first "
                 "(fixed monitor set), then the 5 rotating CV folds.")
    lines.append("")

    for part in PARTITION_ORDER:
        if part not in summary_by_part:
            # B6: a partition missing here means the upstream
            # StratifiedKFold + label loop produced a different set of
            # partition names than PARTITION_ORDER expects. The asserts
            # in partition_mice() should already have caught this --
            # surface the discrepancy loudly rather than dropping the
            # section silently.
            logger.warning(
                "Partition %r expected but not present in "
                "partition_summary; section will be omitted from output.",
                part,
            )
            continue
        summ = summary_by_part[part]
        lines.append("## `%s`" % part)
        lines.append("")
        lines.append(
            "**%d mice** | **%s segments** | **%.2f%% ictal prevalence** "
            "(mean per-mouse prevalence: %.2f%%, mean per-mouse volume: "
            "%.0f segments)"
            % (summ["n_mice"], "{:,}".format(summ["n_total"]),
               summ["prevalence_pct"],
               summ["mean_mouse_prevalence_pct"],
               summ["mean_mouse_volume"])
        )
        lines.append("")
        lines.append(
            "| mouse_id | prevalence_pct | n_ictal | n_nonictal | n_total | stratum |"
        )
        lines.append("|---|---:|---:|---:|---:|---|")

        sub = df[df["partition"] == part]
        mids = sorted(sub.index, key=_mouse_sort_key)
        for mid in mids:
            row = sub.loc[mid]
            lines.append(
                "| `%s` | %.2f | %d | %d | %d | `%s` |"
                % (mid,
                   float(row["prevalence_pct"]),
                   int(row["n_ictal"]),
                   int(row["n_nonictal"]),
                   int(row["n_total"]),
                   row["stratum"])
            )
        lines.append("")

    MOUSE_ASSIGN_MD.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote markdown assignment table: %s", MOUSE_ASSIGN_MD)


def log_mouse_assignment_grouped(df, partition_summary, logger):
    """Compact per-partition per-mouse listing to the log.

    Mirrors the markdown grouping but uses a fixed-width text layout
    suitable for grep / quick eyeballing in the persistent log.
    """
    summary_by_part = {r["partition"]: r for r in partition_summary}

    logger.info("Per-mouse assignment grouped by partition:")
    for part in PARTITION_ORDER:
        if part not in summary_by_part:
            # B6: see write_mouse_assignment_md() for the rationale --
            # warn rather than silently skip.
            logger.warning(
                "Partition %r expected but not present in "
                "partition_summary; section will be omitted from log.",
                part,
            )
            continue
        summ = summary_by_part[part]
        logger.info(
            "  %-12s (%d mice, %s segs, %.2f%% ictal):",
            part, summ["n_mice"],
            "{:,}".format(summ["n_total"]),
            summ["prevalence_pct"],
        )
        sub = df[df["partition"] == part]
        mids = sorted(sub.index, key=_mouse_sort_key)
        for mid in mids:
            row = sub.loc[mid]
            logger.info(
                "      %-6s  prev=%5.2f%%  n_total=%5d  stratum=%s",
                mid,
                float(row["prevalence_pct"]),
                int(row["n_total"]),
                row["stratum"],
            )


def write_bins_csv(df, logger):
    """One row per stratum cell."""
    rows = []
    for stratum, sub in df.groupby("stratum"):
        rows.append({
            "stratum":         stratum,
            "n_mice":          len(sub),
            "mean_prevalence_pct": round(float(sub["prevalence_pct"].mean()), 4),
            "min_prevalence_pct": round(float(sub["prevalence_pct"].min()), 4),
            "max_prevalence_pct": round(float(sub["prevalence_pct"].max()), 4),
            "mean_volume":     round(float(sub["n_total"].mean()), 1),
            "min_volume":      int(sub["n_total"].min()),
            "max_volume":      int(sub["n_total"].max()),
        })
    write_diag_csv(
        BINS_CSV, rows,
        fieldnames=["stratum", "n_mice",
                    "mean_prevalence_pct", "min_prevalence_pct",
                    "max_prevalence_pct",
                    "mean_volume", "min_volume", "max_volume"],
        logger=logger)


def write_stratification_report_csv(partition_summary, fold_summary, logger):
    """Two-section CSV: partition summary (6 rows) + per-fold summary."""
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    with open(STRATIFICATION_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # Section 1: per-partition (the 6 stratified groups)
        w.writerow(["# Partition summary (6 stratified groups)"])
        ps_fields = ["partition", "n_mice", "n_ictal", "n_nonictal",
                     "n_total", "prevalence_pct",
                     "mean_mouse_volume", "mean_mouse_prevalence_pct"]
        w.writerow(ps_fields)
        for r in partition_summary:
            w.writerow([r[k] for k in ps_fields])

        w.writerow([])
        # Section 2: per-fold (5 CV folds; train = the 4 other CV folds,
        # test = the held-out CV fold for that run).
        w.writerow(["# Per-fold summary (train = other 4 CV folds, "
                    "test = held-out CV fold)"])
        fs_fields = ["fold", "n_train_mice", "n_test_mice",
                     "n_train_segments", "n_test_segments",
                     "n_train_ictal", "n_test_ictal",
                     "train_prevalence_pct", "test_prevalence_pct"]
        w.writerow(fs_fields)
        for r in fold_summary:
            w.writerow([r[k] for k in fs_fields])

    logger.info("Wrote stratification report: %s", STRATIFICATION_CSV)


# ---------------------------------------------------------------------------
# Build and write final manifest JSON
# ---------------------------------------------------------------------------
def build_output_manifest(df, fold_defs, partition_summary, fold_summary,
                          stratification_info, train_records,
                          val_holdout, test, source_manifest):
    """Assemble the final JSON dict (single flat `records` list).

    Schema version 2: `fold_definitions[k].test_mice` (was `val_mice` in
    v1); the per-fold summary block uses `n_test_*` and
    `test_prevalence_pct` (was `n_val_*` and `val_prevalence_pct`).
    """
    # N1: emit mouse_partitions in numeric mouse-id order (m2, m4, ...,
    # m10, ..., m375) so the JSON output matches the markdown / log
    # ordering. Python 3.7+ preserves dict insertion order in json.dump.
    raw_partitions = df["partition"].to_dict()
    mouse_partitions = {
        mid: raw_partitions[mid]
        for mid in sorted(raw_partitions, key=_mouse_sort_key)
    }
    early_stop_mice  = sorted(
        df.index[df["partition"] == "early_stop"], key=_mouse_sort_key,
    )

    manifest = {
        "metadata": {
            "created_by":          "create_5fold_cv_splits.py",
            "timestamp":           datetime.now().isoformat(),
            "source_manifest":     str(source_manifest),
            "schema_version":      "2",
            "n_folds":             N_CV_FOLDS,
            "n_partitions":        N_PARTITIONS,
            "seed":                SEED,
            "n_mice_train_total":  len(df),
            "n_mice_val_holdout":  len({r["mouse_id"] for r in val_holdout}),
            "n_mice_test":         len({r["mouse_id"] for r in test}),
            "stratification":      stratification_info,
            "partition_summary":   partition_summary,
            "fold_summary":        fold_summary,
        },
        "mouse_partitions":   mouse_partitions,
        "fold_definitions":   fold_defs,
        "early_stop_mice":    early_stop_mice,
        "records":            train_records,    # flat list; expand via mouse_id
        "val_holdout":        val_holdout,
        "test":               test,
    }
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", type=Path, default=SOURCE_MANIFEST,
                   help="Path to the enriched source manifest JSON.")
    p.add_argument("--output", type=Path, default=OUTPUT_JSON,
                   help="Path to write the 5-fold CV manifest JSON.")
    p.add_argument("--log", type=Path, default=LOG_PATH,
                   help="Path to the persistent log file.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args   = parse_args()
    logger = setup_logging(args.log)

    logger.info("=" * 65)
    logger.info("create_5fold_cv_splits.py")
    logger.info("Timestamp        : %s", datetime.now().isoformat())
    logger.info("Source manifest  : %s", args.source)
    logger.info("Output manifest  : %s", args.output)
    logger.info("Log file         : %s", args.log)
    logger.info("Seed             : %d", SEED)
    logger.info("N folds          : %d (+1 fixed early-stop = %d partitions)",
                N_CV_FOLDS, N_PARTITIONS)
    logger.info("Stratification   : prevalence x volume (2x2 crossed, "
                "median splits; fallback = prevalence tertiles)")
    logger.info("=" * 65)

    # -- 1. Load source manifest ----------------------------------------------
    if not args.source.exists():
        logger.error("Source manifest not found: %s", args.source)
        sys.exit(1)

    splits         = json.loads(args.source.read_text(encoding="utf-8"))
    train_records  = splits.get("train", [])
    val_holdout    = splits.get("val", [])
    test_records   = splits.get("test", [])

    if not train_records:
        logger.error("Source manifest has empty 'train' partition.")
        sys.exit(1)

    logger.info("Loaded source manifest:")
    logger.info("  train       : %d segments", len(train_records))
    logger.info("  val_holdout : %d segments", len(val_holdout))
    logger.info("  test        : %d segments", len(test_records))

    # -- 2. Per-mouse aggregation + sanity check -----------------------------
    df = aggregate_per_mouse(train_records, logger)

    if len(df) != EXPECTED_N_TRAIN_MICE:
        # Not fatal -- log loudly but proceed. The expected count is a
        # documented assumption from the user's project state; if the
        # source manifest changes, we want to see that fact in the log.
        logger.warning(
            "Expected %d train mice but found %d. Proceeding with the "
            "actual count -- verify this matches your current project state.",
            EXPECTED_N_TRAIN_MICE, len(df))

    # -- 3. Assign strata (2x2 crossed, with fallback) -----------------------
    df, stratification_info = assign_crossed_strata(df, logger)

    # -- 4. Partition mice via StratifiedKFold -------------------------------
    df = partition_mice(df, logger)

    # -- 5. Disjointness / leakage check ------------------------------------
    assert_partitions_disjoint(df, val_holdout, test_records, logger)

    # -- 6. Build fold_definitions + summary stats --------------------------
    fold_defs         = build_fold_definitions(df)
    partition_summary = compute_partition_summary(df)
    fold_summary      = compute_fold_summary(df, fold_defs)

    # -- 7. Write diagnostic CSVs + markdown + grouped log -------------------
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    write_bins_csv(df, logger)
    write_mouse_assignment_csv(df, logger)
    write_stratification_report_csv(partition_summary, fold_summary, logger)
    write_mouse_assignment_md(
        df, partition_summary, args.source, stratification_info, logger,
    )
    log_mouse_assignment_grouped(df, partition_summary, logger)

    # -- 8. Assemble + write the manifest JSON -------------------------------
    manifest = build_output_manifest(
        df=df,
        fold_defs=fold_defs,
        partition_summary=partition_summary,
        fold_summary=fold_summary,
        stratification_info=stratification_info,
        train_records=train_records,
        val_holdout=val_holdout,
        test=test_records,
        source_manifest=args.source,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    size_mb = args.output.stat().st_size / 1e6
    logger.info("Wrote CV manifest: %s (%.2f MB)", args.output, size_mb)

    # -- 9. Final summary banner --------------------------------------------
    logger.info("=" * 65)
    logger.info("DONE")
    logger.info("  Train mice (71)  : 5 folds + 1 early-stop partition")
    for r in partition_summary:
        logger.info("    %-12s : %2d mice | %6d segs | %.2f%% ictal",
                    r["partition"], r["n_mice"], r["n_total"],
                    r["prevalence_pct"])
    logger.info("  Stratification   : %s (fallback=%s)",
                stratification_info["binning"],
                stratification_info["fallback_used"])
    logger.info("  Output manifest  : %s", args.output)
    logger.info("  Diagnostics dir  : %s", DIAG_DIR)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
