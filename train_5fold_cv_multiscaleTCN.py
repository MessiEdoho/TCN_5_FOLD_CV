"""
train_5fold_cv_multiscaleTCN.py
================================
5-fold subject-disjoint cross-validation training script for the
MultiScaleTCN architecture, configured to inherit the training
protocol from the parent project DL_WITH_SSL_GA (i.e., TCN_SSL_GA on
the cluster):

  - model            : tcn_utils.MultiScaleTCN
  - optimiser        : AdamW(lr, weight_decay) with values from HPT
  - scheduler        : CosineAnnealingLR(T_max=100, eta_min=lr*0.01)
  - loss             : BCEWithLogitsLoss (unweighted; the input manifest
                       is already proximity-aware non-ictal downsampled
                       upstream by create_balanced_splits.py)
  - mixed precision  : torch.amp autocast + GradScaler on CUDA
  - gradient clip    : max_norm = 1.0
  - batch size       : from HPT
  - max epochs       : 100
  - early stopping   : patience = 15, no warm-up (can fire from epoch 1)

Per fold k in {0..4} the 71-mouse training cohort is partitioned by
mouse_id according to data_splits_5fold_cv.json:

    Train (~47 mice)   = union of CV partitions other than k.
    Early-stop set     = the fixed 'early_stop_mice' partition,
                         identical across all 5 folds. Used every epoch
                         to compute val_loss and val_macro_F1; macro F1
                         drives the early-stopping decision.
    Test fold (~10)    = the held-out CV partition fold_definitions[k].
                         test_mice (manifest schema v2; was `val_mice`
                         in v1). Evaluated exactly ONCE after early
                         stop fires; never seen during training.

The test-fold evaluation deliberately omits all post-processing: no
probability smoothing, no event-level metrics, no threshold sweep.
This script answers the question "does the fixed architecture (with
fixed HPT hyperparameters) generalise robustly to different held-out
subject cohorts, without being tuned to any specific test set?".
Event-level metrics and any post-processing are deferred to the
full-cohort retraining stage that follows model selection.

Reported per fold and aggregated across the 5 folds as mean +/- std:
  accuracy, precision, recall, specificity, F1, MCC, AUROC, PRAUC
IMPORTANT: `precision`, `recall`, and `F1` are MACRO-AVERAGED across
the two classes (sklearn `average="macro"`); they are NOT the binary
positive-class scores. The bare column labels are kept (no `_macro`
suffix) for compatibility with downstream tooling, but the convention
is uniform across the report, the CSV outputs, and this docstring.
`specificity` is the binary class-0 recall TN / (TN + FP), reported
separately because the macro-averaged `recall` above averages it
away. `MCC` is Matthews correlation coefficient. All threshold-
dependent metrics (accuracy, precision, recall, specificity, F1, MCC)
use a FIXED threshold of 0.5 on the raw sigmoid output. AUROC and
PRAUC are threshold-free and computed on the raw probabilities.

Pipeline position
-----------------
After  : create_5fold_cv_splits.py  -> data_splits_5fold_cv.json
         tune_multiscale_tcn.py    -> best_multiscale_params.json
Before : architecture-selection comparisons + full-cohort retraining
         (separate script) for event-level evaluation on the 20-mouse
         independent test set.

Outputs
-------
Under <output-dir>/:
    fold{k}/
        best_weights.pt        - state_dict at best val_macro_F1 epoch
        training_history.csv   - per-epoch train_loss, val_loss, val_f1, lr
        test_predictions.npz   - y_true, y_prob, y_pred, test_mice
        test_metrics.json      - 8 segment-level metrics (precision,
                                 recall, F1 are macro-averaged) + meta
        train.log              - per-epoch log
    cv_summary.csv             - per-fold rows + mean + std rows
    cv_summary.json            - structured per-fold + mean + std
    cv_summary.log             - end-to-end run log

Usage
-----
    python train_5fold_cv_multiscaleTCN.py [options]

Reproducibility
---------------
Single fixed seed (42) for set_seed() at the start of every fold. The
model initialisation, dropout masks, and DataLoader shuffle order are
therefore identical across folds; only the data partition changes.
This isolates fold-level variation to partition variance.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Parent-project import
# ---------------------------------------------------------------------------
# The parent project (DL_WITH_SSL_GA, deployed as TCN_SSL_GA on cluster) is
# the single source of truth for: MultiScaleTCN model class, the per-epoch
# train loop, the DataLoader factory, and the seed helper. Override the path
# with the DL_WITH_SSL_GA_PATH environment variable if your layout differs.
DL_WITH_SSL_GA_PATH = os.environ.get(
    "DL_WITH_SSL_GA_PATH", str(Path.home() / "TCN_SSL_GA")
)
sys.path.insert(0, DL_WITH_SSL_GA_PATH)

from tcn_utils import (  # noqa: E402  (import after sys.path manipulation)
    MultiScaleTCN,
    make_loader,
    set_seed,
    train_one_epoch,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "MultiScaleTCN"
N_FOLDS = 5
MAX_EPOCHS = 100
ES_PATIENCE = 15           # epochs with no val_macro_F1 improvement before stop
GRAD_CLIP = 1.0
THRESHOLD = 0.5            # fixed segment-level decision threshold
SEED = 42

DEFAULT_CV_MANIFEST = (
    "/home/people/22206468/scratch/INPUT_CV_PROJECT/manifest/"
    "data_splits_5fold_cv.json"
)
DEFAULT_BEST_PARAMS = (
    "/home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/"
    "MultiScaleTCNtuning_outputs/best_multiscale_params.json"
)
DEFAULT_OUTPUT_DIR = "/home/people/22206468/scratch/OUTPUT_CV/MultiScaleTCN"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METRIC_COLS = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "mcc",
    "auroc",
    "prauc",
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Data and model construction
# ---------------------------------------------------------------------------
def load_best_params(path: Path, logger: logging.Logger) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    hp = payload["hyperparameters"]
    branches = payload.get(
        "branch_dilations",
        {"branch1": [1, 2, 4], "branch2": [8, 16, 32], "branch3": [32, 64, 128]},
    )
    logger.info(
        "Loaded HPT best params: trial=%s, best_val_f1=%.6f",
        payload.get("best_trial_number", "?"),
        float(payload.get("best_val_f1", float("nan"))),
    )
    logger.info(
        "  num_filters=%s kernel_size=%s dropout=%s fusion=%s "
        "lr=%s wd=%s batch_size=%s",
        hp["num_filters"], hp["kernel_size"], hp["dropout"], hp["fusion"],
        hp["learning_rate"], hp["weight_decay"], hp["batch_size"],
    )
    return {"hp": hp, "branches": branches}


def build_pairs(records: list, mouse_set: set) -> list:
    return [
        (r["filepath"], int(r["label"]))
        for r in records
        if r["mouse_id"] in mouse_set
    ]


def build_model(hp: dict, branches: dict) -> nn.Module:
    return MultiScaleTCN(
        num_filters=int(hp["num_filters"]),
        kernel_size=int(hp["kernel_size"]),
        dropout=float(hp["dropout"]),
        branch1_dilations=branches["branch1"],
        branch2_dilations=branches["branch2"],
        branch3_dilations=branches["branch3"],
        fusion=str(hp["fusion"]),
    )


def count_params(model: nn.Module) -> tuple:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def forward_pass(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple:
    """Run a single inference pass; return y_true, y_prob, mean_loss."""
    model.eval()
    losses: list = []
    ys: list = []
    ps: list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            losses.append(float(loss.item()))
            ys.append(y.detach().cpu().numpy())
            ps.append(torch.sigmoid(logits).float().detach().cpu().numpy())
    y_true = np.concatenate(ys).astype(np.int64).ravel()
    y_prob = np.concatenate(ps).astype(np.float32).ravel()
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return y_true, y_prob, mean_loss


def compute_segment_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Eight segment-level metrics at fixed THRESHOLD; no post-processing.

    MACRO, NOT BINARY. `precision`, `recall`, and `f1` are computed with
    sklearn's `average="macro"` argument, i.e. the unweighted mean of the
    class-0 and class-1 scores. These are NOT the binary positive-class
    metrics returned by sklearn's default `average="binary"`. The bare
    column labels (`precision`, `recall`, `f1`) intentionally omit the
    `_macro` suffix so that downstream CSVs and figure scripts can read
    a stable schema; the macro-averaging convention is documented here
    and in CV_METHODOLOGY_REPORT.md Section 5.3 and applies uniformly.

    `specificity` is the binary class-0 recall, TN / (TN + FP), reported
    separately because the macro `recall` above averages it together
    with sensitivity and so does not expose it on its own.

    `mcc` is Matthews correlation coefficient, a single-number summary
    of the binary confusion matrix that is robust under class imbalance.

    `auroc` and `prauc` are threshold-free and computed directly on the
    raw probabilities.
    """
    y_pred = (y_prob >= THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity": float(specificity),
        "f1":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc":         float(matthews_corrcoef(y_true, y_pred)),
        "auroc":       float(roc_auc_score(y_true, y_prob)),
        "prauc":       float(average_precision_score(y_true, y_prob)),
    }


# ---------------------------------------------------------------------------
# Per-fold training
# ---------------------------------------------------------------------------
def train_one_fold(
    fold_idx: int,
    splits: dict,
    hp_payload: dict,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    fold_def = splits["fold_definitions"][fold_idx]
    assert int(fold_def["fold"]) == fold_idx, (
        f"fold_definitions[{fold_idx}].fold mismatch: {fold_def['fold']}"
    )
    train_mice = set(fold_def["train_mice"])
    test_mice = set(fold_def["test_mice"])
    early_stop_mice = set(splits["early_stop_mice"])
    records = splits["records"]

    # Sanity: train, test, early-stop must be mutually disjoint.
    assert not (train_mice & test_mice), "train and test mice overlap"
    assert not (train_mice & early_stop_mice), "train and early-stop mice overlap"
    assert not (test_mice & early_stop_mice), "test and early-stop mice overlap"

    train_pairs = build_pairs(records, train_mice)
    es_pairs = build_pairs(records, early_stop_mice)
    test_pairs = build_pairs(records, test_mice)

    logger.info(
        "Fold %d composition: train=%d segs / %d mice | early-stop=%d segs / "
        "%d mice | test=%d segs / %d mice",
        fold_idx, len(train_pairs), len(train_mice),
        len(es_pairs), len(early_stop_mice),
        len(test_pairs), len(test_mice),
    )

    # Same seed across folds: same init, same dropout, same shuffle RNG.
    # All fold-level variance is therefore attributable to partition variance.
    set_seed(SEED)

    hp = hp_payload["hp"]
    branches = hp_payload["branches"]
    batch_size = int(hp["batch_size"])
    use_amp = DEVICE.type == "cuda"

    train_loader = make_loader(train_pairs, batch_size, True, DEVICE)
    es_loader = make_loader(es_pairs, batch_size, False, DEVICE)
    test_loader = make_loader(test_pairs, batch_size, False, DEVICE)

    model = build_model(hp, branches).to(DEVICE)
    n_total, n_trainable = count_params(model)
    logger.info("Model: %s, params total=%d trainable=%d", MODEL_NAME, n_total, n_trainable)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(hp["learning_rate"]),
        weight_decay=float(hp["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=MAX_EPOCHS,
        eta_min=float(hp["learning_rate"]) * 0.01,
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    fold_dir = output_dir / f"fold{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = fold_dir / "best_weights.pt"

    history: dict = {
        "epoch": [], "train_loss": [], "val_loss": [], "val_f1": [], "lr": [],
    }
    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    best_state: dict | None = None

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimiser, criterion, DEVICE,
            max_grad_norm=GRAD_CLIP, scaler=scaler,
        )

        y_true_es, y_prob_es, val_loss = forward_pass(
            model, es_loader, criterion, DEVICE, use_amp
        )
        y_pred_es = (y_prob_es >= THRESHOLD).astype(np.int64)
        val_f1 = float(
            f1_score(y_true_es, y_pred_es, average="macro", zero_division=0)
        )

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_f1"].append(float(val_f1))
        history["lr"].append(float(current_lr))

        logger.info(
            "Fold %d Ep %3d | train_loss=%.5f  val_loss=%.5f  "
            "val_macroF1=%.5f  lr=%.2e  (%ds)",
            fold_idx, epoch, train_loss, val_loss, val_f1, current_lr,
            int(time.time() - t0),
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(best_state, best_ckpt_path)
        else:
            no_improve += 1
            if no_improve >= ES_PATIENCE:
                logger.info(
                    "Fold %d: early stop at epoch %d  "
                    "(best epoch %d, val_macroF1=%.5f)",
                    fold_idx, epoch, best_epoch, best_f1,
                )
                break

    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
    logger.info(
        "Fold %d: training done. best_epoch=%d, best_val_macroF1=%.5f",
        fold_idx, best_epoch, best_f1,
    )

    # Restore best weights for the single, final test-fold evaluation.
    assert best_state is not None, "best_state was never set"
    model.load_state_dict(best_state)
    model.to(DEVICE)

    y_true_test, y_prob_test, test_loss = forward_pass(
        model, test_loader, criterion, DEVICE, use_amp
    )
    test_metrics = compute_segment_metrics(y_true_test, y_prob_test)
    test_metrics.update({
        "fold": fold_idx,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_f1),
        "test_loss": float(test_loss),
        "n_test_segments": int(len(y_true_test)),
        "n_test_mice": int(len(test_mice)),
        "threshold": float(THRESHOLD),
    })

    y_pred_test = (y_prob_test >= THRESHOLD).astype(np.int64)
    np.savez_compressed(
        fold_dir / "test_predictions.npz",
        y_true=y_true_test,
        y_prob=y_prob_test,
        y_pred=y_pred_test,
        test_mice=np.array(sorted(test_mice), dtype="<U32"),
        threshold=np.float32(THRESHOLD),
    )
    with open(fold_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info(
        "Fold %d TEST | acc=%.4f  prec=%.4f  rec=%.4f  spec=%.4f  "
        "f1=%.4f  mcc=%.4f  AUROC=%.4f  PRAUC=%.4f",
        fold_idx,
        test_metrics["accuracy"], test_metrics["precision"],
        test_metrics["recall"], test_metrics["specificity"],
        test_metrics["f1"], test_metrics["mcc"],
        test_metrics["auroc"], test_metrics["prauc"],
    )
    return test_metrics


# ---------------------------------------------------------------------------
# Aggregation across folds
# ---------------------------------------------------------------------------
def aggregate_results(
    per_fold: list, output_dir: Path, logger: logging.Logger
) -> None:
    df = pd.DataFrame(per_fold)
    ordered_cols = (
        ["fold", "best_epoch", "best_val_macro_f1",
         "n_test_mice", "n_test_segments", "test_loss"]
        + list(METRIC_COLS)
    )
    df = df[ordered_cols].sort_values("fold").reset_index(drop=True)

    mean_row = {c: float(df[c].mean()) for c in METRIC_COLS}
    std_row = {c: float(df[c].std(ddof=1)) for c in METRIC_COLS}

    # Pad the summary rows with empty strings for non-metric columns.
    mean_row.update({
        "fold": "mean", "best_epoch": "", "best_val_macro_f1": "",
        "n_test_mice": "", "n_test_segments": int(df["n_test_segments"].sum()),
        "test_loss": float(df["test_loss"].mean()),
    })
    std_row.update({
        "fold": "std", "best_epoch": "", "best_val_macro_f1": "",
        "n_test_mice": "", "n_test_segments": "",
        "test_loss": float(df["test_loss"].std(ddof=1)),
    })
    df_full = pd.concat(
        [df, pd.DataFrame([mean_row, std_row])], ignore_index=True
    )
    df_full.to_csv(output_dir / "cv_summary.csv", index=False)

    summary = {
        "model": MODEL_NAME,
        "n_folds": N_FOLDS,
        "threshold": THRESHOLD,
        "per_fold": per_fold,
        "mean": {c: float(df[c].mean()) for c in METRIC_COLS},
        "std":  {c: float(df[c].std(ddof=1)) for c in METRIC_COLS},
    }
    with open(output_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "===== CV summary (%s, n_folds=%d, threshold=%.2f, no post-processing) =====",
        MODEL_NAME, N_FOLDS, THRESHOLD,
    )
    for c in METRIC_COLS:
        logger.info(
            "  %-12s : %.4f +/- %.4f", c, summary["mean"][c], summary["std"][c]
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--cv-manifest", default=DEFAULT_CV_MANIFEST,
                   help="Path to data_splits_5fold_cv.json")
    p.add_argument("--best-params", default=DEFAULT_BEST_PARAMS,
                   help="Path to best_multiscale_params.json")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Root output directory (one subdir per fold)")
    p.add_argument("--folds", default="0,1,2,3,4",
                   help="Comma-separated fold indices to run (default: all 5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_logger = setup_logger("cv_main", output_dir / "cv_summary.log")
    main_logger.info("===== 5-fold CV training : %s =====", MODEL_NAME)
    main_logger.info("device          : %s", DEVICE)
    main_logger.info("parent project  : %s", DL_WITH_SSL_GA_PATH)
    main_logger.info("cv-manifest     : %s", args.cv_manifest)
    main_logger.info("best-params     : %s", args.best_params)
    main_logger.info("output-dir      : %s", args.output_dir)
    main_logger.info(
        "max_epochs=%d  patience=%d (no warm-up)  threshold=%.2f  seed=%d",
        MAX_EPOCHS, ES_PATIENCE, THRESHOLD, SEED,
    )

    with open(args.cv_manifest, "r", encoding="utf-8") as f:
        splits = json.load(f)
    hp_payload = load_best_params(Path(args.best_params), main_logger)

    fold_indices = [int(x) for x in args.folds.split(",") if x.strip()]
    for k in fold_indices:
        assert 0 <= k < N_FOLDS, f"fold index {k} out of range [0,{N_FOLDS})"

    per_fold: list = []
    for k in fold_indices:
        fold_logger = setup_logger(
            f"cv_fold{k}", output_dir / f"fold{k}" / "train.log"
        )
        main_logger.info("--- Starting fold %d ---", k)
        per_fold.append(
            train_one_fold(k, splits, hp_payload, output_dir, fold_logger)
        )
        main_logger.info("--- Finished fold %d ---", k)

    if len(per_fold) == N_FOLDS:
        aggregate_results(per_fold, output_dir, main_logger)
    else:
        main_logger.info(
            "Ran %d/%d folds. Re-run with --folds 0,1,2,3,4 to aggregate.",
            len(per_fold), N_FOLDS,
        )


if __name__ == "__main__":
    main()
