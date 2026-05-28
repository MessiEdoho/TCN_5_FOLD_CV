"""
train_5fold_cv_multiscaleTCNwithATTN.py
=======================================
5-fold subject-disjoint cross-validation training script for the
MultiScaleTCNWithAttention (M4) architecture, inheriting the training
protocol from the parent project DL_WITH_SSL_GA (deployed as TCN_SSL_GA
on the cluster). This script is a sibling of
`train_5fold_cv_multiscaleTCN.py` (M3) and shares the entire
training-loop structure -- only the model class, the hyperparameter
load (two JSONs instead of one), and the final-eval AMP setting differ.

Why two hyperparameter JSON files
---------------------------------
The parent's `tune_multiscale_attention.py` re-tunes only the
attention-specific knobs and the optimisation knobs (`attention_dim`,
`attention_dropout`, `learning_rate`, `weight_decay`, `batch_size`)
while holding the backbone (`num_filters`, `kernel_size`, `dropout`,
`fusion`) fixed at M3's tuned values. This script therefore loads:

  1. `best_multiscale_params.json`      -- backbone HPs (M3 source of truth)
  2. `best_multiscale_attn_params.json` -- attention + lr/wd/batch HPs (M4)

The attn JSON also embeds a `backbone_hyperparameters` snapshot. The
loader asserts that snapshot agrees with the live M3 backbone JSON
and aborts if they diverge -- this catches the case where someone
re-tuned M3 without propagating the change to the M4 snapshot.

Why FP32 for the final test-fold evaluation
-------------------------------------------
The parent's `eval_utils.py` documents a 4-layer NaN-protection scheme
for attention-bearing models, the relevant one here being "Layer 3:
FP32 forward (no AMP) to avoid FP16 overflow inside the attention
softmax". The per-epoch monitor on the early-stop set continues to use
AMP (matching the parent and M3's CV script), and only the single
test-fold pass that produces the reported CV metrics runs in FP32. Any
deviation from this rule risks silent NaNs in attention weights on
adversarial inputs and would invalidate the cross-architecture
comparison with M3.

Per fold k in {0..4}
--------------------
    Train (~47 mice)   = union of CV partitions other than k.
    Early-stop set     = the fixed `early_stop_mice` partition,
                         identical across all 5 folds. Used every epoch
                         to compute val_loss and val_macro_F1; macro F1
                         drives the early-stopping decision (patience=15,
                         no warm-up).
    Test fold (~10)    = the held-out CV partition fold_definitions[k].
                         test_mice (manifest schema v2). Evaluated
                         exactly ONCE after early stop fires, in FP32;
                         never seen during training.

The test-fold evaluation deliberately omits all post-processing: no
probability smoothing, no event-level metrics, no threshold sweep.

Reported per fold and aggregated across the 5 folds as mean +/- std:
  accuracy, precision, recall, specificity, F1, MCC, AUROC, PRAUC
IMPORTANT: `precision`, `recall`, and `F1` are MACRO-AVERAGED across
the two classes (sklearn `average="macro"`); they are NOT the binary
positive-class scores. The bare column labels are kept for
compatibility with downstream tooling. `specificity` is the binary
class-0 recall TN/(TN+FP). All threshold-dependent metrics use a
FIXED threshold of 0.5 on the raw sigmoid output. AUROC and PRAUC are
threshold-free and computed on the raw probabilities.

Pipeline position
-----------------
After  : create_5fold_cv_splits.py    -> data_splits_5fold_cv.json
         tune_multiscale_tcn.py       -> best_multiscale_params.json
         tune_multiscale_attention.py -> best_multiscale_attn_params.json
Before : architecture-selection comparisons (this script vs M3) +
         full-cohort retraining for event-level evaluation on the
         20-mouse independent test set.

Outputs (under <output-dir>/, default OUTPUT_CV_PROJECT/MODEL_4_MTCN_ATTN)
    fold{k}/
        best_weights.pt        - state_dict at best val_macro_F1 epoch
        training_history.csv   - per-epoch train_loss, val_loss, val_f1, lr
        test_predictions.npz   - y_true, y_prob, y_pred, mouse_id, test_mice
        test_metrics.json      - 8 segment-level metrics + meta
        train.log              - per-epoch log
    cv_summary.csv             - per-fold rows + mean + std rows
    cv_summary.json            - structured per-fold + mean + std
    cv_summary.log             - end-to-end run log

Usage
-----
    python train_5fold_cv_multiscaleTCNwithATTN.py [options]
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
DL_WITH_SSL_GA_PATH = os.environ.get(
    "DL_WITH_SSL_GA_PATH", str(Path.home() / "TCN_SSL_GA")
)
sys.path.insert(0, DL_WITH_SSL_GA_PATH)

from tcn_utils import (  # noqa: E402  (import after sys.path manipulation)
    MultiScaleTCNWithAttention,
    make_loader,
    set_seed,
    train_one_epoch,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "MultiScaleTCNWithAttention"
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
DEFAULT_BACKBONE_PARAMS = (
    "/home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/"
    "MultiScaleTCNtuning_outputs/best_multiscale_params.json"
)
DEFAULT_ATTN_PARAMS = (
    "/home/people/22206468/scratch/OUTPUT/MODEL4_OUTPUT/"
    "multiscale_attention_tuning_outputs/best_multiscale_attn_params.json"
)
DEFAULT_OUTPUT_DIR = (
    "/home/people/22206468/scratch/OUTPUT_CV_PROJECT/MODEL_4_MTCN_ATTN"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HP keys this script needs out of each JSON. The backbone JSON contributes
# the architecture defaults the M4 study held fixed; the attn JSON
# contributes the M4-tuned attention + optimisation knobs.
REQUIRED_BACKBONE_HP_KEYS = ("num_filters", "kernel_size", "dropout", "fusion")
REQUIRED_ATTN_HP_KEYS = (
    "attention_dim",
    "attention_dropout",
    "learning_rate",
    "weight_decay",
    "batch_size",
)

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
def load_best_params(
    backbone_path: Path, attn_path: Path, logger: logging.Logger
) -> dict:
    """Load and validate both HPT JSONs; return a merged HP payload.

    Returns
    -------
    {"hp": {<all merged keys>}, "branches": {branch1/2/3 dilation lists}}

    Strategy A (matches the parent training driver
    `MultiScaleTCNAttention.py`): the backbone JSON is the canonical
    source of truth for the four backbone HPs; the attn JSON contributes
    `attention_dim`, `attention_dropout`, `learning_rate`,
    `weight_decay`, `batch_size`. The attn JSON also embeds a
    `backbone_hyperparameters` snapshot for documentation; we assert
    snapshot == live backbone JSON and abort if they disagree, to catch
    the case where someone re-tuned the backbone but did not propagate
    the update into the attn snapshot.
    """
    if not backbone_path.exists():
        raise FileNotFoundError(
            "backbone-params JSON not found at %s." % backbone_path
        )
    if not attn_path.exists():
        raise FileNotFoundError(
            "attn-params JSON not found at %s." % attn_path
        )

    with open(backbone_path, "r", encoding="utf-8") as f:
        backbone_payload = json.load(f)
    with open(attn_path, "r", encoding="utf-8") as f:
        attn_payload = json.load(f)

    # --- backbone block ---------------------------------------------------
    if "hyperparameters" not in backbone_payload:
        raise KeyError(
            "backbone-params JSON %s missing 'hyperparameters' key. "
            "Top-level keys present: %s"
            % (backbone_path, sorted(backbone_payload.keys()))
        )
    backbone_hp = backbone_payload["hyperparameters"]
    missing_backbone = [k for k in REQUIRED_BACKBONE_HP_KEYS
                        if k not in backbone_hp]
    if missing_backbone:
        raise KeyError(
            "backbone-params JSON %s is missing required HP key(s): %s. "
            "Keys present: %s"
            % (backbone_path, missing_backbone, sorted(backbone_hp.keys()))
        )

    # --- attn block -------------------------------------------------------
    if "hyperparameters" not in attn_payload:
        raise KeyError(
            "attn-params JSON %s missing 'hyperparameters' key. "
            "Top-level keys present: %s"
            % (attn_path, sorted(attn_payload.keys()))
        )
    attn_hp = attn_payload["hyperparameters"]
    missing_attn = [k for k in REQUIRED_ATTN_HP_KEYS if k not in attn_hp]
    if missing_attn:
        raise KeyError(
            "attn-params JSON %s is missing required HP key(s): %s. "
            "Keys present: %s"
            % (attn_path, missing_attn, sorted(attn_hp.keys()))
        )

    # --- snapshot agreement check -----------------------------------------
    # The attn JSON snapshots the backbone HPs the study used. If the
    # backbone has been re-tuned but the snapshot is stale, the CV
    # results would silently train on different backbone HPs than the
    # tuning assumed. Abort loudly rather than train on a mismatched
    # configuration.
    attn_backbone_snapshot = attn_payload.get("backbone_hyperparameters", {})
    if attn_backbone_snapshot:
        disagreements = []
        for k in REQUIRED_BACKBONE_HP_KEYS:
            live = backbone_hp.get(k)
            snap = attn_backbone_snapshot.get(k)
            if live != snap:
                disagreements.append((k, live, snap))
        if disagreements:
            details = ", ".join(
                "%s: live=%r vs attn-snapshot=%r" % (k, live, snap)
                for k, live, snap in disagreements
            )
            raise ValueError(
                "Backbone HP disagreement between %s and %s: %s. "
                "Re-run tune_multiscale_attention.py against the current "
                "backbone JSON, or pass --backbone-params pointing to the "
                "version the attn snapshot was taken against."
                % (backbone_path, attn_path, details)
            )
    else:
        logger.warning(
            "attn-params JSON %s has no 'backbone_hyperparameters' "
            "snapshot block; cannot verify backbone agreement.",
            attn_path,
        )

    # --- branch dilations -------------------------------------------------
    # Parent stores branch_dilations at the top level of either JSON. We
    # prefer the attn JSON's copy (most recent), then the backbone JSON,
    # then fall back to the parent's documented defaults.
    branches = (
        attn_payload.get("branch_dilations")
        or backbone_payload.get("branch_dilations")
        or {
            "branch1": [1, 2, 4],
            "branch2": [8, 16, 32],
            "branch3": [32, 64, 128],
        }
    )
    for bk in ("branch1", "branch2", "branch3"):
        if bk not in branches:
            raise KeyError(
                "branch_dilations block missing key %r. Keys present: %s"
                % (bk, sorted(branches.keys()))
            )

    # --- merge ------------------------------------------------------------
    hp = {
        # backbone
        "num_filters":       backbone_hp["num_filters"],
        "kernel_size":       backbone_hp["kernel_size"],
        "dropout":           backbone_hp["dropout"],
        "fusion":            backbone_hp["fusion"],
        # attention + optimisation (from attn study)
        "attention_dim":     attn_hp["attention_dim"],
        "attention_dropout": attn_hp["attention_dropout"],
        "learning_rate":     attn_hp["learning_rate"],
        "weight_decay":      attn_hp["weight_decay"],
        "batch_size":        attn_hp["batch_size"],
    }

    logger.info(
        "Loaded HPT best params:"
    )
    logger.info(
        "  backbone (%s): trial=%s, best_val_f1=%.6f",
        backbone_path.name,
        backbone_payload.get("best_trial_number", "?"),
        float(backbone_payload.get("best_val_f1", float("nan"))),
    )
    logger.info(
        "  attention (%s): trial=%s, best_val_f1=%.6f",
        attn_path.name,
        attn_payload.get("best_trial_number", "?"),
        float(attn_payload.get("best_val_f1", float("nan"))),
    )
    logger.info(
        "  backbone   : num_filters=%s kernel_size=%s dropout=%s fusion=%s",
        hp["num_filters"], hp["kernel_size"], hp["dropout"], hp["fusion"],
    )
    logger.info(
        "  attention  : attention_dim=%s attention_dropout=%s",
        hp["attention_dim"], hp["attention_dropout"],
    )
    logger.info(
        "  optimiser  : lr=%s wd=%s batch_size=%s",
        hp["learning_rate"], hp["weight_decay"], hp["batch_size"],
    )
    return {"hp": hp, "branches": branches}


def build_pairs(records: list, mouse_set: set) -> list:
    return [
        (r["filepath"], int(r["label"]))
        for r in records
        if r["mouse_id"] in mouse_set
    ]


def build_pairs_with_mouse_id(records: list, mouse_set: set) -> tuple:
    """Return (file_label_pairs, mouse_ids) in matching order.

    Used for the test fold so the cached predictions NPZ records the
    source mouse_id alongside each segment. DataLoader iterates in
    dataset order when shuffle=False, so test_mouse_ids[i] matches
    y_true[i].
    """
    pairs: list = []
    mids: list = []
    for r in records:
        if r["mouse_id"] in mouse_set:
            pairs.append((r["filepath"], int(r["label"])))
            mids.append(r["mouse_id"])
    return pairs, mids


def build_model(hp: dict, branches: dict) -> nn.Module:
    """Build MultiScaleTCNWithAttention with the merged HP payload.

    Constructor matches the parent's signature in tcn_utils.py
    (MultiScaleTCNWithAttention class). All four backbone HPs and both
    attention HPs are mandatory positional/keyword args; branch dilations
    fall back to the parent's documented defaults if not provided.
    """
    return MultiScaleTCNWithAttention(
        num_filters=int(hp["num_filters"]),
        kernel_size=int(hp["kernel_size"]),
        dropout=float(hp["dropout"]),
        fusion=str(hp["fusion"]),
        attention_dim=int(hp["attention_dim"]),
        attention_dropout=float(hp["attention_dropout"]),
        branch1_dilations=branches["branch1"],
        branch2_dilations=branches["branch2"],
        branch3_dilations=branches["branch3"],
    )


def count_params(model: nn.Module) -> tuple:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _mouse_sort_key(mid):
    """Numeric mouse-id sort: 'm10' after 'm2'.

    Mirrors the helper of the same name in create_5fold_cv_splits.py and
    train_5fold_cv_multiscaleTCN.py so the mouse-id ordering in
    test_predictions.npz matches the splits manifest, the markdown
    table, and the persistent log.
    """
    s = str(mid)
    rest = s[1:] if s.startswith("m") else s
    try:
        return (0, int(rest))
    except ValueError:
        return (1, s)


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
    sklearn's `average="macro"`, i.e. the unweighted mean of class-0
    and class-1 scores -- NOT the positive-class scores. `specificity`
    is the binary class-0 recall TN/(TN+FP), reported separately. `mcc`
    is Matthews correlation coefficient. `auroc` and `prauc` are
    threshold-free and computed on the raw probabilities. Convention
    documented uniformly in CV_METHODOLOGY_REPORT.md Section 5.3.
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
    es_pairs    = build_pairs(records, early_stop_mice)
    test_pairs, test_mouse_ids = build_pairs_with_mouse_id(records, test_mice)

    logger.info(
        "Fold %d composition: train=%d segs / %d mice | early-stop=%d segs / "
        "%d mice | test=%d segs / %d mice",
        fold_idx, len(train_pairs), len(train_mice),
        len(es_pairs), len(early_stop_mice),
        len(test_pairs), len(test_mice),
    )

    assert train_pairs, "Fold %d: train cohort is empty." % fold_idx
    assert es_pairs,    "Fold %d: early-stop cohort is empty." % fold_idx
    assert test_pairs,  "Fold %d: test cohort is empty." % fold_idx

    # Same seed across folds: same init, same dropout, same main-process
    # shuffle RNG. All fold-level variance is therefore attributable to
    # partition variance.
    # DataLoader worker RNG is intentionally not re-seeded; the dataset
    # (EEGSegmentDataset in tcn_utils.py) performs only deterministic
    # np.load() with no stochastic augmentation, normalisation, or
    # sampling inside __getitem__, so worker RNG state never influences
    # the returned tensors.
    set_seed(SEED)

    hp = hp_payload["hp"]
    branches = hp_payload["branches"]
    batch_size = int(hp["batch_size"])
    use_amp_train = DEVICE.type == "cuda"
    # Final test-fold pass forces FP32 to avoid FP16 overflow inside the
    # attention softmax (parent eval_utils.py "Layer 3" NaN protection).
    # The per-epoch early-stop monitor stays on AMP -- this mirrors the
    # parent driver MultiScaleTCNAttention.py which passes
    # use_amp=True for the monitor pass and use_amp=False for the final
    # full-val pass.
    use_amp_final_eval = False

    train_loader = make_loader(train_pairs, batch_size, True, DEVICE)
    es_loader = make_loader(es_pairs, batch_size, False, DEVICE)
    test_loader = make_loader(test_pairs, batch_size, False, DEVICE)

    model = build_model(hp, branches).to(DEVICE)
    n_total, n_trainable = count_params(model)
    logger.info(
        "Model: %s, params total=%d trainable=%d",
        MODEL_NAME, n_total, n_trainable,
    )

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
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_amp_train)
        if use_amp_train else None
    )

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

        # Per-epoch monitor uses AMP for speed (same as parent).
        y_true_es, y_prob_es, val_loss = forward_pass(
            model, es_loader, criterion, DEVICE, use_amp=use_amp_train,
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
    assert best_state is not None, (
        "Fold %d: best_state was never set. Last seen best_f1=%.5f, "
        "best_epoch=%d, no_improve=%d, last epoch reached=%d. This "
        "typically means the very first val pass raised an exception "
        "before any improvement could be recorded -- inspect the per-"
        "epoch log above for the underlying error."
        % (fold_idx, best_f1, best_epoch, no_improve,
           history["epoch"][-1] if history["epoch"] else 0)
    )
    model.load_state_dict(best_state)
    model.to(DEVICE)

    # FINAL TEST-FOLD EVALUATION: FP32 forward, no AMP. See module
    # docstring for the FP16-softmax-overflow rationale.
    y_true_test, y_prob_test, test_loss = forward_pass(
        model, test_loader, criterion, DEVICE, use_amp=use_amp_final_eval,
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
    assert len(test_mouse_ids) == len(y_true_test), (
        "Fold %d: per-segment mouse_ids length (%d) does not match "
        "y_true length (%d); aborting NPZ write to avoid a corrupted "
        "predictions file." % (fold_idx, len(test_mouse_ids), len(y_true_test))
    )
    np.savez_compressed(
        fold_dir / "test_predictions.npz",
        y_true=y_true_test,
        y_prob=y_prob_test,
        y_pred=y_pred_test,
        mouse_id=np.array(test_mouse_ids, dtype="<U32"),
        test_mice=np.array(sorted(test_mice, key=_mouse_sort_key), dtype="<U32"),
        threshold=np.float32(THRESHOLD),
    )
    with open(fold_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info(
        "Fold %d TEST (FP32) | acc=%.4f  prec=%.4f  rec=%.4f  spec=%.4f  "
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
    p.add_argument("--cv-manifest", type=Path, default=Path(DEFAULT_CV_MANIFEST),
                   help="Path to data_splits_5fold_cv.json")
    p.add_argument("--backbone-params", type=Path,
                   default=Path(DEFAULT_BACKBONE_PARAMS),
                   help="Path to best_multiscale_params.json (M3 backbone)")
    p.add_argument("--attn-params", type=Path,
                   default=Path(DEFAULT_ATTN_PARAMS),
                   help="Path to best_multiscale_attn_params.json (M4 attn)")
    p.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR),
                   help="Root output directory (one subdir per fold)")
    p.add_argument("--folds", default="0,1,2,3,4",
                   help="Comma-separated fold indices to run (default: all 5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    main_logger = setup_logger("cv_main_attn", output_dir / "cv_summary.log")
    main_logger.info("===== 5-fold CV training : %s =====", MODEL_NAME)
    main_logger.info("device          : %s", DEVICE)
    main_logger.info("parent project  : %s", DL_WITH_SSL_GA_PATH)
    main_logger.info("cv-manifest     : %s", args.cv_manifest)
    main_logger.info("backbone-params : %s", args.backbone_params)
    main_logger.info("attn-params     : %s", args.attn_params)
    main_logger.info("output-dir      : %s", args.output_dir)
    main_logger.info(
        "max_epochs=%d  patience=%d (no warm-up)  threshold=%.2f  seed=%d  "
        "final_eval_fp32=True",
        MAX_EPOCHS, ES_PATIENCE, THRESHOLD, SEED,
    )

    if not args.cv_manifest.exists():
        main_logger.error("CV manifest not found: %s", args.cv_manifest)
        sys.exit(1)
    if not args.backbone_params.exists():
        main_logger.error(
            "backbone-params JSON not found: %s", args.backbone_params)
        sys.exit(1)
    if not args.attn_params.exists():
        main_logger.error("attn-params JSON not found: %s", args.attn_params)
        sys.exit(1)

    with open(args.cv_manifest, "r", encoding="utf-8") as f:
        splits = json.load(f)
    hp_payload = load_best_params(
        args.backbone_params, args.attn_params, main_logger,
    )

    fold_indices = [int(x) for x in args.folds.split(",") if x.strip()]
    for k in fold_indices:
        assert 0 <= k < N_FOLDS, f"fold index {k} out of range [0,{N_FOLDS})"

    per_fold: list = []
    for k in fold_indices:
        fold_logger = setup_logger(
            f"cv_fold{k}_attn", output_dir / f"fold{k}" / "train.log"
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
