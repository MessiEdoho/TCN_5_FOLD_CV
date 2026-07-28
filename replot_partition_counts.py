#!/usr/bin/env python3
"""Regenerate the per-partition seizure/segment-count barplot at
publication quality (single-manuscript-column friendly).

This is a standalone re-plotter: it reads the *exact* partition counts
already written by create_5fold_cv_splits.py to the stratification report
CSV (Section 1), so no values are recomputed or assumed. Use it to refresh
the figure without rerunning the whole splits build.

Outputs a vector PDF (preferred for LaTeX) and a 300-dpi PNG next to each
other. Point --csv at fold_stratification_report.csv and --out at the
desired output stem.

Example
-------
    python replot_partition_counts.py \
        --csv  /path/to/diagnostics/fold_stratification_report.csv \
        --out  /path/to/diagnostics/partition_seizure_segment_counts
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe in batch/headless
import matplotlib.pyplot as plt


# Fixed display order: the early-stop partition first, then the CV folds.
PARTITION_ORDER = ["early_stop", "fold_0", "fold_1", "fold_2", "fold_3",
                   "fold_4"]


def read_partition_summary(csv_path: Path) -> list[dict]:
    """Parse Section 1 (the per-partition summary) of the two-section
    stratification report CSV. Returns one dict per partition with the
    integer counts and prevalence needed for the plot."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    # Find the header line of the partition section (starts with "partition")
    header_idx = None
    for i, row in enumerate(rows):
        if row and row[0].strip() == "partition":
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(
            "Could not find a 'partition' header row in %s; is this the "
            "fold_stratification_report.csv?" % csv_path
        )

    header = [h.strip() for h in rows[header_idx]]
    records: list[dict] = []
    for row in rows[header_idx + 1:]:
        if not row or not row[0].strip():
            break  # blank line marks the end of Section 1
        if row[0].strip().startswith("#"):
            break
        rec = dict(zip(header, row))
        records.append({
            "partition":      rec["partition"].strip(),
            "n_mice":         int(float(rec["n_mice"])),
            "n_ictal":        int(float(rec["n_ictal"])),
            "n_nonictal":     int(float(rec["n_nonictal"])),
            "n_total":        int(float(rec["n_total"])),
            "prevalence_pct": float(rec["prevalence_pct"]),
        })
    return records


def make_plot(summary: list[dict], out_stem: Path) -> None:
    by_part = {r["partition"]: r for r in summary}
    labels = [p for p in PARTITION_ORDER if p in by_part]
    if not labels:  # fall back to whatever order the CSV gave us
        labels = [r["partition"] for r in summary]

    ictal     = [by_part[p]["n_ictal"]        for p in labels]
    nonictal  = [by_part[p]["n_nonictal"]     for p in labels]
    totals    = [by_part[p]["n_total"]        for p in labels]
    prev_pcts = [by_part[p]["prevalence_pct"] for p in labels]
    mice      = [by_part[p]["n_mice"]         for p in labels]

    # Large type + near-square aspect so the figure survives being scaled
    # down into one manuscript column and stays legible.
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        13,
        "axes.titlesize":   14,
        "axes.labelsize":   15,
        "xtick.labelsize":  13,
        "ytick.labelsize":  13,
        "legend.fontsize":  13,
    })

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    x = list(range(len(labels)))
    bar_width = 0.74

    ax.bar(
        x, ictal, width=bar_width, color="#d6604d", edgecolor="black",
        linewidth=0.8, label="ictal (seizure)",
    )
    ax.bar(
        x, nonictal, width=bar_width, bottom=ictal, color="#4393c3",
        edgecolor="black", linewidth=0.8, label="non-ictal",
    )

    for i, (tot, prev, nm) in enumerate(zip(totals, prev_pcts, mice)):
        ax.text(
            x[i], tot + max(totals) * 0.012,
            "%d mice\nn=%s\n%.1f%% ictal" % (nm, "{:,}".format(tot), prev),
            ha="center", va="bottom", fontsize=12, linespacing=1.35,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("segment count")
    ax.set_title(
        "Per-partition seizure and segment counts\n"
        "(stratified 5-fold CV + fixed early-stop partition, "
        "n_train_mice=%d)" % sum(mice),
        pad=12,
    )
    ax.set_ylim(0, max(totals) * 1.32)
    ax.margins(x=0.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=4)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.22),
        ncol=2, frameon=False,
    )

    fig.tight_layout()
    pdf = out_stem.with_suffix(".pdf")
    png = out_stem.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")               # vector, for LaTeX
    fig.savefig(png, dpi=300, bbox_inches="tight")       # raster preview
    plt.close(fig)
    print("Wrote:\n  %s\n  %s" % (pdf, png))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--csv", type=Path, required=True,
        help="Path to fold_stratification_report.csv (has the per-partition "
             "counts in Section 1).",
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path("partition_seizure_segment_counts"),
        help="Output path stem; .pdf and .png are appended. "
             "Default: ./partition_seizure_segment_counts",
    )
    args = ap.parse_args()

    if not args.csv.is_file():
        print("ERROR: CSV not found: %s" % args.csv, file=sys.stderr)
        return 1

    summary = read_partition_summary(args.csv)
    if not summary:
        print("ERROR: no partition rows parsed from %s" % args.csv,
              file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_plot(summary, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
