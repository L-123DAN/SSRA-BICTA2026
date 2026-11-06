"""Descriptive full-factorial SSRA analysis; requires all 45 verified cells."""

import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/csv/ssra_support_training_raw.csv"
OUT = ROOT / "results/csv/ssra_support_training_summary.csv"
REPORT = ROOT / "docs/ssra_stage1_report.md"
FIG = ROOT / "results/figures"
CATEGORIES = ("candle", "pcb1", "cashew")
SUPPORTS = (3407, 42, 2026, 123, 2024)
TRAINING = (111, 222, 333)
BLUE, RED, GREEN = "#0F4D92", "#B64342", "#377E49"
FIELDS = ("category", "row_type", "support_seed", "training_seed", "n",
          "mean_I_AUROC", "sd_I_AUROC", "range_I_AUROC", "worst_I_AUROC",
          "best_I_AUROC", "sigma_support", "sigma_training", "Q",
          "SS_support", "SS_training", "SS_residual", "SS_total",
          "E_support", "E_training", "E_residual")


def describe(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), mean_I_AUROC=float(values.mean()),
                sd_I_AUROC=float(values.std(ddof=1)),
                range_I_AUROC=float(values.max() - values.min()),
                worst_I_AUROC=float(values.min()), best_I_AUROC=float(values.max()))


def matrix(rows, category):
    cells = {(int(row["support_seed"]), int(row["training_seed"])): row
             for row in rows if row["category"] == category}
    if len(cells) != 15 or set(cells) != {(s, t) for s in SUPPORTS for t in TRAINING}:
        raise ValueError(f"Incomplete/duplicate factorial matrix: {category}")
    for (support, training), row in cells.items():
        if (row["status"] != "success" or int(row["shot"]) != 1
                or int(row["epoch"]) != 100 or row["checkpoint_policy"] != "fixed-final"):
            raise ValueError(f"Invalid protocol/outcome: {category}/{support}/{training}")
    return np.array([[float(cells[s, t]["I_AUROC"]) for t in TRAINING] for s in SUPPORTS])


def analyze_one(category, a):
    if a.shape != (5, 3) or not np.isfinite(a).all():
        raise ValueError("Expected finite 5×3 I-AUROC matrix")
    grand = float(a.mean())
    by_support = a.mean(axis=1)
    by_training = a.mean(axis=0)
    sigma_support = float(by_support.std(ddof=1))
    sigma_training = float(by_training.std(ddof=1))
    ss_support = float(len(TRAINING) * np.square(by_support - grand).sum())
    ss_training = float(len(SUPPORTS) * np.square(by_training - grand).sum())
    nonadditive = a - by_support[:, None] - by_training[None, :] + grand
    ss_residual = float(np.square(nonadditive).sum())
    ss_total = float(np.square(a - grand).sum())
    if not math.isclose(ss_support + ss_training + ss_residual, ss_total,
                        rel_tol=1e-10, abs_tol=1e-8):
        raise AssertionError(f"SS decomposition does not sum: {category}")
    if ss_total <= 0:
        raise ValueError(f"Zero total variation: {category}; shares undefined")
    shares = [ss_support / ss_total, ss_training / ss_total, ss_residual / ss_total]
    if not math.isclose(sum(shares), 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError(f"Variation shares do not sum to one: {category}")
    overall = {"category": category, "row_type": "overall", "support_seed": "",
               "training_seed": "", **describe(a.ravel()),
               "sigma_support": sigma_support, "sigma_training": sigma_training,
               "Q": sigma_support / (sigma_training + 1e-8),
               "SS_support": ss_support, "SS_training": ss_training,
               "SS_residual": ss_residual, "SS_total": ss_total,
               "E_support": shares[0], "E_training": shares[1], "E_residual": shares[2]}
    details = []
    for ti, training in enumerate(TRAINING):
        details.append({"category": category, "row_type": "per_training_seed",
                        "support_seed": "", "training_seed": training,
                        **describe(a[:, ti])})
    for si, support in enumerate(SUPPORTS):
        details.append({"category": category, "row_type": "per_support_seed",
                        "support_seed": support, "training_seed": "",
                        **describe(a[si, :])})
    return overall, details


def figures(matrices, overall):
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 11,
                         "axes.spines.right": False, "axes.spines.top": False,
                         "axes.linewidth": 1.1, "savefig.facecolor": "white"})
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(13, 5.1), facecolor="white")
    for ax, category in zip(axs, CATEGORIES):
        a = matrices[category]
        im = ax.imshow(a, cmap="Blues", vmin=0, vmax=100)
        for si in range(5):
            for ti in range(3):
                ax.text(ti, si, f"{a[si, ti]:.2f}", ha="center", va="center",
                        color="white" if a[si, ti] > 85 else "black", fontsize=9)
        ax.set(xticks=range(3), xticklabels=TRAINING, yticks=range(5),
               yticklabels=SUPPORTS, xlabel="Training seed", title=category)
    axs[0].set_ylabel("Support seed")
    fig.colorbar(im, ax=axs, label="Image AUROC (%)", shrink=.78, pad=.02)
    fig.subplots_adjust(left=.08, right=.92, bottom=.12, top=.88, wspace=.32)
    fig.savefig(FIG / "ssra_support_training_matrix.png", dpi=300)
    plt.close(fig)

    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.4, 4.8), facecolor="white")
    ax.bar(x-.19, [r["sigma_support"] for r in overall], .38,
           label="Support-mean SD", color=BLUE, edgecolor="black", linewidth=.7)
    ax.bar(x+.19, [r["sigma_training"] for r in overall], .38,
           label="Training-mean SD", color=RED, edgecolor="black", linewidth=.7)
    ax.set(xticks=x, xticklabels=CATEGORIES, ylabel="Sample SD (pp)",
           title="Support versus training-seed variation")
    ax.legend(frameon=False)
    fig.tight_layout(pad=2)
    fig.savefig(FIG / "ssra_support_vs_training_sd.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), facecolor="white")
    left = np.zeros(3)
    for key, label, color in (("E_support", "Support", BLUE),
                              ("E_training", "Training", RED),
                              ("E_residual", "Residual / non-additive", GREEN)):
        vals = np.array([r[key] for r in overall])
        ax.barh(x, vals, left=left, label=label, color=color,
                edgecolor="black", linewidth=.6)
        left += vals
    ax.set(yticks=x, yticklabels=CATEGORIES, xlim=(0, 1),
           xlabel="Descriptive share of total SS", title="Sum-of-squares decomposition")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, -.32), ncol=3)
    fig.tight_layout(pad=2)
    fig.savefig(FIG / "ssra_variation_share.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    with RAW.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = [(r["category"], int(r["support_seed"]), int(r["training_seed"])) for r in rows]
    if len(rows) != 45 or len(set(keys)) != 45:
        raise ValueError(f"Stage 1 requires 45 unique cells, found {len(rows)}")
    matrices = {category: matrix(rows, category) for category in CATEGORIES}
    results = [analyze_one(category, matrices[category]) for category in CATEGORIES]
    overall = [r[0] for r in results]
    details = [row for _, items in results for row in items]
    support_dominant = sum(r["sigma_support"] > r["sigma_training"]
                           and r["E_support"] > r["E_training"] for r in overall)
    training_dominant = sum(r["sigma_training"] > r["sigma_support"]
                            and r["E_training"] > r["E_support"] for r in overall)
    decision = ("SUPPORT-DOMINANT" if support_dominant >= 2 else
                "TRAINING-DOMINANT" if training_dominant >= 2 else "MIXED")
    temp = OUT.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in [*overall, *details]:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    os.replace(temp, OUT)
    figures(matrices, overall)
    lines = ["# SSRA Stage 1 support-versus-training randomness", "",
             "Complete 3-category × 5-support-seed × 3-training-seed fixed-final matrix; 15 verified training-seed-111 results were reused and 30 new runs were executed. Each cell has one formal result. The values below are descriptive, not inferential ANOVA or causal variance attribution.", "",
             "| Category | σ support (pp) | σ training (pp) | Q | SS support | SS training | SS residual | E support | E training | E residual |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in overall:
        lines.append(f"| {r['category']} | {r['sigma_support']:.3f} | {r['sigma_training']:.3f} | {r['Q']:.3f} | {r['SS_support']:.3f} | {r['SS_training']:.3f} | {r['SS_residual']:.3f} | {r['E_support']:.3f} | {r['E_training']:.3f} | {r['E_residual']:.3f} |")
    lines += ["", f"**Overall: {decision}.** Support-dominant conditions hold jointly in {support_dominant}/3 categories; training-dominant conditions in {training_dominant}/3. The Stage 1 decision does not gate BTAD access.", "",
              "`Q=σ_support/(σ_training+1e−8)` is a descriptive variation ratio. The E values are descriptive sum-of-squares shares and sum to one within floating error for each category. With one observation per support×training cell, `SS_residual` combines non-additivity/interaction and unseparated random error. It is not an inferential two-way ANOVA residual. Per-training-seed support SD/range and per-support-seed training SD/range are recorded as dedicated rows in `../results/csv/ssra_support_training_summary.csv`.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": decision, "support_dominant_count": support_dominant,
                      "training_dominant_count": training_dominant,
                      "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
