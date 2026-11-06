# REPRODUCE.md — Reproducing the 315 Evaluated Configurations

This document explains how the 315 distinct evaluated configurations in the paper were specified, how the committed ledgers map to them, and how to reproduce the results at three levels of effort.

**Terminology.** An *evaluated configuration* is one final metric outcome produced under a frozen protocol; it is **not** 315 newly trained networks. Where a configuration reuses an outcome that was already audited (e.g., seed-111 rows inside the support-by-training matrix), the reuse is flagged in the ledger rather than double-counted.

---

## 1. Composition of the 315 configurations

| Arm | Configurations | Notes |
|---|---|---|
| PromptAD, VisA, one-shot | 60 | 12 categories × 5 support seeds |
| PromptAD, VisA, two-/four-shot | 60 | 6 categories × {2, 4} shots × 5 support seeds |
| PromptAD, VisA, support-by-training matrix | 30 | 45-cell matrix minus 15 reused seed-111 outcomes |
| WinCLIP+, VisA (Anomalib 2.6.2 implementation) | 120 | includes a 15-configuration pilot on candle, pcb1, cashew |
| PromptAD, BTAD | 45 | 3 categories × {1, 2, 4} shots × 5 support seeds |
| **Total (distinct)** | **315** | |

The master ledger is `configuration_results_315.csv`; the 45 matrix cells (with 15 reuse flags) are additionally broken out in `support_training_matrix_45.csv`.

## 2. Seed protocol

**Support seeds.** Five prespecified support seeds are used everywhere: `3407, 42, 2026, 123, 2024`. For every (dataset, category, shot, support seed), the corresponding JSON in `frozen_manifests/` lists the exact support image paths (dataset-relative) drawn deterministically, without replacement, from the sorted normal-training file list. Higher-shot supports follow nested manifests (the 1-shot support is contained in the 2-shot and 4-shot sets). **No support is selected by its test score, and all draws are deterministic — there is no fully random sampling anywhere in the protocol.**

**Training seeds.** PromptAD was trained with training seed `111` as the default. The support-by-training matrix crosses the same five support seeds with training seeds `111, 222, 333` in three categories (candle, pcb1, cashew), giving 45 cells; the 15 seed-111 outcomes were verified before reuse and are flagged `matrix_reused=1` in the ledger.

**Shot-study categories.** Six categories — cashew, candle, capsules, pcb1, macaroni1, pipe_fryum — were fixed for the one/two/four-shot analysis before its additional runs (preregistration notes in `protocol_records/`).

**Training protocol.** 100 epochs, batch size four, fixed final checkpoint; intermediate test scores never select the checkpoint. PromptAD source at a pinned revision, ViT-B-16-plus-240/laion400m_e32 backbone. WinCLIP+ via Anomalib 2.6.2 with OpenCLIP ViT-B-16-plus-240/laion400m_e31, frozen support paths passed explicitly, default reference sampling disabled, anomaly maps resized to a 400×400 grid for pixel metrics.

## 3. Ledger field reference

### `configuration_results_315.csv` / `support_training_matrix_45.csv`

| Column | Meaning |
|---|---|
| `detector` | `PromptAD` or `WinCLIP+` |
| `dataset` | `VisA` or `BTAD` |
| `category` | Dataset category label (VisA: 12 industrial categories; BTAD: `01`/`02`/`03`) |
| `shot` | Number of normal support images: 1, 2 or 4 |
| `support_seed` | One of {3407, 42, 2026, 123, 2024}; resolves via `frozen_manifests/` to exact image paths |
| `training_seed` | 111 (default) or 222/333 in the matrix arm |
| `support_paths` | JSON list of dataset-relative support image paths for this trial |
| `I_AUROC` | Image-level AUROC, percent |
| `P_AUROC` | Pixel-level AUROC, percent |
| `AUPRO` | Per-region overlap, percent (frozen `cal_pro_metric` convention; see paper §4.2) |
| `matrix_reused` | `1` if the row is a seed-111 outcome reused inside the matrix arm, else `0` |

### `statistics/per_category_recomputed.csv`

Per-condition summaries over the five support trials: `mean`, sample standard deviation (`sd`, `n-1` denominator), `range` (max − min), `worst` and `best` outcomes, per (detector, category, shot, metric). Score columns are percentages; `sd` and `range` are percentage points (pp). This file is the direct source of Tables 1–4 in the paper.

### `winclip_image_scores/<category>/shot_<k>/support_<seed>/per_image.csv`

| Column | Meaning |
|---|---|
| `image` | Dataset-relative test image path |
| `label` | Ground-truth label (0 = normal, 1 = anomalous) |
| `pred_score` | Retained image-level anomaly score (higher = more anomalous) |
| `anomaly_map_sha256` | Hash of the anomaly map at scoring time (provenance only; maps not distributed) |

### `frozen_manifests/*.json`

Per (dataset, shot, support seed): `source` (original split CSV, host path anonymised), `support_seed`, `shot`, and per-category `support` / `test` image path lists. The `test` lists reproduce the official test partitions exactly. Host-specific absolute paths inside the manifests are anonymised with `LOCAL_USER` / `PROJECT_ROOT` placeholders; dataset-relative paths and SHA256 hashes are unaffected.

## 4. Reproduction tiers

### Tier 1 — Statistical verification (no data, no GPU)

```bash
python reproduce_statistics.py
```

Re-derives, from the committed CSVs only: all per-category five-statistic summaries; the prespecified one-shot sensitivity counts (11/12 for PromptAD, 8/12 for WinCLIP+); Cohen's κ with the (8,3,0,1) table; the 3×3 neighbouring-cutoff grid; exact seeded bootstrap endpoints for the category-reweighting intervals; leave-one-category-out estimates; Spearman mean–dispersion associations; and the 5×3 factorial marginal summaries. Deterministic; completes on any laptop CPU.

### Tier 2 — Score-level recomputation of the 120 WinCLIP+ outcomes

```bash
python recompute_winclip_iauroc.py
```

Reads the 120 retained per-image score/label files under `winclip_image_scores/` and independently recomputes each image-level AUROC, checking it against the ledger. NumPy and scikit-learn required.

### Tier 3 — Full retraining (optional, needs GPU + datasets)

1. Obtain VisA and BTAD from their official channels and place them so that dataset-relative paths in the manifests resolve (e.g., `datasets/VisA/<category>/Data/Images/...`).
2. Clone the PromptAD source at the revision recorded in the paper and the Anomalib 2.6.2 environment for WinCLIP+ (see paper §4.1 for exact library versions).
3. For each arm, iterate the manifest JSONs: train/score with the listed support images and training seed, evaluate on the frozen test list with the final checkpoint, and append one row per trial to the ledger.
4. Compare against `configuration_results_315.csv`; the statistical arm (Tier 1) will confirm that any faithful re-run reproduces the reported summaries.

AUPRO uses the frozen implementation in `evaluation_sources/` (200 score thresholds, 8-connected regions, FPR points ≤ 0.3, min–max rescaling before trapezoidal integration). It is provided to document the convention; full pixel maps were not retained, so AUPRO cannot be recomputed from this package alone.

## 5. Known scope limits

- Results cover two prompt-based pretrained detectors; they do not extend to feature-embedding, reconstruction-based or student–teacher few-shot detectors.
- Five support seeds cover a small fraction of admissible subsets; sample SDs at n = 5 carry wide intervals (95% CI of roughly [0.6, 2.9]× the estimate).
- Deterministic seed-based draws, not random sampling: conclusions describe the recorded support identities.
- BTAD used an adapter with an empty category-specific abnormal vocabulary, limiting direct VisA–BTAD comparison.

See the paper's Limitations section for the full list.
