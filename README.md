# SSRA: Support-Selection Reliability Assessment for Few-Shot Industrial Anomaly Detection

Reproducibility package for the BIC-TA 2026 paper *"How Much Does the Normal Reference Set Matter? A Cross-Detector Assessment of Support-Selection Instability in Few-Shot Industrial Anomaly Detection"*.

SSRA is an assessment and reporting paradigm, not a new detector: it freezes the identities of the few normal reference images (the *support set*), evaluates each detector under five prespecified support choices per condition, and reports the mean, dispersion, range, worst support and provenance of every configuration. Across 315 evaluated configurations on VisA and BTAD with two prompt-based detectors (PromptAD and WinCLIP+), support selection alone moves image-level AUROC by up to 27–29 pp in the most variable categories — a magnitude comparable to the gaps that separate published methods.

## Repository layout

```
SSRA-BICTA2026/
├── README.md                          # this file
├── REPRODUCE.md                       # full reproduction guide (315 configurations)
├── LICENSE                            # MIT
├── configuration_results_315.csv      # master ledger: all 315 distinct evaluated configurations
├── support_training_matrix_45.csv     # 45-cell support x training-seed matrix (15 reuse flags)
├── MANIFEST.csv                       # SHA256 of every file in this package
├── frozen_manifests/                  # JSON support/test splits per (dataset, shot, support seed)
│   ├── visa_1cls_shot{1,2,4}_supportseed{3407,42,2026,123,2024}.json
│   └── btad_shot{1,2,4}_supportseed{3407,42,2026,123,2024}.json
├── test_lists/                        # frozen official test partitions (VisA 1cls, BTAD index)
├── statistics/                        # recomputed summaries (CPU-only, no GPU required)
│   ├── per_category_recomputed.csv    # mean/SD/range/worst/best per condition
│   ├── factorial_recomputed.csv       # 5x3 support-by-training summaries
│   ├── cohen_kappa.csv                # 2x2 label agreement (8/3/0/1)
│   ├── threshold_sensitivity.csv      # 3x3 neighbouring-cutoff grid
│   ├── bootstrap_recomputed.csv       # seeded category-reweighting intervals
│   ├── leave_one_category_out.csv     # fixed-category omission estimates
│   ├── mean_dispersion_association.csv# Spearman mean-vs-dispersion
│   ├── cross_detector_labels.csv      # per-category sensitivity labels, both detectors
│   └── ...
├── sources/                           # frozen copies of raw experiment CSVs
├── evaluation_sources/                # shared metric code (AUPRO convention, WinCLIP+ runner)
├── winclip_image_scores/              # 120 retained per-image score/label files (WinCLIP+)
└── protocol_records/                  # prespecified protocol and preregistration notes
```

## Requirements

Two tiers of reproduction are supported.

**Tier 1 — statistical verification (recommended first).** Python ≥ 3.10, standard library only. No datasets, no pretrained weights, no GPU. This re-derives every per-category summary, the sensitivity-rule counts, Cohen's κ, the threshold grid, seeded bootstrap endpoints, leave-one-category-out estimates, Spearman associations and the 5×3 factorial summaries from the committed ledgers.

**Tier 2 — score-level recomputation.** Python ≥ 3.10 with NumPy and scikit-learn. Re-derives all 120 retained WinCLIP+ image-level AUROC values from `winclip_image_scores/`.

**Tier 3 — full retraining (optional).** Requires the datasets and pretrained weights listed in REPRODUCE.md, a CUDA GPU, and the detector implementations at the pinned versions (PromptAD source; WinCLIP+ via Anomalib 2.6.2 with OpenCLIP 2.24.0).

## Quick start

```bash
# Tier 1: verify all reported statistics from the committed ledgers
python reproduce_statistics.py

# Tier 2: independently recompute the 120 WinCLIP+ image-level AUROC values
python recompute_winclip_iauroc.py
```

Both scripts are deterministic (fixed seeds, no sampling) and print a per-check summary. Tier 1 completes in well under a minute on a laptop CPU.

For full retraining, see REPRODUCE.md for dataset placement, the frozen manifest format, and the seed protocol.

## Data and weights

This repository provides **scripts, frozen manifests, and experiment ledgers only**. The VisA and BTAD datasets and all pretrained weights must be obtained from their official distribution channels by the user. Support and test membership in `frozen_manifests/` and `test_lists/` is expressed as dataset-relative paths and SHA256 hashes, so the exact images used in the paper can be identified after lawful download. Full datasets, pretrained weights, training checkpoints and full WinCLIP+ pixel maps are **not** included.

## Correspondence to the paper

| Paper element | Source file(s) in this repository |
|---|---|
| Table 1 (VisA one-shot five-support summaries) | `statistics/per_category_recomputed.csv` (PromptAD, VisA, shot 1); raw rows in `configuration_results_315.csv` |
| Table 2 (six categories, 1/2/4-shot, both detectors) | `statistics/complete_cross_detector_shot_response.csv`; raw rows in `configuration_results_315.csv` |
| Table 3 (5×3 support-by-training matrix) | `statistics/factorial_recomputed.csv`; all 45 cells with reuse flags in `support_training_matrix_45.csv` |
| Table 4 (BTAD combined) | `statistics/per_category_recomputed.csv` (BTAD rows) |
| Fig. 1 (VisA one-shot per-outcome plot) | `configuration_results_315.csv` (PromptAD, VisA, shot 1) |
| Fig. 2 (cross-detector means and SDs) | `statistics/cross_detector_labels.csv`, `winclip_image_scores/` |
| Fig. 3 (per-metric support SDs) | `statistics/per_category_recomputed.csv` (I_AUROC / P_AUROC / AUPRO) |
| κ = 0.308, counts 11/12 vs 8/12 | `statistics/cohen_kappa.csv`, `statistics/cross_detector_labels.csv` |
| Threshold grid (counts 11–12 / 8–9, κ 0–0.308) | `statistics/threshold_sensitivity.csv` |
| Bootstrap and leave-one-category-out intervals | `statistics/bootstrap_recomputed.csv`, `statistics/leave_one_category_out.csv` |

Every numerical claim in the paper traces to one of these files; `MANIFEST.csv` records the SHA256 of each distributed file.

## Scope and limitations

The study covers two prompt-based pretrained detectors (PromptAD; WinCLIP+ as an Anomalib implementation) on VisA and BTAD; the findings do not extend to other few-shot detector families. All dispersion statistics are descriptive: no ANOVA model is fitted and no significance test is attached to a dispersion contrast. Support samples are drawn deterministically from pre-defined seeds rather than by fully random sampling. See the paper's Limitations section for the complete list.

## License

Released under the [MIT License](LICENSE).

## Citation

```bibtex
@inproceedings{ssra2026bicta,
  title     = {How Much Does the Normal Reference Set Matter? A Cross-Detector
               Assessment of Support-Selection Instability in Few-Shot Industrial
               Anomaly Detection},
  author    = {TODO: author list},
  booktitle = {Proceedings of the International Conference on Bio-inspired
               Computing: Theories and Applications (BIC-TA 2026)},
  series    = {Lecture Notes of the Institute for Computer Sciences, Social
               Informatics and Telecommunications Engineering (LNICST)},
  year      = {2026},
  note      = {Code and evidence package: https://github.com/L-123DAN/SSRA-BICTA2026}
}
```
