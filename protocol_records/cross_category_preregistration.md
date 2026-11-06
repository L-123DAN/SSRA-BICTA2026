# Cross-category support-sampling preregistration

Source: user's pasted request received before the new cross-category runs on 2026-10-04 (Asia/Shanghai). This file records the fixed protocol and decision rule; it is not derived from observed new results.

- Categories, in execution order: cashew, candle, capsules, pcb1, macaroni1, pipe_fryum.
- Support seeds, fixed in advance: 3407, 42, 2026, 123, 2024.
- Dataset: official VisA 1cls split; only one normal training support image changes within a category. No support image may occur in the test partition. Seed collisions must be reported rather than altered.
- Training: PromptAD commit `0f86ce0dc1ed59007d51348d8d566aed31360cf9`, ViT-B-16-plus-240/laion400m_e32, 1-shot, training seed 111, 100 epochs, fixed-final checkpoint. Existing image resize/crop 240, output resolution 400, batch size 4, optimizer/loss/model unchanged.
- Primary metric: I-AUROC. Secondary metrics: P-AUROC and AUPRO. Values are percentages; across-seed standard deviation and range are percentage points (pp). Sample standard deviation uses ddof=1.
- Category label: SUPPORT-SENSITIVE if I-AUROC sample SD ≥1.0 pp **or** range ≥2.0 pp; otherwise RELATIVELY-STABLE.
- GO: at least four of six categories SUPPORT-SENSITIVE and either cross-category median I-AUROC SD ≥0.8 pp or at least four categories have I-AUROC range ≥2.0 pp.
- CONDITIONAL GO: two or three categories SUPPORT-SENSITIVE, or multiple categories fluctuate with cross-category median I-AUROC SD from 0.4 to below 0.8 pp.
- NO-GO: at most one category SUPPORT-SENSITIVE, median I-AUROC SD <0.4 pp, and a clear majority (interpreted conservatively as at least five of six categories) has I-AUROC range <1.0 pp.
- These thresholds are project direction rules, not significance tests. If observations fall outside every specified case, report that gap explicitly rather than invent a new threshold.
- No algorithm changes, hyperparameter tuning, seed selection, 2/4-shot formal runs, test-AUROC checkpoint selection, or early stopping based on results.

The original user's detailed request is stored in the conversation attachment. Command-level execution and any failures are recorded separately.
