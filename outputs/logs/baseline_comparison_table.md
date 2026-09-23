# Baseline Comparison (video-level test-set metrics)

| Model                                         | Accuracy | Precision | Recall |     F1 |    AUC |
|-----------------------------------------------|----------|-----------|--------|--------|--------|
| RGB-only (baseline)                           |   0.5000 |    0.5000 | 1.0000 | 0.6667 | 0.5000 |
| Frequency-only (baseline)                     |   0.5000 |    0.5000 | 1.0000 | 0.6667 | 0.5000 |
| Dual-branch + concat fusion (ablation)        |   0.5000 |    0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Dual-branch + gated fusion (proposed)         |   0.5000 |    0.0000 | 0.0000 | 0.0000 | 0.0000 |
