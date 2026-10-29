# EEG Emotion Project Pipeline Summary

Fusion run: `/Users/likithkumar5178/Desktop/EEG_Emotion_Project/models/fusion/run_20261027_130329`

All reported model comparisons use trial-level predictions on the same fixed validation/test trial assignments. The held-out test set has four trials (one per class); its scores are descriptive and do not establish reliable generalization.

## Training performance

| Branch | Saved training metric |
|---|---|
| EEGNet + GCN | Window accuracy 0.7794 |
| Raw EEG Transformer | Trial accuracy 1.0000 (reported completed baseline) |
| STFT CRNN | Accuracy 0.6261 (reported completed baseline) |
| STFT Transformer | Not present in saved run artifacts |

Training metrics come from existing run metadata or the completed baseline values supplied for this project. Training window accuracy and trial accuracy are different measures and are labeled accordingly; unavailable values are not inferred.

## Validation performance

| Model | Input | Trial accuracy | Macro F1 |
|---|---|---:|---:|
| EEGNet + GCN | Raw EEG | 0.5000 | 0.3333 |
| Raw EEG Transformer | Raw EEG | 0.5000 | 0.3750 |
| STFT CRNN | STFT EEG | 0.5000 | 0.3333 |
| STFT Transformer | STFT EEG | 0.5000 | 0.3750 |
| Fusion | Raw EEG + STFT EEG | 0.7500 | 0.6667 |

## Held-out test performance

| Model | Input | Trial accuracy | Balanced accuracy | Macro F1 |
|---|---|---:|---:|---:|
| EEGNet + GCN | Raw EEG | 0.5000 | 0.5000 | 0.3750 |
| Raw EEG Transformer | Raw EEG | 0.5000 | 0.5000 | 0.3750 |
| STFT CRNN | STFT EEG | 0.2500 | 0.2500 | 0.1250 |
| STFT Transformer | STFT EEG | 0.5000 | 0.5000 | 0.3750 |
| Fusion | Raw EEG + STFT EEG | 0.5000 | 0.5000 | 0.3750 |

## Fusion configuration and status

- Status: `evaluated_once`; test evaluation was already completed once and was not repeated by this report export.
- Class order: `['fear', 'happy', 'neutral', 'sad']`.
- Weights: `{'EEGNet + GCN': 0.25, 'Raw EEG Transformer': 0.25, 'STFT CRNN': 0.25, 'STFT Transformer': 0.25}`.
- Fusion confidence is the maximum fused score, not calibrated probability or uncertainty.
- Individual models and fusion are complete as baseline experiments; no optimization was performed.

## Window counts by split and branch

| Split | EEGNet + GCN | Raw EEG Transformer | STFT CRNN | STFT Transformer |
|---|---:|---:|---:|---:|
| Validation | 700 | 700 | 141 | 141 |
| Test | 1014 | 1014 | 209 | 209 |

Detailed results: `/Users/likithkumar5178/Desktop/EEG_Emotion_Project/models/fusion/run_20261027_130329/results.json`
Human-readable fusion report: `/Users/likithkumar5178/Desktop/EEG_Emotion_Project/models/fusion/run_20261027_130329/fusion_report.md`
