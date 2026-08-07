# UPENN-GBM 10-case validation

This directory contains the lightweight, public evidence for a completed
five-fold nnU-Net inference evaluation. It intentionally excludes source MRI,
model weights, and prediction masks.

## Experiment

- Dataset: UPENN-GBM from The Cancer Imaging Archive (TCIA)
- DOI: `10.7937/TCIA.709X-DN49`
- Source license: CC BY 4.0
- Ground truth: expert-reviewed `images_segm` masks
- Cohort: 10 baseline cases from an institution independent of the model's
  reported BraTS training source
- GPU: Tesla T4
- nnU-Net: `2.8.1`
- Folds: `0,1,2,3,4`
- Output label profile: `brats19_preserved`
- Empty-region policy: prediction and target both empty gives Dice 1.0

## Dice

| Case | WT | TC | ET |
| --- | ---: | ---: | ---: |
| `UPENN-GBM-00002_11` | 0.920917 | 0.961514 | 0.901957 |
| `UPENN-GBM-00006_11` | 0.910697 | 0.959626 | 0.843595 |
| `UPENN-GBM-00008_11` | 0.814566 | 0.912595 | 0.860890 |
| `UPENN-GBM-00009_11` | 0.900697 | 0.935589 | 0.926011 |
| `UPENN-GBM-00011_11` | 0.953159 | 0.911932 | 0.893439 |
| `UPENN-GBM-00013_11` | 0.855954 | 0.919185 | 0.823042 |
| `UPENN-GBM-00014_11` | 0.939691 | 0.933684 | 0.807506 |
| `UPENN-GBM-00016_11` | 0.916600 | 0.936259 | 0.898694 |
| `UPENN-GBM-00017_11` | 0.829754 | 0.841090 | 0.844820 |
| `UPENN-GBM-00018_11` | 0.910835 | 0.875254 | 0.821437 |
| **Macro average** | **0.895287** | **0.918673** | **0.862139** |

Machine-readable results:

- [`evaluation.json`](evaluation.json)
- [`evaluation_cases.csv`](evaluation_cases.csv)

## Scope

This is an independent-institution, 10-case sample and not a clinical
validation. The external model package does not include a complete training
cohort manifest or a clinical review record, so these results must not be used
to claim clinical readiness.
