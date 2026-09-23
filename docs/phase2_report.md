# Phase 2 — Separate chemistry from drift

**Status:** analysis of the completed Phase 2 evaluations, 2026-09-24.

Phase 2 asks whether the regression signal transfers to a station that was not
seen during training, to a month that was not seen during training, or to a
station-month combination that was not seen during training. This separates
station heterogeneity from temporal or instrument effects before investing in
more complex models.

The runs use the processed EEM data and the `EEMpca` representation with 30
components. The nonzero EEM mask, standardization and PCA are fitted inside
each training fold. The processed EEM data had already gone through the data
preparation/calibration workflow; Phase 2 does not add another calibration
step.

Each run evaluated the same four classical models (`linear`, `svr`, `tree` and
`xgboost`), eight EEMpca/tabular feature variants, and both global-mean and
station-mean baselines. The tables below show the plain `EEMpca` representation
so that the transfer comparison is easy to interpret. The other feature
variants remain available in the complete evaluation outputs and were not
selected using held-out results.

## Protocols and data coverage

| Evaluation | Protocol | Groups / split | Effective evaluation set |
|---|---|---|---|
| Leave-one-station-out | LOSO | 45 station groups; two singleton station folds cannot produce R² with the current minimum of two test samples | 43 evaluable folds, 758 test observations |
| Leave-one-month-out | LOMO | 12 month groups | 12 folds, 760 test observations |
| Two-way holdout | unseen station and unseen month, seed 42 | strict station/month holdout | 470 train, 32 test, 258 excluded observations |

For two-way holdout, the validation portion is merged into the training set
before fitting. Samples belonging to only the held-out station or only the
held-out month are excluded; the test set contains samples belonging to both.
There is one two-way test split, so it has no fold-to-fold standard deviation.

## How to read the tables

For LOSO and LOMO, each cell has the form:

```text
arithmetic mean R² ± standard deviation / pooled R²
```

The arithmetic mean gives every held-out group equal weight. Pooled R² is
computed after concatenating all held-out predictions and therefore weights
groups by their number of observations. They answer different questions and
can diverge substantially here. The two-way table reports the single test-set
R².

## Leave-one-station-out (LOSO)

| Target | Global mean | Station mean | EEMpca + LR | EEMpca + SVR | EEMpca + tree | EEMpca + XGB |
|---|---:|---:|---:|---:|---:|---:|
| BOD | -11.012 ± 18.607 / -0.030 | -11.012 ± 18.607 / -0.030 | -6.148 ± 21.615 / -0.439 | -2.851 ± 10.345 / 0.172 | -17.690 ± 82.248 / -0.814 | -7.250 ± 36.613 / -0.202 |
| COD | -18.119 ± 37.995 / -0.033 | -18.119 ± 37.995 / -0.033 | -2.962 ± 8.414 / 0.191 | -1.720 ± 5.632 / 0.410 | -6.654 ± 16.764 / -0.356 | -3.404 ± 10.979 / 0.200 |
| TOC | -13.401 ± 23.393 / -0.030 | -13.401 ± 23.393 / -0.030 | -3.269 ± 11.731 / 0.032 | -1.833 ± 4.781 / 0.360 | -6.066 ± 16.238 / -0.592 | -3.298 ± 12.113 / 0.218 |
| BOD_COD | -1.065 ± 1.660 / -0.024 | -1.065 ± 1.660 / -0.024 | -2.251 ± 7.072 / -0.434 | -0.607 ± 2.355 / 0.194 | -3.206 ± 6.324 / -0.925 | -1.356 ± 5.111 / -0.037 |

The arithmetic mean is negative for every LOSO model and target. Pooled R² is
positive for COD, TOC and BOD with SVR, but the large gap between the mean and
pooled values shows that a small number of difficult stations dominate the
equal-weight summary. BOD and BOD_COD remain below the global baseline even in
the pooled summary. The station baseline is identical to the global baseline
in LOSO because the held-out station has no target values available in the
training fold; the implementation correctly falls back to the global mean.

## Leave-one-month-out (LOMO)

| Target | Global mean | Station mean | EEMpca + LR | EEMpca + SVR | EEMpca + tree | EEMpca + XGB |
|---|---:|---:|---:|---:|---:|---:|
| BOD | -0.083 ± 0.133 / -0.007 | 0.248 ± 0.376 / 0.422 | -0.043 ± 0.710 / 0.202 | 0.410 ± 0.191 / 0.505 | -0.507 ± 0.976 / -0.131 | 0.232 ± 0.291 / 0.370 |
| COD | -0.153 ± 0.199 / -0.014 | 0.328 ± 0.414 / 0.488 | 0.321 ± 0.509 / 0.501 | 0.584 ± 0.194 / 0.614 | -0.333 ± 1.102 / 0.114 | 0.447 ± 0.171 / 0.511 |
| TOC | -0.128 ± 0.137 / -0.014 | 0.389 ± 0.331 / 0.461 | 0.319 ± 0.443 / 0.437 | 0.555 ± 0.224 / 0.539 | -0.369 ± 1.657 / 0.038 | 0.410 ± 0.392 / 0.482 |
| BOD_COD | -0.061 ± 0.061 / -0.009 | 0.226 ± 0.283 / 0.308 | -0.151 ± 0.536 / -0.015 | 0.182 ± 0.206 / 0.256 | -1.108 ± 1.263 / -0.670 | -0.040 ± 0.418 / 0.092 |

Month transfer is substantially stronger than station transfer. SVR and XGB
produce positive pooled R² for BOD, COD and TOC. The station baseline is also
strong in LOMO because each station still appears in the training months; it
captures a station-specific target level without using EEM. BOD_COD remains
harder, with only SVR giving a positive pooled R² of 0.256 among the EEMpca
models shown.

## Two-way holdout: unseen station and unseen month

| Target | Global mean | Station mean | EEMpca + LR | EEMpca + SVR | EEMpca + tree | EEMpca + XGB |
|---|---:|---:|---:|---:|---:|---:|
| BOD | -0.208 | -0.208 | -5.784 | -1.255 | -6.817 | -1.188 |
| COD | -0.221 | -0.221 | -4.287 | -1.066 | -13.331 | -2.319 |
| TOC | -0.282 | -0.282 | -9.881 | -1.685 | -7.027 | -4.992 |
| BOD_COD | -0.049 | -0.049 | -7.986 | -0.176 | -6.631 | -1.929 |

All EEMpca models are below the global baseline in the strict two-way test.
SVR is closest to the baseline, but it is still negative for every target.
This is a strong warning against treating the random or within-station result
as performance for a completely new station and month.

### Descriptive feature-variant check

The runs also contain seven EEMpca plus tabular augmentations. The following
are the highest pooled R² values found among those variants and models; they are
descriptive only because the same held-out folds were used to identify them.

| Protocol | Target | Highest pooled R² | Feature + model |
|---|---|---:|---|
| LOSO | BOD | 0.211 | `EEMpca_Temp_pH_SS_EC` + SVR |
| LOSO | COD | 0.486 | `EEMpca_Temp_pH_SS_EC` + XGB |
| LOSO | TOC | 0.420 | `EEMpca_Temp_SS_EC` + XGB |
| LOSO | BOD_COD | 0.252 | `EEMpca_SS_EC` + SVR |
| LOMO | BOD | 0.549 | `EEMpca_pH_SS_EC` + SVR |
| LOMO | COD | 0.643 | `EEMpca_Temp_pH_SS_EC` + SVR |
| LOMO | TOC | 0.631 | `EEMpca_Temp_SS_EC` + XGB |
| LOMO | BOD_COD | 0.309 | `EEMpca_SS_EC` + SVR |
| Two-way | BOD | -0.366 | `EEMpca_Temp_SS_EC` + tree |
| Two-way | COD | -0.530 | `EEMpca_pH_SS_EC` + XGB |
| Two-way | TOC | -1.625 | `EEMpca_pH_SS_EC` + SVR |
| Two-way | BOD_COD | -0.096 | `EEMpca_Temp_pH_SS_EC` + SVR |

The augmentations improve the descriptive LOMO/LOSO upper bounds for some
targets, but none beats the global baseline in the two-way split. These values
should not replace the fixed EEMpca comparison until the feature choice is
validated with a separate selection procedure.

## Target-range extrapolation by station

The target-by-station diagnostic compares each held-out station's target range
with the range in the other stations. A station is marked when at least one of
its target values is outside the LOSO training range:

| Target | Stations with an outside-range value | Interpretation |
|---|---:|---|
| BOD | 1 / 45 | one value outside the other-station range |
| COD | 1 / 45 | one value outside the other-station range |
| TOC | 2 / 45 | one value in each flagged station |
| BOD_COD | 2 / 45 | 4 values for Man-u stream and 2 for Okgu stream |

No flagged station has its entire target range outside the training range
(`fully_outside_train_range=False`). The result therefore points mainly to a
small number of extreme observations rather than a complete target shift for
an entire station. These observations can nevertheless have a large influence
on R².

## EEM distance and LOSO performance

The EEM-distance diagnostic measures the distance from each held-out station
to the training-station centroids in a fold-local EEM PCA space and joins it to
the per-fold R². The table below uses the
Spearman correlation for distance to the nearest training station; negative
values mean that more distant stations tend to have lower R².

| Target | LR ρ (p) | SVR ρ (p) | Tree ρ (p) | XGB ρ (p) |
|---|---:|---:|---:|---:|
| BOD | -0.195 (0.211) | -0.283 (0.066) | -0.220 (0.156) | **-0.422 (0.005)** |
| COD | 0.000 (0.998) | -0.265 (0.086) | -0.156 (0.319) | **-0.307 (0.045)** |
| TOC | -0.010 (0.948) | -0.164 (0.293) | -0.284 (0.065) | **-0.337 (0.027)** |
| BOD_COD | **-0.434 (0.004)** | **-0.510 (<0.001)** | **-0.507 (<0.001)** | **-0.662 (<0.001)** |

The association is strongest and most consistent for BOD_COD. It is present for
XGB on BOD, COD and TOC, but is weaker or inconsistent for the other models.
This supports station-level domain shift as one explanation for poor LOSO
performance. It does not by itself prove that the shift is biological: station
distance can also reflect sampling conditions, matrix differences or
measurement/session effects.

## Phase 2 conclusion

The completed runs support the following conclusion:

1. **Station transfer is the main limitation.** LOSO arithmetic means are
   negative for all EEMpca models and targets, and many individual station
   folds have negative R².
2. **Month transfer is more successful.** LOMO pooled R² is positive for most
   EEMpca/model combinations, especially SVR and XGB for COD and TOC.
3. **The strict deployment case is not solved.** Two-way R² is negative for
   every target and model in the reported EEMpca comparison.
4. **Station-specific structure is measurable in EEM.** EEM distance has a
   negative monotonic association with LOSO R², particularly for BOD_COD.
5. **The evidence is compatible with either biological heterogeneity or a
   station-linked confounder.** The Phase 2 result alone cannot distinguish
   these explanations.

The practical interpretation is that a model trained on existing stations can
be useful for interpolation to new months at those stations, but current
performance does not support reliable extrapolation to a new station. If the
intended deployment is a new station, LOSO and two-way results should be the
primary performance claims; random or within-station results should not be
used as the deployment estimate.

## Recommended next steps

- Inspect the per-station predictions produced by the LOSO evaluation alongside
  EEM distance, target range and sample count.
- Verify the flagged target extremes and the two singleton stations against
  the mapping metadata and laboratory records.
- Cluster stations in EEM space and run leave-one-cluster-out evaluation. This
  tests whether a new station that is chemically similar to a training station
  transfers better than a distant station.
- Add station-independent environmental covariates or collect a small
  calibration set from a new station if the deployment requires station
  transfer. A station ID or station mean cannot solve a genuinely unseen
  station without calibration observations.
- Keep the same LOSO/LOMO/two-way protocols when comparing later preprocessing
  or model changes. Do not select a model using the held-out folds.

## Limitations

- These are direct, symmetric diagnostic comparisons; no nested 5-fold model
  selection was applied. Every model used the same folds and representation.
- LOSO has two singleton station groups. They are present in the split manifest
  but are omitted from R² aggregation because a one-observation test set cannot
  produce a meaningful R² under the current metric guard.
- Two-way holdout has one seed and only 32 test observations, so its R² is a
  strict diagnostic rather than a precise population estimate.
- EEM-distance correlations use 43 overlapping LOSO training sets and should
  be treated as exploratory associations. The p-values are not corrected for
  the several targets, models and distance definitions tested.
- The report focuses on the plain EEMpca representation. Feature augmentations
  were evaluated as descriptive comparisons but were not ranked as a final
  model from the same held-out results.
