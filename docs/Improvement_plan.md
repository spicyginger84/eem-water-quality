# Improvement Plan

Status: proposed, 2026-09-22
Scope: `eem-water-quality` — predicting BOD, COD, TOC and BOD/COD from EEM fluorescence.

This plan replaces the results framing in `README.md`. It is based on a re-evaluation
of the committed dataset (760 samples, 45 stations) under grouped cross-validation.
Every number below is reproducible from `data/processed/`.

---

## 1. Where the project actually stands

The README reports a fixed random 80/20 split. Under that split **every one of the 41
stations in the test set also appears in training**, so the reported R² measures
interpolation within known sites, not generalisation to new ones. Model selection also
used the test partition, so the reported figures are the maximum over 12
experiment × model combinations evaluated on the same rows they are scored on.

Re-evaluated with grouped 5-fold CV over the 45 stations (test stations never seen in
training), XGBoost, 30 PCA components, pooled out-of-fold R²:

| Target | global mean | station mean | EEM (30 PCs) | EEM + SS + EC |
|---|---:|---:|---:|---:|
| BOD | −0.01 | −0.01 | −0.02 | 0.12 |
| **COD** | 0.00 | 0.00 | 0.29 | **0.34** |
| **TOC** | 0.00 | 0.00 | **0.31** | 0.27 |
| BOD/COD | −0.03 | −0.03 | −0.06 | 0.09 |

**This is good news, correctly stated.** EEM predicts COD and TOC on unseen stations at
R² ≈ 0.3 from the spectra alone. That is a real, transferable result. What does not
survive is BOD regression and BOD/COD regression.

### Results that do not survive re-evaluation

| Claim in README | Honest value | Why it changed |
|---|---|---|
| BOD, EEMpca+XGB = 0.455 | ≈ 0.00 above baseline | A station-mean predictor using **no EEM data** scores 0.452 on the same split |
| BOD/COD, best = 0.425 | ≈ 0.09, unstable | Selection-on-test, single-split luck, and station leakage, in that order |
| `EEMpca_TOC`, BOD = 0.737 / COD = 0.837 | It is the TOC covariate | TOC alone gives 0.700 / 0.862. The 30 spectral components add **+0.037 / −0.025** |

The `EEMpca_TOC*` family should not appear in a table evaluating EEM. TOC, COD and BOD
are mutually correlated at r = 0.78–0.89 because all three measure organic load;
regressing one on another is unit conversion, not modelling.

### The one result that is both robust and under-reported

BOD/COD is conventionally used as a **biodegradability class**, not a precise ratio.
Reframed as binary classification under the same grouped CV (XGBoost, `EEMpca_SS_EC`):

| Threshold | Positive class | Pooled AUC | Per-fold range |
|---|---:|---:|---|
| 0.3 | 64% | 0.696 | 0.63 – 0.72 |
| 0.4 | 37% | 0.785 | 0.73 – 0.83 |
| 0.5 | 24% | 0.789 | 0.73 – 0.83 |

AUC ≈ 0.79 on unseen stations, stable across folds — the most reliable finding in the
project. Physically coherent: fluorescence separates protein-like (labile) from
humic-like (refractory) material well, even when it cannot recover exact oxygen demand.

---

## 2. Three diagnostics that shape the plan

**(a) The spectrum is a strong site fingerprint.** Station identity is predictable from
30 PCs at **45.4%** accuracy across 40 stations (chance 2.5%). This is the mechanism
behind the inflated random-split scores.

**(b) There is a large month effect.** Month is predictable from the spectrum at
**55.9%** (chance 8.3%), and median total EEM intensity varies **2.0×** across months
(1186 → 2381). This is either genuine seasonality, instrument drift between monthly
measurement sessions, or both — and the current design cannot distinguish them.

**(c) The signal lives in absolute intensity, which is never calibrated.** Normalising
each sample by its total intensity destroys performance (TOC 0.31 → −0.05, COD
0.29 → 0.07). So the models are using overall brightness as a proxy for organic load.
That is physically sensible, but **absolute intensity is exactly the quantity that
drifts between sessions**, and the pipeline applies no blank subtraction, no
inner-filter correction and no Raman normalisation.

The central risk: the best result rests on an uncalibrated measurement, in a dataset
with a 2× session-to-session intensity spread.

---

## 3. Plan

### Phase 0 — Make results readable
*Effort: ~2 days. No new science; everything downstream is uninterpretable without it.*

- [ ] Enable grouped splitting from the CLI. `split_indices` already supports
      `mode="group"` on `Point`, but `cli.py` restricts `--split` to `random`, and
      `group_col` is never defined on the parser (it resolves to `None` via `getattr`).
- [ ] Move model selection to the real validation partition. `ml.py` sets
      `val = test = parts["test"]`, so `validation_metrics.csv` and
      `validation_predictions.csv` are byte-for-byte copies of the test artifacts and
      `selected.json["validation"]` holds a test-set score. The 122-row validation
      partition is computed, written to `splits.csv`, then discarded.
- [ ] Add a mandatory baseline column (global mean **and** station mean) to every
      results table, and report EEM contribution as a delta against it.
- [ ] Fix the excitation/emission axis swap (see §4).
- [ ] Resolve the 14 samples with BOD > COD; add `COD >= BOD` as a processing check.
- [ ] Mask the 5,928 of 15,447 flattened cells (38%) that are identically zero in every
      sample — the Rayleigh mask — before PCA.

**Deliverable:** a results table whose numbers mean what their column headers say.

### Phase 1 — Calibrate the spectra
*Effort: 1–2 weeks, conditional on raw files. Highest leverage.*

Apply the standard EEM correction chain, all of which is currently absent:

- [ ] Blank (Milli-Q) subtraction
- [ ] Inner-filter effect correction from absorbance scans
- [ ] **Raman normalisation to Raman Units** — the step that makes intensities
      comparable across sessions
- [ ] Rayleigh and Raman scatter removal with interpolation rather than zero-fill

**Dependency and risk:** this requires raw instrument output, blanks and absorbance
scans. The raw directory (`data/2026ER_data_for Viet`) is not in the repository, so
availability is unverified. If those files were not archived, this becomes a
collection-protocol change for future sampling and a hard ceiling on the present
dataset — worth establishing now rather than after another sampling year.

### Phase 2 — Separate chemistry from drift
*Effort: ~1 day. Cheap and decisive — run before investing in Phases 1, 3 or 4.*

- [ ] Leave-one-month-out CV, reported alongside leave-station-out
- [ ] Leave-one-station-out CV for per-station error, to identify which sites fail
- [ ] Two-way holdout (unseen station **and** unseen month) as the strictest case

If holding out whole months collapses performance while holding out stations does not,
the R² ≈ 0.3 is seasonal or instrumental rather than compositional. Given diagnostic
(b), this is a live possibility, and the answer determines whether Phase 1 is a
refinement or a rescue.

### Phase 3 — Physics-based features instead of PCA
*Effort: 1–2 weeks.*

Thirty opaque components over 15,447 correlated cells is the wrong representation when
the effective sample size is 45 sites.

- [ ] **PARAFAC, properly validated** — split-half analysis, core consistency, 3–7
      components. `ParafacFeatures` is already implemented and the catalog contains
      PARAFAC experiments, but none appear in the README results table. Components map
      to known fluorophores (humic-like C/A, tryptophan-like T, tyrosine-like B).
- [ ] **Fluorescence indices** — FI, BIX, HIX and Coble peak ratios. These are
      *ratios*, so they cancel multiplicative session drift; they are the natural
      complement to an intensity signal that is not fully trusted.
- [ ] **Fluorescence Regional Integration** (5 regions)

Target ~10–20 interpretable features. Lower dimension should transfer better, and peak
assignments can be defended in review in a way that PC7 cannot.

### Phase 4 — Model the actual question
*Effort: ~1 week.*

- [ ] Hierarchical model: station random intercept + EEM fixed effects
- [ ] Equivalently, predict the **within-station anomaly** directly

This separates "this site is usually polluted" from "the spectrum says something extra
today," which is the real research question, and it converts the leakage problem into
the study design. For BOD/COD, 61.4% of variance is within-station (the lowest
between-station share of the four targets: BOD 49.3%, TOC 52.9%, COD 55.8%) — that
within-station portion is precisely what EEM would have to explain, and it has never
been isolated.

### Phase 5 — Reframe the deliverable
*Effort: ~3 days.*

- [ ] Lead with COD and TOC at R² ≈ 0.3 on unseen stations
- [ ] Promote biodegradability classification (AUC ≈ 0.79) to a headline result
- [ ] Report BOD and BOD/COD regression as negative results
- [ ] Present ablations as deltas over baseline, with both split types side by side

A paper concluding *"EEM screens organic load and biodegradability class but cannot
replace BOD assays"* is more useful, and far more defensible, than one claiming
R² = 0.74. Note that much of the published EEM literature reports R² = 0.7–0.9 with the
same station leakage; correcting it will make these numbers look worse than the field's
while actually being better. Decide how to position that up front.

---

## 4. Defects to fix along the way

**Excitation and emission axes are swapped.** `read_eem` labels axis 0 (271 points,
1 nm, 280–550 nm) as excitation and axis 1 (57 points, 5 nm, 220–500 nm) as emission.
The evidence says the reverse:

- The region where axis1 > axis0 is exactly zero everywhere — the standard Rayleigh
  mask, which only makes physical sense if axis 0 is emission (emission cannot fall
  below excitation).
- The strongest scatter ridge sits at axis1 = axis0 / 2, i.e. second-order Rayleigh at
  em = 2 × ex, again requiring axis 0 = emission.
- The global mean peak is at axis0 = 340 nm, axis1 = 230 nm. Read correctly that is
  ex 230 / em 340 — the tryptophan-like peak. Read as labelled it is impossible.
- Fluorometers conventionally scan emission finely (1 nm) and excitation coarsely
  (5 nm), matching 271 vs 57.

No prediction changes (it is a relabelling), but `parafac_loadings.npz` stores emission
loadings under the key `excitation`, and any fluorophore assignment would be wrong. The
README repeats the error.

**Other items**

| Location | Issue |
|---|---|
| `scripts/process_raw_data.py:203` | `float(row[col])` raises an uncaught `ValueError` on a non-numeric required column instead of recording `incomplete_parameters` |
| `src/eem_water_quality/ml.py:111` | Skip row omits the `"model"` key that line 96 includes, producing ragged and order-unstable CSV columns |
| `src/eem_water_quality/ml.py:152` | Recomputes `builder.transform(eem[test])` although `val is test` and the prediction already exists — costly for raw-EEM and non-negative PARAFAC |
| `pyproject.toml` | `line-length = 100` is unenforced (E501 is not in ruff's default select set); 7 lines exceed it |
| `src/eem_water_quality/artifacts.py:71` | `git rev-parse` runs in the process CWD, so provenance is wrong when invoked from outside the repo |

**Data mapping.** 239 of ~1001 raw EEM files are discarded. The dominant bucket — 203
files under "no laboratory row for station/month/round" — is not randomly distributed:
it concentrates on seven stations (ha2, ha3, ho1, ho2, ho4, ho5, ho6) and on the `-2`
through `-5` suffixes. That is the signature of a filename convention `_parse_name` may
be misreading as a round number; `_parse_name`'s own docstring notes that literal labels
such as `Ho1-1` exist. Verify against the raw workbook before accepting the 760-sample
dataset.

**Reproducibility.** `runs/` is gitignored, so the artifacts the README cites
(`runs/selected_4targets_lr_xgb_no_leakage/`, `runs/original_eem_comparison.csv`) do not
exist for anyone cloning the repository. The raw data directory is also absent, so
`process_raw_data.py` cannot be re-run. Meanwhile `eem.npy` (46 MB) is committed
directly; consider Git LFS or DVC.

---

## 5. Deprioritise

**The deep-learning path.** ResNet10/ResNet18 carry millions of parameters against 760
samples with 45 effective independent units. They were never evaluated in the README
table. PARAFAC plus a simple model is the better bet at this sample size; revisit only
if the dataset grows substantially.

**More samples per station.** The binding constraint is **45 stations**, not 760
samples. Roughly 20 new sites would do more for demonstrated generalisation than a
thousand new samples at existing ones.

---

## 6. Recommended sequence

```
Phase 0  ──►  Phase 2  ──►  Phase 1  ──►  Phase 3  ──►  Phase 4  ──►  Phase 5
(2 days)     (1 day)      (1-2 wks)     (1-2 wks)     (1 wk)       (3 days)
             decides
           everything
```

Phase 2 is placed early on purpose: it is one day of work and it determines whether the
R² ≈ 0.3 reflects DOM chemistry or session drift. Everything after it is optimisation on
a foundation that has not yet been verified.

---

## Appendix — Reproducing these numbers

```bash
# after Phase 0 lands:
PYTHONPATH=src python -m eem_water_quality ml \
  --data data/processed --output runs/grouped_cv \
  --targets BOD COD TOC BOD_COD \
  --split group --group-col Point --seed 42 \
  --models linear xgboost \
  --experiments EEMpca EEMpca_SS_EC --pca-components 30
```

Dataset facts used above: 760 samples, 45 stations, 12 months, 1–51 samples per station
(median 12); targets correlated at r(TOC,COD) = 0.891, r(TOC,BOD) = 0.781,
r(BOD,COD) = 0.804; 14 samples with BOD > COD; median BOD 2.0, COD 6.0, TOC 3.6 mg/L.
