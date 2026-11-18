# Sandbox: model experiments (property damage)

Separate work area. Nothing here changes `pipeline/` or `models/`. Plain numpy / pandas, except the
LightGBM challenger (needs `pip install lightgbm`).

## How to run (from the project folder)

    python3 sandbox/make_table.py            # table + splits -> sandbox/data/table.csv
    python3 sandbox/experiments.py round1a   # also round1b, round2, round3a, round3b, round4, round5
    python3 sandbox/lgbm_challenger.py dev   # LightGBM on the same dev folds
    python3 sandbox/compare.py               # finalists with bootstrap error bars (dev storms)
    python3 sandbox/lgbm_challenger.py test  # LightGBM predictions for the locked test
    python3 sandbox/final_test.py            # the locked test (run once, after CHOICE.md)

## Test design

- 194 storms, 2000-2024. Split by whole storm, stratified by storm tier and recorded damage.
- Dev: 134 storms (9 Cat 3+). All experiments use 3 x 5-fold on these, storms held out.
- Locked test: 60 storms (6 Cat 3+: Harvey, Irma, Wilma, Ivan, Milton, Idalia). Opened once, after the
  final model was written down in CHOICE.md.
- Scores: d2 = Tweedie deviance explained vs one national loss ratio; predicted / recorded in total and
  per storm tier; typical storm miss; storm rank correlation; ROC-AUC; top-10% capture; within 3x.

## What we learned on the dev storms (results/experiments.csv, finalists_dev.csv)

- Balanced folds reversed an earlier finding. The base model does not under-predict major hurricanes
  (old lopsided folds: 0.34x). It over-predicts them: 2.35x.
- Cause: the wind term grows without limit. In counties with 50+ m/s wind it predicted $414B vs $93B recorded.
- Fix that worked: hold the wind effect flat above 50 m/s. Total 0.97, tiers 1.08 / 1.20 / 0.86, d2 0.82.
  Cap level matters: 45 -> majors at 0.57x, 55 -> 1.40x.
- Small or no gain on top of the cap: right-of-track, damage-type split, tier correction, storm size.
- Hurt: storm random effect (3x too high), dollar weights (weak storms 1.8x), stacking storm + track features.
- Suspect zeros treated as missing: no gain.
- LightGBM monotone, fixed settings: d2 0.75 vs 0.82, tiers less balanced.
- Simplifying: 4 inputs (capped wind, distance, coastal, coastal x wind) score the same as 6, 7 or 9.
  Without coastal x wind, majors drop to 0.55x.

## Locked test (results/final_test.csv), 60 storms, $115.7B recorded property

| Model | d2 (95% interval) | Predicted / recorded | TD/TS | Cat 1-2 | Cat 3+ | Storm rank corr | ROC-AUC |
|---|---|---|---|---|---|---|---|
| J2 FINAL: 4 inputs | 0.38 (-0.03 to 0.78) | 1.29 | 4.1 | 1.6 | 1.2 | 0.87 | 0.92 |
| Old base (uncapped) | 0.43 (0.08 to 0.79) | 1.43 | 4.1 | 1.4 | 1.4 | 0.87 | 0.92 |
| Damage table | 0.34 (-0.37 to 0.78) | 1.42 | 7.8 | 2.3 | 1.2 | 0.87 | 0.86 |
| Capped, 7 features | 0.42 | 1.33 | 4.1 | 1.5 | 1.3 | 0.87 | 0.92 |
| Capped + right of track | 0.30 | 1.39 | 5.1 | 1.8 | 1.3 | 0.87 | 0.92 |
| Capped, by damage type | 0.43 | 1.02 | 3.5 | 1.4 | 0.9 | 0.87 | 0.92 |
| Capped + tier correction | 0.47 | 1.64 | 3.9 | 1.4 | 1.7 | 0.87 | 0.92 |
| LightGBM monotone | 0.37 (-0.09 to 0.79) | 1.18 | 6.3 | 2.7 | 0.9 | 0.87 | 0.91 |

- No model is distinguishable from another on the test: every interval overlaps. Ranking skill is identical.
- The 4-input model is as good as anything more complex, including LightGBM. That supports choosing it.
- The cap's benefit was smaller on the test (majors 1.4x -> 1.2x) than on dev (2.35x -> 0.86x).
- d2 fell from 0.82 (dev) to about 0.4 (test). The test is dominated by storms no wind model can get:
  Harvey (rain; $63B recorded, $3.6B predicted) and Milton ($1.75B recorded, $33B predicted; labels incomplete).
- Weak storms are 4x over-predicted for every model: the 44 test TD/TS storms recorded only $0.6B.

## Not solved

- Storm-to-storm error is the limit (typical storm off by 15-19x on the test). It comes from rain / surge the
  inputs do not describe and from incomplete labels, not from the model form.
- The recipe was tuned on property only. Applied as-is to crop and people it is not better than the old base
  (results/final_test_all_categories.csv); those need their own dev pass.
- One 60-storm test is a weak judge. A repeated all-storm cross-validation of the fixed final model would give
  tighter numbers (slightly optimistic, since 134 of the storms were used to choose it).

## Round 6: ideas from hurricane-loss research (2026-11-17)

The locked test was already opened in round 5, so it can no longer choose a model. Round 6 ideas were
screened on the dev storms, then checked on a FRESH 3 x 5-fold over all 194 storms (`fold_all_r*`, models
fixed in advance), with the locked test shown for context only.

    python3 sandbox/experiments.py round6      # also round6b
    python3 sandbox/compare.py round6 dev
    python3 sandbox/compare.py round6 all
    python3 sandbox/round6_summary.py          # locked test (context) + results/round6_summary.png

| Idea (source / practice) | Dev D2 | All-storm D2 | Paired diff vs J2, all storms | Verdict |
|---|---|---|---|---|
| K0 J2 reference, 4 inputs | 0.82 | 0.70 | - | - |
| K1 asymmetric wind: + half the forward speed on the right side (standard wind-field practice) | 0.82 | 0.68 | -0.021 (-0.073 to +0.024) | No gain; weak storms worse |
| K2 one-equation Tweedie GLM (insurance pricing standard) | 0.81 | 0.70 | -0.001 (-0.054 to +0.037) | Same skill, totals 1.9x too high |
| K4b Emanuel (2011) curve, thresholds fitted per coast (CLIMADA-style) + floor | 0.79 | 0.71 | +0.007 (-0.032 to +0.032) | Same skill; worse at spotting damage (ROC 0.79); weak storms 3x |
| K5 + region of the country (pooled) | 0.78 | 0.69 | -0.018 (-0.164 to +0.122) | Better typical storm, majors too low; unstable |
| K6 + latitude (vulnerability gradient) | 0.81 | 0.70 | -0.003 (-0.079 to +0.053) | Same skill, best tier balance |
| K7 + county clusters (professor's idea, k-means on county traits) | 0.81 | - | - | No gain on dev; not carried forward |
| K9 average of J2 and the damage-type model (forecast combination) | 0.82 | 0.72 | +0.015 (+0.002 to +0.025) | Small gain, real |
| K12 average of J2+latitude and the damage-type model | 0.82 | 0.73 | +0.026 (+0.000 to +0.044) | Best; small gain, real |
| K10/K11 + reporting era | 0.80 / 0.81 | - | - | No gain; not carried forward |

Locked test (context only): J2 0.38, K6 0.46, K9 0.43, K12 0.46, K4b 0.52; all intervals overlap.

- Averaging two simple models is the only idea that beat J2 by more than the noise on fresh folds.
  The gain is small (+0.02 to +0.03 D2).
- Latitude makes the tiers more balanced (northern counties lose less per unit of wind in the records).
- The physics curve, the one-equation Tweedie, asymmetric wind, regions, clusters and era all tie or lose.
- Scores move a lot between checks (D2 0.82 dev, 0.70 all storms, ~0.4 test): which giant storms fall
  in which fold matters more than any modeling choice. Every claim needs its interval.
- `_to_delete/` holds a backup copy of the table from before round 6 (identical splits); safe to delete.

## Round 7: county clusters and KNN imputation of missing damage (2026-11-17)

    python3 sandbox/knn_explore.py      # clusters, KNN neighbor stats, checks A-C
    python3 sandbox/knn_impute.py       # retrain on imputed labels, score against real records
    python3 sandbox/knn_crop_check.py   # crop flags vs USDA crop insurance claims
    python3 sandbox/knn_chart.py        # results/knn_validation.png

**Clusters** (k-means on density, value per person, farm share, coastal; k = 3 by silhouette 0.35):
rural farm counties (808), coastal counties (372), other inland (1,380). Share of strong-wind hits
(25+ m/s) with any damage record: 48% rural farm, 63% coastal, 53% inland. Rural counties record less.
As a model feature, clusters added nothing (round 6, K7).

**KNN within a storm** (10 most similar counties hit by the same storm: wind, distance, coast, density, value per person):

| Check | Neighbors | J2 model |
|---|---|---|
| A: hide a county's own record; predict whether it had damage (ROC-AUC) | 0.94 | 0.89 |
| A, strong-wind counties only | 0.83 | 0.61 |
| B: hide a damaged county's amount; rank correlation with neighbors' median ratio | 0.66 | 0.32 |

Neighbors use other counties' records from the SAME storm, so this works for filling gaps in past storms,
not for forecasting a new storm.

**Outside checks of the flags** (flag = $0 record, strong wind, half of 10 neighbors recorded damage):
- Property, old wind-only labels: flagged zeros later showed surge / flood damage 7% (25+ m/s) and 8% (33+ m/s)
  of the time, vs 4% and 1% for similar unflagged zeros.
- Crop, vs USDA crop insurance: flagged zeros had a hurricane claim 56% (17.5+ m/s) and 72% (25+ m/s) of the
  time, vs 25% and 44% unflagged.

**Effect on the model** (results/knn_impute_models.csv; scored against REAL records):
- Property: 223 zeros flagged, $3.0B imputed (0.7% of recorded). D2 0.698 -> 0.702 (all storms), 0.818 -> 0.821
  (dev); K12 0.724 -> 0.727. Inside the noise. Full vs cautious imputation, or leaving the zeros out: the same.
- Crop: only 118 flagged, $13M imputed, vs $16.3B recorded. Insurance paid in 25% of farm + strong-wind
  pairs where NOAA shows $0 (about 1,800 pairs), but most sit in storms where neighbors ALSO show $0,
  so KNN cannot see them.

**Takeaway:** KNN is good at finding hidden damage (2x+ enrichment, confirmed by two outside sources), but
the gaps it can fill are small in dollars. Most missing damage is missing for whole storms or whole areas,
which a same-storm neighbor method cannot detect. Useful as a documented data-quality step and as evidence
that labels are a lower bound; not a performance fix.

## Round 8: five model families on the agreed error metrics (2026-11-18)

    python3 sandbox/round8.py fit            # held-out predictions -> data/round8/*.npz (optionally list model codes, e.g. fit G4 Y1)
    python3 sandbox/round8.py score 130      # bootstrap in chunks (cached per model); rerun until the table prints

Metrics agreed for the class: **MAE of county EAL** ($M/yr, 2,560 counties), **bias ratio** (total and by
storm tier), **normalized Gini** of county EAL, **PR-AUC** (any damage, storm x county), **Tweedie D²** (p = 1.5).
Same design as rounds 6-7: 3 x 5-fold over all 194 storms, trained on KNN-filled labels, scored against recorded
damage; 1,000 storm bootstrap draws, PAIRED difference vs R0 (K12, the current model). Settings fixed in advance.
Out: results/round8_metrics.csv (all metrics + intervals), round8_summary.png, round8_stack_weights.csv.

| Model | D² | MAE EAL $M | Gini | PR-AUC | Bias (TS / C1-2 / C3+) | Real difference vs K12 |
|---|---|---|---|---|---|---|
| R0 K12 (current) | 0.732 | 6.27 | 0.966 | 0.408 | 0.76 (1.71 / 0.97 / 0.65) | - |
| T5 Tweedie GLM, 7 capped inputs, recalibrated | 0.746 | 7.32 | 0.964 | 0.425 | 1.04 (3.94 / 1.69 / 0.70) | PR-AUC better, MAE worse |
| G4 LightGBM p=1.3, 15 leaves, recalibrated | 0.712 | 4.96 | 0.980 | 0.419 | 0.93 (1.83 / 1.69 / 0.63) | none (all ties) |
| Y4 Hybrid (XGBoost chance x gamma size) + by-type | 0.750 | 6.23 | 0.966 | 0.449 | 0.74 (1.57 / 0.98 / 0.63) | PR-AUC better (+0.04) |
| D0 Two-part gamma with full distribution | 0.708 | 6.93 | 0.960 | 0.386 | 0.91 (2.02 / 1.08 / 0.81) | MAE, PR-AUC worse |
| D2 Two-part lognormal, modeled spread | 0.697 | 27.2 | 0.966 | 0.423 | 4.68 | MAE much worse |
| E2 Equal average: K12 members + hybrid | 0.745 | 6.45 | 0.965 | 0.433 | 0.80 (1.72 / 1.01 / 0.69) | D² and PR-AUC better, MAE worse (+0.03 to +0.47) |
| E3 Stacked, learned weights (nested CV) | 0.752 | 6.19 | 0.973 | 0.443 | 0.90 (2.05 / 1.31 / 0.71) | PR-AUC better |

- **Dollars: no family beats K12 beyond the noise.** D² and county-EAL MAE intervals overlap for every
  sensible model. LightGBM has the best point MAE (4.96 vs 6.27) and Gini (0.98), but the intervals are wide and
  Cat 1-2 storms run 1.4-1.7x high.
- **Spotting damage is where the real gain is.** A tree classifier on the richer inputs (track, storm size,
  intensity change) lifts PR-AUC by +0.03 to +0.04 (Y1, Y4, E2, E3). It comes from the inputs: the same XGBoost
  on the 4 + latitude inputs (Y3) gains nothing.
- **Tweedie GLM is sensitive to its power:** p = 1.8 blows up (totals 14x), p = 1.2 ties, p = 1.5 needs a
  total recalibration and still loses on MAE and Gini with the 4 + latitude inputs.
- **Distributional:** the lognormal's mean exp(m + s²/2) overshoots 5-11x (large spread); recalibrating fixes
  totals but wrecks D². The gamma two-part's 10-90% band holds 91% of damaged rows (close to the 80% target, a bit wide),
  but only **44% of storm totals**: rows within a storm are treated as independent, so the storm-to-storm
  uncertainty (the main error source) is missing. A shared storm-level shock is needed before intervals can be reported.
- **Stacking:** learned weights jump between folds (LightGBM 0.07-0.98, hybrid 0.02-0.84) and score the same as
  an equal average. Prefer the simple average.
- **Candidate upgrade: Y4** = K12 with member 1's logistic chance swapped for XGBoost on the rich inputs.
  Same dollars (D² 0.750, MAE 6.23, same tiers), clearly better at spotting damage. Only one piece changes.
