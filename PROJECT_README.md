# Hurricane Risk by County: Project Overview

BC Data Science Capstone (CSCI4900, Fall 2026). Hurricane part: Michael. Wildfire: Lukas. Dashboard: Nathan.
Last updated 2026-11-18. For how to run the data pipeline, see `README.md`.

## Goal

Estimate each US county's **expected annual hurricane loss (EAL, $ per year)** for property, crops and
people, built bottom-up from storm physics and recorded damage. Then compare it with **FEMA's National
Risk Index** hurricane EAL to find where, and why, FEMA's estimate differs:
exposure x how often a county is hit x share of value lost per hit.

Group deliverable: one row per county,
`fips, hazard, crop_eal_usd, property_eal_usd, people_eal_usd, total_eal_usd` (2023 dollars).

## General direction

1. **Learn** how storms turn into damage from recorded losses (NY Fed / NOAA, 2000-2024).
2. **Validate** on storms the model never saw, against those records, judged against simple baselines.
3. **Estimate** each county's EAL from the validated model (2024 exposure).
4. **Compare** with FEMA factor by factor; use recorded and outside data (e.g. crop insurance) to judge
   who is closer where we disagree.

Check 2 is accuracy against real records. Check 4 is not an accuracy test: FEMA is another model, and our
differences from it are the project's findings.

## Data

| Piece | Source | Coverage |
|---|---|---|
| Storm tracks, wind at each county | NOAA HURDAT2 (wind estimated with a Rankine vortex, Willoughby storm size) | 1975-2025, 839 storms |
| Damage ("truth") | NY Fed Losses from Natural Disasters (NOAA Storm Events): hurricane records + surge, coastal flood, flood, flash flood, high wind, tornado records matched to a storm | 1996-2024 |
| Exposure | Census housing and population, FEMA building value per unit, USDA farm, crop and timber value | 2000-2024 |
| Benchmark | FEMA NRI v1.20 (converted from Dec 2024 to 2023 dollars) | one snapshot |

Model table: one row per storm x county within 500 km, 2000-2024: 97,179 rows, 194 storms, 94% with $0.

## Work done

- Data pipeline (`pipeline/`): storms, county wind, damage matched to storms, yearly exposure, FEMA values.
- EDA (`analysis/`): damage distribution, label completeness, matching rules, crop check vs USDA crop insurance.
- About 60 model variants (`models/`, `sandbox/`), all tested by holding out whole storms.
- County EAL file for the group (`sandbox/results/hurricane_eal_by_county.csv`).

## Current model

For property, crops and people separately, two simple steps learned from past storms:

**predicted damage = chance of any damage x share of value lost if damaged x value at risk**

- Chance of damage: logistic regression. Share lost: gamma regression on damage / exposure. Every storm
  weighted equally.
- Inputs: wind at the county (held flat above Category 3), distance to the storm, coastal, coastal x wind,
  latitude.
- Property uses the average of this model and one built from separate wind, surge and rain models.
- Trained on records with KNN-filled gaps (counties with strong wind and a $0 record whose similar
  neighbors, hit by the same storm, recorded damage).
- EAL = predicted damage for every 2000-2024 storm, priced on 2024 values, averaged over 25 seasons.
  It is not a plain average of recorded damage.

## Results

Held out: each storm predicted by a model that never saw it, scored against recorded damage (property).

| What | Result |
|---|---|
| Spots which county-storms had damage (probability step) | ROC-AUC 0.92 |
| Recorded losses in the model's riskiest 10% of counties | 95% |
| Ranks storms by total damage | rank correlation 0.86 |
| Ranks counties by yearly risk | rank correlation 0.77 |
| Deviance explained (skill score for zero-heavy, skewed data) | 0.73 |
| Individual county amounts (counties with $1M+/yr recorded) | typical miss 4.3x; 38% within 3x |
| National total vs recorded | 0.76x (low on giant storms) |

| EAL, $B per year | Ours | FEMA |
|---|---|---|
| Property | 12.27 | 10.86 |
| Crop | 0.97 | 0.44 |
| People | 0.71 | 0.65 |
| Total | 13.95 | 11.94 |

Against FEMA: same county ranking (0.94), typical gap 2.2x, 67% of high-risk counties within 3x.
Our total includes surge and rain damage; FEMA's hurricane EAL does not.

**Bottom line:** the model reliably finds **where** hurricane risk is and **which storms** are worst;
its estimate of **how much** a single county loses is rough.

## Issues and what we did

| Issue | What we did | Result |
|---|---|---|
| 94% of rows have $0 damage | Two-step model: chance of damage, then size | Works well for spotting damage |
| A few storms dominate (Katrina = 27% of dollars, top 3 = 50%) | Weight every storm equally; report by storm tier | Typical storm better; giants still hard |
| Dollars 13-39x too high in early models | Predict share of value lost, weighted by exposure | Fixed most of the bias |
| Hurricane-only records missed water damage (Sandy was $0) | Added surge, flood, wind, tornado records with a tight storm match | Property labels $231B -> $426B |
| Year-based test was unfair (records get less complete each year) | Hold out whole storms instead of years | Train and test share label quality |
| Test groups were lopsided (one held 7 of 15 major storms) | Balanced groups by storm category and damage | Revealed the true error pattern |
| Damage curve grew without limit at high wind | Held wind effect flat above 50 m/s (Cat 3) | Storm tiers balanced on development storms |
| Some $0 records hide real damage | KNN on similar counties in the same storm | Finds hidden damage (confirmed by two outside sources); small dollar effect |
| NOAA under-records crop damage | Checked against USDA crop-insurance payouts | Crop EAL treated as a lower bound |
| FEMA values in Dec 2024 dollars | Converted to 2023 dollars | Comparison no longer off by 4% |

## What worked and what did not

**Worked:** predicting share of value lost with exposure weights; wider damage labels; balanced
storm-held-out testing; capping the wind effect; a simple 4-input model (as good as 6-9 inputs);
averaging two simple models (the only idea that beat the 4-input model beyond the noise:
0.70 -> 0.73); latitude (best balance across storm tiers).

**Did not work (tied or worse):** LightGBM, GAM, random effects per storm, top-down storm totals,
fitted Emanuel damage curve, one-equation Tweedie model, asymmetric wind, region and cluster inputs,
reporting-era adjustment, dollar weighting. Random forest and XGBoost slightly improve the "any damage"
step (ROC-AUC 0.917 -> 0.931), mostly from extra inputs; not yet tested in dollars.

**Main lesson:** fixes to the data and the test design mattered far more than the choice of model.
Results shift more with which giant storms land in which test group than between models.

## Open issues and next steps

- Giant, water-driven storms (Katrina surge, Harvey rain) are missed: no surge or rainfall inputs yet.
- Labels are incomplete, worst for crops and 2023-24; all estimates are lower bounds.
- Locked test storms were used once; future claims rely on all-storm cross-validation or new storms.
- Next: split our EAL into FEMA's three factors (exposure, hit rate, share lost per hit) to say which
  factor FEMA gets wrong; compare like for like (our wind part vs FEMA hurricane); try storm pressure and
  housing age; per-county under-reporting score from KNN.

## Where things are

| Folder | Contents |
|---|---|
| `source/` | Raw downloads, never edited (`SOURCES.md`) |
| `pipeline/` | Builds `clean/` and `final/` tables |
| `final/` | Tables we model on (`final/data_dictionary.csv`) |
| `analysis/` | EDA scripts and outputs |
| `models/` | First model rounds (two-part GLM, damage table, 8-strategy comparison) |
| `sandbox/` | Experiments, final model, group EAL file (`sandbox/README.md` has every round) |
