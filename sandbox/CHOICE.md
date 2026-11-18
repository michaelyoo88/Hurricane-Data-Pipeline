# Final model choice (written BEFORE the locked test was opened)

Date: 2026-11-16. Chosen on the 134 dev storms only (3 x stratified 5-fold, whole storms held out).

**Final model: J2, a two-part GLM with four inputs**
- wind_cap: wind at the county, held flat above 50 m/s (Cat 3)
- log_dist_km: closest distance to the storm center
- coastal: coastal county yes/no
- coastal_x_wind_cap: extra effect of wind above tropical-storm force in coastal counties (capped the same way)
- Part 1 logistic (any damage?), part 2 gamma on damage / building value (how much?), every storm weighted equally.

**Rule used to choose:** best deviance explained (d2) among models with balanced storm tiers; if several are
within the bootstrap noise of each other, take the one with the fewest inputs.

**Why this one:** d2 0.82 on dev, same as the 6-, 7- and 9-feature versions (paired differences inside the
noise); tiers at 1.07 / 1.14 / 0.81 of recorded; total at 0.91.

**Comparison models scored on the locked test for context only:** old base (uncapped wind), damage table,
C4, G2, D7, E1, E2, LightGBM monotone. None of them can replace the choice after seeing the test.
