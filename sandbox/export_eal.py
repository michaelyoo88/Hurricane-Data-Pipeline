"""Hurricane EAL per county for the group file: fips, hazard, crop_eal_usd, property_eal_usd, people_eal_usd, total_eal_usd.

Model (sandbox round 6/7 recommendation), fit on ALL 194 storms 2000-2024:
  property  average of (two-part GLM on J2 + latitude) and (wind / surge / rain two-part models),
            trained on KNN-filled property labels (suspicious zeros at 25+ m/s, cautious "expected" amount)
  crop      two-part GLM on J2 + latitude, exposure = farm + timber sales, trained on KNN-filled crop labels
            (zeros at 17.5+ m/s with farms, half of 10 same-storm neighbors recorded crop damage)
  people    two-part GLM on J2 + latitude, exposure = population; target = direct deaths + injuries / 10,
            priced at FEMA's value of a statistical life in 2023 dollars (no imputation)
EAL = every storm x county prediction 2000-2024, priced on 2024 exposure, summed per county / 25 seasons.
All dollars are 2023 dollars. Counties no storm came within 500 km of get 0.
Out: results/hurricane_eal_by_county.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from knn_explore import knn_stats
from knn_impute import K12_IMP, add_imputation
from lib import J2, load, predict, two_part

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from common import VSL, cpi_factors   # noqa: E402

N_SEASONS = 25
VSL_2023 = VSL * cpi_factors()[2024]

df = load()
df["log_value_pc"] = np.log(df.bldg_value / df.population)
everything = np.ones(len(df), bool)

# ---- KNN-filled labels ----
nb = knn_stats(df, "prop_usd")
df["nb_share"], df["nb_median_pos"] = nb.nb_share, nb.nb_median_pos
p_flag, p_add = add_imputation(df, 25.0, "expected")                    # sets prop_usd_imp, prop_wind_usd_imp
nc = knn_stats(df, "crop_usd")
c_flag = ((df.crop_usd == 0) & (df.farm_timber_value > 0) & (df.wind_ms >= 17.5) & (nc.nb_share >= 0.5) & nc.nb_median_pos.notna()).to_numpy()
c_add = np.where(c_flag, (nc.nb_share * nc.nb_median_pos).fillna(0) * df.farm_timber_value, 0.0)
df["crop_usd_imp"] = df.crop_usd + c_add

# ---- fit on all storms, predict every storm x county ----
FEATS = J2 + ["pop_lat"]
prop_ratio = predict(df, everything, K12_IMP)
crop_ratio = two_part(df, everything, "crop_usd_imp", features=FEATS, exposure="farm_timber_value", cap=10.0)
people_ratio = two_part(df, everything, "people_equiv", features=FEATS, exposure="population", cap=1.0)

rows = df[["fips"]].assign(
    property=prop_ratio * df.bldg_value_2024,
    crop=crop_ratio * df.farm_timber_value_2024,
    people=people_ratio * df.population_2024 * VSL_2023)
eal = rows.groupby("fips")[["crop", "property", "people"]].sum() / N_SEASONS

counties = pd.read_csv(ROOT / "final" / "counties.csv", dtype={"fips": str}, usecols=["fips"])
out = counties.set_index("fips").join(eal).fillna(0.0).round(0).astype("int64")
out = out.rename(columns={"crop": "crop_eal_usd", "property": "property_eal_usd", "people": "people_eal_usd"})
out["total_eal_usd"] = out.crop_eal_usd + out.property_eal_usd + out.people_eal_usd
out.insert(0, "hazard", "hurricane")
out = out.reset_index()[["fips", "hazard", "crop_eal_usd", "property_eal_usd", "people_eal_usd", "total_eal_usd"]]

# ---- checks ----
assert len(out) == 3_109 and out.fips.is_unique and out.fips.str.len().eq(5).all()
assert (out[["crop_eal_usd", "property_eal_usd", "people_eal_usd"]] >= 0).all().all()
assert (out.total_eal_usd == out.crop_eal_usd + out.property_eal_usd + out.people_eal_usd).all()
out.to_csv(HERE / "results" / "hurricane_eal_by_county.csv", index=False)

rec = {"crop": df.crop_usd.sum(), "property": df.prop_usd.sum(), "people": (df.people_equiv * VSL_2023).sum()}
print(f"wrote results/hurricane_eal_by_county.csv: {len(out):,} counties, {(out.total_eal_usd > 0).sum():,} with EAL > 0")
print(f"KNN-filled: property {p_flag.sum()} records (+${p_add.sum() / 1e9:.2f}B), crop {c_flag.sum()} records (+${c_add.sum() / 1e6:.1f}M)")
for c in ["crop", "property", "people"]:
    print(f"  {c:9s} EAL ${out[f'{c}_eal_usd'].sum() / 1e9:6.2f}B/yr  (recorded average, that year's exposure: ${rec[c] / N_SEASONS / 1e9:.2f}B/yr)")
print(f"  total     EAL ${out.total_eal_usd.sum() / 1e9:6.2f}B/yr")
print(out.sort_values("total_eal_usd", ascending=False).head(5).to_string(index=False))
