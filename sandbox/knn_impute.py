"""Round 7, part 2: fill in suspicious $0 records with KNN, retrain, and score against the REAL records.

Suspicious zero (rule fixed before modeling): $0 recorded, 25+ m/s wind at the county, and at least half of
its 10 most similar counties hit by the same storm recorded damage. Sensitivity: 33+ m/s.
Imputed amount, two versions:
  expected   neighbor share with damage x neighbors' median loss ratio x building value   (cautious)
  full       neighbors' median loss ratio x building value                                  (assumes damage)
Variants are trained on imputed labels (or with suspicious zeros left out) and ALWAYS scored against the
recorded labels of held-out storms; scores against the imputed labels are shown as a sensitivity.
Imputation of a training row only uses rows of the same storm, so no held-out storm's label leaks in.
Out: results/knn_impute_models.csv, results/knn_impute_flags.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from lib import J2, capped, cross_validate, load

HERE = Path(__file__).resolve().parent


def add_imputation(df, min_wind, how):
    flag = ((df.prop_usd == 0) & (df.wind_ms >= min_wind) & (df.nb_share >= 0.5) & df.nb_median_pos.notna()).to_numpy()
    ratio = df.nb_median_pos.fillna(0) * (df.nb_share.fillna(0) if how == "expected" else 1.0)
    add = np.where(flag, ratio * df.bldg_value, 0.0)
    df["suspect"] = flag
    df["prop_usd_imp"] = df.prop_usd + add
    df["prop_wind_usd_imp"] = df.prop_wind_usd + add          # unrecorded damage counted as wind damage
    return flag, add


K12 = {"kind": "average", "members": [{"features": J2 + ["pop_lat"]}, {"features": capped(), "by_type": True}]}
K12_IMP = {"kind": "average", "members": [{"features": J2 + ["pop_lat"], "impute": True}, {"features": capped(), "by_type": True, "impute": True}]}
VARIANTS = [  # label, imputation rule (min wind, amount), model spec
    ("M0 J2, recorded labels", None, {"features": J2}),
    ("M1 J2, suspicious zeros left out", (25.0, "expected"), {"features": J2, "exclude_col": "suspect"}),
    ("M2 J2, KNN impute expected", (25.0, "expected"), {"features": J2, "impute": True}),
    ("M3 J2, KNN impute full", (25.0, "full"), {"features": J2, "impute": True}),
    ("M4 J2, KNN impute expected, 33+ m/s", (33.0, "expected"), {"features": J2, "impute": True}),
    ("M5 K12 average, recorded labels", None, K12),
    ("M6 K12 average, KNN impute expected", (25.0, "expected"), K12_IMP),
]
if __name__ == "__main__":
    df = load()
    st = pd.read_csv(HERE / "data" / "knn_stats.csv", dtype={"fips": str})
    df = df.merge(st[["storm_id", "fips", "nb_share", "nb_median_pos"]], on=["storm_id", "fips"], how="left", validate="one_to_one")
    pd.set_option("display.width", 250)
    rows, flags = [], []
    for label, rule, spec in VARIANTS:
        if rule:
            flag, add = add_imputation(df, *rule)
            flags.append({"rule": f"{rule[0]}+ m/s, {rule[1]}", "rows_flagged": int(flag.sum()), "storms": df.storm_id[flag].nunique(),
                          "added_$B": add.sum() / 1e9, "added_pct_of_recorded": 100 * add.sum() / df.prop_usd.sum(),
                          "by_cluster_rows": df.loc[flag].groupby(df.coastal[flag]).size().rename({0: "inland", 1: "coastal"}).to_dict()})
        else:
            df["suspect"], df["prop_usd_imp"], df["prop_wind_usd_imp"] = False, df.prop_usd, df.prop_wind_usd
        for scope in ("all", "dev"):
            m, _ = cross_validate(df, spec, scope=scope)
            m_imp, _ = cross_validate(df, spec, scope=scope, eval_col="prop_usd_imp") if rule else (m, None)
            rows.append({"model": label, "scope": scope, **{k: m[k] for k in ["d2", "pred_rec", "tier_ts", "tier_c12", "tier_c3", "tier_miss",
                                                                              "storm_miss", "rho_storm", "roc_auc", "top10", "within3x"]},
                         "d2_vs_imputed_labels": m_imp["d2"]})
        print(f"  done {label}", flush=True)
    out = pd.DataFrame(rows)
    out.round(3).to_csv(HERE / "results" / "knn_impute_models.csv", index=False)
    pd.DataFrame(flags).round(3).to_csv(HERE / "results" / "knn_impute_flags.csv", index=False)
    print("\nimputation rules:"); print(pd.DataFrame(flags).round(2).to_string(index=False))
    for scope in ("all", "dev"):
        print(f"\n{scope.upper()} storms (scored against RECORDED labels; last column = against imputed labels):")
        print(out[out.scope == scope].drop(columns="scope").set_index("model").round(3).to_string())
