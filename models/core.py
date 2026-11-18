"""Reusable model pieces. Each takes the table and a mask of rows to FIT on, and returns a
predicted damage RATIO (damage / exposure) for EVERY row, so held-out rows are easy to score."""
import numpy as np
import pandas as pd

from glm import add_intercept, fit_gamma, fit_logistic

EDGES = [0, 10, 17.5, 25, 33, 43, 50, 999]     # wind bins, m/s (33 = hurricane force)


def two_part(df, fit, target, exposure, features, cap=1.0, alpha=1.0, rows=None, group="storm_id"):
    """chance of any damage (logistic, all rows) x size if damaged (gamma on the ratio, damaged rows).

    alpha sets how much each storm counts in the size model: weight = exposure / (storm's exposure)^alpha
      alpha = 1    every storm counts equally
      alpha = 0    rows count by dollars at risk (storms with big footprints dominate)
      alpha = 0.5  halfway
    rows  = optional mask of rows the model applies to (others are predicted 0)
    group = what "a storm" is for the weights (a year, for the county-year model)
    """
    usable = (df[exposure] > 0).to_numpy()
    if rows is not None:
        usable = usable & rows
    ratio = np.where(usable, df[target] / df[exposure].where(df[exposure] > 0, 1), 0.0)
    fit = fit & usable
    X = df[features].to_numpy(dtype=float)
    mean, sd = X[fit].mean(axis=0), X[fit].std(axis=0)
    X = (X - mean) / np.where(sd > 0, sd, 1)                      # scale with the fit rows only

    b_chance = fit_logistic(X[fit], (ratio[fit] > 0).astype(float))
    pos = fit & (ratio > 0)
    e = df.loc[pos, exposure]
    weight = e / e.groupby(df.loc[pos, group]).transform("sum") ** alpha
    b_size = fit_gamma(X[pos], ratio[pos], weight.to_numpy())

    X1 = add_intercept(X)
    chance = 1 / (1 + np.exp(-np.clip(X1 @ b_chance, -30, 30)))
    pred = chance * np.exp(np.clip(X1 @ b_size, -40, 10))
    return np.where(usable, np.minimum(pred, cap if cap else ratio[pos].max()), 0.0)


def damage_table(df, fit, target, exposure, min_storms=3):
    """Loss ratio by box (wind bin x coastal) = total damage / total exposure in the box, from fit rows.
    Boxes with fewer than min_storms storms use the wind bin's ratio regardless of coast."""
    wind_bin = pd.cut(df.wind_ms, EDGES, labels=False, right=False)
    coast = df.coastal.astype(int)
    d = pd.DataFrame({"w": wind_bin, "c": coast, "storm": df.storm_id, "y": df[target], "e": df[exposure]})[
        fit & (df[exposure] > 0).to_numpy()]
    grid = pd.MultiIndex.from_product([range(len(EDGES) - 1), [0, 1]])
    g = d.groupby(["w", "c"])
    ratio = (g.y.sum() / g.e.sum()).reindex(grid)
    n_storms = g.storm.nunique().reindex(grid).fillna(0)
    gw = d.groupby("w")
    by_wind = (gw.y.sum() / gw.e.sum()).reindex(range(len(EDGES) - 1)).ffill().fillna(0)
    table = ratio.where(n_storms >= min_storms, pd.Series(by_wind.reindex(grid.get_level_values(0)).to_numpy(), index=grid))
    return table


def table_lookup(table, df):
    wind_bin = pd.cut(df.wind_ms, EDGES, labels=False, right=False)
    return table.reindex(pd.MultiIndex.from_arrays([wind_bin, df.coastal.astype(int)])).to_numpy()
