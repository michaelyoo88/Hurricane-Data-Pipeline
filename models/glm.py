"""The two small models behind the basic model, in plain numpy, plus two metrics.

  fit_logistic   chance that a row has ANY damage (0/1 target)
  fit_gamma      size of damage when there is some (positive, very skewed target; log link,
                 so effects multiply and predictions can never be negative)
Both add a small "ridge" penalty that keeps coefficients stable when features overlap.
"""
import numpy as np
import pandas as pd


def add_intercept(X):
    return np.column_stack([np.ones(len(X)), X])


def fit_logistic(X, y, l2=1.0, iters=50):
    """Logistic regression by Newton's method."""
    X = add_intercept(X)
    b = np.zeros(X.shape[1])
    penalty = l2 * np.eye(X.shape[1]); penalty[0, 0] = 0          # never shrink the intercept
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        step = np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + penalty, X.T @ (y - p) - penalty @ b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def fit_gamma(X, y, w, l2=1.0, iters=200):
    """Gamma regression with log link, by repeated weighted least squares.
    w = how much each row counts (we use it to make every STORM count equally)."""
    X = add_intercept(X)
    w = w / w.mean()
    penalty = l2 * np.eye(X.shape[1]); penalty[0, 0] = 0
    b = np.zeros(X.shape[1]); b[0] = np.log(np.sum(w * y) / np.sum(w))

    def loss(b):
        mu = np.exp(np.clip(X @ b, -40, 10))
        return 2 * np.sum(w * ((y - mu) / mu - np.log(y / mu))) + b @ penalty @ b

    current = loss(b)
    for _ in range(iters):
        eta = np.clip(X @ b, -40, 10)
        z = eta + (y - np.exp(eta)) / np.exp(eta)                   # "working" target
        step = np.linalg.solve((X * w[:, None]).T @ X + penalty, X.T @ (w * z)) - b
        for _ in range(30):                                         # halve the step if the fit got worse
            new = loss(b + step)
            if new <= current:
                break
            step /= 2
        b, previous, current = b + step, current, new
        if abs(previous - current) < 1e-9 * abs(previous):
            break
    return b


def pr_auc(y, score):
    """Area under the precision-recall curve; tied scores count as one step."""
    d = pd.DataFrame({"y": np.asarray(y, dtype=float), "s": np.asarray(score, dtype=float)})
    g = d.groupby("s", sort=True).y.agg(["sum", "size"]).iloc[::-1]
    precision = g["sum"].cumsum() / g["size"].cumsum()
    return float((precision * g["sum"] / d.y.sum()).sum())


def roc_auc(y, score):
    """ROC-AUC: chance that a random damaged row is scored above a random undamaged one (0.5 = coin flip)."""
    y = np.asarray(y, dtype=bool)
    ranks = pd.Series(np.asarray(score, dtype=float)).rank().to_numpy()
    n_pos, n_neg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(a, b):
    """Rank correlation."""
    return float(pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank()))
